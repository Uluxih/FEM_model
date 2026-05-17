import numpy as np

# Импортируем модули для поиска критической плоскости
from FEM.Integration_Point_Level.CriticalPlane.criterion import (
    find_critical_plane_shear,
    find_critical_plane_tensile
)
from FEM.Integration_Point_Level.CriticalPlane.tensor import StressTensor
from FEM.Abstract.Integration_Point_Level import ConstitutiveModel


class UbiquitousJointModel3D(ConstitutiveModel):
    """
    Адаптивная модель сплошной среды с фиксацией трещины (Fixed Smeared Crack).
    - До разрушения: Изотропная упругая порода.
    - При разрушении: Ищет критическую плоскость и фиксирует её.
    - После разрушения: Строгая модель Ubiquitous Joint с согласованной матрицей.
    """

    def __init__(self, material):
        super().__init__(material)

        E = self.material.E
        nu = self.material.nu
        jp = self.material.joint_params

        # --- 1. Параметры трещины (включаются ПОСЛЕ фиксации) ---
        self.kn, self.ks, self.kt, self.S = jp['kn'], jp['ks'], jp['kt'], jp['spacing']
        self.c = jp.get('c', 0.0)
        self.phi = np.radians(jp.get('phi', 0.0))
        self.psi = np.radians(jp.get('psi', 0.0))
        self.t_limit = jp.get('t', 0.0)

        # Ограничение на растяжение (Apex correction)
        if self.phi > 0:
            t_max = self.c / np.tan(self.phi)
            self.t_limit = min(self.t_limit, t_max)

        # --- 2. Параметры поиска (Critical Plane Material) ---
        self.cp_material = jp.get('cp_material', None)
        if self.cp_material is None:
            raise ValueError("Для адаптивной модели необходимо передать 'cp_material' в joint_params!")

        # --- 3. Упругая матрица целой породы (действует ДО фиксации) ---
        self.C_rock = self._build_isotropic_compliance(E, nu)
        self.D_rock = np.linalg.inv(self.C_rock)

        # --- 4. Переменные состояния фиксации ---
        self.is_locked = False
        self.fixed_normal = None

        # Заглушки для матриц, которые будут построены при фиксации
        self.R = np.eye(3)
        self.D_eq_local = None
        self.D_eq_global = self.D_rock.copy()
        self.alpha_1 = 0.0
        self.alpha_2 = 0.0

        # Переменные состояния напряжений/деформаций
        self.stress_old = np.zeros(6)
        self.strain_old = np.zeros(6)
        self.stress = np.zeros(6)
        self.strain = np.zeros(6)

    def _build_isotropic_compliance(self, E, nu):
        C = np.zeros((6, 6))
        G = E / (2.0 * (1.0 + nu))
        C[0, 0] = C[1, 1] = C[2, 2] = 1.0 / E
        C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = -nu / E
        C[3, 3] = C[4, 4] = C[5, 5] = 1.0 / G
        return C

    def _build_joint_compliance(self, kn, ks, kt, S):
        C = np.zeros((6, 6))
        C[2, 2] = 1.0 / (kn * S)
        C[4, 4] = 1.0 / (kt * S)
        C[5, 5] = 1.0 / (ks * S)
        return C

    def _build_rotation_matrix(self, n):
        nz = np.array(n) / np.linalg.norm(n)
        if abs(nz[2]) > 0.999:
            nx = np.array([1.0, 0.0, 0.0])
            ny = np.cross(nz, nx)  # <--- ИСПРАВЛЕНИЕ ОШИБКИ ЗДЕСЬ
        else:
            ny = np.cross(nz, [0.0, 0.0, 1.0])
            ny /= np.linalg.norm(ny)
            nx = np.cross(ny, nz)
        nx /= np.linalg.norm(nx)
        return np.column_stack((nx, ny, nz))

    def _rotate_matrix(self, D, R):
        """Вращение матрицы жесткости 6x6"""
        D_glob = np.zeros((6, 6))
        for j in range(6):
            e_g = np.zeros(6);
            e_g[j] = 1.0
            et_g = np.array(
                [[e_g[0], e_g[3] / 2, e_g[5] / 2], [e_g[3] / 2, e_g[1], e_g[4] / 2], [e_g[5] / 2, e_g[4] / 2, e_g[2]]])
            et_l = R.T @ et_g @ R
            ev_l = np.array([et_l[0, 0], et_l[1, 1], et_l[2, 2], 2 * et_l[0, 1], 2 * et_l[1, 2], 2 * et_l[0, 2]])

            sv_l = D @ ev_l
            st_l = np.array([[sv_l[0], sv_l[3], sv_l[5]], [sv_l[3], sv_l[1], sv_l[4]], [sv_l[5], sv_l[4], sv_l[2]]])
            st_g = R @ st_l @ R.T
            D_glob[:, j] = np.array([st_g[0, 0], st_g[1, 1], st_g[2, 2], st_g[0, 1], st_g[1, 2], st_g[0, 2]])
        return D_glob

    def _voigt_to_stresstensor(self, v):
        """Конвертация вектора Фойгта в объект StressTensor"""
        return StressTensor(v[0], v[1], v[2], v[3], v[4], v[5])

    def _lock_plane(self, normal):
        """Процедура фиксации трещины и перестроения матриц жесткости"""
        self.fixed_normal = normal
        self.R = self._build_rotation_matrix(normal)

        # Добавляем податливость трещины к породе
        C_joint_local = self._build_joint_compliance(self.kn, self.ks, self.kt, self.S)
        C_eq_local = self.C_rock + C_joint_local
        self.D_eq_local = np.linalg.inv(C_eq_local)

        # Модули для Return Mapping
        self.alpha_1 = self.D_eq_local[2, 2]
        self.alpha_2 = (self.D_eq_local[4, 4] + self.D_eq_local[5, 5]) / 2.0

        # Глобальная эквивалентная матрица с учетом трещины
        self.D_eq_global = self._rotate_matrix(self.D_eq_local, self.R)
        self.is_locked = True

        print(f" [!] ОБРАЗОВАНИЕ ТРЕЩИНЫ. Нормаль зафиксирована: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")

    def update_state(self, current_strain):
        self.strain = current_strain
        d_strain = current_strain - self.strain_old

        # ==========================================================
        # ЭТАП 1: ПОИСК И ФИКСАЦИЯ (Если трещина еще не образовалась)
        # ==========================================================
        if not self.is_locked:
            # Считаем пробное напряжение как для целой породы
            stress_trial_rock = self.stress_old + self.D_rock @ d_strain
            st = self._voigt_to_stresstensor(stress_trial_rock)

            # Ищем критическую плоскость
            f_sh, n_sh, util_sh = find_critical_plane_shear(st, self.cp_material, mode='3D')
            f_t, n_t, util_t = find_critical_plane_tensile(st, self.cp_material, mode='3D')

            max_f = max(f_sh, f_t)

            if max_f > 0:
                # Порода не выдержала! Фиксируем плоскость.
                best_n = n_sh if f_sh > f_t else n_t
                self._lock_plane(best_n)
            else:
                # Порода держит нагрузку. Возвращаем упругое состояние.
                self.stress = stress_trial_rock
                return self.stress, self.D_rock

        # ==========================================================
        # ЭТАП 2: РАБОТА С ЗАФИКСИРОВАННОЙ ТРЕЩИНОЙ (Return Mapping)
        # ==========================================================
        # Считаем пробное напряжение уже с учетом податливости трещины (D_eq_global)
        sig_tr_v = self.stress_old + self.D_eq_global @ d_strain

        # Перевод в локальные оси трещины
        st_t = np.array([[sig_tr_v[0], sig_tr_v[3], sig_tr_v[5]],
                         [sig_tr_v[3], sig_tr_v[1], sig_tr_v[4]],
                         [sig_tr_v[5], sig_tr_v[4], sig_tr_v[2]]])
        sig_tr_l_t = self.R.T @ st_t @ self.R

        s33, s13, s23 = sig_tr_l_t[2, 2], sig_tr_l_t[0, 2], sig_tr_l_t[1, 2]
        tau = np.sqrt(s13 ** 2 + s23 ** 2)

        # Проверка критериев (на уже зафиксированной плоскости)
        f_s = tau + s33 * np.tan(self.phi) - self.c
        f_t = s33 - self.t_limit

        if f_s <= 0 and f_t <= 0:
            self.stress = sig_tr_v
            return self.stress, self.D_eq_global

        # --- АЛГОРИТМИЧЕСКИЙ ВОЗВРАТ И СОГЛАСОВАННАЯ МАТРИЦА ---
        D_tan_l = self.D_eq_local.copy()

        if f_t > 0 or (f_s > 0 and (s33 + (f_s / self.alpha_2) * self.alpha_1 * np.tan(self.psi)) > self.t_limit):
            # Отрыв
            sig_tr_l_t[2, 2], sig_tr_l_t[0, 2], sig_tr_l_t[1, 2] = self.t_limit, 0, 0
            D_tan_l[2, :] = 0; D_tan_l[:, 2] = 0
            D_tan_l[4, :] = 0; D_tan_l[:, 4] = 0
            D_tan_l[5, :] = 0; D_tan_l[:, 5] = 0
        else:
            # Сдвиг
            tan_phi, tan_psi = np.tan(self.phi), np.tan(self.psi)
            lam = f_s / (self.alpha_2 + self.alpha_1 * tan_phi * tan_psi)

            sig_33_new = s33 - lam * self.alpha_1 * tan_psi
            tau_new = tau - lam * self.alpha_2

            apex_stress = self.c / tan_phi if self.phi > 0 else float('inf')

            if sig_33_new > apex_stress:
                sig_tr_l_t[2, 2], sig_tr_l_t[0, 2], sig_tr_l_t[1, 2] = apex_stress, 0, 0
                D_tan_l[2, :] = 0; D_tan_l[:, 2] = 0
                D_tan_l[4, :] = 0; D_tan_l[:, 4] = 0
                D_tan_l[5, :] = 0; D_tan_l[:, 5] = 0
            else:
                sig_tr_l_t[2, 2] = sig_33_new
                if tau > 0:
                    factor = tau_new / tau
                    sig_tr_l_t[0, 2] *= factor
                    sig_tr_l_t[1, 2] *= factor

                n_vec, m_vec = np.zeros(6), np.zeros(6)
                n_vec[2], m_vec[2] = tan_phi, tan_psi
                if tau > 0:
                    n_vec[4] = m_vec[4] = s23 / tau
                    n_vec[5] = m_vec[5] = s13 / tau

                D_m = self.D_eq_local @ m_vec
                n_D = n_vec @ self.D_eq_local
                denom = np.dot(n_vec, D_m)

                if abs(denom) > 1e-12:
                    D_ep = self.D_eq_local - np.outer(D_m, n_D) / denom
                    if tau > 0:
                        beta = tau_new / tau
                        delta_D = np.zeros((6, 6))
                        delta_D[4, 4] = self.alpha_2 * (1 - (s23 / tau) ** 2)
                        delta_D[5, 5] = self.alpha_2 * (1 - (s13 / tau) ** 2)
                        delta_D[4, 5] = delta_D[5, 4] = -self.alpha_2 * (s23 / tau) * (s13 / tau)
                        D_tan_l = D_ep - (1 - beta) * delta_D
                    else:
                        D_tan_l = D_ep

        # Обратный перевод напряжений и касательной матрицы
        st_g = self.R @ sig_tr_l_t @ self.R.T
        self.stress = np.array([st_g[0, 0], st_g[1, 1], st_g[2, 2], st_g[0, 1], st_g[1, 2], st_g[0, 2]])

        return self.stress, self._rotate_matrix(D_tan_l, self.R)

    def get_tangent_matrix(self):
        return self.D_eq_global

    def get_stress(self, strain):
        return self.stress

    def commit(self):
        self.stress_old = self.stress.copy()
        self.strain_old = self.strain.copy()
