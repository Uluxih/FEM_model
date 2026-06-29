import numpy as np

# Импортируем модули для поиска критической плоскости и вычисления параметров
from FEM.Integration_Point_Level.CriticalPlane.criterion import (
    find_critical_plane_shear,
    find_critical_plane_tensile,
    get_tensile_limit,
    get_compression_limit,
    get_cohesion_limit
)
from FEM.Integration_Point_Level.CriticalPlane.tensor import StressTensor
from FEM.Abstract.Integration_Point_Level import ConstitutiveModel


class UbiquitousJointModel3D(ConstitutiveModel):
    """
    Адаптивная упругопластическая модель с фиксацией трещины (Fixed Smeared Crack).
    Трещина рассматривается как поверхность пластического скольжения/отрыва.

    ОСОБЕННОСТИ:
    - До разрушения: Изотропная линейно-упругая порода.
    - При разрушении: Поиск критической плоскости, фиксация и расчет C, T, C_c.
    - После разрушения: Return Mapping на зафиксированной плоскости.
    - РАСТЯЖЕНИЕ: Внедрено только нелинейное изотропное упрочнение (Hardening).
      Разупрочнение (Softening/Damage) исключено.
    """

    def __init__(self, material):
        super().__init__(material)

        E = self.material.E
        nu = self.material.nu
        jp = self.material.joint_params

        # --- 1. Базовые параметры пластичности ---
        self.phi = np.radians(jp.get('phi', 0.0))
        self.psi = np.radians(jp.get('psi', 0.0))

        # Пределы прочности инициализируются нулями
        self.t_limit_base = 0.0
        self.t_limit = 0.0
        self.c_limit = 0.0
        self.c = 0.0

        # --- 2. Параметры поиска (Critical Plane Material) ---
        self.cp_material = jp.get('cp_material', None)
        if self.cp_material is None:
            raise ValueError("Для адаптивной модели необходимо передать 'cp_material' в joint_params!")

        # --- 3. Параметры Упрочнения при растяжении ---
        # Нелинейное упрочнение: R(p) = R_inf * (1 - exp(-b * p))
        self.R_inf = jp.get('R_inf', 0.0)  # Напряжение насыщения
        self.b_param = jp.get('b_param', 0.0)  # Скорость насыщения

        # Переменные состояния для растяжения
        self.eps_p_t = 0.0  # Текущая накопленная пластическая деформация отрыва
        self.eps_p_t_old = 0.0  # Пластическая деформация с прошлого шага

        # --- 4. Упругая матрица целой породы ---
        self.C_rock = self._build_isotropic_compliance(E, nu)
        self.D_rock = np.linalg.inv(self.C_rock)

        # Текущая касательная матрица жесткости
        self.D_tangent = self.D_rock.copy()

        # --- 5. Переменные состояния фиксации ---
        self.is_locked = False
        self.fixed_normal = None
        self.R = np.eye(3)
        self.D_local = self.D_rock.copy()

        # Модули для алгоритма Return Mapping (упругие константы породы)
        self.alpha_1 = self.D_local[2, 2]
        self.alpha_2 = (self.D_local[4, 4] + self.D_local[5, 5]) / 2.0

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

    def _build_rotation_matrix(self, n):
        nz = np.array(n, dtype=float).flatten()
        nz /= np.linalg.norm(nz)

        if abs(nz[2]) > 0.999:
            nx = np.array([1.0, 0.0, 0.0])
            ny = np.cross(nz, nx)
        else:
            ny = np.cross(nz, [0.0, 0.0, 1.0])
            ny /= np.linalg.norm(ny)
            nx = np.cross(ny, nz)

        nx /= np.linalg.norm(nx)
        return np.column_stack((nx, ny, nz))

    def _rotate_matrix(self, D, R):
        D_glob = np.zeros((6, 6))
        for j in range(6):
            e_g = np.zeros(6)
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
        return StressTensor(v[0], v[1], v[2], v[3], v[4], v[5])

    def _lock_plane(self, normal, stress_tensor):
        """Процедура фиксации трещины и вычисления параметров прочности"""
        self.fixed_normal = normal
        self.R = self._build_rotation_matrix(normal)

        self.t_limit_base = get_tensile_limit(normal, self.cp_material)
        self.c_limit = get_compression_limit(normal, self.cp_material)
        self.c = get_cohesion_limit(normal, stress_tensor, self.cp_material)

        # Пересчитываем ограничение на растяжение (Apex correction)
        self.t_limit = self.t_limit_base
        if self.phi > 0:
            t_max = self.c / np.tan(self.phi)
            self.t_limit = min(self.t_limit, t_max)

        self.is_locked = True
        print(f" [!] ОБРАЗОВАНИЕ ТРЕЩИНЫ. Нормаль: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")

    def update_state(self, current_strain):
        self.strain = current_strain
        d_strain = current_strain - self.strain_old

        # Упругий предиктор ВСЕГДА строится по жесткости неповрежденной породы
        sig_tr_v = self.stress_old + self.D_rock @ d_strain

        # ==========================================================
        # ЭТАП 1: ПОИСК И ФИКСАЦИЯ
        # ==========================================================
        if not self.is_locked:
            st = self._voigt_to_stresstensor(sig_tr_v)
            f_sh, n_sh, _ = find_critical_plane_shear(st, self.cp_material, mode='3D')
            f_t, n_t, _ = find_critical_plane_tensile(st, self.cp_material, mode='3D')

            if max(f_sh, f_t) > 0:
                best_n = n_sh if f_sh > f_t else n_t
                self._lock_plane(best_n, st)
            else:
                self.stress = sig_tr_v
                self.D_tangent = self.D_rock
                return self.stress, self.D_tangent

        # ==========================================================
        # ЭТАП 2: РАБОТА С ЗАФИКСИРОВАННОЙ ТРЕЩИНОЙ (Return Mapping)
        # ==========================================================
        st_t = np.array([[sig_tr_v[0], sig_tr_v[3], sig_tr_v[5]],
                         [sig_tr_v[3], sig_tr_v[1], sig_tr_v[4]],
                         [sig_tr_v[5], sig_tr_v[4], sig_tr_v[2]]])
        sig_tr_l_t = self.R.T @ st_t @ self.R

        s33, s13, s23 = sig_tr_l_t[2, 2], sig_tr_l_t[0, 2], sig_tr_l_t[1, 2]
        tau = np.sqrt(s13 ** 2 + s23 ** 2)

        # ----------------------------------------------------------
        # ВЫЧИСЛЕНИЕ ЭФФЕКТИВНОГО ПРЕДЕЛА ПРОЧНОСТИ (Только Упрочнение)
        # Eq (2): R(p) = R_inf * (1 - exp(-b * p))
        # ----------------------------------------------------------
        R_hardening = self.R_inf * (1.0 - np.exp(-self.b_param * self.eps_p_t_old))
        t_limit_eff = self.t_limit + R_hardening

        # Проверка критериев (Отрыв, Сдвиг, Сжатие)
        f_s = tau + s33 * np.tan(self.phi) - self.c
        f_t = s33 - t_limit_eff
        f_c = -s33 - self.c_limit

        # Упругий шаг на зафиксированной плоскости
        if f_s <= 0 and f_t <= 0 and f_c <= 0:
            self.stress = sig_tr_v
            self.D_tangent = self.D_rock
            return self.stress, self.D_tangent

        # Инициализация локальной касательной матрицы
        D_tan_l = self.D_local.copy()

        # --- АЛГОРИТМИЧЕСКИЙ ВОЗВРАТ (Return Mapping) ---

        # 1. ОТРЫВ (Tension Cutoff) с учетом упрочнения
        if f_t > 0 or (f_s > 0 and (s33 + (f_s / self.alpha_2) * self.alpha_1 * np.tan(self.psi)) > t_limit_eff):

            # Производная функции упрочнения: H = dR/dp
            H = self.R_inf * self.b_param * np.exp(-self.b_param * self.eps_p_t_old)

            # Приращение пластической деформации отрыва
            delta_lam_t = f_t / (self.alpha_1 + H)
            self.eps_p_t = self.eps_p_t_old + delta_lam_t

            # Обновление напряжений
            sig_tr_l_t[2, 2] = s33 - self.alpha_1 * delta_lam_t
            sig_tr_l_t[0, 2] = 0.0
            sig_tr_l_t[1, 2] = 0.0

            # Формирование консистентной касательной матрицы для упрочнения
            D_epc = (self.alpha_1 * H) / (self.alpha_1 + H) if (self.alpha_1 + H) > 1e-12 else 0.0

            D_tan_l[2, :] = 0;
            D_tan_l[:, 2] = 0
            D_tan_l[2, 2] = D_epc

            # Сдвиговая жесткость полностью обнуляется при отрыве
            D_tan_l[4, :] = 0;
            D_tan_l[:, 4] = 0
            D_tan_l[5, :] = 0;
            D_tan_l[:, 5] = 0

        # 2. СЖАТИЕ (Compression Cutoff)
        elif f_c > 0 or (f_s > 0 and (s33 - (f_s / self.alpha_2) * self.alpha_1 * np.tan(self.psi)) < -self.c_limit):
            sig_tr_l_t[2, 2] = -self.c_limit
            sig_tr_l_t[0, 2] = 0.0
            sig_tr_l_t[1, 2] = 0.0
            D_tan_l[2, :] = 0;
            D_tan_l[:, 2] = 0
            D_tan_l[4, :] = 0;
            D_tan_l[:, 4] = 0
            D_tan_l[5, :] = 0;
            D_tan_l[:, 5] = 0

        # 3. СДВИГ (Shear)
        else:
            tan_phi, tan_psi = np.tan(self.phi), np.tan(self.psi)
            lam = f_s / (self.alpha_2 + self.alpha_1 * tan_phi * tan_psi)

            sig_33_new = s33 - lam * self.alpha_1 * tan_psi
            tau_new = tau - lam * self.alpha_2

            apex_stress = self.c / tan_phi if self.phi > 0 else float('inf')

            # Проверка перехода из сдвига в отрыв (через Apex)
            if sig_33_new > apex_stress:
                sig_tr_l_t[2, 2] = apex_stress
                sig_tr_l_t[0, 2] = 0.0
                sig_tr_l_t[1, 2] = 0.0
                D_tan_l[2, :] = 0;
                D_tan_l[:, 2] = 0
                D_tan_l[4, :] = 0;
                D_tan_l[:, 4] = 0
                D_tan_l[5, :] = 0;
                D_tan_l[:, 5] = 0

            # Проверка перехода из сдвига в сжатие
            elif sig_33_new < -self.c_limit:
                sig_tr_l_t[2, 2] = -self.c_limit
                sig_tr_l_t[0, 2] = 0.0
                sig_tr_l_t[1, 2] = 0.0
                D_tan_l[2, :] = 0;
                D_tan_l[:, 2] = 0
                D_tan_l[4, :] = 0;
                D_tan_l[:, 4] = 0
                D_tan_l[5, :] = 0;
                D_tan_l[:, 5] = 0

            # Чистый сдвиг
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

                # Формирование упругопластической касательной матрицы для сдвига
                D_m = self.D_local @ m_vec
                n_D = n_vec @ self.D_local
                denom = np.dot(n_vec, D_m)

                if abs(denom) > 1e-12:
                    D_ep = self.D_local - np.outer(D_m, n_D) / denom
                    if tau > 0:
                        beta = tau_new / tau
                        delta_D = np.zeros((6, 6))
                        delta_D[4, 4] = self.alpha_2 * (1 - (s23 / tau) ** 2)
                        delta_D[5, 5] = self.alpha_2 * (1 - (s13 / tau) ** 2)
                        delta_D[4, 5] = delta_D[5, 4] = -self.alpha_2 * (s23 / tau) * (s13 / tau)
                        D_tan_l = D_ep - (1 - beta) * delta_D
                    else:
                        D_tan_l = D_ep

        # Возврат напряжений и касательной матрицы в глобальную систему
        st_g = self.R @ sig_tr_l_t @ self.R.T
        self.stress = np.array([st_g[0, 0], st_g[1, 1], st_g[2, 2], st_g[0, 1], st_g[1, 2], st_g[0, 2]])

        self.D_tangent = self._rotate_matrix(D_tan_l, self.R)

        return self.stress, self.D_tangent

    def get_tangent_matrix(self):
        return self.D_tangent

    def get_stress(self, strain):
        return self.stress

    def commit(self):
        """Фиксация состояния в конце шага нагружения"""
        self.stress_old = self.stress.copy()
        self.strain_old = self.strain.copy()

        # Фиксация переменной состояния (накопленная пластика)
        self.eps_p_t_old = self.eps_p_t
