import numpy as np

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
    def __init__(self, material):
        super().__init__(material)

        E = self.material.E
        nu = self.material.nu
        jp = self.material.joint_params

        self.E_min = 1e-5 * E

        # Базовые параметры
        self.phi = np.radians(jp.get('phi', 30.0))
        self.psi = np.radians(jp.get('psi', 10.0))
        self.tan_phi = np.tan(self.phi)
        self.tan_psi = np.tan(self.psi)

        self.cp_material = jp.get('cp_material', None)
        if self.cp_material is None:
            raise ValueError("Требуется 'cp_material' в joint_params!")

        # Энергии разрушения
        self.l_c = jp.get('l_c', 1.0)
        self.Gf_t = jp.get('Gf_t', 100.0) / self.l_c
        self.Gf_c = jp.get('Gf_c', 5000.0) / self.l_c
        self.Gf_s = jp.get('Gf_s', 500.0) / self.l_c

        self.H_t = 0.0
        self.H_c = 0.0
        self.H_s = 0.0

        # Переменные состояния
        self._init_history()

        self.D_rock = self._build_isotropic_stiffness(E, nu)
        self.D_tangent = self.D_rock.copy()

        self.is_locked = False
        self.fixed_normal = None
        self.R = np.eye(3)
        self.T_sig = np.eye(6)
        self.T_eps = np.eye(6)
        self.D_local = self.D_rock.copy()

        self.E_n = self.D_local[2, 2]
        self.G_s = self.D_local[4, 4]

        # Флаг для выбора метода вычисления CTO
        # 'numerical' гарантирует идеальную сходимость для сложных Damage моделей
        self.tangent_type = 'numerical'

    def _init_history(self):
        self.lam_t_old = self.lam_c_old = self.lam_s_old = 0.0
        self.eps_p_n_old = self.eps_p_23_old = self.eps_p_13_old = 0.0
        self.W_pl_t_old = self.W_pl_c_old = self.W_pl_s_old = 0.0
        self.D_n_old = self.D_s_old = 0.0
        self.stress_old = np.zeros(6)
        self.strain_old = np.zeros(6)

        self.stress = np.zeros(6)
        self.strain = np.zeros(6)

        self._reset_trial()

    def _reset_trial(self):
        self.lam_t_trial = self.lam_t_old
        self.lam_c_trial = self.lam_c_old
        self.lam_s_trial = self.lam_s_old
        self.eps_p_n_trial = self.eps_p_n_old
        self.eps_p_23_trial = self.eps_p_23_old
        self.eps_p_13_trial = self.eps_p_13_old
        self.W_pl_t_trial = self.W_pl_t_old
        self.W_pl_c_trial = self.W_pl_c_old
        self.W_pl_s_trial = self.W_pl_s_old
        self.D_n_trial = self.D_n_old
        self.D_s_trial = self.D_s_old

    def _build_isotropic_stiffness(self, E, nu):
        D = np.zeros((6, 6))
        c1 = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
        c2 = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        G = E / (2.0 * (1.0 + nu))
        D[0:3, 0:3] = c2
        D[0, 0] = D[1, 1] = D[2, 2] = c1
        D[3, 3] = D[4, 4] = D[5, 5] = G
        return D

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

    def _build_voigt_transformations(self, R):
        """
        ИСПРАВЛЕНО: Строгое построение матриц трансформации Фойгта.
        sig_global = T_sig^T * sig_local
        eps_local = T_eps * eps_global
        """
        T_sig = np.zeros((6, 6))
        for i in range(3):
            for j in range(3): T_sig[i, j] = R[i, j] ** 2

        T_sig[0, 3] = 2 * R[0, 0] * R[0, 1];
        T_sig[0, 4] = 2 * R[0, 1] * R[0, 2];
        T_sig[0, 5] = 2 * R[0, 0] * R[0, 2]
        T_sig[1, 3] = 2 * R[1, 0] * R[1, 1];
        T_sig[1, 4] = 2 * R[1, 1] * R[1, 2];
        T_sig[1, 5] = 2 * R[1, 0] * R[1, 2]
        T_sig[2, 3] = 2 * R[2, 0] * R[2, 1];
        T_sig[2, 4] = 2 * R[2, 1] * R[2, 2];
        T_sig[2, 5] = 2 * R[2, 0] * R[2, 2]

        T_sig[3, 0] = R[0, 0] * R[1, 0];
        T_sig[3, 1] = R[0, 1] * R[1, 1];
        T_sig[3, 2] = R[0, 2] * R[1, 2]
        T_sig[4, 0] = R[1, 0] * R[2, 0];
        T_sig[4, 1] = R[1, 1] * R[2, 1];
        T_sig[4, 2] = R[1, 2] * R[2, 2]
        T_sig[5, 0] = R[0, 0] * R[2, 0];
        T_sig[5, 1] = R[0, 1] * R[2, 1];
        T_sig[5, 2] = R[0, 2] * R[2, 2]

        T_sig[3, 3] = R[0, 0] * R[1, 1] + R[0, 1] * R[1, 0]
        T_sig[3, 4] = R[0, 1] * R[1, 2] + R[0, 2] * R[1, 1]
        T_sig[3, 5] = R[0, 0] * R[1, 2] + R[0, 2] * R[1, 0]

        T_sig[4, 3] = R[1, 0] * R[2, 1] + R[1, 1] * R[2, 0]
        T_sig[4, 4] = R[1, 1] * R[2, 2] + R[1, 2] * R[2, 1]
        T_sig[4, 5] = R[1, 0] * R[2, 2] + R[1, 2] * R[2, 0]

        T_sig[5, 3] = R[0, 0] * R[2, 1] + R[0, 1] * R[2, 0]
        T_sig[5, 4] = R[0, 1] * R[2, 2] + R[0, 2] * R[2, 1]
        T_sig[5, 5] = R[0, 0] * R[2, 2] + R[0, 2] * R[2, 0]

        T_eps = np.zeros((6, 6))
        T_eps[0:3, 0:3] = T_sig[0:3, 0:3]
        T_eps[0:3, 3:6] = T_sig[0:3, 3:6] / 2.0
        T_eps[3:6, 0:3] = T_sig[3:6, 0:3] * 2.0
        T_eps[3:6, 3:6] = T_sig[3:6, 3:6]

        return T_sig, T_eps

    def _lock_plane(self, normal, stress_tensor):
        self.fixed_normal = normal
        self.R = self._build_rotation_matrix(normal)
        self.T_sig, self.T_eps = self._build_voigt_transformations(self.R)

        # Перевод матрицы жесткости в локальные оси трещины (ось Z - нормаль)
        self.D_local = self.T_sig @ self.D_rock @ self.T_eps.T
        self.E_n = self.D_local[2, 2]
        self.G_s = self.D_local[4, 4]

        self.f_t = get_tensile_limit(normal, self.cp_material)
        self.f_c = get_compression_limit(normal, self.cp_material)
        self.c = get_cohesion_limit(normal, stress_tensor, self.cp_material)

        eps_f_t = 3.0 * self.Gf_t / self.f_t if self.f_t > 0 else 1e-6
        self.H_t = -self.f_t / eps_f_t if eps_f_t > 0 else 0.0

        eps_f_c = 2.0 * self.Gf_c / self.f_c if self.f_c > 0 else 1e-6
        self.H_c = -self.f_c / eps_f_c if eps_f_c > 0 else 0.0

        gamma_f_s = 2.0 * self.Gf_s / max(self.c, 1e-6)
        self.H_s = -self.c / gamma_f_s if gamma_f_s > 0 else 0.0

        self.is_locked = True

    def _integrate_stress(self, current_strain):
        """
        Чистая функция интегрирования напряжений.
        Принимает деформации, возвращает напряжения. Никаких побочных эффектов на state.
        """
        d_strain = current_strain - self.strain_old
        sig_tr_v = self.stress_old + self.D_rock @ d_strain

        if not self.is_locked:
            return sig_tr_v

        # Перевод деформаций в локальные оси (T_eps)
        e_l = self.T_eps @ current_strain

        # Упругие предикторы
        sig_n_tr = self.E_n * (e_l[2] - self.eps_p_n_old)
        tau_23_tr = self.G_s * (e_l[4] - self.eps_p_23_old)
        tau_13_tr = self.G_s * (e_l[5] - self.eps_p_13_old)
        tau_tr = np.sqrt(tau_23_tr ** 2 + tau_13_tr ** 2)

        ft_curr = max(self.f_t + self.H_t * self.lam_t_old, 0.0)
        fc_curr = max(self.f_c + self.H_c * self.lam_c_old, 0.1 * self.f_c)
        c_curr = max(self.c + self.H_s * self.lam_s_old, 0.0)

        f_t_val = sig_n_tr - ft_curr
        f_c_val = -sig_n_tr - fc_curr
        f_s_val = tau_tr + sig_n_tr * self.tan_phi - c_curr

        d_lam_t = d_lam_c = d_lam_s = 0.0

        if f_t_val > 0 and f_s_val > 0:
            det = (self.E_n + self.H_t) * (self.G_s + self.E_n * self.tan_phi * self.tan_psi + self.H_s) - \
                  (self.E_n * self.tan_psi) * (self.E_n * self.tan_phi)
            d_lam_t = ((self.G_s + self.E_n * self.tan_phi * self.tan_psi + self.H_s) * f_t_val - \
                       (self.E_n * self.tan_psi) * f_s_val) / det
            d_lam_s = (-(self.E_n * self.tan_phi) * f_t_val + (self.E_n + self.H_t) * f_s_val) / det
            if d_lam_t <= 0:
                d_lam_t = 0;
                d_lam_s = f_s_val / (self.G_s + self.E_n * self.tan_phi * self.tan_psi + self.H_s)
            elif d_lam_s <= 0:
                d_lam_s = 0;
                d_lam_t = f_t_val / (self.E_n + self.H_t)

        elif f_c_val > 0 and f_s_val > 0:
            det = (self.E_n + self.H_c) * (self.G_s - self.E_n * self.tan_phi * self.tan_psi + self.H_s) + \
                  (self.E_n * self.tan_psi) * (self.E_n * self.tan_phi)
            d_lam_c = ((self.G_s - self.E_n * self.tan_phi * self.tan_psi + self.H_s) * f_c_val + \
                       (self.E_n * self.tan_psi) * f_s_val) / det
            d_lam_s = (-(self.E_n * self.tan_phi) * f_c_val + (self.E_n + self.H_c) * f_s_val) / det
            if d_lam_c <= 0:
                d_lam_c = 0;
                d_lam_s = f_s_val / (self.G_s + self.E_n * self.tan_phi * self.tan_psi + self.H_s)
            elif d_lam_s <= 0:
                d_lam_s = 0;
                d_lam_c = f_c_val / (self.E_n + self.H_c)

        elif f_t_val > 0:
            d_lam_t = f_t_val / (self.E_n + self.H_t)
        elif f_c_val > 0:
            d_lam_c = f_c_val / (self.E_n + self.H_c)
        elif f_s_val > 0:
            d_lam_s = f_s_val / (self.G_s + self.E_n * self.tan_phi * self.tan_psi + self.H_s)

        # Эффективные напряжения
        sig_n = sig_n_tr - self.E_n * (d_lam_t - d_lam_c + d_lam_s * self.tan_psi)
        tau = max(tau_tr - self.G_s * d_lam_s, 0.0)

        shear_ratio = tau / tau_tr if tau_tr > 1e-12 else 0.0
        tau_23 = tau_23_tr * shear_ratio
        tau_13 = tau_13_tr * shear_ratio

        # Обновление Damage
        W_pl_t = self.W_pl_t_old + (sig_n * d_lam_t if d_lam_t > 0 else 0)
        W_pl_c = self.W_pl_c_old + (abs(sig_n) * d_lam_c if d_lam_c > 0 else 0)
        W_pl_s = self.W_pl_s_old + (tau * d_lam_s if d_lam_s > 0 else 0)

        r_t = min(W_pl_t / self.Gf_t, 1.0)
        r_c = min(W_pl_c / self.Gf_c, 1.0)
        D_n_new = max(r_t * (2.0 - r_t), 0.5 * np.sin(np.pi * r_c - np.pi / 2.0) + 0.5)
        D_n = min(max(D_n_new, self.D_n_old), 0.999)

        r_s = min(W_pl_s / self.Gf_s, 1.0)
        D_s = min(max(r_s * (2.0 - r_s), self.D_s_old), 0.999)

        # Собираем локальные напряжения
        sig_l = self.D_local @ e_l
        sig_l[2] = (1.0 - D_n) * sig_n
        sig_l[4] = (1.0 - D_s) * tau_23
        sig_l[5] = (1.0 - D_s) * tau_13

        # Возвращаем глобальные напряжения (T_sig^T)
        return self.T_sig.T @ sig_l

    def _compute_numerical_tangent(self, current_strain, eps=1e-8):
        """
        ВЫЧИСЛЕНИЕ ЧИСЛЕННОГО ЯКОБИАНА (ТОЧНЫЙ CTO)
        Гарантирует квадратичную сходимость Ньютона-Рафсона.
        D_ij = d(sigma_i) / d(eps_j)
        """
        D_num = np.zeros((6, 6))
        stress_base = self._integrate_stress(current_strain)

        for j in range(6):
            strain_pert = current_strain.copy()
            strain_pert[j] += eps
            stress_pert = self._integrate_stress(strain_pert)
            D_num[:, j] = (stress_pert - stress_base) / eps

        # Симметризация малой численной асимметрии (по желанию, для неассоциированного течения матрица несимметрична!)
        return D_num

    def update_state(self, current_strain):
        self.strain = current_strain
        self._reset_trial()

        # 1. Поиск критической плоскости
        if not self.is_locked:
            sig_tr_v = self.stress_old + self.D_rock @ (current_strain - self.strain_old)
            st = StressTensor(sig_tr_v[0], sig_tr_v[1], sig_tr_v[2], sig_tr_v[3], sig_tr_v[4], sig_tr_v[5])
            f_t_scaled, n_t, _ = find_critical_plane_tensile(StressTensor(*(sig_tr_v / 0.8)), self.cp_material,
                                                             mode='3D')
            f_sh, n_sh, _ = find_critical_plane_shear(st, self.cp_material, mode='3D')

            if f_sh > 0 or f_t_scaled > 0:
                f_t_real, _, _ = find_critical_plane_tensile(st, self.cp_material, mode='3D')
                best_n = n_sh if (f_sh > f_t_real and f_sh > 0) else n_t
                self._lock_plane(best_n, st)
            else:
                self.stress = sig_tr_v
                self.D_tangent = self.D_rock
                return self.stress, self.D_tangent

        # 2. Интегрирование напряжений (Return Mapping)
        self.stress = self._integrate_stress(current_strain)

        # 3. Вычисление точного касательного оператора
        if self.tangent_type == 'numerical':
            self.D_tangent = self._compute_numerical_tangent(current_strain)
        else:
            # Здесь можно оставить ваш аналитический метод, но численный надежнее
            self.D_tangent = self._compute_numerical_tangent(current_strain)

        # Защита от нулей на диагонали
        self.D_tangent += np.eye(6) * self.E_min

        # 4. Обновление trial переменных для коммита
        # (Повторяем расчет, чтобы зафиксировать внутренние переменные)
        e_l = self.T_eps @ current_strain
        sig_n_tr = self.E_n * (e_l[2] - self.eps_p_n_old)
        tau_23_tr = self.G_s * (e_l[4] - self.eps_p_23_old)
        tau_13_tr = self.G_s * (e_l[5] - self.eps_p_13_old)
        tau_tr = np.sqrt(tau_23_tr ** 2 + tau_13_tr ** 2)

        ft_curr = max(self.f_t + self.H_t * self.lam_t_old, 0.0)
        fc_curr = max(self.f_c + self.H_c * self.lam_c_old, 0.1 * self.f_c)
        c_curr = max(self.c + self.H_s * self.lam_s_old, 0.0)

        f_t_val = sig_n_tr - ft_curr
        f_c_val = -sig_n_tr - fc_curr
        f_s_val = tau_tr + sig_n_tr * self.tan_phi - c_curr
        d_lam_t = d_lam_c = d_lam_s = 0.0

        if f_t_val > 0 and f_s_val > 0:
            det = (self.E_n + self.H_t) * (self.G_s + self.E_n * self.tan_phi * self.tan_psi + self.H_s) - (
                        self.E_n * self.tan_psi) * (self.E_n * self.tan_phi)
            d_lam_t = ((self.G_s + self.E_n * self.tan_phi * self.tan_psi + self.H_s) * f_t_val - (
                        self.E_n * self.tan_psi) * f_s_val) / det
            d_lam_s = (-(self.E_n * self.tan_phi) * f_t_val + (self.E_n + self.H_t) * f_s_val) / det
            if d_lam_t <= 0:
                d_lam_t = 0;
                d_lam_s = f_s_val / (self.G_s + self.E_n * self.tan_phi * self.tan_psi + self.H_s)
            elif d_lam_s <= 0:
                d_lam_s = 0;
                d_lam_t = f_t_val / (self.E_n + self.H_t)
        elif f_c_val > 0 and f_s_val > 0:
            det = (self.E_n + self.H_c) * (self.G_s - self.E_n * self.tan_phi * self.tan_psi + self.H_s) + (
                        self.E_n * self.tan_psi) * (self.E_n * self.tan_phi)
            d_lam_c = ((self.G_s - self.E_n * self.tan_phi * self.tan_psi + self.H_s) * f_c_val + (
                        self.E_n * self.tan_psi) * f_s_val) / det
            d_lam_s = (-(self.E_n * self.tan_phi) * f_c_val + (self.E_n + self.H_c) * f_s_val) / det
            if d_lam_c <= 0:
                d_lam_c = 0;
                d_lam_s = f_s_val / (self.G_s + self.E_n * self.tan_phi * self.tan_psi + self.H_s)
            elif d_lam_s <= 0:
                d_lam_s = 0;
                d_lam_c = f_c_val / (self.E_n + self.H_c)
        elif f_t_val > 0:
            d_lam_t = f_t_val / (self.E_n + self.H_t)
        elif f_c_val > 0:
            d_lam_c = f_c_val / (self.E_n + self.H_c)
        elif f_s_val > 0:
            d_lam_s = f_s_val / (self.G_s + self.E_n * self.tan_phi * self.tan_psi + self.H_s)

        sig_n = sig_n_tr - self.E_n * (d_lam_t - d_lam_c + d_lam_s * self.tan_psi)
        tau = max(tau_tr - self.G_s * d_lam_s, 0.0)

        self.lam_t_trial += d_lam_t
        self.lam_c_trial += d_lam_c
        self.lam_s_trial += d_lam_s

        self.eps_p_n_trial += d_lam_t - d_lam_c + d_lam_s * self.tan_psi
        if tau_tr > 1e-12:
            self.eps_p_23_trial += d_lam_s * (tau_23_tr / tau_tr)
            self.eps_p_13_trial += d_lam_s * (tau_13_tr / tau_tr)

        self.W_pl_t_trial += (sig_n * d_lam_t if d_lam_t > 0 else 0)
        self.W_pl_c_trial += (abs(sig_n) * d_lam_c if d_lam_c > 0 else 0)
        self.W_pl_s_trial += (tau * d_lam_s if d_lam_s > 0 else 0)

        r_t = min(self.W_pl_t_trial / self.Gf_t, 1.0)
        r_c = min(self.W_pl_c_trial / self.Gf_c, 1.0)
        D_n_new = max(r_t * (2.0 - r_t), 0.5 * np.sin(np.pi * r_c - np.pi / 2.0) + 0.5)
        self.D_n_trial = min(max(D_n_new, self.D_n_old), 0.999)

        r_s = min(self.W_pl_s_trial / self.Gf_s, 1.0)
        self.D_s_trial = min(max(r_s * (2.0 - r_s), self.D_s_old), 0.999)

        return self.stress, self.D_tangent

    def get_tangent_matrix(self):
        return self.D_tangent

    def get_stress(self, strain):
        return self.stress

    def commit(self):
        self.stress_old = self.stress.copy()
        self.strain_old = self.strain.copy()
        self.lam_t_old = self.lam_t_trial
        self.lam_c_old = self.lam_c_trial
        self.lam_s_old = self.lam_s_trial
        self.eps_p_n_old = self.eps_p_n_trial
        self.eps_p_23_old = self.eps_p_23_trial
        self.eps_p_13_old = self.eps_p_13_trial
        self.W_pl_t_old = self.W_pl_t_trial
        self.W_pl_c_old = self.W_pl_c_trial
        self.W_pl_s_old = self.W_pl_s_trial
        self.D_n_old = self.D_n_trial
        self.D_s_old = self.D_s_trial
