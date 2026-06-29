import numpy as np

from FEM.Integration_Point_Level.CriticalPlane.criterion import (
    get_tensile_limit,
    get_compression_limit,
    get_cohesion_limit
)
from FEM.Integration_Point_Level.CriticalPlane.tensor import StressTensor
from FEM.Abstract.Integration_Point_Level import ConstitutiveModel


class UbiquitousJointModel3DFixed(ConstitutiveModel):
    """
    Строгая 3D модель Damage-Plasticity с ЗАФИКСИРОВАННОЙ плоскостью ослабления.
    Идеально подходит для тестирования сходимости решателя и верификации.
    - Учтена дилатансия (неассоциированное течение).
    - Совместный (Coupled) алгоритм Return Mapping.
    - Строгий несимметричный алгоритмический касательный оператор.
    - Разупрочнение при сжатии и сдвиге.
    """

    def __init__(self, material, fixed_normal=None):
        """
        :param material: Объект материала
        :param fixed_normal: Вектор нормали фиксированной плоскости (по умолчанию [0, 0, 1])
        """
        super().__init__(material)

        if fixed_normal is None:
            fixed_normal = [0.0, 0.0, 1.0]

        E = self.material.E
        nu = self.material.nu
        jp = self.material.joint_params

        self.E_min = 1e-5 * E

        # --- 1. Базовые параметры (Трение и Дилатансия) ---
        self.phi = np.radians(jp.get('phi', 30.0))
        self.psi = np.radians(jp.get('psi', 10.0))
        self.tan_phi = np.tan(self.phi)
        self.tan_psi = np.tan(self.psi)

        self.cp_material = jp.get('cp_material', None)
        if self.cp_material is None:
            raise ValueError("Требуется 'cp_material' в joint_params!")

        # --- 2. Энергии разрушения и модули разупрочнения ---
        self.l_c = jp.get('l_c', 1.0)

        self.Gf_t = jp.get('Gf_t', 100.0) / self.l_c
        self.Gf_c = jp.get('Gf_c', 5000.0) / self.l_c
        self.Gf_s = jp.get('Gf_s', 500.0) / self.l_c

        self.lambda_param = jp.get('lambda_param', 0.1)

        self.H_t = 0.0
        self.H_c = 0.0
        self.H_s = 0.0

        # --- 3. Переменные состояния ---
        self.lam_t_old = 0.0
        self.lam_c_old = 0.0
        self.lam_s_old = 0.0

        self.eps_p_n_old = 0.0
        self.eps_p_23_old = 0.0
        self.eps_p_13_old = 0.0

        self.W_pl_t_old = 0.0
        self.W_pl_c_old = 0.0
        self.W_pl_s_old = 0.0

        self.D_n_old = 0.0
        self.D_s_old = 0.0

        self.stress_old = np.zeros(6)
        self.strain_old = np.zeros(6)

        self._init_trial_states()
        self.stress = np.zeros(6)
        self.strain = np.zeros(6)
        self.D_tangent = np.zeros((6, 6))

        self.D_rock = self._build_isotropic_stiffness(E, nu)
        self.D_tangent = self.D_rock.copy()

        self.is_locked = False
        self.fixed_normal = None
        self.R = np.eye(3)
        self.T_eps = np.eye(6)
        self.T_sig = np.eye(6)
        self.D_local = self.D_rock.copy()

        # ПРИНУДИТЕЛЬНАЯ ФИКСАЦИЯ ПЛОСКОСТИ ПРИ ИНИЦИАЛИЗАЦИИ
        dummy_stress = StressTensor(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._lock_plane(fixed_normal, dummy_stress)

    def _init_trial_states(self):
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
        T_sig = np.zeros((6, 6))
        for i in range(3):
            for j in range(3): T_sig[i, j] = R[j, i] ** 2
        T_sig[0, 3] = 2 * R[0, 0] * R[1, 0];
        T_sig[0, 4] = 2 * R[1, 0] * R[2, 0];
        T_sig[0, 5] = 2 * R[0, 0] * R[2, 0]
        T_sig[1, 3] = 2 * R[0, 1] * R[1, 1];
        T_sig[1, 4] = 2 * R[1, 1] * R[2, 1];
        T_sig[1, 5] = 2 * R[0, 1] * R[2, 1]
        T_sig[2, 3] = 2 * R[0, 2] * R[1, 2];
        T_sig[2, 4] = 2 * R[1, 2] * R[2, 2];
        T_sig[2, 5] = 2 * R[0, 2] * R[2, 2]
        T_sig[3, 0] = R[0, 0] * R[0, 1];
        T_sig[3, 1] = R[1, 0] * R[1, 1];
        T_sig[3, 2] = R[2, 0] * R[2, 1]
        T_sig[4, 0] = R[0, 1] * R[0, 2];
        T_sig[4, 1] = R[1, 1] * R[1, 2];
        T_sig[4, 2] = R[2, 1] * R[2, 2]
        T_sig[5, 0] = R[0, 0] * R[0, 2];
        T_sig[5, 1] = R[1, 0] * R[1, 2];
        T_sig[5, 2] = R[2, 0] * R[2, 2]
        T_sig[3, 3] = R[0, 0] * R[1, 1] + R[1, 0] * R[0, 1];
        T_sig[3, 4] = R[1, 0] * R[2, 1] + R[2, 0] * R[1, 1];
        T_sig[3, 5] = R[0, 0] * R[2, 1] + R[2, 0] * R[0, 1]
        T_sig[4, 3] = R[0, 1] * R[1, 2] + R[1, 1] * R[0, 2];
        T_sig[4, 4] = R[1, 1] * R[2, 2] + R[2, 1] * R[1, 2];
        T_sig[4, 5] = R[0, 1] * R[2, 2] + R[2, 1] * R[0, 2]
        T_sig[5, 3] = R[0, 0] * R[1, 2] + R[1, 0] * R[0, 2];
        T_sig[5, 4] = R[1, 0] * R[2, 2] + R[2, 0] * R[1, 2];
        T_sig[5, 5] = R[0, 0] * R[2, 2] + R[2, 0] * R[0, 2]

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

        self.D_local = self.T_sig @ self.D_rock @ self.T_sig.T
        self.E_n = self.D_local[2, 2]
        self.G_s = self.D_local[4, 4]

        self.f_t = get_tensile_limit(normal, self.cp_material)
        self.f_c = get_compression_limit(normal, self.cp_material)
        self.c = get_cohesion_limit(normal, stress_tensor, self.cp_material)

        # Вычисление модулей разупрочнения (Отрицательные значения)
        eps_f_t = 3.0 * self.Gf_t / self.f_t if self.f_t > 0 else 1e-6
        self.H_t = -self.f_t / eps_f_t if eps_f_t > 0 else 0.0

        eps_f_c = 2.0 * self.Gf_c / self.f_c if self.f_c > 0 else 1e-6
        self.H_c = -self.f_c / eps_f_c if eps_f_c > 0 else 0.0

        gamma_f_s = 2.0 * self.Gf_s / max(self.c, 1e-6)
        self.H_s = -self.c / gamma_f_s if gamma_f_s > 0 else 0.0

        self.is_locked = True

    def update_state(self, current_strain):
        self.strain = current_strain
        self._init_trial_states()

        # --- ЭТАП 1: ПОИСК УДАЛЕН ---
        # Плоскость уже зафиксирована, работаем сразу в локальной системе

        # --- ЭТАП 2: COUPLED RETURN MAPPING ---
        e_l = self.T_eps @ self.strain

        # Упругие предикторы в локальной системе
        sig_n_tr = self.E_n * (e_l[2] - self.eps_p_n_old)
        tau_23_tr = self.G_s * (e_l[4] - self.eps_p_23_old)
        tau_13_tr = self.G_s * (e_l[5] - self.eps_p_13_old)
        tau_tr = np.sqrt(tau_23_tr ** 2 + tau_13_tr ** 2)

        # Текущие пределы прочности
        ft_curr = max(self.f_t + self.H_t * self.lam_t_old, 0.0)
        fc_curr = max(self.f_c + self.H_c * self.lam_c_old, 0.1 * self.f_c)
        c_curr = max(self.c + self.H_s * self.lam_s_old, 0.0)

        # Функции текучести
        f_t_val = sig_n_tr - ft_curr
        f_c_val = -sig_n_tr - fc_curr
        f_s_val = tau_tr + sig_n_tr * self.tan_phi - c_curr

        d_lam_t = d_lam_c = d_lam_s = 0.0
        active_t = active_c = active_s = False

        if f_t_val > 0 and f_s_val > 0:
            det = (self.E_n + self.H_t) * (self.G_s + self.E_n * self.tan_phi * self.tan_psi + self.H_s) - \
                  (self.E_n * self.tan_psi) * (self.E_n * self.tan_phi)
            d_lam_t = ((self.G_s + self.E_n * self.tan_phi * self.tan_psi + self.H_s) * f_t_val - \
                       (self.E_n * self.tan_psi) * f_s_val) / det
            d_lam_s = (-(self.E_n * self.tan_phi) * f_t_val + (self.E_n + self.H_t) * f_s_val) / det

            if d_lam_t > 0 and d_lam_s > 0:
                active_t = active_s = True
            elif d_lam_t <= 0:
                d_lam_t = 0
                active_s = True
                d_lam_s = f_s_val / (self.G_s + self.E_n * self.tan_phi * self.tan_psi + self.H_s)
            else:
                d_lam_s = 0
                active_t = True
                d_lam_t = f_t_val / (self.E_n + self.H_t)

        elif f_c_val > 0 and f_s_val > 0:
            det = (self.E_n + self.H_c) * (self.G_s - self.E_n * self.tan_phi * self.tan_psi + self.H_s) + \
                  (self.E_n * self.tan_psi) * (self.E_n * self.tan_phi)
            d_lam_c = ((self.G_s - self.E_n * self.tan_phi * self.tan_psi + self.H_s) * f_c_val + \
                       (self.E_n * self.tan_psi) * f_s_val) / det
            d_lam_s = (-(self.E_n * self.tan_phi) * f_c_val + (self.E_n + self.H_c) * f_s_val) / det

            if d_lam_c > 0 and d_lam_s > 0:
                active_c = active_s = True
            elif d_lam_c <= 0:
                d_lam_c = 0
                active_s = True
                d_lam_s = f_s_val / (self.G_s + self.E_n * self.tan_phi * self.tan_psi + self.H_s)
            else:
                d_lam_s = 0
                active_c = True
                d_lam_c = f_c_val / (self.E_n + self.H_c)

        elif f_t_val > 0:
            active_t = True
            d_lam_t = f_t_val / (self.E_n + self.H_t)
        elif f_c_val > 0:
            active_c = True
            d_lam_c = f_c_val / (self.E_n + self.H_c)
        elif f_s_val > 0:
            active_s = True
            d_lam_s = f_s_val / (self.G_s + self.E_n * self.tan_phi * self.tan_psi + self.H_s)

        # Обновление эффективных напряжений
        sig_n = sig_n_tr - self.E_n * (d_lam_t - d_lam_c + d_lam_s * self.tan_psi)
        tau = max(tau_tr - self.G_s * d_lam_s, 0.0)

        shear_ratio = tau / tau_tr if tau_tr > 1e-12 else 0.0
        tau_23 = tau_23_tr * shear_ratio
        tau_13 = tau_13_tr * shear_ratio

        # Обновление trial-переменных
        self.lam_t_trial += d_lam_t
        self.lam_c_trial += d_lam_c
        self.lam_s_trial += d_lam_s

        self.eps_p_n_trial += d_lam_t - d_lam_c + d_lam_s * self.tan_psi
        if tau_tr > 1e-12:
            self.eps_p_23_trial += d_lam_s * (tau_23_tr / tau_tr)
            self.eps_p_13_trial += d_lam_s * (tau_13_tr / tau_tr)

        # --- ЭТАП 3: ЭВОЛЮЦИЯ ПОВРЕЖДЕНИЙ ---
        if active_t: self.W_pl_t_trial += sig_n * d_lam_t
        if active_c: self.W_pl_c_trial += abs(sig_n) * d_lam_c
        if active_s: self.W_pl_s_trial += tau * d_lam_s

        r_t = min(self.W_pl_t_trial / self.Gf_t, 1.0)
        r_c = min(self.W_pl_c_trial / self.Gf_c, 1.0)
        D_n_new = max(r_t * (2.0 - r_t), 0.5 * np.sin(np.pi * r_c - np.pi / 2.0) + 0.5)
        self.D_n_trial = min(max(D_n_new, self.D_n_old), 0.999)

        r_s = min(self.W_pl_s_trial / self.Gf_s, 1.0)
        D_s_new = r_s * (2.0 - r_s)
        self.D_s_trial = min(max(D_s_new, self.D_s_old), 0.999)

        # --- ЭТАП 4: АЛГОРИТМИЧЕСКИЙ КАСАТЕЛЬНЫЙ ОПЕРАТОР ---
        D_ep_2D = np.array([[self.E_n, 0.0], [0.0, self.G_s]])
        dlam_deps = np.zeros((2, 2))

        if active_t and active_s:
            A_inv = np.linalg.inv([[self.E_n + self.H_t, self.E_n * self.tan_psi],
                                   [self.E_n * self.tan_phi,
                                    self.G_s + self.E_n * self.tan_phi * self.tan_psi + self.H_s]])
            N_T_De = np.array([[self.E_n, 0.0], [self.E_n * self.tan_phi, self.G_s]])
            De_M = np.array([[self.E_n, self.E_n * self.tan_psi], [0.0, self.G_s]])
            dlam_deps = A_inv @ N_T_De
            D_ep_2D -= De_M @ dlam_deps

        elif active_c and active_s:
            A_inv = np.linalg.inv([[self.E_n + self.H_c, -self.E_n * self.tan_psi],
                                   [-self.E_n * self.tan_phi,
                                    self.G_s + self.E_n * self.tan_phi * self.tan_psi + self.H_s]])
            N_T_De = np.array([[-self.E_n, 0.0], [self.E_n * self.tan_phi, self.G_s]])
            De_M = np.array([[-self.E_n, self.E_n * self.tan_psi], [0.0, self.G_s]])
            dlam_deps = A_inv @ N_T_De
            D_ep_2D -= De_M @ dlam_deps

        elif active_t:
            dlam_deps[0, 0] = self.E_n / (self.E_n + self.H_t)
            D_ep_2D[0, 0] -= self.E_n ** 2 / (self.E_n + self.H_t)

        elif active_c:
            dlam_deps[0, 0] = -self.E_n / (self.E_n + self.H_c)
            D_ep_2D[0, 0] -= self.E_n ** 2 / (self.E_n + self.H_c)

        elif active_s:
            A = self.G_s + self.E_n * self.tan_phi * self.tan_psi + self.H_s
            dlam_deps[1, 0] = (self.E_n * self.tan_phi) / A
            dlam_deps[1, 1] = self.G_s / A
            D_ep_2D[0, 0] -= (self.E_n ** 2 * self.tan_phi * self.tan_psi) / A
            D_ep_2D[0, 1] -= (self.E_n * self.G_s * self.tan_psi) / A
            D_ep_2D[1, 0] -= (self.E_n * self.G_s * self.tan_phi) / A
            D_ep_2D[1, 1] -= self.G_s ** 2 / A

        D_tan_l = self.D_local.copy()

        # Secant update
        D_tan_l[2, :] *= (1.0 - self.D_n_trial)
        D_tan_l[:, 2] *= (1.0 - self.D_n_trial)
        D_tan_l[4, :] *= (1.0 - self.D_s_trial)
        D_tan_l[:, 4] *= (1.0 - self.D_s_trial)
        D_tan_l[5, :] *= (1.0 - self.D_s_trial)
        D_tan_l[:, 5] *= (1.0 - self.D_s_trial)

        v1 = tau_23_tr / tau_tr if tau_tr > 1e-12 else 1.0
        v2 = tau_13_tr / tau_tr if tau_tr > 1e-12 else 0.0

        D_tan_l[2, 2] = (1.0 - self.D_n_trial) * max(D_ep_2D[0, 0], self.E_min)
        D_tan_l[2, 4] = (1.0 - self.D_n_trial) * D_ep_2D[0, 1] * v1
        D_tan_l[2, 5] = (1.0 - self.D_n_trial) * D_ep_2D[0, 1] * v2
        D_tan_l[4, 2] = (1.0 - self.D_s_trial) * D_ep_2D[1, 0] * v1
        D_tan_l[5, 2] = (1.0 - self.D_s_trial) * D_ep_2D[1, 0] * v2

        G_alg = max(D_ep_2D[1, 1], self.E_min)
        G_sec = max(self.G_s * shear_ratio, self.E_min)

        D_tan_l[4, 4] = (1.0 - self.D_s_trial) * (G_alg * v1 ** 2 + G_sec * v2 ** 2)
        D_tan_l[5, 5] = (1.0 - self.D_s_trial) * (G_alg * v2 ** 2 + G_sec * v1 ** 2)
        D_tan_l[4, 5] = D_tan_l[5, 4] = (1.0 - self.D_s_trial) * (G_alg - G_sec) * v1 * v2

        # Damage Tangent (Производная Damage по деформациям)
        if active_t and r_t < 1.0:
            dDn_dWp = (2.0 - 2.0 * r_t) / self.Gf_t
            dDn_deps_n = dDn_dWp * sig_n * dlam_deps[0, 0]
            dDn_deps_s = dDn_dWp * sig_n * dlam_deps[0, 1]

            D_tan_l[2, 2] -= sig_n * dDn_deps_n
            D_tan_l[2, 4] -= sig_n * dDn_deps_s * v1
            D_tan_l[2, 5] -= sig_n * dDn_deps_s * v2

        if active_s and r_s < 1.0:
            dDs_dWp = (2.0 - 2.0 * r_s) / self.Gf_s
            dDs_deps_n = dDs_dWp * tau * dlam_deps[1, 0]
            dDs_deps_s = dDs_dWp * tau * dlam_deps[1, 1]

            D_tan_l[4, 2] -= tau_23 * dDs_deps_n
            D_tan_l[4, 4] -= tau_23 * dDs_deps_s * v1
            D_tan_l[4, 5] -= tau_23 * dDs_deps_s * v2
            D_tan_l[5, 2] -= tau_13 * dDs_deps_n
            D_tan_l[5, 4] -= tau_13 * dDs_deps_s * v1
            D_tan_l[5, 5] -= tau_13 * dDs_deps_s * v2

        # Обновление глобальных напряжений
        sig_l = self.D_local @ e_l
        sig_l[2] = (1.0 - self.D_n_trial) * sig_n
        sig_l[4] = (1.0 - self.D_s_trial) * tau_23
        sig_l[5] = (1.0 - self.D_s_trial) * tau_13

        self.stress = self.T_eps.T @ sig_l
        self.D_tangent = self.T_eps.T @ D_tan_l @ self.T_eps

        self.D_tangent += np.eye(6) * 1e-9

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
