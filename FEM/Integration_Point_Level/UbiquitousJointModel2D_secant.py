import numpy as np

from FEM.Integration_Point_Level.CriticalPlane.criterion import (
    find_critical_plane_shear,
    find_critical_plane_tensile,
    find_critical_plane_compression,
    get_tensile_limit,
    get_compression_limit,
    get_cohesion_limit
)
from FEM.Integration_Point_Level.CriticalPlane.tensor import StressTensor
from FEM.Abstract.Integration_Point_Level import ConstitutiveModel


class UbiquitousJointModel2D(ConstitutiveModel):
    """
    2D Ubiquitous-Joint Damage-Plasticity модель.
    Реализован точный аналитический Return Mapping для напряжений.
    Пластическая работа вычисляется через эффективные напряжения для
    соответствия алгоритмическому разделению (decoupling) по Minga et al. 2017.
    """

    def __init__(self, material):
        super().__init__(material)

        E = self.material.E
        nu = self.material.nu
        jp = self.material.joint_params

        self.E_min = 1e-5 * E

        self.phi = np.radians(jp.get('phi', 30.0))
        self.psi = np.radians(jp.get('psi', 10.0))
        self.phi_r = np.radians(jp.get('phi_r', np.degrees(self.phi)))
        self.tan_phi = np.tan(self.phi)
        self.tan_psi = np.tan(self.psi)
        self.tan_phi_r = np.tan(self.phi_r)

        self.cp_material = jp.get('cp_material', None)
        if self.cp_material is None:
            raise ValueError("Требуется 'cp_material' в joint_params!")

        self.cp_num_planes = jp.get('cp_num_planes', 100)
        self.lock_on_yield = jp.get('lock_on_yield', True)

        self.preset_plane_normal = jp.get('preset_plane_normal', None)
        self.force_horizontal = jp.get('force_horizontal', False)
        if self.force_horizontal:
            self.preset_plane_normal = np.array([0.0, 1.0, 0])

        self.l_c = jp.get('l_c', 1.0)
        self.Gf_t = jp.get('Gf_t', 1.0) / self.l_c
        self.Gf_c = jp.get('Gf_c', 1.0) / self.l_c
        self.Gf_s = jp.get('Gf_s', 1.0) / self.l_c

        self.a_t = jp.get('a_t', 1.0)
        self.a_s = jp.get('a_s', 1.0)
        self.mu = jp.get('mu', 0.1)
        self.fcr_over_fc = jp.get('fcr_over_fc', 0.0)

        self.D_rock = self._build_plane_stress_stiffness(E, nu)
        self.D_tangent = self.D_rock.copy()

        self.is_locked = False
        self.fixed_normal = None
        self.T_sig = np.eye(3)
        self.T_eps = np.eye(3)
        self.D_local = self.D_rock.copy()

        self.E_n = self.D_local[0, 0]
        self.G_s = self.D_local[2, 2]

        self.f_t = self.f_c = self.c = self.q_lim = self.H_t = 0.0

        self._init_history()

    def _init_history(self):
        self.sig_eff_old = np.zeros(3)
        self.eps_p_old = np.zeros(3)
        self.q_old = 0.0
        self.W_pl_t_old = self.W_pl_c_old = self.W_pl_s_old = 0.0
        self.stress_old = np.zeros(3)
        self.strain_old = np.zeros(3)
        self.stress = np.zeros(3)
        self.strain = np.zeros(3)
        self._reset_trial()

    def _reset_trial(self):
        self.sig_eff_trial = self.sig_eff_old.copy()
        self.eps_p_trial = self.eps_p_old.copy()
        self.q_trial = self.q_old
        self.W_pl_t_trial = self.W_pl_t_old
        self.W_pl_c_trial = self.W_pl_c_old
        self.W_pl_s_trial = self.W_pl_s_old

    def _build_plane_stress_stiffness(self, E, nu):
        D = np.zeros((3, 3))
        c1 = E / (1.0 - nu ** 2)
        c2 = E * nu / (1.0 - nu ** 2)
        G = E / (2.0 * (1.0 + nu))
        D[0, 0] = D[1, 1] = c1
        D[0, 1] = D[1, 0] = c2
        D[2, 2] = G
        return D

    def _build_2d_voigt_transformation(self, nx, ny):
        T_sig = np.zeros((3, 3))
        T_sig[0, 0] = nx ** 2
        T_sig[0, 1] = ny ** 2
        T_sig[0, 2] = 2.0 * nx * ny
        T_sig[1, 0] = ny ** 2
        T_sig[1, 1] = nx ** 2
        T_sig[1, 2] = -2.0 * nx * ny
        T_sig[2, 0] = -nx * ny
        T_sig[2, 1] = nx * ny
        T_sig[2, 2] = nx ** 2 - ny ** 2

        N = np.diag([1.0, 1.0, 2.0])
        N_inv = np.diag([1.0, 1.0, 0.5])
        T_eps = N @ T_sig @ N_inv
        return T_sig, T_eps

    def _lock_plane(self, normal, stress_tensor_3d):
        nx, ny = normal[0], normal[1]
        norm = np.hypot(nx, ny)
        if norm < 1e-12:
            nx, ny = 1.0, 0.0
        else:
            nx /= norm
            ny /= norm

        self.fixed_normal = np.array([nx, ny])
        self.T_sig, self.T_eps = self._build_2d_voigt_transformation(nx, ny)

        self.D_local = self.T_sig @ self.D_rock @ np.linalg.inv(self.T_eps)
        self.E_n = self.D_local[0, 0]
        self.G_s = self.D_local[2, 2]

        self.f_t = max(get_tensile_limit(normal, self.cp_material), 1e-12)
        self.f_c = max(get_compression_limit(normal, self.cp_material), 1e-12)
        self.c = max(get_cohesion_limit(normal, stress_tensor_3d, self.cp_material), 1e-12)

        denom = (1.0 + self.f_t ** 2 / (3.0 * self.E_n * self.Gf_t)) * self.mu - 1.0
        self.H_t = abs(self.E_n / denom) if denom != 0 else self.E_n * 0.1
        self.H_t = min(self.H_t, self.E_n * 0.9)

        self.q_lim = np.inf if self.tan_phi < 1e-12 else self.c / self.tan_phi - self.f_t
        self.is_locked = True

    def _c_curr(self, q):
        if q <= self.q_lim: return self.c
        return self.c + (q - self.q_lim) * self.tan_phi

    def _return_mapping_stress(self, deps_l):
        sig_tr = self.sig_eff_old + self.D_local @ deps_l
        sn_tr = sig_tr[0]
        tau_tr = sig_tr[2]

        abs_tau_tr = abs(tau_tr)
        sign_tau = 1.0 if tau_tr >= 0 else -1.0

        ft_curr = self.f_t + self.q_old
        c_curr = self._c_curr(self.q_old)

        F1_tr = sn_tr - ft_curr
        F3_tr = -sn_tr - self.f_c
        F2_tr = abs_tau_tr + sn_tr * self.tan_phi - c_curr

        sig_eff_new = sig_tr.copy()
        dlams = np.zeros(3)

        if F1_tr <= 1e-12 and F2_tr <= 1e-12 and F3_tr <= 1e-12:
            return sig_eff_new, dlams, 0.0

        denom_s = self.G_s + self.E_n * self.tan_phi * self.tan_psi

        if F1_tr > 1e-12 and (abs_tau_tr + ft_curr * self.tan_phi - c_curr) <= 1e-12:
            dlam_t = F1_tr / (self.E_n + self.H_t)
            m_vec = np.array([1.0, 0.0, 0.0])
            sig_eff_new -= dlam_t * (self.D_local @ m_vec)
            dlams[0] = dlam_t

        elif F3_tr > 1e-12 and (abs_tau_tr - self.f_c * self.tan_phi - c_curr) <= 1e-12:
            dlam_c = F3_tr / self.E_n
            m_vec = np.array([-1.0, 0.0, 0.0])
            sig_eff_new -= dlam_c * (self.D_local @ m_vec)
            dlams[1] = dlam_c

        elif (F1_tr > 1e-12 and (abs_tau_tr + ft_curr * self.tan_phi - c_curr > 1e-12)) or \
             (F2_tr > 1e-12 and (F1_tr - (F2_tr / denom_s) * self.E_n * self.tan_psi > 1e-12)):
            A = np.array([
                [self.E_n + self.H_t, self.E_n * self.tan_psi],
                [self.E_n * self.tan_phi, self.G_s + self.E_n * self.tan_phi * self.tan_psi]
            ])
            B = np.array([F1_tr, F2_tr])
            try:
                x = np.linalg.solve(A, B)
                dlam_t, dlam_s = x[0], x[1]
                if dlam_t < 0 or dlam_s < 0:
                    dlam_t = max(dlam_t, 0.0)
                    dlam_s = max(dlam_s, 0.0)
            except np.linalg.LinAlgError:
                dlam_t, dlam_s = 0.0, 0.0

            M_mat = np.column_stack([
                np.array([1.0, 0.0, 0.0]),
                np.array([self.tan_psi, 0.0, sign_tau])
            ])
            sig_eff_new -= self.D_local @ M_mat @ np.array([dlam_t, dlam_s])
            dlams[0] = dlam_t
            dlams[2] = dlam_s

        elif (F2_tr > 1e-12 and (F3_tr + (F2_tr / denom_s) * self.E_n * self.tan_psi > 1e-12)) or \
             (F3_tr > 1e-12 and (abs_tau_tr - self.f_c * self.tan_phi - c_curr > 1e-12)):
            A = np.array([
                [self.E_n, -self.E_n * self.tan_psi],
                [-self.E_n * self.tan_phi, self.G_s + self.E_n * self.tan_phi * self.tan_psi]
            ])
            B = np.array([F3_tr, F2_tr])
            try:
                x = np.linalg.solve(A, B)
                dlam_c, dlam_s = x[0], x[1]
                if dlam_c < 0 or dlam_s < 0:
                    dlam_c = max(dlam_c, 0.0)
                    dlam_s = max(dlam_s, 0.0)
            except np.linalg.LinAlgError:
                dlam_c, dlam_s = 0.0, 0.0

            M_mat = np.column_stack([
                np.array([-1.0, 0.0, 0.0]),
                np.array([self.tan_psi, 0.0, sign_tau])
            ])
            sig_eff_new -= self.D_local @ M_mat @ np.array([dlam_c, dlam_s])
            dlams[1] = dlam_c
            dlams[2] = dlam_s

        elif F2_tr > 1e-12:
            dlam_s = F2_tr / denom_s
            m_vec = np.array([self.tan_psi, 0.0, sign_tau])
            sig_eff_new -= dlam_s * (self.D_local @ m_vec)
            dlams[2] = dlam_s

        return sig_eff_new, dlams, dlams[0] * self.H_t

    def _calculate_damage(self, W_pl_t, W_pl_c, W_pl_s, q_new, sn_eff):
        r_t = min(W_pl_t / self.Gf_t, 1.0)
        r_c = min(W_pl_c / self.Gf_c, 1.0)
        r_s = min(W_pl_s / self.Gf_s, 1.0)

        Fp_t = r_t * (2.0 - r_t)
        Fp_c = 0.5 * (np.sin(np.pi * r_c - 0.5 * np.pi) + 1.0)
        Fp_s = r_s * (2.0 - r_s)

        dt = Fp_t + self.a_t * Fp_s * (1.0 - Fp_t)
        D_nt = 1.0 - (1.0 - dt) * self.f_t / (self.f_t + q_new + 1e-12)

        D_nc = (1.0 - self.fcr_over_fc) * Fp_c

        ds_base = min(max(self.a_s * Fp_t * (1 - Fp_s) * (1 - Fp_c) + Fp_s + Fp_c - Fp_s * Fp_c, 0.0), 1.0)
        if sn_eff < 0.0:
            abs_sn = -sn_eff
            D_s = ds_base * (self.c + abs_sn * (self.tan_phi - self.tan_phi_r)) / (
                    self.c + abs_sn * self.tan_phi + 1e-12)
        else:
            D_s = ds_base

        return min(D_nt, 0.999), min(D_nc, 0.999), min(D_s, 0.999)

    def _evaluate_local_stress(self, deps_l):
        # 1. Возврат на поверхность текучести для эффективных напряжений
        sig_eff_new, dlams, dq = self._return_mapping_stress(deps_l)

        # 2. ОЦЕНКА РАБОТЫ ПО MINGA ET AL. (Eq. 15):
        # Пластическая работа вычисляется строго через ЭФФЕКТИВНЫЕ напряжения.
        dW_t = max(sig_eff_new[0] * dlams[0], 0.0)
        dW_c = max(abs(sig_eff_new[0]) * dlams[1], 0.0)
        dW_s = max((abs(sig_eff_new[2]) + sig_eff_new[0] * self.tan_psi) * dlams[2], 0.0)

        # 3. Обновление значений работы и упрочнения
        W_pl_t = self.W_pl_t_old + dW_t
        W_pl_c = self.W_pl_c_old + dW_c
        W_pl_s = self.W_pl_s_old + dW_s
        q_new = self.q_old + dq

        # 4. Расчет новой поврежденности (напрямую от новой работы)
        D_nt, D_nc, D_s = self._calculate_damage(W_pl_t, W_pl_c, W_pl_s, q_new, sig_eff_new[0])
        D_n = D_nt if sig_eff_new[0] >= 0 else D_nc

        # 5. Итоговые номинальные напряжения
        sig_l_nom = sig_eff_new.copy()
        sig_l_nom[0] *= (1.0 - D_n)
        sig_l_nom[2] *= (1.0 - D_s)

        return sig_l_nom, sig_eff_new, dlams, dq, W_pl_t, W_pl_c, W_pl_s, q_new, D_n, D_s

    def update_state(self, current_strain_voigt):
        self.strain = current_strain_voigt.copy()
        self._reset_trial()

        if not self.is_locked:
            sig_tr = self.stress_old + self.D_rock @ (self.strain - self.strain_old)
            sig_tr_3d = np.array([sig_tr[0], sig_tr[1], 0.0, sig_tr[2], 0.0, 0.0])
            st = StressTensor(*sig_tr_3d)

            if self.preset_plane_normal is not None:
                self._lock_plane(self.preset_plane_normal, st)
            else:
                f_s, n_s, u_s = find_critical_plane_shear(st, self.cp_material, self.cp_num_planes)
                f_t, n_t, u_t = find_critical_plane_tensile(st, self.cp_material, self.cp_num_planes)
                f_c, n_c, u_c = find_critical_plane_compression(st, self.cp_material, self.cp_num_planes)

                max_f = max(f_s, f_t, f_c)
                if self.lock_on_yield and max_f <= 1e-10:
                    self.stress = sig_tr
                    self.D_tangent = self.D_rock.copy()
                    return self.stress.copy(), self.D_tangent
                else:
                    max_u = max(u_s, u_t, u_c)
                    best_n = n_s if max_u == u_s else (n_t if max_u == u_t else n_c)
                    self._lock_plane(best_n, st)

        deps_global = self.strain - self.strain_old
        deps_l = self.T_eps @ deps_global

        # 1. Получение текущего состояния
        res = self._evaluate_local_stress(deps_l)
        sig_l_nom = res[0]
        sig_eff_new = res[1]
        dlams = res[2]
        D_n = res[8]
        D_s = res[9]

        # 2. Построение матриц жесткости (Алгоритмическая Minga 2017)
        # Поскольку _evaluate_local_stress теперь строго соответствует decoupling,
        # численная производная автоматически даст корректную матрицу жесткости.
        h = 1e-8
        K_ep = np.zeros((3, 3))  # d(sig_eff)/d(eps)
        dD_deps = np.zeros((3, 3))  # d(D)/d(eps)

        for j in range(3):
            # Вперед
            deps_pos = deps_l.copy()
            deps_pos[j] += h
            res_pos = self._evaluate_local_stress(deps_pos)

            # Назад
            deps_neg = deps_l.copy()
            deps_neg[j] -= h
            res_neg = self._evaluate_local_stress(deps_neg)

            # Центральные разности
            K_ep[:, j] = (res_pos[1] - res_neg[1]) / (2.0 * h)
            dD_deps[0, j] = (res_pos[8] - res_neg[8]) / (2.0 * h)
            dD_deps[1, j] = 0.0
            dD_deps[2, j] = (res_pos[9] - res_neg[9]) / (2.0 * h)

        # Матрица (I - D) * K_ep
        I_minus_D = np.diag([1.0 - D_n, 1.0, 1.0 - D_s])
        K_ed = I_minus_D @ K_ep

        # Точная алгоритмическая матрица
        K_c_exact = K_ed.copy()
        for i in [0, 2]:
            for j in range(3):
                K_c_exact[i, j] -= dD_deps[i, j] * sig_eff_new[i]

        # Включение стабилизации (секущая упруго-поврежденная жесткость)
        use_secant_stabilization = True

        if use_secant_stabilization:
            K_local_tangent = K_ed
        else:
            K_local_tangent = K_c_exact

        # 3. Трансформация обратно в глобальные координаты
        self.stress = self.T_eps.T @ sig_l_nom
        self.D_tangent = self.T_eps.T @ K_local_tangent @ self.T_eps

        # 4. Сохранение Trial-значений
        self.sig_eff_trial = sig_eff_new
        self.q_trial = res[7]
        self.W_pl_t_trial = res[4]
        self.W_pl_c_trial = res[5]
        self.W_pl_s_trial = res[6]

        self.eps_p_trial = self.eps_p_old + np.array([
            dlams[0] - dlams[1] + dlams[2] * self.tan_psi,
            0,
            dlams[2] * (1.0 if sig_eff_new[2] >= 0 else -1.0)
        ])

        return self.stress.copy(), self.D_tangent

    def get_tangent_matrix(self):
        return self.D_tangent

    def get_stress(self, strain):
        return self.stress

    def commit(self):
        self.stress_old = self.stress.copy()
        self.strain_old = self.strain.copy()
        self.sig_eff_old = self.sig_eff_trial.copy()
        self.eps_p_old = self.eps_p_trial.copy()
        self.q_old = self.q_trial
        self.W_pl_t_old = self.W_pl_t_trial
        self.W_pl_c_old = self.W_pl_c_trial
        self.W_pl_s_old = self.W_pl_s_trial