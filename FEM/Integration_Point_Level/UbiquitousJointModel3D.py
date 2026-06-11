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
    """
    3D Ubiquitous-Joint Damage-Plasticity модель в нотации Кельвина.

    Нотация Кельвина используется для представления напряжений, деформаций и
    матрицы жесткости, что обеспечивает их тензорные свойства в 6D пространстве.
    - Векторы напряжений/деформаций: [σxx, σyy, σzz, √2*τxy, √2*τyz, √2*τxz]
    - Матрица жесткости (изотропная): диагональные сдвиговые члены равны 2G.
    - Преобразования выполняются единой ортогональной матрицей T_k.

    Алгоритм:
      1. До локализации:        чисто упругий отклик породы D_rock_k.
      2. При срабатывании       критерия — фиксация критической плоскости
                                 (one-shot, навсегда).
      3. На зафиксированной плоскости:
           - F1 (растяжение) с линейным хардеингом H_t (через параметр mu).
           - F2 (Mohr-Coulomb сдвиг), non-associated с дилатансией psi.
           - F3 (сжатие) идеально-пластичное.
           - Damage Dnt, Dnc, Ds — нарастают по пластической работе и
             переводят эффективные напряжения в номинальные.
           - При сжатии на сдвиге сохраняется остаточное Coulomb-трение
             (residual friction angle phi_r).
    """

    # ============================ INIT ============================

    def __init__(self, material):
        super().__init__(material)

        E = self.material.E
        nu = self.material.nu
        jp = self.material.joint_params

        self.E_min = 1e-5 * E

        # --- Углы трения / дилатансии ---
        self.phi = np.radians(jp.get('phi', 30.0))
        self.psi = np.radians(jp.get('psi', 10.0))
        self.phi_r = np.radians(jp.get('phi_r', np.degrees(self.phi)))
        self.tan_phi = np.tan(self.phi)
        self.tan_psi = np.tan(self.psi)
        self.tan_phi_r = np.tan(self.phi_r)

        # --- Critical-plane материал ---
        self.cp_material = jp.get('cp_material', None)
        if self.cp_material is None:
            raise ValueError("Требуется 'cp_material' в joint_params!")

        # --- Энергии разрушения, регуляризованные l_c ---
        self.l_c = jp.get('l_c', 1.0)
        self.Gf_t = jp.get('Gf_t', 100.0) / self.l_c
        self.Gf_c = jp.get('Gf_c', 5000.0) / self.l_c
        self.Gf_s = jp.get('Gf_s', 500.0) / self.l_c

        # --- Перекрёстные коэффициенты в damage (Minga, Eq. 20, 29) ---
        self.a_t = jp.get('a_t', 1.0)  # влияние моды II на Dnt
        self.a_s = jp.get('a_s', 1.0)  # влияние моды I  на Ds

        # --- Параметр μ для остаточной нормальной деформации (Minga Eq. 23) ---
        self.mu = jp.get('mu', 0.1)

        # --- Доля остаточной прочности на сжатие fcr/fc (Minga Eq. 28) ---
        self.fcr_over_fc = jp.get('fcr_over_fc', 0.0)

        self.H_t = 0.0
        self.H_c = 0.0
        self.H_s = 0.0

        # --- Переменные состояния ---
        self._init_history()

        # Жёсткость породы и тангенциальная (в нотации Кельвина)
        self.D_rock = self._build_isotropic_stiffness_kelvin(E, nu)
        self.D_tangent = self.D_rock.copy()

        # Локализация трещины
        self.is_locked = False
        self.fixed_normal = None
        self.R = np.eye(3)
        self.T_k = np.eye(6)  # Единая матрица преобразования для напряжений и деформаций
        self.D_local = self.D_rock.copy()
        self.E_n = self.D_local[2, 2]
        # Модуль сдвига G_s извлекается из матрицы Кельвина (2*G)
        self.G = self.D_local[4, 4] / 2.0

        # Прочностные пределы на плоскости (заполняются при locked)
        self.f_t = 0.0
        self.f_c = 0.0
        self.c = 0.0
        self.q_lim = 0.0

        self.tangent_type = 'numerical'

    # ============================ HISTORY ============================

    def _init_history(self):
        # Пластические деформации и множители хранятся в физических величинах
        self.lam_t_old = self.lam_c_old = self.lam_s_old = 0.0
        self.eps_p_n_old = 0.0
        self.eps_p_23_old = 0.0  # phys
        self.eps_p_13_old = 0.0  # phys
        self.W_pl_t_old = self.W_pl_c_old = self.W_pl_s_old = 0.0
        self.D_nt_old = self.D_nc_old = self.D_s_old = 0.0
        self.stress_old = np.zeros(6)  # Kelvin
        self.strain_old = np.zeros(6)  # Kelvin
        self.stress = np.zeros(6)  # Kelvin
        self.strain = np.zeros(6)  # Kelvin
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
        self.D_nt_trial = self.D_nt_old
        self.D_nc_trial = self.D_nc_old
        self.D_s_trial = self.D_s_old

    # ===================== ELASTICITY & ROTATIONS =====================

    def _build_isotropic_stiffness_kelvin(self, E, nu):
        """Строит изотропную матрицу жесткости в нотации Кельвина."""
        D = np.zeros((6, 6))
        c1 = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
        c2 = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        G = E / (2.0 * (1.0 + nu))
        D[0:3, 0:3] = c2
        D[0, 0] = D[1, 1] = D[2, 2] = c1
        # В нотации Кельвина диагональные сдвиговые элементы равны 2G
        D[3, 3] = D[4, 4] = D[5, 5] = 2.0 * G
        return D

    def _build_rotation_matrix(self, n):
        nz = np.array(n, dtype=float).flatten()
        nz /= np.linalg.norm(nz)
        if abs(nz[2]) > 0.999:
            nx = np.array([1.0, 0.0, 0.0])
            ny = np.cross(nz, nx);
            ny /= np.linalg.norm(ny)
            nx = np.cross(ny, nz)
        else:
            ny = np.cross(nz, [0.0, 0.0, 1.0]);
            ny /= np.linalg.norm(ny)
            nx = np.cross(ny, nz)
        nx /= np.linalg.norm(nx)
        return np.column_stack((nx, ny, nz))

    def _build_kelvin_transformation_matrix(self, R):
        """
        Строит матрицу преобразования T_k для нотации Кельвина.
        В этой нотации T_k одинакова для напряжений и деформаций и является
        ортогональной (T_k.T @ T_k = I).
        sigma_local = T_k @ sigma_global
        eps_local   = T_k @ eps_global
        """
        Q = R.T
        T_sig_voigt = np.zeros((6, 6))

        # Построение матрицы преобразования для напряжений в нотации Войгта (T_sig)
        # (этот код остался без изменений)
        for i in range(3):
            for j in range(3):
                T_sig_voigt[i, j] = Q[i, j] ** 2
        T_sig_voigt[0, 3] = 2 * Q[0, 0] * Q[0, 1]
        T_sig_voigt[0, 4] = 2 * Q[0, 1] * Q[0, 2]
        T_sig_voigt[0, 5] = 2 * Q[0, 0] * Q[0, 2]
        T_sig_voigt[1, 3] = 2 * Q[1, 0] * Q[1, 1]
        T_sig_voigt[1, 4] = 2 * Q[1, 1] * Q[1, 2]
        T_sig_voigt[1, 5] = 2 * Q[1, 0] * Q[1, 2]
        T_sig_voigt[2, 3] = 2 * Q[2, 0] * Q[2, 1]
        T_sig_voigt[2, 4] = 2 * Q[2, 1] * Q[2, 2]
        T_sig_voigt[2, 5] = 2 * Q[2, 0] * Q[2, 2]
        T_sig_voigt[3, 0] = Q[0, 0] * Q[1, 0]
        T_sig_voigt[3, 1] = Q[0, 1] * Q[1, 1]
        T_sig_voigt[3, 2] = Q[0, 2] * Q[1, 2]
        T_sig_voigt[4, 0] = Q[1, 0] * Q[2, 0]
        T_sig_voigt[4, 1] = Q[1, 1] * Q[2, 1]
        T_sig_voigt[4, 2] = Q[1, 2] * Q[2, 2]
        T_sig_voigt[5, 0] = Q[0, 0] * Q[2, 0]
        T_sig_voigt[5, 1] = Q[0, 1] * Q[2, 1]
        T_sig_voigt[5, 2] = Q[0, 2] * Q[2, 2]
        T_sig_voigt[3, 3] = Q[0, 0] * Q[1, 1] + Q[0, 1] * Q[1, 0]
        T_sig_voigt[3, 4] = Q[0, 1] * Q[1, 2] + Q[0, 2] * Q[1, 1]
        T_sig_voigt[3, 5] = Q[0, 0] * Q[1, 2] + Q[0, 2] * Q[1, 0]
        T_sig_voigt[4, 3] = Q[1, 0] * Q[2, 1] + Q[1, 1] * Q[2, 0]
        T_sig_voigt[4, 4] = Q[1, 1] * Q[2, 2] + Q[1, 2] * Q[2, 1]
        T_sig_voigt[4, 5] = Q[1, 0] * Q[2, 2] + Q[1, 2] * Q[2, 0]
        T_sig_voigt[5, 3] = Q[0, 0] * Q[2, 1] + Q[0, 1] * Q[2, 0]
        T_sig_voigt[5, 4] = Q[0, 1] * Q[2, 2] + Q[0, 2] * Q[2, 1]
        T_sig_voigt[5, 5] = Q[0, 0] * Q[2, 2] + Q[0, 2] * Q[2, 0]

        # Преобразование из Войгта в Кельвина: T_k = M @ T_sig_voigt @ M_inv
        s2 = np.sqrt(2.0)
        M = np.diag([1.0, 1.0, 1.0, s2, s2, s2])
        M_inv = np.diag([1.0, 1.0, 1.0, 1.0 / s2, 1.0 / s2, 1.0 / s2])

        T_k = M @ T_sig_voigt @ M_inv
        return T_k

    # ============================ LOCKING ============================

    def _lock_plane(self, normal, stress_tensor):
        print("locked", normal)
        self.fixed_normal = normal
        self.R = self._build_rotation_matrix(normal)
        self.T_k = self._build_kelvin_transformation_matrix(self.R)

        # В нотации Кельвина D_local = T_k @ D_rock @ T_k.T
        # Для изотропного D_rock, D_local всегда равно D_rock.
        self.D_local = self.D_rock.copy()
        self.E_n = self.D_local[2, 2]
        self.G = self.D_local[4, 4] / 2.0  # Физический модуль сдвига

        #TODO переписать проверку на ноль
        # Прочностные пределы на плоскости
        self.f_t = max(get_tensile_limit(normal, self.cp_material), 1e-12)
        self.f_c = max(get_compression_limit(normal, self.cp_material), 1e-12)
        self.c = max(get_cohesion_limit(normal, stress_tensor, self.cp_material), 1e-12)

        # Модуль хардеинга H_t
        denom = (1.0 + self.f_t ** 2 / (3.0 * self.E_n * self.Gf_t)) * self.mu - 1.0
        # if denom >= -1e-12:
        #     self.H_t = -0.999 * self.E_n
        # else:
        #     self.H_t = self.E_n / denom
        self.H_t = self.E_n / denom
        self.H_t = abs(self.H_t)

        self.q_lim = self.c / self.tan_phi - self.f_t
        self.H_c = 0.0
        self.H_s = 0.0
        self.is_locked = True

    # ===================== UTILITY: yield/q evolution =====================

    def _c_curr(self, q):
        if q <= self.q_lim: return self.c
        return self.c + (q - self.q_lim) * self.tan_phi

    def _ft_curr(self, q):
        if q <= self.q_lim: return self.f_t + q
        return np.inf

    # ===================== STRESS INTEGRATION =====================

    def _integrate_stress(self, current_strain_k, update_history=False):
        if not self.is_locked:
            d_strain_k = current_strain_k - self.strain_old
            return self.stress_old + np.dot(self.D_rock, d_strain_k)

        # e_l_k = self.T_k @ current_strain_k
        e_l_k = np.dot(self.T_k, current_strain_k)
        s2 = np.sqrt(2.0)

        # Эффективные деформации (физические) для вычисления пробных напряжений
        e_l_eff = e_l_k.copy()
        e_l_eff[2] -= self.eps_p_n_old
        e_l_eff[4] -= self.eps_p_23_old * s2  # -> Kelvin
        e_l_eff[5] -= self.eps_p_13_old * s2  # -> Kelvin

        sig_l_tr_k = self.D_local @ e_l_eff
        sig_n_tr = sig_l_tr_k[2]

        # Физические касательные напряжения из вектора Кельвина
        tau_23_tr = sig_l_tr_k[4] / s2
        tau_13_tr = sig_l_tr_k[5] / s2
        tau_tr = np.sqrt(tau_23_tr ** 2 + tau_13_tr ** 2)

        q_old = self.H_t * self.lam_t_old
        ft_yld = self._ft_curr(q_old)
        fc_yld = self.f_c
        c_yld = self._c_curr(q_old)

        f_t_val = sig_n_tr - ft_yld if np.isfinite(ft_yld) else -1.0
        f_c_val = -sig_n_tr - fc_yld
        f_s_val = tau_tr + sig_n_tr * self.tan_phi - c_yld

        d_lam_t = d_lam_c = d_lam_s = 0.0
        tol = 1e-10 * max(self.f_t, self.f_c, self.c, 1.0)

        # Return mapping (используем физический модуль сдвига G)
        G_s = self.G
        if f_t_val > tol and f_s_val > tol:
            A = self.E_n + self.H_t
            B = self.E_n * self.tan_psi
            C = self.E_n * self.tan_phi
            D = G_s + self.E_n * self.tan_phi * self.tan_psi + self.H_s
            det = A * D - B * C
            d_lam_t = (D * f_t_val - B * f_s_val) / det
            d_lam_s = (-C * f_t_val + A * f_s_val) / det
            if d_lam_t < 0.0:
                d_lam_t = 0.0; d_lam_s = f_s_val / D
            elif d_lam_s < 0.0:
                d_lam_s = 0.0; d_lam_t = f_t_val / A
        elif f_c_val > tol and f_s_val > tol:
            A = self.E_n + self.H_c;
            B = -self.E_n * self.tan_psi
            C = -self.E_n * self.tan_phi;
            D = G_s + self.E_n * self.tan_phi * self.tan_psi + self.H_s
            det = A * D - B * C
            d_lam_c = (D * f_c_val - B * f_s_val) / det
            d_lam_s = (-C * f_c_val + A * f_s_val) / det
            if d_lam_c < 0.0:
                d_lam_c = 0.0; d_lam_s = f_s_val / D
            elif d_lam_s < 0.0:
                d_lam_s = 0.0; d_lam_c = f_c_val / A
        elif f_t_val > tol:
            d_lam_t = f_t_val / (self.E_n + self.H_t)
        elif f_c_val > tol:
            d_lam_c = f_c_val / (self.E_n + self.H_c)
        elif f_s_val > tol:
            d_lam_s = f_s_val / (G_s + self.E_n * self.tan_phi * self.tan_psi + self.H_s)

        # Обновление эффективных напряжений (физических)
        d_eps_p_n = d_lam_t - d_lam_c + d_lam_s * self.tan_psi
        sig_n = sig_n_tr - self.E_n * d_eps_p_n
        tau = max(tau_tr - G_s * d_lam_s, 0.0)

        shear_ratio = tau / tau_tr if tau_tr > 1e-12 else 0.0
        tau_23 = tau_23_tr * shear_ratio
        tau_13 = tau_13_tr * shear_ratio

        lam_t_new = self.lam_t_old + d_lam_t
        q_new = self.H_t * lam_t_new

        dW_t = max(sig_n * d_lam_t, 0.0) if d_lam_t > 0 else 0.0
        dW_c = max(abs(sig_n) * d_lam_c, 0.0) if d_lam_c > 0 else 0.0
        dW_s = max((tau + sig_n * self.tan_psi) * d_lam_s, 0.0) if d_lam_s > 0 else 0.0

        W_pl_t = self.W_pl_t_old + dW_t;
        W_pl_c = self.W_pl_c_old + dW_c;
        W_pl_s = self.W_pl_s_old + dW_s
        r_t = min(W_pl_t / self.Gf_t, 1.0);
        r_c = min(W_pl_c / self.Gf_c, 1.0);
        r_s = min(W_pl_s / self.Gf_s, 1.0)
        Fp_t = r_t * (2.0 - r_t);
        Fp_c = 0.5 * (np.sin(np.pi * r_c - 0.5 * np.pi) + 1.0);
        Fp_s = r_s * (2.0 - r_s)
        dt = Fp_t + self.a_t * Fp_s * (1.0 - Fp_t)
        D_nt_new = 1.0 - (1.0 - dt) * self.f_t / (self.f_t + q_new + 1e-12)
        D_nc_new = (1.0 - self.fcr_over_fc) * Fp_c
        ds_base = min(
            max((self.a_s * Fp_t * (1 - Fp_s) * (1 - Fp_c) + Fp_s * Fp_c + Fp_s * (1 - Fp_c) + Fp_c * (1 - Fp_s)), 0.0),
            1.0)

        if sig_n < 0.0:
            abs_sn = -sig_n
            D_s_new = ds_base * (self.c + abs_sn * (self.tan_phi - self.tan_phi_r)) / (
                        self.c + abs_sn * self.tan_phi + 1e-12)
        else:
            D_s_new = ds_base

        D_nt = min(max(D_nt_new, self.D_nt_old), 0.999)
        D_nc = min(max(D_nc_new, self.D_nc_old), 0.999)
        D_s = min(max(D_s_new, self.D_s_old), 0.999)

        # Сборка локальных напряжений в нотации Кельвина
        sig_l_k = self.D_local @ e_l_k
        if sig_n >= 0.0:
            sig_l_k[2] = (1.0 - D_nt) * sig_n
        else:
            sig_l_k[2] = (1.0 - D_nc) * sig_n
        sig_l_k[4] = (1.0 - D_s) * tau_23 * s2
        sig_l_k[5] = (1.0 - D_s) * tau_13 * s2

        if update_history:
            self.lam_t_trial = lam_t_new
            self.lam_c_trial = self.lam_c_old + d_lam_c
            self.lam_s_trial = self.lam_s_old + d_lam_s
            self.eps_p_n_trial = self.eps_p_n_old + d_eps_p_n
            if tau_tr > 1e-12:  # Обновляем физические пластические деформации
                self.eps_p_23_trial = self.eps_p_23_old + d_lam_s * (tau_23_tr / tau_tr)
                self.eps_p_13_trial = self.eps_p_13_old + d_lam_s * (tau_13_tr / tau_tr)
            else:
                self.eps_p_23_trial = self.eps_p_23_old
                self.eps_p_13_trial = self.eps_p_13_old
            self.W_pl_t_trial = W_pl_t;
            self.W_pl_c_trial = W_pl_c;
            self.W_pl_s_trial = W_pl_s
            self.D_nt_trial = D_nt;
            self.D_nc_trial = D_nc;
            self.D_s_trial = D_s

        # Обратное преобразование в глобальные координаты
        return self.T_k.T @ sig_l_k

    # ===================== NUMERICAL TANGENT =====================

    def _compute_numerical_tangent(self, current_strain_k, eps=1e-7):
        D_num = np.zeros((6, 6))
        scale = max(np.linalg.norm(current_strain_k), 1.0)
        h = eps * scale
        for j in range(6):
            ep_k = current_strain_k.copy();
            ep_k[j] += h
            em_k = current_strain_k.copy();
            em_k[j] -= h
            sp_k = self._integrate_stress(ep_k, update_history=False)
            sm_k = self._integrate_stress(em_k, update_history=False)
            D_num[:, j] = (sp_k - sm_k) / (2.0 * h)
        return D_num

    # ===================== MAIN ENTRY =====================

    def update_state(self, current_strain_k):
        self.strain = current_strain_k
        self._reset_trial()

        s2 = np.sqrt(2.0)
        if not self.is_locked:
            sig_tr_k = self.stress_old + self.D_rock @ (current_strain_k - self.strain_old)
            # Для функций поиска крит. плоскости нужен Voigt-вектор или тензор
            sig_tr_v = [sig_tr_k[0], sig_tr_k[1], sig_tr_k[2], sig_tr_k[3] / s2, sig_tr_k[4] / s2, sig_tr_k[5] / s2]
            st = StressTensor(*sig_tr_v)

            f_t_scaled, n_t, _ = find_critical_plane_tensile(
                StressTensor(*(np.array(sig_tr_v) / 1.0)), self.cp_material, mode='3D')
            f_sh, n_sh, _ = find_critical_plane_shear(st, self.cp_material, mode='3D')

            S_tensor = np.array([
                [sig_tr_v[0], sig_tr_v[3], sig_tr_v[5]],
                [sig_tr_v[3], sig_tr_v[1], sig_tr_v[4]],
                [sig_tr_v[5], sig_tr_v[4], sig_tr_v[2]]
            ])
            eigvals, eigvecs = np.linalg.eigh(S_tensor)
            min_stress = eigvals[0]
            n_c = eigvecs[:, 0]
            f_c_limit = get_compression_limit(n_c, self.cp_material)
            v_c = -min_stress - f_c_limit

            if f_sh > 0 or f_t_scaled > 0 or v_c > 0:
                f_t_real, _, _ = find_critical_plane_tensile(st, self.cp_material, mode='3D')
                max_violation = max(f_sh, f_t_real, v_c)
                if max_violation == v_c:
                    best_n = n_c
                elif max_violation == f_sh:
                    best_n = n_sh
                else:
                    best_n = n_t
                self._lock_plane(best_n, st)
            else:
                self.stress = sig_tr_k
                self.D_tangent = self.D_rock
                return self.stress, self.D_tangent

        self.stress = self._integrate_stress(current_strain_k, update_history=True)
        self.D_tangent = self._compute_numerical_tangent(current_strain_k)
        self.D_tangent += np.eye(6) * self.E_min

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
        self.D_nt_old = self.D_nt_trial
        self.D_nc_old = self.D_nc_trial
        self.D_s_old = self.D_s_trial
