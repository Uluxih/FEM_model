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


class UbiquitousJointModel3D(ConstitutiveModel):
    """
    3D Ubiquitous-Joint Damage-Plasticity модель в нотации Войгта.
    Включает аналитический расчет согласованной матрицы жёсткости (Consistent Tangent Stiffness).
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

        # --- Перекрёстные коэффициенты damage (Minga, Eq. 20, 29) ---
        self.a_t = jp.get('a_t', 1.0)
        self.a_s = jp.get('a_s', 1.0)

        # --- Параметр μ для остаточной нормальной деформации (Minga Eq. 23) ---
        self.mu = jp.get('mu', 0.1)

        # --- Доля остаточной прочности на сжатие fcr/fc (Minga Eq. 28) ---
        self.fcr_over_fc = jp.get('fcr_over_fc', 0.0)

        self.H_t = self.H_c = self.H_s = 0.0

        # --- Параметры Newton-Raphson ---
        self.nr_tol = jp.get('nr_tol', 1e-10)
        self.nr_max_iter = jp.get('nr_max_iter', 25)

        # --- Переменные состояния ---
        self._init_history()

        # --- Матрица жёсткости породы (Войгт) ---
        self.D_rock = self._build_isotropic_stiffness_voigt(E, nu)
        self.D_tangent = self.D_rock.copy()

        # --- Состояние локализации ---
        self.is_locked = False
        self.fixed_normal = None
        self.R = np.eye(3)
        self.T_sig = np.eye(6)
        self.T_eps = np.eye(6)

        self.D_local = self.D_rock.copy()
        self.E_n = self.D_local[2, 2]
        self.G_s = self.D_local[4, 4]

        # --- Прочностные пределы на плоскости ---
        self.f_t = 0.0
        self.f_c = 0.0
        self.c = 0.0
        self.q_lim = 0.0

        # Переменные для передачи в Tangent Stiffness
        self._last_J_inv = None
        self._last_active_flags = (False, False, False)
        self._last_sig_eff = np.zeros(3)
        self._last_d_lams = np.zeros(3)
        self._last_W_pls = np.zeros(3)
        self._last_q_new = 0.0

    # ============================ HISTORY ============================

    def _init_history(self):
        self.lam_t_old = self.lam_c_old = self.lam_s_old = 0.0
        self.eps_p_n_old = 0.0
        self.gamma_p_23_old = 0.0
        self.gamma_p_13_old = 0.0
        self.W_pl_t_old = self.W_pl_c_old = self.W_pl_s_old = 0.0
        self.D_nt_old = self.D_nc_old = self.D_s_old = 0.0
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
        self.gamma_p_23_trial = self.gamma_p_23_old
        self.gamma_p_13_trial = self.gamma_p_13_old
        self.W_pl_t_trial = self.W_pl_t_old
        self.W_pl_c_trial = self.W_pl_c_old
        self.W_pl_s_trial = self.W_pl_s_old
        self.D_nt_trial = self.D_nt_old
        self.D_nc_trial = self.D_nc_old
        self.D_s_trial = self.D_s_old

    # ===================== УПРУГОСТЬ И ПОВОРОТЫ =====================

    def _build_isotropic_stiffness_voigt(self, E, nu):
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
            ny = np.cross(nz, nx);
            ny /= np.linalg.norm(ny)
            nx = np.cross(ny, nz)
        else:
            ny = np.cross(nz, [0.0, 0.0, 1.0]);
            ny /= np.linalg.norm(ny)
            nx = np.cross(ny, nz)
        nx /= np.linalg.norm(nx)
        return np.column_stack((nx, ny, nz))

    def _build_voigt_transformation_matrices(self, R):
        Q = R.T
        T_sig = np.zeros((6, 6))
        for i in range(3):
            for j in range(3):
                T_sig[i, j] = Q[i, j] ** 2
        T_sig[0, 3] = 2 * Q[0, 0] * Q[0, 1];
        T_sig[0, 4] = 2 * Q[0, 1] * Q[0, 2];
        T_sig[0, 5] = 2 * Q[0, 0] * Q[0, 2]
        T_sig[1, 3] = 2 * Q[1, 0] * Q[1, 1];
        T_sig[1, 4] = 2 * Q[1, 1] * Q[1, 2];
        T_sig[1, 5] = 2 * Q[1, 0] * Q[1, 2]
        T_sig[2, 3] = 2 * Q[2, 0] * Q[2, 1];
        T_sig[2, 4] = 2 * Q[2, 1] * Q[2, 2];
        T_sig[2, 5] = 2 * Q[2, 0] * Q[2, 2]
        T_sig[3, 0] = Q[0, 0] * Q[1, 0];
        T_sig[3, 1] = Q[0, 1] * Q[1, 1];
        T_sig[3, 2] = Q[0, 2] * Q[1, 2]
        T_sig[4, 0] = Q[1, 0] * Q[2, 0];
        T_sig[4, 1] = Q[1, 1] * Q[2, 1];
        T_sig[4, 2] = Q[1, 2] * Q[2, 2]
        T_sig[5, 0] = Q[0, 0] * Q[2, 0];
        T_sig[5, 1] = Q[0, 1] * Q[2, 1];
        T_sig[5, 2] = Q[0, 2] * Q[2, 2]
        T_sig[3, 3] = Q[0, 0] * Q[1, 1] + Q[0, 1] * Q[1, 0]
        T_sig[3, 4] = Q[0, 1] * Q[1, 2] + Q[0, 2] * Q[1, 1]
        T_sig[3, 5] = Q[0, 0] * Q[1, 2] + Q[0, 2] * Q[1, 0]
        T_sig[4, 3] = Q[1, 0] * Q[2, 1] + Q[1, 1] * Q[2, 0]
        T_sig[4, 4] = Q[1, 1] * Q[2, 2] + Q[1, 2] * Q[2, 1]
        T_sig[4, 5] = Q[1, 0] * Q[2, 2] + Q[1, 2] * Q[2, 0]
        T_sig[5, 3] = Q[0, 0] * Q[2, 1] + Q[0, 1] * Q[2, 0]
        T_sig[5, 4] = Q[0, 1] * Q[2, 2] + Q[0, 2] * Q[2, 1]
        T_sig[5, 5] = Q[0, 0] * Q[2, 2] + Q[0, 2] * Q[2, 0]

        N = np.diag([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        N_inv = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
        T_eps = N @ T_sig @ N_inv
        return T_sig, T_eps

    # ============================ LOCKING ============================

    def _lock_plane(self, normal, stress_tensor):
        self.fixed_normal = normal
        self.R = self._build_rotation_matrix(normal)
        self.T_sig, self.T_eps = self._build_voigt_transformation_matrices(self.R)

        self.D_local = self.T_sig @ self.D_rock @ self.T_sig.T
        self.E_n = self.D_local[2, 2]
        self.G_s = self.D_local[4, 4]

        self.f_t = max(get_tensile_limit(normal, self.cp_material), 1e-12)
        self.f_c = max(get_compression_limit(normal, self.cp_material), 1e-12)
        self.c = max(get_cohesion_limit(normal, stress_tensor, self.cp_material), 1e-12)

        denom = (1.0 + self.f_t ** 2 / (3.0 * self.E_n * self.Gf_t)) * self.mu - 1.0
        self.H_t = abs(self.E_n / denom) if denom != 0 else 0.0

        if self.tan_phi < 1e-12:
            self.q_lim = np.inf
        else:
            self.q_lim = self.c / self.tan_phi - self.f_t

        self.H_c = self.H_s = 0.0
        self.is_locked = True

    # ===================== ВСПОМОГАТЕЛЬНЫЕ =====================

    def _c_curr(self, q):
        if q <= self.q_lim:
            return self.c
        return self.c + (q - self.q_lim) * self.tan_phi

    def _ft_curr(self, q):
        return self.f_t + q

    # ===================== NEWTON-RAPHSON RETURN MAPPING =====================

    def _return_mapping_nr(self, sig_n_tr, tau_23_tr, tau_13_tr):
        tol = self.nr_tol
        max_iter = self.nr_max_iter

        q_old = self.H_t * self.lam_t_old
        tau_tr = np.sqrt(tau_23_tr ** 2 + tau_13_tr ** 2)

        ft_yld = self._ft_curr(q_old)
        c_yld = self._c_curr(q_old)

        f_t_tr = (sig_n_tr - ft_yld) if np.isfinite(ft_yld) else -1.0
        f_c_tr = -sig_n_tr - self.f_c
        f_s_tr = tau_tr + sig_n_tr * self.tan_phi - c_yld

        tol_f_t = tol * max(self.f_t, 1.0)
        tol_f_c = tol * max(self.f_c, 1.0)
        tol_f_s = tol * max(self.c, 1.0)

        if f_t_tr <= tol_f_t and f_c_tr <= tol_f_c and f_s_tr <= tol_f_s:
            return sig_n_tr, tau_23_tr, tau_13_tr, 0.0, 0.0, 0.0, None, (False, False, False)

        act_t = f_t_tr > tol_f_t and np.isfinite(ft_yld)
        act_c = f_c_tr > tol_f_c and not act_t
        sn_proj = ft_yld if act_t else (-self.f_c if act_c else sig_n_tr)
        act_s = (tau_tr + sn_proj * self.tan_phi - c_yld) > tol_f_s

        scale_nr = max(self.E_n, self.G_s, 1.0)
        max_dlam = ((abs(sig_n_tr) + tau_tr + max(self.f_t, self.f_c, self.c)) / min(self.E_n, self.G_s)) * 1000.0
        nr_abs_tol = tol * max(self.f_t, self.f_c, self.c, abs(sig_n_tr), tau_tr, 1.0) * 1e4

        def _nr_solve(act_t, act_c, act_s):
            col_t = 3 if act_t else None
            col_c = (3 + int(act_t)) if act_c else None
            col_s = (3 + int(act_t) + int(act_c)) if act_s else None
            n = 3 + int(act_t) + int(act_c) + int(act_s)

            x = np.zeros(n)
            x[0], x[1], x[2] = sig_n_tr, tau_23_tr, tau_13_tr
            R_vec = np.zeros(n)
            J = np.zeros((n, n))

            converged = False
            for _ in range(max_iter):
                sn_i = x[0];
                t23_i = x[1];
                t13_i = x[2]
                dlt_i = x[col_t] if act_t else 0.0
                dlc_i = x[col_c] if act_c else 0.0
                dls_i = x[col_s] if act_s else 0.0

                tau_i = np.sqrt(t23_i ** 2 + t13_i ** 2)
                tau_si = max(tau_i, 1e-14)
                n23_i = t23_i / tau_si
                n13_i = t13_i / tau_si

                q_i = self.H_t * (self.lam_t_old + max(dlt_i, 0.0))
                ft_i = self._ft_curr(q_i)
                c_i = self._c_curr(q_i)

                R_vec = np.zeros(n)
                R_vec[0] = sn_i - sig_n_tr + self.E_n * (dlt_i - dlc_i + dls_i * self.tan_psi)
                R_vec[1] = t23_i - tau_23_tr + self.G_s * dls_i * n23_i
                R_vec[2] = t13_i - tau_13_tr + self.G_s * dls_i * n13_i
                row = 3
                if act_t: R_vec[row] = sn_i - (ft_i if np.isfinite(ft_i) else 1e30); row += 1
                if act_c: R_vec[row] = -sn_i - self.f_c; row += 1
                if act_s: R_vec[row] = tau_i + sn_i * self.tan_phi - c_i; row += 1

                if np.linalg.norm(R_vec) / scale_nr < tol:
                    converged = True
                    break

                J = np.zeros((n, n))
                J[0, 0] = 1.0
                if tau_si > 1e-10:
                    J[1, 1] = 1.0 + self.G_s * dls_i * n13_i ** 2 / tau_si
                    J[1, 2] = -self.G_s * dls_i * n23_i * n13_i / tau_si
                    J[2, 1] = J[1, 2]
                    J[2, 2] = 1.0 + self.G_s * dls_i * n23_i ** 2 / tau_si
                else:
                    J[1, 1] = 1.0;
                    J[2, 2] = 1.0

                if act_t: J[0, col_t] = self.E_n
                if act_c: J[0, col_c] = -self.E_n
                if act_s:
                    J[0, col_s] = self.E_n * self.tan_psi
                    if tau_si > 1e-10:
                        J[1, col_s] = self.G_s * n23_i
                        J[2, col_s] = self.G_s * n13_i

                row = 3
                if act_t:
                    J[row, 0] = 1.0;
                    J[row, col_t] = -self.H_t;
                    row += 1
                if act_c:
                    J[row, 0] = -1.0;
                    row += 1
                if act_s:
                    J[row, 0] = self.tan_phi
                    if tau_si > 1e-10:
                        J[row, 1] = n23_i;
                        J[row, 2] = n13_i
                    if act_t and q_i > self.q_lim:
                        J[row, col_t] = -self.H_t * self.tan_phi
                    row += 1

                try:
                    x -= np.linalg.solve(J, R_vec)
                except np.linalg.LinAlgError:
                    break

            converged = converged or (np.linalg.norm(R_vec) / scale_nr < tol * 1e5)

            # БЕЗОПАСНОЕ ОБРАЩЕНИЕ МАТРИЦЫ
            J_inv = None
            if converged:
                try:
                    J_inv = np.linalg.inv(J)
                except np.linalg.LinAlgError:
                    try:
                        # Если обычное обращение не удалось, используем псевдообратную матрицу
                        J_inv = np.linalg.pinv(J)
                    except np.linalg.LinAlgError:
                        pass


            return (x[0], x[1], x[2], x[col_t] if act_t else 0.0, x[col_c] if act_c else 0.0,
                    x[col_s] if act_s else 0.0, converged, J_inv)

        def _apex_solve():
            if self.G_s < 1e-12: return 0, 0, 0, 0, 0, 0, False, None
            dls = tau_tr / self.G_s
            dlt = (sig_n_tr - self.E_n * dls * self.tan_psi - self.f_t - self.H_t * self.lam_t_old) / (
                        self.E_n + self.H_t)
            if dlt < -1e-6: return 0, 0, 0, 0, 0, 0, False, None
            dlt = max(dlt, 0.0)
            sn = self.f_t + self.H_t * (self.lam_t_old + dlt)

            # --- Сборка Якобиана для APEX (активны растяжение и сдвиг, размер 5x5) ---
            J = np.zeros((5, 5))
            J[0, 0] = 1.0;
            J[0, 3] = self.E_n;
            J[0, 4] = self.E_n * self.tan_psi
            J[1, 1] = 1.0;
            J[2, 2] = 1.0
            if tau_tr > 1e-14:
                J[1, 4] = self.G_s * (tau_23_tr / tau_tr)
                J[2, 4] = self.G_s * (tau_13_tr / tau_tr)
            J[3, 0] = 1.0;
            J[3, 3] = -self.H_t

            # Уравнение для сдвига в вершине (tau = 0)
            if tau_tr > 1e-14:
                J[4, 1] = tau_23_tr / tau_tr
                J[4, 2] = tau_13_tr / tau_tr
            else:
                J[4, 1] = 1.0  # Заглушка от деления на ноль
            J[4, 4] = self.G_s

            try:
                J_inv = np.linalg.inv(J)
            except np.linalg.LinAlgError:
                # Псевдообратная матрица (Мура-Пенроуза) на случай плохой обусловленности
                J_inv = np.linalg.pinv(J)

            return sn, 0.0, 0.0, dlt, 0.0, dls, True, J_inv

        def _is_valid(sn, t23, t13, dl_t):
            tau = np.sqrt(t23 ** 2 + t13 ** 2)
            q = self.H_t * (self.lam_t_old + max(dl_t, 0.0))
            ft_v = self._ft_curr(q)
            return ((sn - ft_v) <= nr_abs_tol and
                    -sn - self.f_c <= nr_abs_tol and
                    tau + sn * self.tan_phi - self._c_curr(q) <= nr_abs_tol)

        def _check_multipliers(act_t, act_c, act_s, dlt, dlc, dls):
            tol_lam = 1e-7 + 1e-5 * max_dlam
            if act_t and (dlt < -tol_lam or dlt > max_dlam): return False
            if act_c and (dlc < -tol_lam or dlc > max_dlam): return False
            if act_s and (dls < -tol_lam or dls > max_dlam): return False
            return True

        sig_n, tau_23, tau_13, d_lam_t, d_lam_c, d_lam_s, conv, J_inv = _nr_solve(act_t, act_c, act_s)
        if conv and _check_multipliers(act_t, act_c, act_s, d_lam_t, d_lam_c, d_lam_s) and _is_valid(sig_n, tau_23,
                                                                                                     tau_13, d_lam_t):
            return sig_n, tau_23, tau_13, max(d_lam_t, 0.0), max(d_lam_c, 0.0), max(d_lam_s, 0.0), J_inv, (act_t, act_c,
                                                                                                           act_s)

        combinations = [
            (True, False, True), (False, True, True), (True, False, False),
            (False, True, False), (False, False, True), ("APEX", False, False)
        ]

        for test_t, test_c, test_s in combinations:
            if test_t == "APEX":
                sn, t23, t13, dlt, dlc, dls, conv, J_inv = _apex_solve()
                if conv:
                    q_i = self.H_t * (self.lam_t_old + dlt)
                    if (sn * self.tan_phi - self._c_curr(q_i)) < -nr_abs_tol:
                        conv = False
            else:
                sn, t23, t13, dlt, dlc, dls, conv, J_inv = _nr_solve(test_t, test_c, test_s)

            if not conv: continue
            check_t = True if test_t == "APEX" else test_t
            check_s = True if test_t == "APEX" else test_s

            if not _check_multipliers(check_t, test_c, check_s, dlt, dlc, dls): continue
            if _is_valid(sn, t23, t13, dlt):
                return sn, t23, t13, max(dlt, 0.0), max(dlc, 0.0), max(dls, 0.0), J_inv, (check_t, test_c, check_s)

        raise RuntimeError("Return mapping did not converge")

    # ===================== ИНТЕГРАЦИЯ НАПРЯЖЕНИЙ =====================

    def _integrate_stress(self, current_strain, update_history=False):
        if not self.is_locked:
            self._last_J_inv = None
            self._last_active_flags = (False, False, False)
            return self.stress_old + self.D_rock @ (current_strain - self.strain_old)

        e_l = self.T_eps @ current_strain
        e_l_eff = e_l.copy()
        e_l_eff[2] -= self.eps_p_n_old
        e_l_eff[4] -= self.gamma_p_23_old
        e_l_eff[5] -= self.gamma_p_13_old

        sig_l_tr = self.D_local @ e_l_eff
        sig_n_tr = sig_l_tr[2]
        tau_23_tr = sig_l_tr[4]
        tau_13_tr = sig_l_tr[5]

        sig_n, tau_23, tau_13, d_lam_t, d_lam_c, d_lam_s, J_inv, active_flags = \
            self._return_mapping_nr(sig_n_tr, tau_23_tr, tau_13_tr)

        tau = np.sqrt(tau_23 ** 2 + tau_13 ** 2)
        d_eps_p_n = d_lam_t - d_lam_c + d_lam_s * self.tan_psi

        lam_t_new = self.lam_t_old + d_lam_t
        q_new = self.H_t * lam_t_new

        dW_t = max(sig_n * d_lam_t, 0.0) if d_lam_t > 0 else 0.0
        dW_c = max(abs(sig_n) * d_lam_c, 0.0) if d_lam_c > 0 else 0.0
        dW_s = max((tau + sig_n * self.tan_psi) * d_lam_s, 0.0) if d_lam_s > 0 else 0.0

        W_pl_t = self.W_pl_t_old + dW_t
        W_pl_c = self.W_pl_c_old + dW_c
        W_pl_s = self.W_pl_s_old + dW_s

        r_t = min(W_pl_t / self.Gf_t, 1.0)
        r_c = min(W_pl_c / self.Gf_c, 1.0)
        r_s = min(W_pl_s / self.Gf_s, 1.0)

        Fp_t = r_t * (2.0 - r_t)
        Fp_c = 0.5 * (np.sin(np.pi * r_c - 0.5 * np.pi) + 1.0)
        Fp_s = r_s * (2.0 - r_s)

        dt = Fp_t + self.a_t * Fp_s * (1.0 - Fp_t)
        D_nt_new = 1.0 - (1.0 - dt) * self.f_t / (self.f_t + q_new + 1e-12)
        D_nc_new = (1.0 - self.fcr_over_fc) * Fp_c

        ds_base = min(max(self.a_s * Fp_t * (1 - Fp_s) * (1 - Fp_c) + Fp_s + Fp_c - Fp_s * Fp_c, 0.0), 1.0)

        if sig_n < 0.0:
            abs_sn = -sig_n
            D_s_new = ds_base * (self.c + abs_sn * (self.tan_phi - self.tan_phi_r)) / (
                        self.c + abs_sn * self.tan_phi + 1e-12)
        else:
            D_s_new = ds_base

        D_nt = min(max(D_nt_new, self.D_nt_old), 0.9)
        D_nc = min(max(D_nc_new, self.D_nc_old), 0.9)
        D_s = min(max(D_s_new, self.D_s_old), 0.9)

        sig_l = self.D_local @ e_l
        sig_l[2] = (1.0 - D_nt) * sig_n if sig_n >= 0.0 else (1.0 - D_nc) * sig_n
        sig_l[4] = (1.0 - D_s) * tau_23
        sig_l[5] = (1.0 - D_s) * tau_13

        if update_history:
            self.lam_t_trial = lam_t_new
            self.lam_c_trial = self.lam_c_old + d_lam_c
            self.lam_s_trial = self.lam_s_old + d_lam_s
            self.eps_p_n_trial = self.eps_p_n_old + d_eps_p_n
            tau_safe = max(tau, 1e-14)
            if d_lam_s > 0.0:
                self.gamma_p_23_trial = self.gamma_p_23_old + d_lam_s * (tau_23 / tau_safe)
                self.gamma_p_13_trial = self.gamma_p_13_old + d_lam_s * (tau_13 / tau_safe)
            self.W_pl_t_trial = W_pl_t
            self.W_pl_c_trial = W_pl_c
            self.W_pl_s_trial = W_pl_s
            self.D_nt_trial = D_nt
            self.D_nc_trial = D_nc
            self.D_s_trial = D_s

            # Сохранение данных для аналитической касательной
            self._last_J_inv = J_inv
            self._last_active_flags = active_flags
            self._last_sig_eff = np.array([sig_n, tau_23, tau_13])
            self._last_d_lams = np.array([d_lam_t, d_lam_c, d_lam_s])
            self._last_W_pls = np.array([W_pl_t, W_pl_c, W_pl_s])
            self._last_q_new = q_new

        return self.T_eps.T @ sig_l

    # ===================== АНАЛИТИЧЕСКИЙ КАСАТЕЛЬНЫЙ МОДУЛЬ =====================

    def _compute_analytical_tangent(self):
        J_inv = self._last_J_inv
        if J_inv is None:
            return None  # Переход к численному модулю

        act_t, act_c, act_s = self._last_active_flags
        sig_eff = self._last_sig_eff
        d_lams = self._last_d_lams
        W_pls = self._last_W_pls
        q_new = self._last_q_new

        col_t = 3 if act_t else None
        col_c = (3 + int(act_t)) if act_c else None
        col_s = (3 + int(act_t) + int(act_c)) if act_s else None

        D_e_loc = np.diag([self.E_n, self.G_s, self.G_s])
        tau_eff = np.sqrt(sig_eff[1] ** 2 + sig_eff[2] ** 2)

        D_ep_loc = J_inv[0:3, 0:3] @ D_e_loc

        d_lam_deps = np.zeros((3, 3))
        if act_t: d_lam_deps[0, :] = J_inv[col_t, 0:3] @ D_e_loc
        if act_c: d_lam_deps[1, :] = J_inv[col_c, 0:3] @ D_e_loc
        if act_s: d_lam_deps[2, :] = J_inv[col_s, 0:3] @ D_e_loc

        dW_deps = np.zeros((3, 3))
        if act_t and d_lams[0] > 0:
            dW_deps[0, :] = d_lams[0] * D_ep_loc[0, :] + sig_eff[0] * d_lam_deps[0, :]
        if act_c and d_lams[1] > 0:
            sign_c = -1.0 if sig_eff[0] < 0 else 1.0
            dW_deps[1, :] = d_lams[1] * sign_c * D_ep_loc[0, :] + abs(sig_eff[0]) * d_lam_deps[1, :]
        if act_s and d_lams[2] > 0:
            dtau_dsig = np.array([0.0, sig_eff[1] / max(tau_eff, 1e-14), sig_eff[2] / max(tau_eff, 1e-14)])
            term1 = dtau_dsig @ D_ep_loc + self.tan_psi * D_ep_loc[0, :]
            dW_deps[2, :] = d_lams[2] * term1 + (tau_eff + sig_eff[0] * self.tan_psi) * d_lam_deps[2, :]

        r_t = W_pls[0] / self.Gf_t
        dFpt_dWt = (2.0 - 2.0 * r_t) / self.Gf_t if r_t < 1.0 else 0.0
        Fpt = r_t * (2.0 - r_t) if r_t < 1.0 else 1.0

        r_c = W_pls[1] / self.Gf_c
        dFpc_dWc = 0.5 * np.pi * np.cos(np.pi * r_c - 0.5 * np.pi) / self.Gf_c if r_c < 1.0 else 0.0
        Fpc = 0.5 * (np.sin(np.pi * r_c - 0.5 * np.pi) + 1.0) if r_c < 1.0 else 1.0

        r_s = W_pls[2] / self.Gf_s
        dFps_dWs = (2.0 - 2.0 * r_s) / self.Gf_s if r_s < 1.0 else 0.0
        Fps = r_s * (2.0 - r_s) if r_s < 1.0 else 1.0

        dt = Fpt + self.a_t * Fps * (1.0 - Fpt)
        ddt_dWt = (1.0 - self.a_t * Fps) * dFpt_dWt
        ddt_dWs = self.a_t * (1.0 - Fpt) * dFps_dWs

        denom_t = self.f_t + q_new + 1e-12
        dDnt_ddt = self.f_t / denom_t
        dDnt_dlamt = (1.0 - dt) * self.f_t * self.H_t / (denom_t ** 2)
        dDnt_deps = dDnt_ddt * (ddt_dWt * dW_deps[0, :] + ddt_dWs * dW_deps[2, :]) + dDnt_dlamt * d_lam_deps[0, :]

        dDnc_deps = (1.0 - self.fcr_over_fc) * dFpc_dWc * dW_deps[1, :]
        Dnc_val = (1.0 - self.fcr_over_fc) * Fpc

        ddsbase_dWt = self.a_s * (1.0 - Fps) * (1.0 - Fpc) * dFpt_dWt
        ddsbase_dWc = (-self.a_s * Fpt * (1.0 - Fps) + 1.0 - Fps) * dFpc_dWc
        ddsbase_dWs = (-self.a_s * Fpt * (1.0 - Fpc) + 1.0 - Fpc) * dFps_dWs
        dDsbase_deps = ddsbase_dWt * dW_deps[0, :] + ddsbase_dWc * dW_deps[1, :] + ddsbase_dWs * dW_deps[2, :]
        ds_base = min(max(self.a_s * Fpt * (1 - Fps) * (1 - Fpc) + Fps + Fpc - Fps * Fpc, 0.0), 1.0)

        if sig_eff[0] < 0.0:
            x = -(1.0 - Dnc_val) * sig_eff[0]
            denom_s = self.c + x * self.tan_phi + 1e-12
            K_fric = (self.c + x * (self.tan_phi - self.tan_phi_r)) / denom_s
            dK_dx = -self.c * self.tan_phi_r / (denom_s ** 2)
            dx_dsign = -(1.0 - Dnc_val)
            dx_dDnc = sig_eff[0]
            dK_deps = dK_dx * (dx_dsign * D_ep_loc[0, :] + dx_dDnc * dDnc_deps)
            dDs_deps = dDsbase_deps * K_fric + ds_base * dK_deps
            D_s_new = ds_base * K_fric
        else:
            dDs_deps = dDsbase_deps
            D_s_new = ds_base

        D_nt_new = 1.0 - (1.0 - dt) * self.f_t / denom_t
        if D_nt_new > 0.899 or D_nt_new < self.D_nt_old: dDnt_deps *= 0.0
        if Dnc_val > 0.899 or Dnc_val < self.D_nc_old: dDnc_deps *= 0.0
        if D_s_new > 0.899 or D_s_new < self.D_s_old: dDs_deps *= 0.0

        D_nt = min(max(D_nt_new, self.D_nt_old), 0.899)
        D_nc = min(max(Dnc_val, self.D_nc_old), 0.899)
        D_s = min(max(D_s_new, self.D_s_old), 0.899)

        K_loc = np.zeros((3, 3))
        D_n = D_nt if sig_eff[0] >= 0.0 else D_nc
        dDn_deps = dDnt_deps if sig_eff[0] >= 0.0 else dDnc_deps

        K_loc[0, :] = (1.0 - D_n) * D_ep_loc[0, :] - sig_eff[0] * dDn_deps
        K_loc[1, :] = (1.0 - D_s) * D_ep_loc[1, :] - sig_eff[1] * dDs_deps
        K_loc[2, :] = (1.0 - D_s) * D_ep_loc[2, :] - sig_eff[2] * dDs_deps

        K_loc_6x6 = self.D_local.copy()
        idx = [2, 4, 5]
        for i in range(3):
            for j in range(3):
                K_loc_6x6[idx[i], idx[j]] = K_loc[i, j]

        return self.T_eps.T @ K_loc_6x6 @ self.T_eps

    def _compute_numerical_tangent(self, current_strain, eps=1e-7):
        # print("пришлось вызвать численную матрицу")
        D_num = np.zeros((6, 6))
        scale = max(np.linalg.norm(current_strain), 1.0)
        h = eps * scale
        for j in range(6):
            ep = current_strain.copy();
            ep[j] += h
            em = current_strain.copy();
            em[j] -= h
            D_num[:, j] = (self._integrate_stress(ep, update_history=False)
                           - self._integrate_stress(em, update_history=False)) / (2.0 * h)
        return D_num

    # ===================== ГЛАВНЫЙ ВХОД =====================

    def update_state(self, current_strain_voigt):
        current_strain = current_strain_voigt.copy()
        self.strain = current_strain
        self._reset_trial()

        if not self.is_locked:
            sig_tr = self.stress_old + self.D_rock @ (current_strain - self.strain_old)
            st = StressTensor(*sig_tr)
            self._lock_plane(np.array([0.0, 0.0, 1.0]), st)

        self.stress = self._integrate_stress(current_strain, update_history=True)

        # Пытаемся получить аналитическую матрицу
        D_alg = None
        if self.is_locked:
            if self._last_active_flags == (False, False, False):
                # Если новой пластики нет, матрица считается мгновенно
                D_alg = self._compute_damaged_elastic_tangent()
            else:
                try:
                    D_alg = self._compute_analytical_tangent()
                except Exception:
                    D_alg = None


        # Fallback на численную, если аналитическая не может быть вычислена (например, apex)
        if D_alg is None:
            D_alg = self._compute_numerical_tangent(current_strain)

        D_alg += np.eye(6) * self.E_min
        self.D_tangent = D_alg

        return self.stress.copy(), self.D_tangent

    def _compute_damaged_elastic_tangent(self):
        """Аналитическая матрица для упругого шага (с учетом накопленных повреждений)"""
        K_loc_6x6 = self.D_local.copy()
        sig_n = self._last_sig_eff[0]

        # Берем текущие значения повреждений (trial)
        D_n = self.D_nt_trial if sig_n >= 0.0 else self.D_nc_trial
        D_s = self.D_s_trial

        K_loc_6x6[2, 2] *= (1.0 - D_n)
        K_loc_6x6[4, 4] *= (1.0 - D_s)
        K_loc_6x6[5, 5] *= (1.0 - D_s)

        return self.T_eps.T @ K_loc_6x6 @ self.T_eps

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
        self.gamma_p_23_old = self.gamma_p_23_trial
        self.gamma_p_13_old = self.gamma_p_13_trial
        self.W_pl_t_old = self.W_pl_t_trial
        self.W_pl_c_old = self.W_pl_c_trial
        self.W_pl_s_old = self.W_pl_s_trial
        self.D_nt_old = self.D_nt_trial
        self.D_nc_old = self.D_nc_trial
        self.D_s_old = self.D_s_trial
