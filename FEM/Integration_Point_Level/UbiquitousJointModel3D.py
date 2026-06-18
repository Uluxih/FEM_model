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

    Соглашение по нотации Войгта:
      - Вектор напряжений : [σxx, σyy, σzz, τxy,  τyz,  τxz]
      - Вектор деформаций : [εxx, εyy, εzz, γxy,  γyz,  γxz]  (инженерные γ = 2ε)
      - Матрица жёсткости : диагональные сдвиговые члены = G  (не 2G!)

    В Войгте напряжения и деформации преобразуются РАЗНЫМИ матрицами:
      T_sig : σ_local = T_sig @ σ_global          (Bond matrix)
      T_eps : ε_local = T_eps @ ε_global           T_eps = N @ T_sig @ N_inv,
                                                   N = diag(1,1,1,2,2,2)
    Обратное преобразование напряжений:
      σ_global = T_eps.T @ σ_local                 (следует из ортогональности T_k)

    Алгоритм:
      1. До локализации         : чисто упругий отклик D_rock.
      2. При срабатывании       : фиксация критической плоскости (one-shot).
      3. На зафиксированной плоскости:
           F1 — растяжение, линейный хардеинг H_t (параметр mu).
           F2 — Mohr-Coulomb сдвиг, non-associated, дилатансия psi.
           F3 — сжатие, идеально-пластичное.
           Return mapping — Newton-Raphson по (σ_n, τ_23, τ_13).
           Damage Dnt / Dnc / Ds — по пластической работе.
           Остаточное трение phi_r при сжатии.
    """

    # ============================ INIT ============================

    def __init__(self, material):
        super().__init__(material)

        E  = self.material.E
        nu = self.material.nu
        jp = self.material.joint_params

        self.E_min = 1e-5 * E

        # --- Углы трения / дилатансии ---
        self.phi   = np.radians(jp.get('phi',   30.0))
        self.psi   = np.radians(jp.get('psi',   10.0))
        self.phi_r = np.radians(jp.get('phi_r', np.degrees(self.phi)))
        self.tan_phi   = np.tan(self.phi)
        self.tan_psi   = np.tan(self.psi)
        self.tan_phi_r = np.tan(self.phi_r)

        # --- Critical-plane материал ---
        self.cp_material = jp.get('cp_material', None)
        if self.cp_material is None:
            raise ValueError("Требуется 'cp_material' в joint_params!")

        # --- Энергии разрушения, регуляризованные l_c ---
        self.l_c  = jp.get('l_c',    1.0)
        self.Gf_t = jp.get('Gf_t',  100.0) / self.l_c
        self.Gf_c = jp.get('Gf_c', 5000.0) / self.l_c
        self.Gf_s = jp.get('Gf_s',  500.0) / self.l_c

        # --- Перекрёстные коэффициенты damage (Minga, Eq. 20, 29) ---
        self.a_t = jp.get('a_t', 1.0)   # влияние моды II на Dnt
        self.a_s = jp.get('a_s', 1.0)   # влияние моды I  на Ds

        # --- Параметр μ для остаточной нормальной деформации (Minga Eq. 23) ---
        self.mu = jp.get('mu', 0.1)

        # --- Доля остаточной прочности на сжатие fcr/fc (Minga Eq. 28) ---
        self.fcr_over_fc = jp.get('fcr_over_fc', 0.0)

        self.H_t = self.H_c = self.H_s = 0.0

        # --- Параметры Newton-Raphson ---
        self.nr_tol      = jp.get('nr_tol',      1e-10)
        self.nr_max_iter = jp.get('nr_max_iter',     25)

        # --- Переменные состояния ---
        self._init_history()

        # --- Матрица жёсткости породы (Войгт) ---
        self.D_rock    = self._build_isotropic_stiffness_voigt(E, nu)
        self.D_tangent = self.D_rock.copy()

        # --- Состояние локализации ---
        self.is_locked    = False
        self.fixed_normal = None
        self.R     = np.eye(3)
        self.T_sig = np.eye(6)   # преобразование напряжений
        self.T_eps = np.eye(6)   # преобразование деформаций

        self.D_local = self.D_rock.copy()
        self.E_n  = self.D_local[2, 2]
        self.G_s  = self.D_local[4, 4]   # G (Войгт: не 2G!)

        # --- Прочностные пределы на плоскости ---
        self.f_t   = 0.0
        self.f_c   = 0.0
        self.c     = 0.0
        self.q_lim = 0.0

        self.tangent_type = 'numerical'

    # ============================ HISTORY ============================

    def _init_history(self):
        self.lam_t_old = self.lam_c_old = self.lam_s_old = 0.0

        # Нормальная пластическая деформация (физическая = инженерная для нормали)
        self.eps_p_n_old = 0.0
        # Инженерные сдвиговые пластические деформации γ^p = 2ε^p
        self.gamma_p_23_old = 0.0
        self.gamma_p_13_old = 0.0

        self.W_pl_t_old = self.W_pl_c_old = self.W_pl_s_old = 0.0
        self.D_nt_old = self.D_nc_old = self.D_s_old = 0.0

        self.stress_old = np.zeros(6)   # Войгт
        self.strain_old = np.zeros(6)   # Войгт
        self.stress     = np.zeros(6)
        self.strain     = np.zeros(6)
        self._reset_trial()

    def _reset_trial(self):
        self.lam_t_trial = self.lam_t_old
        self.lam_c_trial = self.lam_c_old
        self.lam_s_trial = self.lam_s_old

        self.eps_p_n_trial    = self.eps_p_n_old
        self.gamma_p_23_trial = self.gamma_p_23_old
        self.gamma_p_13_trial = self.gamma_p_13_old

        self.W_pl_t_trial = self.W_pl_t_old
        self.W_pl_c_trial = self.W_pl_c_old
        self.W_pl_s_trial = self.W_pl_s_old

        self.D_nt_trial = self.D_nt_old
        self.D_nc_trial = self.D_nc_old
        self.D_s_trial  = self.D_s_old

    # ===================== УПРУГОСТЬ И ПОВОРОТЫ =====================

    def _build_isotropic_stiffness_voigt(self, E, nu):
        """
        Изотропная матрица жёсткости в нотации Войгта.
        Деформации: [εxx, εyy, εzz, γxy, γyz, γxz]  (инженерные γ).
        Диагональные сдвиговые члены = G  (не 2G, как в нотации Кельвина).
        """
        D  = np.zeros((6, 6))
        c1 = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
        c2 = E * nu          / ((1.0 + nu) * (1.0 - 2.0 * nu))
        G  = E / (2.0 * (1.0 + nu))
        D[0:3, 0:3] = c2
        D[0, 0] = D[1, 1] = D[2, 2] = c1
        D[3, 3] = D[4, 4] = D[5, 5] = G   # именно G, не 2G
        return D

    def _build_rotation_matrix(self, n):
        nz = np.array(n, dtype=float).flatten()
        nz /= np.linalg.norm(nz)
        if abs(nz[2]) > 0.999:
            nx = np.array([1.0, 0.0, 0.0])
            ny = np.cross(nz, nx);  ny /= np.linalg.norm(ny)
            nx = np.cross(ny, nz)
        else:
            ny = np.cross(nz, [0.0, 0.0, 1.0]);  ny /= np.linalg.norm(ny)
            nx = np.cross(ny, nz)
        nx /= np.linalg.norm(nx)
        return np.column_stack((nx, ny, nz))

    def _build_voigt_transformation_matrices(self, R):
        """
        Строит матрицы преобразования для нотации Войгта.

        T_sig  — Bond matrix для напряжений:   σ_local = T_sig @ σ_global
        T_eps  — матрица для деформаций:        ε_local = T_eps @ ε_global
                 T_eps = N @ T_sig @ N_inv,    N = diag(1,1,1,2,2,2)

        Ключевые свойства (следствие ортогональности T_k в нотации Кельвина):
          T_sig^{-1}  = T_eps^T   →  σ_global = T_eps.T @ σ_local
          T_eps^{-1}  = T_sig^T   →  D_local  = T_sig  @ D_global @ T_sig.T

        T_sig НЕ ортогональна (T_sig.T ≠ T_sig^{-1}).
        """
        Q = R.T   # переход из глобальной СК в локальную

        T_sig = np.zeros((6, 6))

        # нормальные строки → нормальные столбцы
        for i in range(3):
            for j in range(3):
                T_sig[i, j] = Q[i, j] ** 2

        # нормальные строки → сдвиговые столбцы (коэффициент 2)
        T_sig[0, 3] = 2*Q[0, 0]*Q[0, 1];  T_sig[0, 4] = 2*Q[0, 1]*Q[0, 2];  T_sig[0, 5] = 2*Q[0, 0]*Q[0, 2]
        T_sig[1, 3] = 2*Q[1, 0]*Q[1, 1];  T_sig[1, 4] = 2*Q[1, 1]*Q[1, 2];  T_sig[1, 5] = 2*Q[1, 0]*Q[1, 2]
        T_sig[2, 3] = 2*Q[2, 0]*Q[2, 1];  T_sig[2, 4] = 2*Q[2, 1]*Q[2, 2];  T_sig[2, 5] = 2*Q[2, 0]*Q[2, 2]

        # сдвиговые строки → нормальные столбцы (коэффициент 1/2 от Bond matrix)
        T_sig[3, 0] = Q[0, 0]*Q[1, 0];  T_sig[3, 1] = Q[0, 1]*Q[1, 1];  T_sig[3, 2] = Q[0, 2]*Q[1, 2]
        T_sig[4, 0] = Q[1, 0]*Q[2, 0];  T_sig[4, 1] = Q[1, 1]*Q[2, 1];  T_sig[4, 2] = Q[1, 2]*Q[2, 2]
        T_sig[5, 0] = Q[0, 0]*Q[2, 0];  T_sig[5, 1] = Q[0, 1]*Q[2, 1];  T_sig[5, 2] = Q[0, 2]*Q[2, 2]

        # сдвиговые строки → сдвиговые столбцы
        T_sig[3, 3] = Q[0, 0]*Q[1, 1] + Q[0, 1]*Q[1, 0]
        T_sig[3, 4] = Q[0, 1]*Q[1, 2] + Q[0, 2]*Q[1, 1]
        T_sig[3, 5] = Q[0, 0]*Q[1, 2] + Q[0, 2]*Q[1, 0]
        T_sig[4, 3] = Q[1, 0]*Q[2, 1] + Q[1, 1]*Q[2, 0]
        T_sig[4, 4] = Q[1, 1]*Q[2, 2] + Q[1, 2]*Q[2, 1]
        T_sig[4, 5] = Q[1, 0]*Q[2, 2] + Q[1, 2]*Q[2, 0]
        T_sig[5, 3] = Q[0, 0]*Q[2, 1] + Q[0, 1]*Q[2, 0]
        T_sig[5, 4] = Q[0, 1]*Q[2, 2] + Q[0, 2]*Q[2, 1]
        T_sig[5, 5] = Q[0, 0]*Q[2, 2] + Q[0, 2]*Q[2, 0]

        # T_eps = N @ T_sig @ N_inv,  N = diag(1,1,1,2,2,2)
        N     = np.diag([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        N_inv = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
        T_eps = N @ T_sig @ N_inv

        return T_sig, T_eps

    # ============================ LOCKING ============================

    def _lock_plane(self, normal, stress_tensor):
        print("locked", normal)
        self.fixed_normal = normal
        self.R = self._build_rotation_matrix(normal)
        self.T_sig, self.T_eps = self._build_voigt_transformation_matrices(self.R)

        # D_local = T_sig @ D_rock @ T_sig.T
        # Для изотропного D_rock результат совпадает с D_rock,
        # но формула корректна и в анизотропном случае.
        self.D_local = self.T_sig @ self.D_rock @ self.T_sig.T
        self.E_n = self.D_local[2, 2]
        self.G_s = self.D_local[4, 4]   # G (Войгт), НЕ делим на 2

        # Прочностные пределы
        self.f_t = max(get_tensile_limit(normal, self.cp_material),                    1e-12)
        self.f_c = max(get_compression_limit(normal, self.cp_material),                1e-12)
        self.c   = max(get_cohesion_limit(normal, stress_tensor, self.cp_material),    1e-12)

        # Модуль хардеинга H_t
        denom  = (1.0 + self.f_t**2 / (3.0 * self.E_n * self.Gf_t)) * self.mu - 1.0
        self.H_t = abs(self.E_n / denom)

        self.q_lim = self.c / self.tan_phi - self.f_t
        self.H_c = self.H_s = 0.0
        self.is_locked = True

    # ===================== ВСПОМОГАТЕЛЬНЫЕ =====================

    def _c_curr(self, q):
        if q <= self.q_lim:
            return self.c
        return self.c + (q - self.q_lim) * self.tan_phi

    def _ft_curr(self, q):
        if q <= self.q_lim:
            return self.f_t + q
        return np.inf

    # ===================== NEWTON-RAPHSON RETURN MAPPING =====================

    def _return_mapping_nr(self, sig_n_tr, tau_23_tr, tau_13_tr):
        """
        Newton–Raphson return mapping на критической плоскости (Minga 2017, Sec. 3.3).

        Все компоненты — физические (Войгт: τ без множителей √2).

        Поверхности текучести:
          F1 :  σ_n − ft_curr(λ_t) = 0              растяжение, лин. хардеинг
          F2 :  τ + σ_n·tan(φ) − c_curr = 0         Mohr-Coulomb (τ = ‖τ_23, τ_13‖)
          F3 : −σ_n − fc = 0                         сжатие, идеально-пластичное

        Правила течения (F2 — неассоциированное, дилатансия ψ):
          dε_n^p  = dλ_t − dλ_c + dλ_s·tan(ψ)    нормальная деформация
          dγ_23^p = dλ_s · τ_23/τ                  инженерная сдвиговая (γ = 2ε)
          dγ_13^p = dλ_s · τ_13/τ                  инженерная сдвиговая (γ = 2ε)

        Возвращает: (σ_n, τ_23, τ_13, dλ_t, dλ_c, dλ_s)
        """
        tol      = self.nr_tol
        max_iter = self.nr_max_iter

        q_old  = self.H_t * self.lam_t_old
        tau_tr = np.sqrt(tau_23_tr**2 + tau_13_tr**2)

        ft_yld = self._ft_curr(q_old)
        c_yld  = self._c_curr(q_old)

        f_t_tr = (sig_n_tr - ft_yld) if np.isfinite(ft_yld) else -1.0
        f_c_tr = -sig_n_tr - self.f_c
        f_s_tr = tau_tr + sig_n_tr * self.tan_phi - c_yld

        tol_f = tol * max(self.f_t, self.f_c, self.c, 1.0)

        # Упругий шаг
        if f_t_tr <= tol_f and f_c_tr <= tol_f and f_s_tr <= tol_f:
            return sig_n_tr, tau_23_tr, tau_13_tr, 0.0, 0.0, 0.0

        # Предсказание активного набора
        act_t = f_t_tr > tol_f and np.isfinite(ft_yld)
        act_c = f_c_tr > tol_f and not act_t
        act_s = f_s_tr > tol_f

        def _nr_solve(act_t, act_c, act_s):
            col_t = 3                              if act_t else None
            col_c = (3 + int(act_t))               if act_c else None
            col_s = (3 + int(act_t) + int(act_c)) if act_s else None
            n     = 3 + int(act_t) + int(act_c) + int(act_s)

            x = np.zeros(n)
            x[0], x[1], x[2] = sig_n_tr, tau_23_tr, tau_13_tr

            for _ in range(max_iter):
                sig_n  = x[0];  tau_23 = x[1];  tau_13 = x[2]
                d_lam_t = x[col_t] if act_t else 0.0
                d_lam_c = x[col_c] if act_c else 0.0
                d_lam_s = x[col_s] if act_s else 0.0

                tau   = np.sqrt(tau_23**2 + tau_13**2)
                tau_s = max(tau, 1e-14)
                n23, n13 = tau_23 / tau_s, tau_13 / tau_s

                q_curr  = self.H_t * (self.lam_t_old + d_lam_t)
                ft_curr = self._ft_curr(q_curr) if act_t else 0.0
                c_curr  = self._c_curr(q_curr)

                # ---- Вектор невязок ----
                # Упругий предиктор: dσ_n = E_n·dε_n^p,  dτ = G_s·dγ^p
                R_vec = np.zeros(n)
                R_vec[0] = sig_n - sig_n_tr  + self.E_n * (d_lam_t - d_lam_c + d_lam_s * self.tan_psi)
                R_vec[1] = tau_23 - tau_23_tr + self.G_s * d_lam_s * n23
                R_vec[2] = tau_13 - tau_13_tr + self.G_s * d_lam_s * n13
                row = 3
                if act_t:
                    R_vec[row] = sig_n  - ft_curr;                         row += 1
                if act_c:
                    R_vec[row] = -sig_n - self.f_c;                        row += 1
                if act_s:
                    R_vec[row] = tau + sig_n * self.tan_phi - c_curr;      row += 1

                if np.linalg.norm(R_vec) < tol:
                    break

                # ---- Якобиан ----
                J = np.zeros((n, n))
                J[0, 0] = 1.0
                J[1, 1] = 1.0 + self.G_s * d_lam_s * n13**2 / tau_s
                J[1, 2] = -self.G_s * d_lam_s * n23 * n13 / tau_s
                J[2, 1] = J[1, 2]
                J[2, 2] = 1.0 + self.G_s * d_lam_s * n23**2 / tau_s
                if act_t:
                    J[0, col_t] = self.E_n
                if act_c:
                    J[0, col_c] = -self.E_n
                if act_s:
                    J[0, col_s] = self.E_n * self.tan_psi
                    J[1, col_s] = self.G_s * n23
                    J[2, col_s] = self.G_s * n13

                row = 3
                if act_t:
                    J[row, 0]     =  1.0
                    J[row, col_t] = -self.H_t
                    row += 1
                if act_c:
                    J[row, 0] = -1.0
                    row += 1
                if act_s:
                    J[row, 0] = self.tan_phi
                    J[row, 1] = n23
                    J[row, 2] = n13
                    if act_t and q_curr > self.q_lim:
                        J[row, col_t] = -self.H_t * self.tan_phi
                    row += 1

                try:
                    x -= np.linalg.solve(J, R_vec)
                except np.linalg.LinAlgError:
                    break

            d_lam_t = x[col_t] if act_t else 0.0
            d_lam_c = x[col_c] if act_c else 0.0
            d_lam_s = x[col_s] if act_s else 0.0
            return x[0], x[1], x[2], d_lam_t, d_lam_c, d_lam_s

        sig_n, tau_23, tau_13, d_lam_t, d_lam_c, d_lam_s = _nr_solve(act_t, act_c, act_s)

        # Коррекция активного набора (corner return)
        if act_t and act_s:
            if d_lam_t < 0.0:
                sig_n, tau_23, tau_13, d_lam_t, d_lam_c, d_lam_s = _nr_solve(False, False, True)
            elif d_lam_s < 0.0:
                sig_n, tau_23, tau_13, d_lam_t, d_lam_c, d_lam_s = _nr_solve(True,  False, False)
        elif act_c and act_s:
            if d_lam_c < 0.0:
                sig_n, tau_23, tau_13, d_lam_t, d_lam_c, d_lam_s = _nr_solve(False, False, True)
            elif d_lam_s < 0.0:
                sig_n, tau_23, tau_13, d_lam_t, d_lam_c, d_lam_s = _nr_solve(False, True,  False)

        return sig_n, tau_23, tau_13, d_lam_t, d_lam_c, d_lam_s

    # ===================== ИНТЕГРАЦИЯ НАПРЯЖЕНИЙ =====================

    # ===================== ИНТЕГРАЦИЯ НАПРЯЖЕНИЙ =====================

    def _integrate_stress(self, current_strain, update_history=False):
        """
        Интегрирует напряжения в нотации Войгта.
          Вход  : current_strain = [εxx, εyy, εzz, γxy, γyz, γxz]
          Выход : [σxx, σyy, σzz, τxy, τyz, τxz]
        """
        if not self.is_locked:
            return self.stress_old + self.D_rock @ (current_strain - self.strain_old)

        # --- Переход в локальную СК ---
        e_l = self.T_eps @ current_strain

        # Эффективные деформации (вычет пластических)
        e_l_eff = e_l.copy()
        e_l_eff[2] -= self.eps_p_n_old
        e_l_eff[4] -= self.gamma_p_23_old
        e_l_eff[5] -= self.gamma_p_13_old

        # Пробные напряжения
        sig_l_tr = self.D_local @ e_l_eff
        sig_n_tr = sig_l_tr[2]
        tau_23_tr = sig_l_tr[4]
        tau_13_tr = sig_l_tr[5]

        # =================== NEWTON-RAPHSON RETURN MAPPING ===================
        sig_n, tau_23, tau_13, d_lam_t, d_lam_c, d_lam_s = \
            self._return_mapping_nr(sig_n_tr, tau_23_tr, tau_13_tr)
        # =====================================================================

        tau = np.sqrt(tau_23 ** 2 + tau_13 ** 2)
        d_eps_p_n = d_lam_t - d_lam_c + d_lam_s * self.tan_psi

        lam_t_new = self.lam_t_old + d_lam_t
        q_new = self.H_t * lam_t_new

        # Пластические работы
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

        # Параметры повреждения
        dt = Fp_t + self.a_t * Fp_s * (1.0 - Fp_t)
        D_nt_new = 1.0 - (1.0 - dt) * self.f_t / (self.f_t + q_new + 1e-12)
        D_nc_new = (1.0 - self.fcr_over_fc) * Fp_c

        ds_base = min(
            max(
                self.a_s * Fp_t * (1 - Fp_s) * (1 - Fp_c)
                + Fp_s * Fp_c
                + Fp_s * (1 - Fp_c)
                + Fp_c * (1 - Fp_s),
                0.0
            ),
            1.0
        )

        if sig_n < 0.0:
            abs_sn = -sig_n
            D_s_new = ds_base * (self.c + abs_sn * (self.tan_phi - self.tan_phi_r)) \
                      / (self.c + abs_sn * self.tan_phi + 1e-12)
        else:
            D_s_new = ds_base

        D_nt = min(max(D_nt_new, self.D_nt_old), 0.999)
        D_nc = min(max(D_nc_new, self.D_nc_old), 0.999)
        D_s = min(max(D_s_new, self.D_s_old), 0.999)

        # --- ОТЛАДОЧНЫЙ ВЫВОД (только на основном шаге) ---
        if update_history and (d_lam_s > 0 or self.D_s_old > 0):
            print(f"\n[DEBUG] --- Шаг интеграции ---")
            print(f"  sig_n = {sig_n:.2f}, tau = {tau:.2f}")
            print(f"  d_lam_s = {d_lam_s:.4e}, dW_s = {dW_s:.4e}")
            print(f"  W_pl_s_old = {self.W_pl_s_old:.4e}, W_pl_s = {W_pl_s:.4e}")
            print(f"  Gf_s = {self.Gf_s:.2f}, r_s = {r_s:.4f}, Fp_s = {Fp_s:.4f}")
            print(f"  Fp_t = {Fp_t:.4f}, Fp_c = {Fp_c:.4f}, ds_base = {ds_base:.4f}")
            if sig_n < 0.0:
                num = (self.c + abs_sn * (self.tan_phi - self.tan_phi_r))
                den = (self.c + abs_sn * self.tan_phi + 1e-12)
                print(f"  [Сжатие] num = {num:.2f}, den = {den:.2f}, D_s_new = {D_s_new:.4f}")
            print(f"  D_s_old = {self.D_s_old:.4f}, Итоговый D_s = {D_s:.4f}")
        # --------------------------------------------------a

        # Сборка локальных напряжений
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
            else:
                self.gamma_p_23_trial = self.gamma_p_23_old
                self.gamma_p_13_trial = self.gamma_p_13_old
            self.W_pl_t_trial = W_pl_t;
            self.W_pl_c_trial = W_pl_c;
            self.W_pl_s_trial = W_pl_s
            self.D_nt_trial = D_nt;
            self.D_nc_trial = D_nc;
            self.D_s_trial = D_s

        return self.T_eps.T @ sig_l

    # ===================== ЧИСЛЕННЫЙ КАСАТЕЛЬНЫЙ МОДУЛЬ =====================

    def _compute_numerical_tangent(self, current_strain, eps=1e-7):
        """Центральные разности в нотации Войгта — никаких конвертаций."""
        D_num = np.zeros((6, 6))
        scale = max(np.linalg.norm(current_strain), 1.0)
        h = eps * scale
        for j in range(6):
            ep = current_strain.copy();  ep[j] += h
            em = current_strain.copy();  em[j] -= h
            D_num[:, j] = (self._integrate_stress(ep, update_history=False)
                         - self._integrate_stress(em, update_history=False)) / (2.0 * h)
        return D_num

    # ===================== ГЛАВНЫЙ ВХОД =====================

    def update_state(self, current_strain_voigt):
        """
        Публичный интерфейс — нотация Войгта на входе и выходе:
          Вход  : [εxx, εyy, εzz, γxy,  γyz,  γxz]   инженерные γ = 2ε
          Выход : [σxx, σyy, σzz, τxy,  τyz,  τxz]   физические τ

        Никаких внутренних конвертаций √2 — всё в Войгте от начала до конца.
        """
        current_strain = current_strain_voigt.copy()
        self.strain = current_strain
        self._reset_trial()

        if not self.is_locked:
            sig_tr = self.stress_old + self.D_rock @ (current_strain - self.strain_old)
            st = StressTensor(*sig_tr)

            f_sh, n_sh, _ = find_critical_plane_shear(st, self.cp_material, mode='3D')
            f_t, n_t, _ = find_critical_plane_tensile(st, self.cp_material, mode='3D')
            f_c, n_c, _ = find_critical_plane_compression(st, self.cp_material, mode='3D')

            if f_sh > 0 or f_t > 0 or f_c > 0:
                max_v = max(f_sh, f_t, f_c)
                best_n = (n_c if max_v == f_c else
                          n_sh if max_v == f_sh else
                          n_t)
                self._lock_plane(best_n, st)
                print(f"[LOCK] Стресс при锁定: {st}")
                print(f"[LOCK] Когезия c = {self.c:.6f}")
            else:
                self.stress = sig_tr
                self.D_tangent = self.D_rock.copy()
                return sig_tr.copy(), self.D_tangent

        # Интеграция (внутри — чистый Войгт)
        self.stress = self._integrate_stress(current_strain, update_history=True)

        # Численный касательный модуль + минимальная жёсткость
        D_num = self._compute_numerical_tangent(current_strain)
        D_num += np.eye(6) * self.E_min
        self.D_tangent = D_num   # уже в Войгте

        return self.stress.copy(), self.D_tangent

    def get_tangent_matrix(self):
        return self.D_tangent

    def get_stress(self, strain):
        return self.stress

    def commit(self):
        self.stress_old     = self.stress.copy()
        self.strain_old     = self.strain.copy()
        self.lam_t_old      = self.lam_t_trial
        self.lam_c_old      = self.lam_c_trial
        self.lam_s_old      = self.lam_s_trial
        self.eps_p_n_old    = self.eps_p_n_trial
        self.gamma_p_23_old = self.gamma_p_23_trial
        self.gamma_p_13_old = self.gamma_p_13_trial
        self.W_pl_t_old     = self.W_pl_t_trial
        self.W_pl_c_old     = self.W_pl_c_trial
        self.W_pl_s_old     = self.W_pl_s_trial
        self.D_nt_old       = self.D_nt_trial
        self.D_nc_old       = self.D_nc_trial
        self.D_s_old        = self.D_s_trial
