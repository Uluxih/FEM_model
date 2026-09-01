
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
    2D Ubiquitous-Joint Damage-Plasticity модель с пластичностью матрицы.

    Архитектура (переписанная):
      * Главная переменная состояния — ЭФФЕКТИВНОЕ глобальное напряжение sig_eff_glob
        (неповрежденное). Вся пластичность (Друкер-Прагер матрицы и return mapping шва)
        работает в эффективном пространстве.
      * Номинальное (наблюдаемое) напряжение — выходная величина: (1-D) * sig_eff,
        поврежденность применяется РОВНО ОДИН РАЗ в локальных осях трещины.
      * Секущая матрица использует те же коэффициенты (1-D), что и номинальные
        напряжения (с нижним полом min_stiff_factor только для устойчивости решателя).

    Матрица: идеальная пластичность Друкера-Прагера (2D сглаженный Мор-Кулон).
    Трещина: аналитический return mapping с поврежденностью (Minga et al. 2017).
    """

    def __init__(self, material):
        super().__init__(material)

        E = self.material.E
        nu = self.material.nu
        jp = self.material.joint_params

        # --- Параметры матрицы (Друкер-Прагер) ---
        mp = getattr(self.material, 'matrix_params', {})
        self.phi_m = np.radians(mp.get('phi', 30.0))
        self.psi_m = np.radians(mp.get('psi', 0.0))
        self.c_m = mp.get('c', 1e6)
        self.sin_phi_m = np.sin(self.phi_m)
        self.cos_phi_m = np.cos(self.phi_m)
        self.sin_psi_m = np.sin(self.psi_m)

        # --- Параметры шва ---
        self.phi = np.radians(jp.get('phi', 30.0))
        self.psi = np.radians(jp.get('psi', 0.0))
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
            self.preset_plane_normal = np.array([0.0, 1.0, 0.0])

        # --- Энергии разрушения (регуляризация по l_c) ---
        self.l_c = jp.get('l_c', 1.0)
        self.Gf_t = jp.get('Gf_t', 1.0) / self.l_c
        self.Gf_c = jp.get('Gf_c', 1.0) / self.l_c
        self.Gf_s = jp.get('Gf_s', 1.0) / self.l_c

        self.a_t = jp.get('a_t', 1.0)
        self.a_s = jp.get('a_s', 1.0)
        self.mu = self.tan_phi
        self.fcr_over_fc = jp.get('fcr_over_fc', 0.0)

        # --- Численные параметры (вынесены из "магических чисел") ---
        # Капы поврежденности (единые для напряжений И жесткости)
        self.d_nt_max = jp.get('d_nt_max', 0.99)
        self.d_nc_max = jp.get('d_nc_max', 0.99)
        self.d_s_max = jp.get('d_s_max', 0.99)
        # Регуляризация: деградация матрицы вдоль трещины (0 = выключено)
        self.matrix_parallel_damage_cap = jp.get('matrix_parallel_damage_cap', 0.0)
        # Нижний пол секущей жесткости (только для устойчивости глобального Ньютона)
        self.min_stiff_factor = jp.get('min_stiff_factor', 0.05)
        self.debug = jp.get('debug', False)

        self.D_rock = self._build_plane_stress_stiffness(E, nu)
        self.D_tangent = self.D_rock.copy()

        # Флаги блокировки трещины (committed и trial)
        self.is_locked = False
        self.is_locked_trial = False

        self.fixed_normal = None
        self.T_sig = np.eye(3)
        self.T_eps = np.eye(3)
        self.D_local = self.D_rock.copy()

        self.E_n = self.D_local[0, 0]
        self.G_s = self.D_local[2, 2]

        self.f_t = self.f_c = self.c = self.q_lim = self.H_t = 0.0

        self.cp_material.mu = self.tan_phi

        self._init_history()

    # ------------------------------------------------------------------
    # История / trial-состояние
    # ------------------------------------------------------------------
    def _init_history(self):
        # ЭФФЕКТИВНОЕ глобальное напряжение — главная переменная состояния
        self.sig_eff_glob_old = np.zeros(3)
        self.eps_p_old = np.zeros(3)          # пластическая деформация, ГЛОБАЛЬНЫЕ оси
        self.q_old = 0.0
        self.W_pl_t_old = self.W_pl_c_old = self.W_pl_s_old = 0.0

        self.D_nt_old = 0.0
        self.D_nc_old = 0.0
        self.D_s_old = 0.0

        # Номинальные величины — только для вывода
        self.stress_old = np.zeros(3)
        self.strain_old = np.zeros(3)
        self.stress = np.zeros(3)
        self.strain = np.zeros(3)
        self._reset_trial()

    def _reset_trial(self):
        self.sig_eff_glob_trial = self.sig_eff_glob_old.copy()
        self.eps_p_trial = self.eps_p_old.copy()
        self.q_trial = self.q_old
        self.W_pl_t_trial = self.W_pl_t_old
        self.W_pl_c_trial = self.W_pl_c_old
        self.W_pl_s_trial = self.W_pl_s_old

        self.D_nt_trial = self.D_nt_old
        self.D_nc_trial = self.D_nc_old
        self.D_s_trial = self.D_s_old
        self.D_n_trial = 0.0

        # Сброс пробного флага трещины к подтвержденному состоянию
        self.is_locked_trial = self.is_locked

    # ------------------------------------------------------------------
    # Упругость / преобразования
    # ------------------------------------------------------------------
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
        T_eps = N @ T_sig @ N_inv          # T_eps = T_sig^{-T} (энергетическая сопряженность)
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
        if self.debug:
            print(f"[UJ2D] lock: n=({nx:.3f},{ny:.3f}) f_t={self.f_t:.3g} "
                  f"f_c={self.f_c:.3g} c={self.c:.3g}")

        # ВНИМАНИЕ: множитель self.mu в формуле H_t требует сверки с Minga et al. (2017).
        # Сохранен из исходной версии; добавлены защиты от вырождения знаменателя.
        term = (1.0 + self.f_t ** 2 / (3.0 * self.E_n * self.Gf_t)) * self.mu
        if term > 1e-9:
            self.H_t = self.E_n * (1.0 / term - 1.0)
        else:
            self.H_t = self.E_n * 0.1
        # Защита: E_n + H_t не должен вырождаться (return mapping делит на него)
        self.H_t = max(self.H_t, -0.5 * self.E_n)

        self.q_lim = np.inf if self.tan_phi < 1e-12 else self.c / self.tan_phi - self.f_t

        # Устанавливаем только TRIAL флаг; фиксация — в commit()
        self.is_locked_trial = True

    def _c_curr(self, q):
        if q <= self.q_lim:
            return self.c
        return self.c + (q - self.q_lim) * self.tan_phi

    # ------------------------------------------------------------------
    # Пластичность матрицы (Друкер-Прагер, эффективные напряжения)
    # ------------------------------------------------------------------
    def _return_mapping_matrix(self, sig_tr):
        p_tr = 0.5 * (sig_tr[0] + sig_tr[1])
        sx = 0.5 * (sig_tr[0] - sig_tr[1])
        txy = sig_tr[2]
        q_tr = np.hypot(sx, txy)

        F_tr = q_tr + p_tr * self.sin_phi_m - self.c_m * self.cos_phi_m
        if F_tr <= 1e-10:
            return sig_tr.copy()

        K_2D = 0.5 * (self.D_rock[0, 0] + self.D_rock[0, 1])
        G = self.D_rock[2, 2]

        denom = G + K_2D * self.sin_phi_m * self.sin_psi_m
        dlam = F_tr / denom

        p_new = p_tr - K_2D * self.sin_psi_m * dlam
        q_new = q_tr - G * dlam

        p_apex = (self.c_m * self.cos_phi_m / self.sin_phi_m
                  if self.sin_phi_m > 1e-12 else np.inf)

        if q_new < 0 or p_new > p_apex:
            p_new = min(p_apex, p_tr)
            q_new = 0.0

        sig_mat = np.zeros(3)
        if q_tr > 1e-12:
            sx_new = sx * (q_new / q_tr)
            txy_new = txy * (q_new / q_tr)
        else:
            sx_new = 0.0
            txy_new = 0.0

        sig_mat[0] = p_new + sx_new
        sig_mat[1] = p_new - sx_new
        sig_mat[2] = txy_new

        return sig_mat

    # ------------------------------------------------------------------
    # Return mapping шва (локальные ЭФФЕКТИВНЫЕ напряжения)
    # ------------------------------------------------------------------
    def _return_mapping_stress(self, sig_tr_local):
        """
        Возвращает: sig_eff_new (локальное эффективное), dlams=[t, c, s],
                    dq, deps_p_local (приращение пластической деформации, локальные оси).
        """
        sn_tr = sig_tr_local[0]
        tau_tr = sig_tr_local[2]

        abs_tau_tr = abs(tau_tr)
        sign_tau = 1.0 if tau_tr >= 0 else -1.0

        ft_curr = self.f_t + self.q_old
        c_curr = self._c_curr(self.q_old)

        F1_tr = sn_tr - ft_curr                                 # растяжение
        F3_tr = -sn_tr - self.f_c                               # сжатие
        F2_tr = abs_tau_tr + sn_tr * self.tan_phi - c_curr      # сдвиг

        dlams = np.zeros(3)
        tol = 1e-8

        if F1_tr <= tol and F2_tr <= tol and F3_tr <= tol:
            return sig_tr_local.copy(), dlams, 0.0, np.zeros(3)

        denom_t = self.E_n + self.H_t
        denom_s = self.G_s + self.E_n * self.tan_phi * self.tan_psi

        # --- Выбор активного набора поверхностей ---
        if F1_tr > tol and (abs_tau_tr + ft_curr * self.tan_phi - c_curr) <= tol:
            # Чистое растяжение
            dlams[0] = F1_tr / denom_t

        elif F3_tr > tol and (abs_tau_tr - self.f_c * self.tan_phi - c_curr) <= tol:
            # Чистое сжатие
            dlams[1] = F3_tr / self.E_n

        elif (F1_tr > tol and (abs_tau_tr + ft_curr * self.tan_phi - c_curr > tol)) or \
                (F2_tr > tol and (F1_tr - (F2_tr / denom_s) * self.E_n * self.tan_psi > tol)):
            # Угол растяжение + сдвиг
            A = np.array([
                [denom_t,                self.E_n * self.tan_psi],
                [self.E_n * self.tan_phi, denom_s]
            ])
            B = np.array([F1_tr, F2_tr])
            try:
                x = np.linalg.solve(A, B)
            except np.linalg.LinAlgError:
                x = np.zeros(2)

            if x[0] < 0.0:
                # Активна только сдвиговая поверхность
                dlams[2] = max(F2_tr, 0.0) / denom_s
            elif x[1] < 0.0:
                # Активна только поверхность растяжения
                dlams[0] = max(F1_tr, 0.0) / denom_t
            else:
                dlams[0], dlams[2] = x[0], x[1]

        elif (F2_tr > tol and (F3_tr + (F2_tr / denom_s) * self.E_n * self.tan_psi > tol)) or \
                (F3_tr > tol and (abs_tau_tr - self.f_c * self.tan_phi - c_curr > tol)):
            # Угол сжатие + сдвиг
            A = np.array([
                [self.E_n, -self.E_n * self.tan_psi],
                [-self.E_n * self.tan_phi,  denom_s]
            ])
            B = np.array([F3_tr, F2_tr])
            try:
                x = np.linalg.solve(A, B)
            except np.linalg.LinAlgError:
                x = np.zeros(2)

            if x[0] < 0.0:
                dlams[2] = max(F2_tr, 0.0) / denom_s
            elif x[1] < 0.0:
                dlams[1] = max(F3_tr, 0.0) / self.E_n
            else:
                dlams[1], dlams[2] = x[0], x[1]

        elif F2_tr > tol:
            # Чистый сдвиг
            dlams[2] = F2_tr / denom_s

        # --- Единое вычисление пластической деформации и возврата напряжений ---
        m_t = np.array([1.0, 0.0, 0.0])
        m_c = np.array([-1.0, 0.0, 0.0])
        m_s = np.array([self.tan_psi, 0.0, sign_tau])
        deps_p_local = dlams[0] * m_t + dlams[1] * m_c + dlams[2] * m_s

        sig_eff_new = sig_tr_local - self.D_local @ deps_p_local
        dq = dlams[0] * self.H_t

        return sig_eff_new, dlams, dq, deps_p_local

    # ------------------------------------------------------------------
    # Поврежденность
    # ------------------------------------------------------------------
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

        ds_base = min(max(self.a_s * Fp_t * (1 - Fp_s) * (1 - Fp_c)
                          + Fp_s + Fp_c - Fp_s * Fp_c, 0.0), 1.0)
        if sn_eff < 0.0:
            abs_sn = -sn_eff
            D_s = ds_base * (self.c + abs_sn * (self.tan_phi - self.tan_phi_r)) / (
                    self.c + abs_sn * self.tan_phi + 1e-12)
        else:
            D_s = ds_base

        return (min(D_nt, self.d_nt_max),
                min(D_nc, self.d_nc_max),
                min(D_s, self.d_s_max))

    def _apply_damage(self, sig_eff_local, D_n, D_s):
        """Единственное место, где эффективные напряжения переводятся в номинальные."""
        sig_nom = sig_eff_local.copy()
        sig_nom[0] *= (1.0 - D_n)
        sig_nom[2] *= (1.0 - D_s)

        # Регуляризация: деградация матрицы вдоль трещины, ограниченная капом.
        # matrix_parallel_damage_cap = 0 полностью отключает механизм.
        if self.matrix_parallel_damage_cap > 0.0:
            D_parallel = min(max(D_n, D_s), self.matrix_parallel_damage_cap)
            sig_nom[1] *= (1.0 - D_parallel)

        return sig_nom

    # ------------------------------------------------------------------
    # Основной расчет напряженного состояния (эффективное пространство)
    # ------------------------------------------------------------------
    def _evaluate_joint(self, sig_tr_local):
        """Return mapping шва + обновление энергий/поврежденности (trial-значения)."""
        sig_eff_new, dlams, dq, deps_p_local = self._return_mapping_stress(sig_tr_local)

        sig_n_yield_start = self.f_t + self.q_old
        dW_t = max(0.5 * (sig_n_yield_start + sig_eff_new[0]) * dlams[0], 0.0)
        dW_c = max(abs(sig_eff_new[0]) * dlams[1], 0.0)
        dW_s = max((abs(sig_eff_new[2]) + sig_eff_new[0] * self.tan_psi) * dlams[2], 0.0)

        W_pl_t = self.W_pl_t_old + dW_t
        W_pl_c = self.W_pl_c_old + dW_c
        W_pl_s = self.W_pl_s_old + dW_s
        q_new = self.q_old + dq

        D_nt_calc, D_nc_calc, D_s_calc = self._calculate_damage(
            W_pl_t, W_pl_c, W_pl_s, q_new, sig_eff_new[0])

        # Необратимость поврежденности
        D_nt = max(D_nt_calc, self.D_nt_old)
        D_nc = max(D_nc_calc, self.D_nc_old)
        D_s = max(D_s_calc, self.D_s_old)

        D_n = D_nt if sig_eff_new[0] >= 0 else D_nc

        sig_nom_local = self._apply_damage(sig_eff_new, D_n, D_s)

        return (sig_nom_local, sig_eff_new, deps_p_local,
                W_pl_t, W_pl_c, W_pl_s, q_new, D_nt, D_nc, D_s, D_n)

    # ------------------------------------------------------------------
    # Секущая матрица (согласована с _apply_damage)
    # ------------------------------------------------------------------
    def _compute_secant_matrix(self):
        if not self.is_locked_trial:
            return self.D_rock.copy()

        D_sec_local = self.D_local.copy()

        # Те же (1-D), что и в номинальных напряжениях;
        # min_stiff_factor — пол исключительно для устойчивости решателя.
        ms = self.min_stiff_factor
        f_n = max(1.0 - self.D_n_trial, ms)
        f_s = max(1.0 - self.D_s_trial, ms)

        if self.matrix_parallel_damage_cap > 0.0:
            D_parallel = min(max(self.D_n_trial, self.D_s_trial),
                             self.matrix_parallel_damage_cap)
            f_p = max(1.0 - D_parallel, ms)
        else:
            f_p = 1.0

        D_sec_local[0, :] *= f_n
        D_sec_local[:, 0] *= f_n
        D_sec_local[1, :] *= f_p
        D_sec_local[:, 1] *= f_p
        D_sec_local[2, :] *= f_s
        D_sec_local[:, 2] *= f_s

        return self.T_eps.T @ D_sec_local @ self.T_eps

    # ------------------------------------------------------------------
    # Интерфейс ConstitutiveModel
    # ------------------------------------------------------------------
    def update_state(self, current_strain_voigt):
        self.strain = current_strain_voigt.copy()
        self._reset_trial()

        deps_global = self.strain - self.strain_old

        # 1. Trial в ЭФФЕКТИВНОМ пространстве (от неповрежденной истории!)
        sig_tr_global = self.sig_eff_glob_old + self.D_rock @ deps_global

        # 2. Пластичность матрицы (эффективные напряжения)
        sig_mat_global = self._return_mapping_matrix(sig_tr_global)

        # 3. Проверка критерия образования трещины (по эффективным напряжениям)
        if not self.is_locked_trial:
            sig_3d = np.array([sig_mat_global[0], sig_mat_global[1], 0.0,
                               sig_mat_global[2], 0.0, 0.0])
            st = StressTensor(*sig_3d)

            if self.preset_plane_normal is not None:
                self._lock_plane(self.preset_plane_normal, st)
            else:
                f_s, n_s, u_s = find_critical_plane_shear(st, self.cp_material)
                f_t, n_t, u_t = find_critical_plane_tensile(st, self.cp_material)
                f_c, n_c, u_c = find_critical_plane_compression(st, self.cp_material)

                max_f = max(f_s, f_t, f_c)
                if self.lock_on_yield and max_f <= 1e-10:
                    # Трещины нет: эффективное = номинальное
                    self.stress = sig_mat_global.copy()
                    self.sig_eff_glob_trial = sig_mat_global.copy()
                    self.D_tangent = self.D_rock.copy()
                    return self.stress.copy(), self.D_tangent
                else:
                    max_u = max(u_s, u_t, u_c)
                    if max_u == u_s:
                        best_n, mode = n_s, "СДВИГ"
                    elif max_u == u_t:
                        best_n, mode = n_t, "РАСТЯЖЕНИЕ"
                    else:
                        best_n, mode = n_c, "СЖАТИЕ"
                    if self.debug:
                        print(f"[UJ2D] Трещина образовалась по режиму: {mode}")
                    # Устанавливает is_locked_trial=True; фиксация — в commit()
                    self._lock_plane(best_n, st)

        # 4. Return mapping шва в локальных ЭФФЕКТИВНЫХ напряжениях
        sig_tr_local = self.T_sig @ sig_mat_global
        (sig_nom_local, sig_eff_local, deps_p_local,
         W_pl_t, W_pl_c, W_pl_s, q_new, D_nt, D_nc, D_s, D_n) = \
            self._evaluate_joint(sig_tr_local)

        # 5. Обновление trial-состояния
        # Эффективное напряжение — обратно в глобальные оси (T_eps.T = T_sig^{-1})
        self.sig_eff_glob_trial = self.T_eps.T @ sig_eff_local
        self.q_trial = q_new
        self.W_pl_t_trial = W_pl_t
        self.W_pl_c_trial = W_pl_c
        self.W_pl_s_trial = W_pl_s
        self.D_nt_trial = D_nt
        self.D_nc_trial = D_nc
        self.D_s_trial = D_s
        self.D_n_trial = D_n

        # Пластическая деформация: локальные -> глобальные оси (eps_glob = T_sig^T eps_loc)
        self.eps_p_trial = self.eps_p_old + self.T_sig.T @ deps_p_local

        # 6. Номинальное напряжение и секущая матрица (согласованы между собой)
        self.stress = self.T_eps.T @ sig_nom_local
        self.D_tangent = self._compute_secant_matrix()

        return self.stress.copy(), self.D_tangent

    def get_tangent_matrix(self):
        return self.D_tangent

    def get_stress(self, strain):
        # Возвращает напряжение последнего update_state; аргумент не используется
        return self.stress

    def commit(self):
        self.stress_old = self.stress.copy()
        self.strain_old = self.strain.copy()
        self.sig_eff_glob_old = self.sig_eff_glob_trial.copy()
        self.eps_p_old = self.eps_p_trial.copy()

        self.q_old = self.q_trial
        self.W_pl_t_old = self.W_pl_t_trial
        self.W_pl_c_old = self.W_pl_c_trial
        self.W_pl_s_old = self.W_pl_s_trial

        self.D_nt_old = self.D_nt_trial
        self.D_nc_old = self.D_nc_trial
        self.D_s_old = self.D_s_trial

        # Окончательная фиксация трещины только при сходимости шага
        self.is_locked = self.is_locked_trial