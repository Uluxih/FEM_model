import numpy as np


class XFEM_CohesiveDamagePlasticity2D:
    """
    Когезионная модель трещины для XFEM.
    Связывает скачки перемещений (jumps) с тяжениями (tractions) на берегах трещины.
    """

    def __init__(self, material_params):
        # Штрафные жесткости (Penalty stiffness) - заменяют упругость матрицы.
        # Обычно задаются очень большими (например, 10^4 * E / L_elem),
        # чтобы до разрушения берега не "пружинили" фиктивно.
        self.K_n = material_params.get('K_n', 1e11)
        self.K_s = material_params.get('K_s', 1e11)

        self.f_t = material_params.get('f_t', 1e6)
        self.f_c = material_params.get('f_c', 1e7)
        self.c = material_params.get('c', 1e6)

        self.phi = np.radians(material_params.get('phi', 30.0))
        self.psi = np.radians(material_params.get('psi', 0.0))
        self.phi_r = np.radians(material_params.get('phi_r', 30.0))
        self.tan_phi = np.tan(self.phi)
        self.tan_psi = np.tan(self.psi)
        self.tan_phi_r = np.tan(self.phi_r)

        # Энергии разрушения на единицу ПЛОЩАДИ (Дж/м2).
        # В XFEM делить на l_c больше НЕ НУЖНО!
        self.Gf_t = material_params.get('Gf_t', 100.0)
        self.Gf_c = material_params.get('Gf_c', 500.0)
        self.Gf_s = material_params.get('Gf_s', 200.0)

        self.a_t = material_params.get('a_t', 1.0)
        self.a_s = material_params.get('a_s', 1.0)
        self.fcr_over_fc = material_params.get('fcr_over_fc', 0.0)

        # Модуль разупрочнения для эффективных тяжений
        term = (1.0 + self.f_t ** 2 / (3.0 * self.K_n * self.Gf_t)) * self.tan_phi
        self.H_t = self.K_n * (1.0 / term - 1.0) if term > 0 else self.K_n * 0.1
        self.q_lim = np.inf if self.tan_phi < 1e-12 else self.c / self.tan_phi - self.f_t

        self._init_history()

    def _init_history(self):
        self.jump_p_old = np.zeros(2)  # [plastic_jump_n, plastic_jump_s]
        self.q_old = 0.0
        self.W_pl_t_old = self.W_pl_c_old = self.W_pl_s_old = 0.0
        self.D_nt_old = self.D_nc_old = self.D_s_old = 0.0

        # Для trial-состояний
        self.jump_p_trial = np.zeros(2)
        self.q_trial = 0.0
        self.W_pl_t_trial = self.W_pl_c_trial = self.W_pl_s_trial = 0.0
        self.D_nt_trial = self.D_nc_trial = self.D_s_trial = 0.0

        self.traction = np.zeros(2)  # [t_n, t_s]
        self.K_secant = np.diag([self.K_n, self.K_s])

    def _c_curr(self, q):
        if q <= self.q_lim: return self.c
        return self.c + (q - self.q_lim) * self.tan_phi

    def update_state(self, current_jump):
        """
        current_jump: np.array([delta_u_n, delta_u_s]) - скачок перемещений в локальных координатах
        Возвращает: тяжения (tractions) и касательную матрицу 2x2.
        """
        jump_n, jump_s = current_jump[0], current_jump[1]

        # 1. Пробные тяжения (Упругий шаг)
        t_n_tr = self.K_n * (jump_n - self.jump_p_old[0])
        t_s_tr = self.K_s * (jump_s - self.jump_p_old[1])

        abs_ts_tr = abs(t_s_tr)
        sign_ts = 1.0 if t_s_tr >= 0 else -1.0

        ft_curr = self.f_t + self.q_old
        c_curr = self._c_curr(self.q_old)

        F1_tr = t_n_tr - ft_curr
        F3_tr = -t_n_tr - self.f_c
        F2_tr = abs_ts_tr + t_n_tr * self.tan_phi - c_curr

        t_n_eff = t_n_tr
        t_s_eff = t_s_tr
        dlam_t = dlam_c = dlam_s = 0.0

        e_okrugl = 1e-8
        denom_s = self.K_s + self.K_n * self.tan_phi * self.tan_psi

        # 2. Return Mapping (Возврат на поверхность текучести)
        if F1_tr > e_okrugl or F2_tr > e_okrugl or F3_tr > e_okrugl:
            if F1_tr > e_okrugl and (abs_ts_tr + ft_curr * self.tan_phi - c_curr) <= e_okrugl:
                dlam_t = F1_tr / (self.K_n + self.H_t)
                t_n_eff -= dlam_t * self.K_n

            elif F3_tr > e_okrugl and (abs_ts_tr - self.f_c * self.tan_phi - c_curr) <= e_okrugl:
                dlam_c = F3_tr / self.K_n
                t_n_eff += dlam_c * self.K_n  # m_vec = -1

            elif (F1_tr > e_okrugl and (abs_ts_tr + ft_curr * self.tan_phi - c_curr > e_okrugl)) or \
                    (F2_tr > e_okrugl and (F1_tr - (F2_tr / denom_s) * self.K_n * self.tan_psi > e_okrugl)):
                A = np.array([[self.K_n + self.H_t, self.K_n * self.tan_psi],
                              [self.K_n * self.tan_phi, denom_s]])
                B = np.array([F1_tr, F2_tr])
                try:
                    x = np.linalg.solve(A, B)
                    dlam_t, dlam_s = max(x[0], 0.0), max(x[1], 0.0)
                except np.linalg.LinAlgError:
                    pass
                t_n_eff -= (dlam_t * self.K_n + dlam_s * self.K_n * self.tan_psi)
                t_s_eff -= dlam_s * self.K_s * sign_ts

            elif (F2_tr > e_okrugl and (F3_tr + (F2_tr / denom_s) * self.K_n * self.tan_psi > e_okrugl)) or \
                    (F3_tr > e_okrugl and (abs_ts_tr - self.f_c * self.tan_phi - c_curr > e_okrugl)):
                A = np.array([[self.K_n, -self.K_n * self.tan_psi],
                              [-self.K_n * self.tan_phi, denom_s]])
                B = np.array([F3_tr, F2_tr])
                try:
                    x = np.linalg.solve(A, B)
                    dlam_c, dlam_s = max(x[0], 0.0), max(x[1], 0.0)
                except np.linalg.LinAlgError:
                    pass
                t_n_eff -= (-dlam_c * self.K_n + dlam_s * self.K_n * self.tan_psi)
                t_s_eff -= dlam_s * self.K_s * sign_ts

            elif F2_tr > e_okrugl:
                dlam_s = F2_tr / denom_s
                t_n_eff -= dlam_s * self.K_n * self.tan_psi
                t_s_eff -= dlam_s * self.K_s * sign_ts

        # 3. Накопление работы и расчет поврежденности (Damage)
        dq = dlam_t * self.H_t
        sig_n_yield_start = self.f_t + self.q_old
        dW_t = max(0.5 * (sig_n_yield_start + t_n_eff) * dlam_t, 0.0)
        dW_c = max(abs(t_n_eff) * dlam_c, 0.0)
        dW_s = max((abs(t_s_eff) + t_n_eff * self.tan_psi) * dlam_s, 0.0)

        self.W_pl_t_trial = self.W_pl_t_old + dW_t
        self.W_pl_c_trial = self.W_pl_c_old + dW_c
        self.W_pl_s_trial = self.W_pl_s_old + dW_s
        self.q_trial = self.q_old + dq

        # Функции повреждения
        r_t = min(self.W_pl_t_trial / self.Gf_t, 1.0)
        r_c = min(self.W_pl_c_trial / self.Gf_c, 1.0)
        r_s = min(self.W_pl_s_trial / self.Gf_s, 1.0)

        Fp_t = r_t * (2.0 - r_t)
        Fp_c = 0.5 * (np.sin(np.pi * r_c - 0.5 * np.pi) + 1.0)
        Fp_s = r_s * (2.0 - r_s)

        dt = Fp_t + self.a_t * Fp_s * (1.0 - Fp_t)
        D_nt_calc = 1.0 - (1.0 - dt) * self.f_t / (self.f_t + self.q_trial + 1e-12)
        D_nc_calc = (1.0 - self.fcr_over_fc) * Fp_c

        ds_base = min(max(self.a_s * Fp_t * (1 - Fp_s) * (1 - Fp_c) + Fp_s + Fp_c - Fp_s * Fp_c, 0.0), 1.0)
        if t_n_eff < 0.0:
            abs_sn = -t_n_eff
            D_s_calc = ds_base * (self.c + abs_sn * (self.tan_phi - self.tan_phi_r)) / (
                        self.c + abs_sn * self.tan_phi + 1e-12)
        else:
            D_s_calc = ds_base

        self.D_nt_trial = max(min(D_nt_calc, 0.999), self.D_nt_old)
        self.D_nc_trial = max(min(D_nc_calc, 0.999), self.D_nc_old)
        self.D_s_trial = max(min(D_s_calc, 0.999), self.D_s_old)

        D_n = self.D_nt_trial if t_n_eff >= 0 else self.D_nc_trial
        D_s = self.D_s_trial

        # 4. Номинальные тяжения
        t_n_nom = t_n_eff * (1.0 - D_n)
        t_s_nom = t_s_eff * (1.0 - D_s)
        self.traction = np.array([t_n_nom, t_s_nom])

        # Обновление пластических скачков
        self.jump_p_trial = self.jump_p_old + np.array([
            dlam_t - dlam_c + dlam_s * self.tan_psi,
            dlam_s * sign_ts
        ])

        # 5. Секущая матрица (Secant Stiffness)
        min_stiff = 1e-4
        self.K_secant = np.diag([
            self.K_n * max(1.0 - D_n, min_stiff),
            self.K_s * max(1.0 - D_s, min_stiff)
        ])

        return self.traction.copy(), self.K_secant.copy()

    def commit(self):
        self.jump_p_old = self.jump_p_trial.copy()
        self.q_old = self.q_trial
        self.W_pl_t_old = self.W_pl_t_trial
        self.W_pl_c_old = self.W_pl_c_trial
        self.W_pl_s_old = self.W_pl_s_trial
        self.D_nt_old = self.D_nt_trial
        self.D_nc_old = self.D_nc_trial
        self.D_s_old = self.D_s_trial