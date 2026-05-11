import numpy as np
from FEM.Abstract.Integration_Point_Level import ConstitutiveModel
from FEM.Integration_Point_Level.CriticalPlane.tensor import StressTensor
from FEM.Integration_Point_Level.CriticalPlane import criterion as cr


class CriticalPlanePlasticity3D(ConstitutiveModel):
    def __init__(self, material):
        super().__init__(material)

        E = self.material.E
        nu = self.material.nu
        C_el = E / ((1 + nu) * (1 - 2 * nu))

        self.D_e = np.zeros((6, 6))
        self.D_e[0:3, 0:3] = C_el * nu
        self.D_e[0, 0] = self.D_e[1, 1] = self.D_e[2, 2] = C_el * (1 - nu)
        G = E / (2 * (1 + nu))
        self.D_e[3, 3] = self.D_e[4, 4] = self.D_e[5, 5] = G

        self.stress_old = np.zeros(6)
        self.strain_old = np.zeros(6)
        self.kappa_old = 0.0
        self.is_cracked_old = False
        self.fixed_normal_old = None

        self.stress = np.zeros(6)
        self.strain = np.zeros(6)
        self.kappa = 0.0
        self.is_cracked = False
        self.fixed_normal = None

        # --- ПАРАМЕТРЫ ВАШЕЙ МОДЕЛИ УПРОЧНЕНИЯ ---
        self.hardening_type = 'cohesion'  # или 'friction'
        self.c_fixed = 2.0
        self.mu_c = 0.6
        self.A_mu = 0.005
        self.mu_fixed = 0.4
        self.c_0 = 1.0
        self.c_u = 4.0
        self.A_c = 0.005
        self.sigma_0 = 1.0
        self.eta_c = 0.7

    def _voigt_to_tensor(self, v):
        return np.array([
            [v[0], v[3], v[5]],
            [v[3], v[1], v[4]],
            [v[5], v[4], v[2]]
        ])

    def _get_stress_on_plane(self, stress_voigt, normal):
        st_obj = StressTensor.from_matrix(self._voigt_to_tensor(stress_voigt))
        sigma_n = st_obj.normal_stress(normal)
        tau_n = st_obj.shear_stress_magnitude(normal)

        if tau_n > 1e-12:
            s_geom = st_obj.shear_stress_vector(normal) / tau_n
        else:
            s_geom = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            s_geom = s_geom - np.dot(s_geom, normal) * normal
            s_geom /= np.linalg.norm(s_geom)

        return sigma_n, tau_n, s_geom

    def _get_friction_and_hardening(self, kappa):
        mu = self.mu_c * kappa / (self.A_mu + kappa)
        dmu_dkappa = self.mu_c * self.A_mu / ((self.A_mu + kappa) ** 2)
        return mu, dmu_dkappa

    def _get_cohesion_and_hardening(self, kappa):
        c = self.c_0 + (self.c_u - self.c_0) * kappa / (self.A_c + kappa)
        dc_dkappa = (self.c_u - self.c_0) * self.A_c / ((self.A_c + kappa) ** 2)
        return c, dc_dkappa

    def _get_gradients_voigt(self, n, s, mu_fric, sigma_n, tau):
        N_tens = np.outer(n, n)
        S_tens = 0.5 * (np.outer(n, s) + np.outer(s, n))

        df_dsigma_tens = mu_fric * N_tens + S_tens

        # ЗАЩИТА 1: От сингулярности при сильном растяжении
        denom = self.sigma_0 - sigma_n
        if denom < 1e-4:
            denom = 1e-4

        dpsi_dsigma_scalar = (tau / denom) - self.eta_c
        dpsi_dsigma_tens = dpsi_dsigma_scalar * N_tens + S_tens

        q = np.array([
            df_dsigma_tens[0, 0], df_dsigma_tens[1, 1], df_dsigma_tens[2, 2],
            2.0 * df_dsigma_tens[0, 1], 2.0 * df_dsigma_tens[1, 2], 2.0 * df_dsigma_tens[0, 2]
        ])
        r = np.array([
            dpsi_dsigma_tens[0, 0], dpsi_dsigma_tens[1, 1], dpsi_dsigma_tens[2, 2],
            2.0 * dpsi_dsigma_tens[0, 1], 2.0 * dpsi_dsigma_tens[1, 2], 2.0 * dpsi_dsigma_tens[0, 2]
        ])
        return q, r

    def get_tangent_matrix(self):
        return self.D_e

    def get_stress(self, strain):
        return self.stress

    def update_state(self, current_strain):
        d_strain = current_strain - self.strain_old
        stress_trial = self.stress_old + self.D_e @ d_strain

        if not self.is_cracked_old:
            st_obj_trial = StressTensor.from_matrix(self._voigt_to_tensor(stress_trial))
            _, best_n, _ = cr.find_critical_plane_shear(st_obj_trial, self.material.crit_mat, mode='3D')
        else:
            best_n = self.fixed_normal_old

        sigma_n_trial, tau_n_trial, s_geom_trial = self._get_stress_on_plane(stress_trial, best_n)

        # Вычисляем начальную прочность
        if self.hardening_type == 'friction':
            mu_trial, _ = self._get_friction_and_hardening(self.kappa_old)
            c_trial = self.c_fixed
        else:
            mu_trial = self.mu_fixed
            c_trial, _ = self._get_cohesion_and_hardening(self.kappa_old)

        f_yield_trial = tau_n_trial + mu_trial * sigma_n_trial - c_trial

        # Упругий шаг
        if f_yield_trial <= 1e-6:
            self.stress = stress_trial
            self.strain = current_strain
            self.kappa = self.kappa_old
            self.is_cracked = self.is_cracked_old
            self.fixed_normal = self.fixed_normal_old
            return self.stress, self.D_e

        # --- ПЛАСТИЧЕСКИЙ ШАГ (Cutting Plane Algorithm) ---
        self.is_cracked = True
        self.fixed_normal = best_n.copy() if not self.is_cracked_old else self.fixed_normal_old

        stress_k = stress_trial.copy()
        kappa_k = self.kappa_old
        delta_gamma = 0.0

        max_iters = 40
        tol = 1e-5 * self.material.E

        for it in range(max_iters):
            sigma_n_k, tau_n_k, s_geom_k = self._get_stress_on_plane(stress_k, self.fixed_normal)

            if self.hardening_type == 'friction':
                mu_k, dmu_dkappa = self._get_friction_and_hardening(kappa_k)
                c_k = self.c_fixed
                H = - (dmu_dkappa * sigma_n_k)
            else:
                mu_k = self.mu_fixed
                c_k, dc_dkappa = self._get_cohesion_and_hardening(kappa_k)
                H = dc_dkappa

            f_k = tau_n_k + mu_k * sigma_n_k - c_k

            # Критерий сходимости
            if abs(f_k) < tol:
                break

            q, r = self._get_gradients_voigt(self.fixed_normal, s_geom_k, mu_k, sigma_n_k, tau_n_k)

            denom = np.dot(q, self.D_e @ r) + H
            if denom <= 1e-10:
                denom = 1e-10

            d_gamma = f_k / denom

            # ЗАЩИТА 2: Демпфирование (Ограничение максимального скачка за 1 итерацию)
            max_step = 0.01
            if d_gamma > max_step:
                d_gamma = max_step
            elif d_gamma < -max_step:
                d_gamma = -max_step

            delta_gamma += d_gamma

            # ЗАЩИТА 3: Пластическая деформация НЕ МОЖЕТ быть отрицательной!
            if delta_gamma < 0.0:
                delta_gamma = 0.0

            kappa_k = self.kappa_old + delta_gamma

            # Инкрементальное обновление напряжений (Стягиваем их к поверхности)
            stress_k -= d_gamma * (self.D_e @ r)

        # Вычисление финальной касательной матрицы по сошедшемуся состоянию
        sigma_n_k, tau_n_k, s_geom_k = self._get_stress_on_plane(stress_k, self.fixed_normal)
        if self.hardening_type == 'friction':
            mu_k, dmu_dkappa = self._get_friction_and_hardening(kappa_k)
            H = - (dmu_dkappa * sigma_n_k)
        else:
            mu_k = self.mu_fixed
            _, dc_dkappa = self._get_cohesion_and_hardening(kappa_k)
            H = dc_dkappa

        q, r = self._get_gradients_voigt(self.fixed_normal, s_geom_k, mu_k, sigma_n_k, tau_n_k)

        D_e_r = self.D_e @ r
        q_T_D_e = q.T @ self.D_e
        denom_final = np.dot(q, D_e_r) + H
        if denom_final <= 1e-10: denom_final = 1e-10

        D_ep = self.D_e - np.outer(D_e_r, q_T_D_e) / denom_final
        # Принудительная симметризация (спасает глобальный решатель)
        D_ep = 0.5 * (D_ep + D_ep.T)

        self.stress = stress_k
        self.strain = current_strain
        self.kappa = kappa_k

        return self.stress, D_ep

    def commit(self):
        self.stress_old = self.stress.copy()
        self.strain_old = self.strain.copy()
        self.kappa_old = self.kappa
        self.is_cracked_old = self.is_cracked
        if self.fixed_normal is not None:
            self.fixed_normal_old = self.fixed_normal.copy()
