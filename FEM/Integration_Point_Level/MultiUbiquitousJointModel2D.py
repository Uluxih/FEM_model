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


class PlaneState:
    """Вспомогательный класс для хранения истории и свойств отдельной плоскости (трещины)"""

    def __init__(self, normal, cp_material, D_rock, mu, l_c, Gf_t, Gf_c, Gf_s, a_t, a_s, fcr_over_fc, phi, psi, phi_r):
        self.normal = np.array(normal)
        self.nx, self.ny = self.normal[0], self.normal[1]

        # Матрицы трансформации
        self.T_sig, self.T_eps = self._build_2d_voigt_transformation(self.nx, self.ny)
        self.T_eps_inv = np.linalg.inv(self.T_eps)

        # Локальная жесткость
        self.D_local = self.T_sig @ D_rock @ self.T_eps_inv
        self.E_n = self.D_local[0, 0]
        self.G_s = self.D_local[2, 2]

        # Пределы прочности
        self.f_t = max(get_tensile_limit(self.normal, cp_material), 1e-12)
        self.f_c = max(get_compression_limit(self.normal, cp_material), 1e-12)
        # Базовое сцепление (будет обновляться при необходимости, но здесь берем начальное)
        # Для упрощения инициализации передаем фиктивный тензор, так как c обычно константа
        st_dummy = StressTensor(0, 0, 0, 0, 0, 0)
        self.c = max(get_cohesion_limit(self.normal, st_dummy, cp_material), 1e-12)

        # Параметры разупрочнения
        denom = (1.0 + self.f_t ** 2 / (3.0 * self.E_n * Gf_t)) * mu - 1.0
        self.H_t = abs(self.E_n / denom) if denom != 0 else 0.0

        self.tan_phi = np.tan(phi)
        self.tan_psi = np.tan(psi)
        self.tan_phi_r = np.tan(phi_r)

        self.q_lim = (self.c / self.tan_phi - self.f_t) if self.tan_phi > 1e-12 else np.inf

        # Параметры Damage
        self.Gf_t, self.Gf_c, self.Gf_s = Gf_t, Gf_c, Gf_s
        self.a_t, self.a_s = a_t, a_s
        self.fcr_over_fc = fcr_over_fc

        # Переменные состояния (Old)
        self.lam_t_old = self.lam_c_old = self.lam_s_old = 0.0
        self.W_pl_t_old = self.W_pl_c_old = self.W_pl_s_old = 0.0
        self.D_nt_old = self.D_nc_old = self.D_s_old = 0.0

        # Переменные состояния (Trial)
        self.lam_t = self.lam_c = self.lam_s = 0.0
        self.W_pl_t = self.W_pl_c = self.W_pl_s = 0.0
        self.D_nt = self.D_nc = self.D_s = 0.0

        # Текущие эффективные напряжения на плоскости (для Damage)
        self.sig_n_eff = 0.0
        self.tau_eff = 0.0

    def _build_2d_voigt_transformation(self, nx, ny):
        T_sig = np.zeros((3, 3))
        T_sig[0, 0] = nx ** 2;
        T_sig[0, 1] = ny ** 2;
        T_sig[0, 2] = 2.0 * nx * ny
        T_sig[1, 0] = ny ** 2;
        T_sig[1, 1] = nx ** 2;
        T_sig[1, 2] = -2.0 * nx * ny
        T_sig[2, 0] = -nx * ny;
        T_sig[2, 1] = nx * ny;
        T_sig[2, 2] = nx ** 2 - ny ** 2

        N = np.diag([1.0, 1.0, 2.0])
        N_inv = np.diag([1.0, 1.0, 0.5])
        T_eps = N @ T_sig @ N_inv
        return T_sig, T_eps

    def reset_trial(self):
        self.lam_t = self.lam_t_old
        self.lam_c = self.lam_c_old
        self.lam_s = self.lam_s_old
        self.W_pl_t = self.W_pl_t_old
        self.W_pl_c = self.W_pl_c_old
        self.W_pl_s = self.W_pl_s_old
        self.D_nt = self.D_nt_old
        self.D_nc = self.D_nc_old
        self.D_s = self.D_s_old

    def commit(self):
        self.lam_t_old = self.lam_t
        self.lam_c_old = self.lam_c
        self.lam_s_old = self.lam_s
        self.W_pl_t_old = self.W_pl_t
        self.W_pl_c_old = self.W_pl_c
        self.W_pl_s_old = self.W_pl_s
        self.D_nt_old = self.D_nt
        self.D_nc_old = self.D_nc
        self.D_s_old = self.D_s

    def c_curr(self, q):
        return self.c if q <= self.q_lim else self.c + (q - self.q_lim) * self.tan_phi

    def ft_curr(self, q):
        return self.f_t + q

    def update_damage(self, d_lam_t, d_lam_c, d_lam_s):
        """Вычисляет повреждение для данной плоскости"""
        q_new = self.H_t * self.lam_t

        dW_t = max(self.sig_n_eff * d_lam_t, 0.0) if d_lam_t > 0 else 0.0
        dW_c = max(abs(self.sig_n_eff) * d_lam_c, 0.0) if d_lam_c > 0 else 0.0
        dW_s = max((abs(self.tau_eff) + self.sig_n_eff * self.tan_psi) * d_lam_s, 0.0) if d_lam_s > 0 else 0.0

        self.W_pl_t = self.W_pl_t_old + dW_t
        self.W_pl_c = self.W_pl_c_old + dW_c
        self.W_pl_s = self.W_pl_s_old + dW_s

        r_t = min(self.W_pl_t / self.Gf_t, 1.0)
        r_c = min(self.W_pl_c / self.Gf_c, 1.0)
        r_s = min(self.W_pl_s / self.Gf_s, 1.0)

        Fp_t = r_t * (2.0 - r_t)
        Fp_c = 0.5 * (np.sin(np.pi * r_c - 0.5 * np.pi) + 1.0)
        Fp_s = r_s * (2.0 - r_s)

        dt = Fp_t + self.a_t * Fp_s * (1.0 - Fp_t)
        D_nt_new = 1.0 - (1.0 - dt) * self.f_t / (self.f_t + q_new + 1e-12)
        D_nc_new = (1.0 - self.fcr_over_fc) * Fp_c

        ds_base = min(max(self.a_s * Fp_t * (1 - Fp_s) * (1 - Fp_c) + Fp_s + Fp_c - Fp_s * Fp_c, 0.0), 1.0)

        if self.sig_n_eff < 0.0:
            abs_sn = -self.sig_n_eff
            D_s_new = ds_base * (self.c + abs_sn * (self.tan_phi - self.tan_phi_r)) / (
                        self.c + abs_sn * self.tan_phi + 1e-12)
        else:
            D_s_new = ds_base

        self.D_nt = min(max(D_nt_new, self.D_nt_old), 0.99)
        self.D_nc = min(max(D_nc_new, self.D_nc_old), 0.99)
        self.D_s = min(max(D_s_new, self.D_s_old), 0.99)


class MultiUbiquitousJointModel2D(ConstitutiveModel):
    def __init__(self, material):
        super().__init__(material)

        E = self.material.E
        nu = self.material.nu
        jp = self.material.joint_params

        self.E_min = 1e-5 * E
        self.D_rock = self._build_plane_stress_stiffness(E, nu)
        self.C_rock = np.linalg.inv(self.D_rock)

        # Параметры мульти-модели
        self.max_planes = 3
        self.angle_tolerance = np.radians(30.0)
        self.planes = []

        # Извлечение параметров материала
        self.phi = np.radians(jp.get('phi', 30.0))
        self.psi = np.radians(jp.get('psi', 10.0))
        self.phi_r = np.radians(jp.get('phi_r', np.degrees(self.phi)))

        self.cp_material = jp.get('cp_material', None)
        if self.cp_material is None:
            raise ValueError("Требуется 'cp_material' в joint_params!")

        self.cp_num_planes = jp.get('cp_num_planes', 100)
        self.l_c = jp.get('l_c', 1.0)
        self.Gf_t = jp.get('Gf_t', 100.0) / self.l_c
        self.Gf_c = jp.get('Gf_c', 5000.0) / self.l_c
        self.Gf_s = jp.get('Gf_s', 500.0) / self.l_c
        self.a_t = jp.get('a_t', 1.0)
        self.a_s = jp.get('a_s', 1.0)
        self.mu = jp.get('mu', 0.1)
        self.fcr_over_fc = jp.get('fcr_over_fc', 0.0)

        self.nr_tol = jp.get('nr_tol', 1e-8)
        self.nr_max_iter = jp.get('nr_max_iter', 25)

        # Глобальные переменные
        self.eps_p_old = np.zeros(3)
        self.eps_p = np.zeros(3)
        self.stress_old = np.zeros(3)
        self.strain_old = np.zeros(3)
        self.stress = np.zeros(3)
        self.strain = np.zeros(3)
        self.D_tangent = self.D_rock.copy()

    def _build_plane_stress_stiffness(self, E, nu):
        D = np.zeros((3, 3))
        c1 = E / (1.0 - nu ** 2)
        c2 = E * nu / (1.0 - nu ** 2)
        G = E / (2.0 * (1.0 + nu))
        D[0, 0] = D[1, 1] = c1
        D[0, 1] = D[1, 0] = c2
        D[2, 2] = G
        return D

    def _check_new_plane_allowed(self, new_normal):
        if len(self.planes) >= self.max_planes:
            return False
        for p in self.planes:
            cos_theta = abs(np.dot(p.normal, new_normal))
            if cos_theta > np.cos(self.angle_tolerance):
                return False
        return True

    def _global_return_mapping(self, sig_trial_eff):
        """
        Совместный Return Mapping для всех плоскостей.
        Работает в эффективных напряжениях (до применения Damage).
        """
        tol = self.nr_tol

        # Активные механизмы: список кортежей (plane_idx, mech_type)
        # mech_type: 't', 'c', 's'
        active_mechs = []

        # Инициализируем trial состояния
        for p in self.planes:
            p.reset_trial()
            sig_l_tr = p.T_sig @ sig_trial_eff
            sn_tr, tau_tr = sig_l_tr[0], sig_l_tr[2]

            q_old = p.H_t * p.lam_t_old
            ft_yld = p.ft_curr(q_old)
            c_yld = p.c_curr(q_old)

            if sn_tr - ft_yld > tol:
                active_mechs.append((p, 't'))
            elif -sn_tr - p.f_c > tol:
                active_mechs.append((p, 'c'))

            if abs(tau_tr) + sn_tr * p.tan_phi - c_yld > tol: active_mechs.append((p, 's'))

        if not active_mechs:
            # Все упруго
            for p in self.planes:
                sig_l = p.T_sig @ sig_trial_eff
                p.sig_n_eff, p.tau_eff = sig_l[0], sig_l[2]
            return sig_trial_eff, np.zeros(3)

        # Вектор неизвестных d_lam
        n_vars = len(active_mechs)
        d_lams = np.zeros(n_vars)

        # Newton-Raphson
        for _ in range(self.nr_max_iter):
            # 1. Вычисляем текущие пластические деформации и напряжения
            d_eps_p = np.zeros(3)
            for i, (p, mech) in enumerate(active_mechs):
                dl = d_lams[i]
                v_local = np.zeros(3)
                if mech == 't':
                    v_local[0] = dl
                elif mech == 'c':
                    v_local[0] = -dl
                elif mech == 's':
                    sig_l_curr = p.T_sig @ (sig_trial_eff - self.D_rock @ d_eps_p)
                    sign_tau = 1.0 if sig_l_curr[2] >= 0 else -1.0
                    v_local[0] = dl * p.tan_psi
                    v_local[2] = dl * sign_tau

                d_eps_p += p.T_eps_inv.T @ v_local

            sig_curr = sig_trial_eff - self.D_rock @ d_eps_p

            # 2. Сборка невязок (Residual) и Якобиана
            R = np.zeros(n_vars)
            J = np.zeros((n_vars, n_vars))

            for i, (p_i, mech_i) in enumerate(active_mechs):
                sig_l_i = p_i.T_sig @ sig_curr
                sn_i, tau_i = sig_l_i[0], sig_l_i[2]

                # Поиск d_lam_t для текущей плоскости (нужен для q)
                dl_t_idx = next((idx for idx, (p, m) in enumerate(active_mechs) if p == p_i and m == 't'), None)
                dl_t_val = d_lams[dl_t_idx] if dl_t_idx is not None else 0.0
                q_i = p_i.H_t * (p_i.lam_t_old + dl_t_val)

                if mech_i == 't':
                    R[i] = sn_i - p_i.ft_curr(q_i)
                elif mech_i == 'c':
                    R[i] = -sn_i - p_i.f_c
                elif mech_i == 's':
                    R[i] = abs(tau_i) + sn_i * p_i.tan_phi - p_i.c_curr(q_i)

                # Градиенты для Якобиана
                for j, (p_j, mech_j) in enumerate(active_mechs):
                    # Направление пластического течения плоскости j
                    df_dlam = 0.0

                    v_j = np.zeros(3)
                    if mech_j == 't':
                        v_j[0] = 1.0
                    elif mech_j == 'c':
                        v_j[0] = -1.0
                    elif mech_j == 's':
                        sig_l_j = p_j.T_sig @ sig_curr
                        v_j[0] = p_j.tan_psi
                        v_j[2] = 1.0 if sig_l_j[2] >= 0 else -1.0

                    # Влияние j на напряжения
                    d_sig_d_lam_j = -self.D_rock @ (p_j.T_eps_inv.T @ v_j)
                    d_sig_l_i = p_i.T_sig @ d_sig_d_lam_j

                    if mech_i == 't':
                        df_dlam = d_sig_l_i[0]
                        if p_i == p_j and mech_j == 't': df_dlam -= p_i.H_t
                    elif mech_i == 'c':
                        df_dlam = -d_sig_l_i[0]
                    elif mech_i == 's':
                        sign_tau_i = 1.0 if tau_i >= 0 else -1.0
                        df_dlam = sign_tau_i * d_sig_l_i[2] + d_sig_l_i[0] * p_i.tan_phi
                        if p_i == p_j and mech_j == 't' and q_i > p_i.q_lim:
                            df_dlam -= p_i.H_t * p_i.tan_phi

                    J[i, j] = df_dlam

            if np.linalg.norm(R) < tol:
                break

            try:
                delta = np.linalg.solve(J, -R)
            except np.linalg.LinAlgError:
                delta = np.linalg.pinv(J) @ -R

            d_lams += delta
            d_lams = np.maximum(d_lams, 0.0)  # Проекция для предотвращения отрицательных множителей

        # Обновление состояний плоскостей (в эффективном пространстве)
        for i, (p, mech) in enumerate(active_mechs):
            if mech == 't':
                p.lam_t = p.lam_t_old + d_lams[i]
            elif mech == 'c':
                p.lam_c = p.lam_c_old + d_lams[i]
            elif mech == 's':
                p.lam_s = p.lam_s_old + d_lams[i]

        for p in self.planes:
            sig_l = p.T_sig @ sig_curr
            p.sig_n_eff, p.tau_eff = sig_l[0], sig_l[2]

            # Извлекаем приращения для Damage
            dl_t = p.lam_t - p.lam_t_old
            dl_c = p.lam_c - p.lam_c_old
            dl_s = p.lam_s - p.lam_s_old
            p.update_damage(dl_t, dl_c, dl_s)

        return sig_curr, d_eps_p

    def update_state(self, current_strain_voigt, compute_tangent=True):
        self.strain = current_strain_voigt.copy()
        delta_strain = self.strain - self.strain_old

        # Trial эффективное напряжение
        sig_trial_eff = self.stress_old + self.D_rock @ delta_strain

        # 1. Проверка новых критических плоскостей
        st_trial = StressTensor(sig_trial_eff[0], sig_trial_eff[1], 0.0, sig_trial_eff[2], 0.0, 0.0)
        f_s, n_s, util_s = find_critical_plane_shear(st_trial, self.cp_material, num_planes=self.cp_num_planes)
        f_t, n_t, util_t = find_critical_plane_tensile(st_trial, self.cp_material, num_planes=self.cp_num_planes)
        max_f = max(f_s, f_t)

        if max_f > 1e-10:
            best_n = n_s if util_s > util_t else n_t
            if self._check_new_plane_allowed(best_n):
                new_plane = PlaneState(
                    best_n, self.cp_material, self.D_rock, self.mu, self.l_c,
                    self.Gf_t, self.Gf_c, self.Gf_s, self.a_t, self.a_s,
                    self.fcr_over_fc, self.phi, self.psi, self.phi_r
                )
                self.planes.append(new_plane)
                # Выводим сообщение только на реальном шаге, а не при расчете тангенса
                if compute_tangent:
                    print(f"Добавлена плоскость {len(self.planes)} с нормалью {best_n}")

        # 2. Глобальный Return Mapping
        sig_eff, d_eps_p = self._global_return_mapping(sig_trial_eff)
        self.eps_p = self.eps_p_old + d_eps_p

        # 3. Применение повреждения через секущую податливость
        C_global = self.C_rock.copy()
        for p in self.planes:
            D_n = p.D_nt if p.sig_n_eff >= 0.0 else p.D_nc
            D_s = p.D_s

            C_extra_loc = np.zeros((3, 3))
            if D_n > 1e-6: C_extra_loc[0, 0] = (D_n / (1.0 - D_n)) / p.E_n
            if D_s > 1e-6: C_extra_loc[2, 2] = (D_s / (1.0 - D_s)) / p.G_s

            C_global += p.T_eps_inv @ C_extra_loc @ p.T_sig

        D_secant = np.linalg.inv(C_global)
        eps_e = self.strain - self.eps_p
        self.stress = D_secant @ eps_e

        # 4. Вычисление численного тангенса
        if compute_tangent:
            self.D_tangent = self._compute_numerical_tangent(self.strain) + np.eye(3) * self.E_min

        return self.stress.copy(), self.D_tangent

    def _compute_numerical_tangent(self, current_strain, eps=1e-7):
        D_num = np.zeros((3, 3))
        scale = max(np.linalg.norm(current_strain), 1.0)
        h = eps * scale

        # Сохраняем состояние
        saved_stress = self.stress.copy()
        saved_eps_p = self.eps_p.copy()
        saved_num_planes = len(self.planes)
        saved_planes = [(p.lam_t, p.lam_c, p.lam_s, p.W_pl_t, p.W_pl_c, p.W_pl_s, p.D_nt, p.D_nc, p.D_s) for p in
                        self.planes]

        for j in range(3):
            ep = current_strain.copy();
            ep[j] += h
            em = current_strain.copy();
            em[j] -= h

            # Временный апдейт БЕЗ расчета тангенса (прерываем рекурсию)
            self.update_state(ep, compute_tangent=False);
            sig_p = self.stress.copy()
            self._restore_state(saved_stress, saved_eps_p, saved_planes, saved_num_planes)

            self.update_state(em, compute_tangent=False);
            sig_m = self.stress.copy()
            self._restore_state(saved_stress, saved_eps_p, saved_planes, saved_num_planes)

            D_num[:, j] = (sig_p - sig_m) / (2.0 * h)

        return D_num

    def _restore_state(self, s_stress, s_eps_p, s_planes, s_num_planes):
        self.stress = s_stress.copy()
        self.eps_p = s_eps_p.copy()

        # Если при возмущении создалась ложная плоскость, удаляем её
        if len(self.planes) > s_num_planes:
            self.planes = self.planes[:s_num_planes]

        for p, state in zip(self.planes, s_planes):
            p.lam_t, p.lam_c, p.lam_s, p.W_pl_t, p.W_pl_c, p.W_pl_s, p.D_nt, p.D_nc, p.D_s = state

    def get_tangent_matrix(self):
        return self.D_tangent

    def get_stress(self, strain):
        return self.stress

    def commit(self):
        self.stress_old = self.stress.copy()
        self.strain_old = self.strain.copy()
        self.eps_p_old = self.eps_p.copy()
        for p in self.planes:
            p.commit()
