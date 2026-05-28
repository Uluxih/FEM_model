import numpy as np
from FEM.Abstract.Integration_Point_Level import ConstitutiveModel


class MingaMasonryJointModel3DFixed(ConstitutiveModel):
    """
    3D Mesoscale Damage-Plasticity Model for Masonry Joints with a FIXED weak plane.
    Based on Minga et al. (2017) "A 3D mesoscale damage-plasticity approach for masonry structures under cyclic loading".

    Features:
    - Multi-surface plasticity (Tensile cap, Shear Coulomb, Compressive cap).
    - Hardening in tension to control permanent strains.
    - Decoupled anisotropic damage evolution based on plastic work.
    - Captures stiffness degradation and unilateral contact (crack closure).
    - Fixed plane orientation (ideal for embedded smeared cracks or specific joint directions).
    """

    def __init__(self, material, fixed_normal=None):
        super().__init__(material)

        if fixed_normal is None:
            fixed_normal = [0.0, 0.0, 1.0]

        E = self.material.E
        nu = self.material.nu
        jp = self.material.joint_params

        self.E_min = 1e-5 * E

        # --- 1. Base Parameters ---
        self.ft = jp.get('ft', 0.25)
        self.fc = jp.get('fc', 10.0)
        self.c = jp.get('c', 0.5)

        self.phi = np.radians(jp.get('phi', 30.0))
        self.phi_g = np.radians(jp.get('phi_g', 0.0))
        self.phi_r = np.radians(jp.get('phi_r', self.phi))

        self.tan_phi = np.tan(self.phi)
        self.tan_phi_g = np.tan(self.phi_g)
        self.tan_phi_r = np.tan(self.phi_r)

        # --- 2. Fracture Energies and Damage Parameters ---
        self.Gf_1 = jp.get('Gf_1', 0.05)  # Tension
        self.Gf_2 = jp.get('Gf_2', 0.1)  # Shear
        self.Gf_3 = jp.get('Gf_3', 1.0)  # Compression

        self.l_param = jp.get('l', 0.1)  # Controls permanent normal strains
        self.alpha_t = jp.get('alpha_t', 1.0)
        self.alpha_s = jp.get('alpha_s', 1.0)
        self.fc_r = jp.get('fc_r', 0.1 * self.fc)

        # Elastic stiffness of the rock
        self.D_rock = self._build_isotropic_stiffness(E, nu)

        # Plane orientation
        self.fixed_normal = fixed_normal
        self.R = self._build_rotation_matrix(fixed_normal)
        self.T_sig, self.T_eps = self._build_voigt_transformations(self.R)
        self.D_local = self.T_sig @ self.D_rock @ self.T_sig.T

        self.E_n = self.D_local[2, 2]
        self.G_s = self.D_local[4, 4]

        # --- 3. Hardening Modulus (Eq 27) ---
        factor = self.l_param * (1.0 + (self.ft ** 2) / (3.0 * self.E_n * self.Gf_1))
        if factor > 0 and factor < 1.0:
            self.H = self.E_n * (1.0 / factor - 1.0)
        else:
            self.H = 0.0

        self.q_lim = self.c / self.tan_phi - self.ft if self.tan_phi > 1e-6 else 1e9

        # --- 4. History Variables ---
        self._init_history()

        self.stress = np.zeros(6)
        self.strain = np.zeros(6)
        self.D_tangent = self.D_rock.copy()
        self.is_locked = True

    def _init_history(self):
        self.W_pl_1_old = 0.0
        self.W_pl_2_old = 0.0
        self.W_pl_3_old = 0.0

        self.ep_n_old = 0.0
        self.ep_23_old = 0.0
        self.ep_13_old = 0.0

        self.q_old = 0.0
        self.D_n_old = 0.0
        self.D_s_old = 0.0

        self.stress_old = np.zeros(6)
        self.strain_old = np.zeros(6)

        self._reset_trial()

    def _reset_trial(self):
        self.W_pl_1_trial = self.W_pl_1_old
        self.W_pl_2_trial = self.W_pl_2_old
        self.W_pl_3_trial = self.W_pl_3_old

        self.ep_n_trial = self.ep_n_old
        self.ep_23_trial = self.ep_23_old
        self.ep_13_trial = self.ep_13_old

        self.q_trial = self.q_old
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
        T_sig[3, 3] = R[0, 0] * R[1, 1] + R[1, 0] * R[0, 1]
        T_sig[3, 4] = R[1, 0] * R[2, 1] + R[2, 0] * R[1, 1]
        T_sig[3, 5] = R[0, 0] * R[2, 1] + R[2, 0] * R[0, 1]
        T_sig[4, 3] = R[0, 1] * R[1, 2] + R[1, 1] * R[0, 2]
        T_sig[4, 4] = R[1, 1] * R[2, 2] + R[2, 1] * R[1, 2]
        T_sig[4, 5] = R[0, 1] * R[2, 2] + R[2, 1] * R[0, 2]
        T_sig[5, 3] = R[0, 0] * R[1, 2] + R[1, 0] * R[0, 2]
        T_sig[5, 4] = R[1, 0] * R[2, 2] + R[2, 0] * R[1, 2]
        T_sig[5, 5] = R[0, 0] * R[2, 2] + R[2, 0] * R[0, 2]

        T_eps = np.zeros((6, 6))
        T_eps[0:3, 0:3] = T_sig[0:3, 0:3]
        T_eps[0:3, 3:6] = T_sig[0:3, 3:6] / 2.0
        T_eps[3:6, 0:3] = T_sig[3:6, 0:3] * 2.0
        T_eps[3:6, 3:6] = T_sig[3:6, 3:6]
        return T_sig, T_eps

    def _compute_state(self, current_strain):
        """
        Core return mapping and damage calculation.
        Pure function: does not modify historical states directly.
        """
        e_l = self.T_eps @ current_strain

        # 1. Elastic predictors for effective stresses
        sig_n_tr = self.E_n * (e_l[2] - self.ep_n_old)
        tau_23_tr = self.G_s * (e_l[4] - self.ep_23_old)
        tau_13_tr = self.G_s * (e_l[5] - self.ep_13_old)
        tau_tr = np.sqrt(tau_23_tr ** 2 + tau_13_tr ** 2)

        # 2. Plasticity Return Mapping
        dlam1 = dlam2 = dlam3 = 0.0
        sig_n = sig_n_tr
        tau = tau_tr
        q_new = self.q_old

        f1_tr = sig_n_tr - self.ft - self.q_old
        c_prime_tr = self.c if self.q_old <= self.q_lim else self.c + (self.q_old - self.q_lim) * self.tan_phi
        f2_tr = tau_tr - c_prime_tr + sig_n_tr * self.tan_phi
        f3_tr = -sig_n_tr - self.fc

        if f1_tr > 1e-9 or f2_tr > 1e-9 or f3_tr > 1e-9:
            valid = False

            # --- Try {1, 2} (Tension + Shear) ---
            det = (self.E_n + self.H) * self.G_s + self.E_n * self.H * self.tan_phi_g * self.tan_phi
            dl1 = (self.G_s * f1_tr - self.E_n * self.tan_phi_g * (
                        tau_tr - self.c + (self.ft + self.q_old) * self.tan_phi)) / det
            dl2 = (self.H * self.tan_phi * f1_tr + (self.E_n + self.H) * (
                        tau_tr - self.c + (self.ft + self.q_old) * self.tan_phi)) / det

            if dl1 > -1e-9 and dl2 > -1e-9:
                q_tmp = self.q_old + self.H * dl1
                if q_tmp > self.q_lim:
                    # Cone reduces to a point
                    dl2 = tau_tr / self.G_s
                    dl1 = (sig_n_tr - self.ft - self.q_old - self.E_n * self.tan_phi_g * dl2) / (self.E_n + self.H)
                    q_tmp = self.q_old + self.H * dl1

                if dl1 > -1e-9 and dl2 > -1e-9:
                    sn_tmp = sig_n_tr - self.E_n * (dl1 + dl2 * self.tan_phi_g)
                    if -sn_tmp - self.fc <= 1e-9:
                        dlam1, dlam2 = max(0.0, dl1), max(0.0, dl2)
                        sig_n = sn_tmp
                        tau = max(0.0, tau_tr - self.G_s * dlam2)
                        q_new = q_tmp
                        valid = True

            # --- Try {2, 3} (Compression + Shear) ---
            if not valid:
                dl2 = (tau_tr - self.c - self.fc * self.tan_phi) / self.G_s
                dl3 = (-self.fc - sig_n_tr + self.E_n * self.tan_phi_g * dl2) / self.E_n
                if dl2 > -1e-9 and dl3 > -1e-9:
                    sn_tmp = -self.fc
                    if sn_tmp - (self.ft + self.q_old) <= 1e-9:
                        dlam2, dlam3 = max(0.0, dl2), max(0.0, dl3)
                        sig_n = sn_tmp
                        tau = max(0.0, tau_tr - self.G_s * dlam2)
                        valid = True

            # --- Try {1} (Pure Tension) ---
            if not valid:
                dl1 = f1_tr / (self.E_n + self.H)
                if dl1 > -1e-9:
                    q_tmp = self.q_old + self.H * dl1
                    cp_tmp = self.c + max(0.0, q_tmp - self.q_lim) * self.tan_phi
                    sn_tmp = sig_n_tr - self.E_n * dl1
                    if tau_tr - cp_tmp + sn_tmp * self.tan_phi <= 1e-9:
                        dlam1 = max(0.0, dl1)
                        sig_n = sn_tmp
                        q_new = q_tmp
                        valid = True

            # --- Try {3} (Pure Compression) ---
            if not valid:
                dl3 = f3_tr / self.E_n
                if dl3 > -1e-9:
                    sn_tmp = -self.fc
                    if tau_tr - self.c + sn_tmp * self.tan_phi <= 1e-9:
                        dlam3 = max(0.0, dl3)
                        sig_n = sn_tmp
                        valid = True

            # --- Try {2} (Pure Shear) ---
            if not valid:
                dl2 = f2_tr / (self.G_s + self.E_n * self.tan_phi_g * self.tan_phi)
                if dl2 > -1e-9:
                    sn_tmp = sig_n_tr - self.E_n * dl2 * self.tan_phi_g
                    if sn_tmp - (self.ft + self.q_old) <= 1e-9 and -sn_tmp - self.fc <= 1e-9:
                        dlam2 = max(0.0, dl2)
                        sig_n = sn_tmp
                        tau = max(0.0, tau_tr - self.G_s * dlam2)
                        valid = True

        # 3. Update Plastic Strains and Work
        ep_n_new = self.ep_n_old + dlam1 - dlam3 + dlam2 * self.tan_phi_g
        if tau_tr > 1e-12:
            ep_23_new = self.ep_23_old + dlam2 * (tau_23_tr / tau_tr)
            ep_13_new = self.ep_13_old + dlam2 * (tau_13_tr / tau_tr)
            tau_23 = tau_23_tr * (tau / tau_tr)
            tau_13 = tau_13_tr * (tau / tau_tr)
        else:
            ep_23_new = self.ep_23_old
            ep_13_new = self.ep_13_old
            tau_23 = 0.0
            tau_13 = 0.0

        dW1 = sig_n * dlam1 if dlam1 > 0 else 0.0
        dW2 = tau * dlam2 + sig_n * dlam2 * self.tan_phi_g if dlam2 > 0 else 0.0
        dW3 = abs(sig_n) * dlam3 if dlam3 > 0 else 0.0

        W1 = self.W_pl_1_old + dW1
        W2 = self.W_pl_2_old + dW2
        W3 = self.W_pl_3_old + dW3

        # 4. Damage Evolution
        r1 = min(W1 / self.Gf_1, 1.0)
        r2 = min(W2 / self.Gf_2, 1.0)
        r3 = min(W3 / self.Gf_3, 1.0)

        Fp1 = r1 * (2.0 - r1)
        Fp2 = r2 * (2.0 - r2)
        Fp3 = r3 * (2.0 - r3)
        Fs3 = 0.5 * np.sin(np.pi * r3 - np.pi / 2.0) + 0.5

        D_nt = (Fp1 * self.ft + q_new) / (self.ft + q_new) + self.alpha_t * Fp2 * (
                    1.0 - (Fp1 * self.ft + q_new) / (self.ft + q_new))
        D_nc = ((self.fc - self.fc_r) / self.fc) * Fs3

        qs = max(0.0, q_new - self.q_lim) * self.tan_phi
        ds = self.alpha_s * ((Fp1 * self.c + qs) / (self.c + qs)) * (
                    1.0 - Fp2 - Fp3 + Fp2 * Fp3) + Fp2 + Fp3 - Fp2 * Fp3

        if sig_n >= 0:
            D_n_new = D_nt
            D_s_new = ds
        else:
            D_n_new = D_nc
            denom = self.c + abs(sig_n) * self.tan_phi
            D_s_new = ds * (self.c + abs(sig_n) * (self.tan_phi - self.tan_phi_r)) / max(denom, 1e-9)

        # Prevent damage healing
        D_n_new = min(max(D_n_new, self.D_n_old), 0.999)
        D_s_new = min(max(D_s_new, self.D_s_old), 0.999)

        # 5. Nominal Stress Assembly
        sig_l = self.D_local @ e_l
        sig_l[2] = (1.0 - D_n_new) * sig_n
        sig_l[4] = (1.0 - D_s_new) * tau_23
        sig_l[5] = (1.0 - D_s_new) * tau_13

        stress_global = self.T_sig.T @ sig_l

        trial_dict = {
            'q': q_new,
            'ep_n': ep_n_new,
            'ep_23': ep_23_new,
            'ep_13': ep_13_new,
            'W1': W1,
            'W2': W2,
            'W3': W3,
            'D_n': D_n_new,
            'D_s': D_s_new
        }
        return stress_global, trial_dict

    def _integrate_stress(self, current_strain):
        """Helper for numerical tangent."""
        stress, _ = self._compute_state(current_strain)
        return stress

    def _compute_numerical_tangent(self, current_strain, eps=1e-8):
        """Computes the numerical Jacobian ensuring quadratic convergence in NR."""
        D_num = np.zeros((6, 6))
        stress_base = self._integrate_stress(current_strain)

        for j in range(6):
            strain_pert = current_strain.copy()
            strain_pert[j] += eps
            stress_pert = self._integrate_stress(strain_pert)
            D_num[:, j] = (stress_pert - stress_base) / eps

        return D_num

    def update_state(self, current_strain):
        self.strain = current_strain

        # 1. Compute state and trial variables
        self.stress, trial_dict = self._compute_state(current_strain)

        # 2. Save trial variables for commit
        self.q_trial = trial_dict['q']
        self.ep_n_trial = trial_dict['ep_n']
        self.ep_23_trial = trial_dict['ep_23']
        self.ep_13_trial = trial_dict['ep_13']
        self.W_pl_1_trial = trial_dict['W1']
        self.W_pl_2_trial = trial_dict['W2']
        self.W_pl_3_trial = trial_dict['W3']
        self.D_n_trial = trial_dict['D_n']
        self.D_s_trial = trial_dict['D_s']

        # 3. Compute numerical tangent
        self.D_tangent = self._compute_numerical_tangent(current_strain)

        # Add minimal stiffness to diagonal to prevent singular matrices
        self.D_tangent += np.eye(6) * self.E_min

        return self.stress, self.D_tangent

    def get_tangent_matrix(self):
        return self.D_tangent

    def get_stress(self, strain):
        return self.stress

    def commit(self):
        self.stress_old = self.stress.copy()
        self.strain_old = self.strain.copy()

        self.W_pl_1_old = self.W_pl_1_trial
        self.W_pl_2_old = self.W_pl_2_trial
        self.W_pl_3_old = self.W_pl_3_trial

        self.ep_n_old = self.ep_n_trial
        self.ep_23_old = self.ep_23_trial
        self.ep_13_old = self.ep_13_trial

        self.q_old = self.q_trial
        self.D_n_old = self.D_n_trial
        self.D_s_old = self.D_s_trial
