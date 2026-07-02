import numpy as np


class DruckerPragerMatrix2D:
    """
    Модель сплошной среды (матрицы) для XFEM.
    Описывает поведение породы до и после образования трещины (вдали от нее).
    """

    def __init__(self, E, nu, matrix_params):
        self.E = E
        self.nu = nu

        self.phi_m = np.radians(matrix_params.get('phi', 30.0))
        self.psi_m = np.radians(matrix_params.get('psi', 0.0))
        self.c_m = matrix_params.get('c', 1e6)
        self.sin_phi_m = np.sin(self.phi_m)
        self.cos_phi_m = np.cos(self.phi_m)
        self.sin_psi_m = np.sin(self.psi_m)

        self.D_elastic = self._build_plane_stress_stiffness(E, nu)

        self.stress_old = np.zeros(3)
        self.strain_old = np.zeros(3)
        self.stress = np.zeros(3)
        self.D_tangent = self.D_elastic.copy()

    def _build_plane_stress_stiffness(self, E, nu):
        D = np.zeros((3, 3))
        c1 = E / (1.0 - nu ** 2)
        c2 = E * nu / (1.0 - nu ** 2)
        G = E / (2.0 * (1.0 + nu))
        D[0, 0] = D[1, 1] = c1
        D[0, 1] = D[1, 0] = c2
        D[2, 2] = G
        return D

    def update_state(self, current_strain_voigt):
        deps = current_strain_voigt - self.strain_old
        sig_tr = self.stress_old + self.D_elastic @ deps

        p_tr = 0.5 * (sig_tr[0] + sig_tr[1])
        sx = 0.5 * (sig_tr[0] - sig_tr[1])
        txy = sig_tr[2]
        q_tr = np.hypot(sx, txy)

        F_tr = q_tr + p_tr * self.sin_phi_m - self.c_m * self.cos_phi_m
        if F_tr <= 1e-10:
            self.stress = sig_tr.copy()
            self.D_tangent = self.D_elastic.copy()
            return self.stress, self.D_tangent

        # Return mapping Друкера-Прагера
        K_2D = 0.5 * (self.D_elastic[0, 0] + self.D_elastic[0, 1])
        G = self.D_elastic[2, 2]

        denom = G + K_2D * self.sin_phi_m * self.sin_psi_m
        dlam = F_tr / denom

        p_new = p_tr - K_2D * self.sin_psi_m * dlam
        q_new = q_tr - G * dlam

        p_apex = self.c_m * self.cos_phi_m / (self.sin_phi_m + 1e-12) if self.sin_phi_m > 1e-12 else np.inf

        if q_new < 0 or p_new > p_apex:
            p_new = p_apex
            q_new = 0.0

        if q_tr > 1e-12:
            sx_new = sx * (q_new / q_tr)
            txy_new = txy * (q_new / q_tr)
        else:
            sx_new = 0.0
            txy_new = 0.0

        self.stress[0] = p_new + sx_new
        self.stress[1] = p_new - sx_new
        self.stress[2] = txy_new

        # Для простоты касательная матрица остается упругой (можно добавить консистентную)
        self.D_tangent = self.D_elastic.copy()

        return self.stress.copy(), self.D_tangent

    def commit(self):
        self.stress_old = self.stress.copy()
        # Внешний решатель должен обновить self.strain_old