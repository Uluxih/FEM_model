import numpy as np


class JointedRockModel:
    def __init__(self, E, nu, c_joint, phi_joint, theta):
        self.E = E
        self.nu = nu
        self.c_joint = c_joint
        self.phi_joint = phi_joint
        self.theta = np.radians(theta)

        # Упругая матрица (плоская деформация, 2D)
        factor = E / ((1 + nu) * (1 - 2 * nu))
        self.D_el = factor * np.array([
            [1 - nu, nu, 0],
            [nu, 1 - nu, 0],
            [0, 0, (1 - 2 * nu) / 2]
        ])

    def get_rotation_matrix(self):
        c, s = np.cos(self.theta), np.sin(self.theta)
        return np.array([[c ** 2, s ** 2, 2 * c * s],
                         [s ** 2, c ** 2, -2 * c * s],
                         [-c * s, c * s, c ** 2 - s ** 2]])

    def get_flow_vector_n(self, stress_local):
        """
        Градиент функции текучести f = |tau| - c - sigma_n * tan(phi)
        f = |sigma_local[2]| - c - sigma_local[0] * tan(phi)
        """
        phi_rad = np.radians(self.phi_joint)
        # df/d(sigma_n) = -tan(phi)
        # df/d(sigma_t) = 0
        # df/d(tau) = sign(tau)
        n = np.array([-np.tan(phi_rad), 0, np.sign(stress_local[2])])
        return n

    def get_Dep_local(self, stress_local, is_plastic):
        if not is_plastic:
            return self.D_el

        # n - вектор нормали к поверхности текучести
        n = self.get_flow_vector_n(stress_local)

        # Формула: Dep = De - (De * n * n^T * De) / (n^T * De * n)
        # (предполагая ассоциированное течение m=n и H=0)

        De_n = np.dot(self.D_el, n)
        denom = np.dot(n, De_n)

        if abs(denom) < 1e-12:  # Защита от деления на ноль
            return self.D_el

        Dep = self.D_el - np.outer(De_n, De_n) / denom
        return Dep

    def get_Dep_global(self, stress_global, is_plastic):
        # 1. Поворот напряжений в локальные
        R = self.get_rotation_matrix()
        stress_local = np.dot(R, stress_global)

        # 2. Вычисление Dep в локальных
        Dep_local = self.get_Dep_local(stress_local, is_plastic)

        # 3. Поворот Dep обратно: Dep_global = R^T * Dep_local * R
        Dep_global = np.dot(R.T, np.dot(Dep_local, R))
        return Dep_global


# --- ПРИМЕР ИСПОЛЬЗОВАНИЯ ---
model = JointedRockModel(E=20e9, nu=0.25, c_joint=1.0e6, phi_joint=30, theta=30)
current_stress = np.array([10e6, 5e6, 2e6])  # [sigma_xx, sigma_yy, tau_xy]

# Допустим, мы проверили критерий и знаем, что трещина скользит
is_plastic = True

Dep = model.get_Dep_global(current_stress, is_plastic)

print("Касательная матрица жесткости (Dep) в глобальных координатах:")
print(np.round(Dep, 2))
