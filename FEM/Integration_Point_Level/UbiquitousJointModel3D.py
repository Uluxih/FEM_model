import numpy as np
from FEM.Abstract.Integration_Point_Level import ConstitutiveModel


class UbiquitousJointModel3D(ConstitutiveModel):
    """
    Модель эквивалентной сплошной среды с одной системой трещин (Ubiquitous Joint).
    Трещина фиксирована горизонтально (нормаль совпадает с глобальной осью Z).
    """

    def __init__(self, material):
        super().__init__(material)

        E = self.material.E
        nu = self.material.nu
        joint_params = self.material.joint_params

        # Упругие константы для алгоритма Return Mapping (Формула 6 из док. FLAC3D)
        self.K = E / (3.0 * (1.0 - 2.0 * nu))
        self.G = E / (2.0 * (1.0 + nu))
        self.alpha_1 = self.K + (4.0 / 3.0) * self.G
        self.alpha_2 = 2.0 * self.G

        # Параметры трещины
        self.kn = joint_params['kn']
        self.ks = joint_params['ks']
        self.kt = joint_params['kt']
        self.S = joint_params['spacing']

        self.c = joint_params['c']
        self.phi = np.radians(joint_params['phi'])
        self.psi = np.radians(joint_params['psi'])
        self.t_limit = joint_params['t']

        # Вычисление эквивалентной матрицы жесткости D_eq
        self.D_eq = self._build_equivalent_stiffness(E, nu)

        # Переменные состояния
        self.stress_old = np.zeros(6)
        self.strain_old = np.zeros(6)
        self.stress = np.zeros(6)
        self.strain = np.zeros(6)

    def _build_equivalent_stiffness(self, E, nu):
        """Строит эквивалентную упругую матрицу (Порода + Трещина)"""
        # 1. Податливость целой породы
        C_rock = np.zeros((6, 6))
        C_rock[0, 0] = C_rock[1, 1] = C_rock[2, 2] = 1.0 / E
        C_rock[0, 1] = C_rock[0, 2] = C_rock[1, 0] = C_rock[1, 2] = C_rock[2, 0] = C_rock[2, 1] = -nu / E
        C_rock[3, 3] = C_rock[4, 4] = C_rock[5, 5] = 1.0 / self.G

        # 2. Податливость трещины (Нормаль по Z, поэтому локальные оси = глобальные)
        # Вектор Фойгта в проекте: [xx, yy, zz, xy, yz, xz] -> Индексы: 0, 1, 2, 3, 4, 5
        C_joint = np.zeros((6, 6))
        C_joint[2, 2] = 1.0 / (self.kn * self.S)  # Индекс 2: zz (Нормаль)
        C_joint[4, 4] = 1.0 / (self.kt * self.S)  # Индекс 4: yz (Сдвиг)
        C_joint[5, 5] = 1.0 / (self.ks * self.S)  # Индекс 5: xz (Сдвиг)

        # 3. Эквивалентная жесткость
        C_eq = C_rock + C_joint
        return np.linalg.inv(C_eq)

    def _voigt_to_tensor(self, v):
        """Перевод вектора Фойгта проекта в тензор 3x3"""
        return np.array([
            [v[0], v[3], v[5]],
            [v[3], v[1], v[4]],
            [v[5], v[4], v[2]]
        ])

    def _tensor_to_voigt(self, t):
        """Перевод тензора 3x3 в вектор Фойгта проекта"""
        return np.array([t[0, 0], t[1, 1], t[2, 2], t[0, 1], t[1, 2], t[0, 2]])

    def get_tangent_matrix(self):
        return self.D_eq

    def get_stress(self, strain):
        return self.stress

    def update_state(self, current_strain):
        self.strain = current_strain
        d_strain = current_strain - self.strain_old

        # 1. Упругий предиктор (Elastic guess)
        stress_trial = self.stress_old + self.D_eq @ d_strain

        # Перевод в тензор (оси совпадают, т.к. трещина по Z)
        sig_tensor = self._voigt_to_tensor(stress_trial)

        # Компоненты на плоскости трещины
        sig_33 = sig_tensor[2, 2]  # Нормальное напряжение (сжатие < 0)
        sig_13 = sig_tensor[0, 2]  # Сдвиг XZ
        sig_23 = sig_tensor[1, 2]  # Сдвиг YZ

        tau = np.sqrt(sig_13 ** 2 + sig_23 ** 2)

        # 2. Функции текучести (Уравнения 8 и 9)
        f_s = tau + sig_33 * np.tan(self.phi) - self.c
        f_t = sig_33 - self.t_limit

        is_shear = False
        is_tension = False

        if f_t > 0:
            is_tension = True
        elif f_s > 0:
            is_shear = True

        # 3. Пластическая коррекция (Return Mapping)
        if is_tension:
            # Разрушение при растяжении
            lam_t = f_t / self.alpha_1
            sig_33_new = self.t_limit
            sig_13_new = 0.0
            sig_23_new = 0.0

            sig_tensor[2, 2] = sig_33_new
            sig_tensor[0, 2] = sig_tensor[2, 0] = sig_13_new
            sig_tensor[1, 2] = sig_tensor[2, 1] = sig_23_new

        elif is_shear:
            # Сдвиговое разрушение
            lam_s = f_s / (self.alpha_2 - self.alpha_1 * np.tan(self.psi) * np.tan(self.phi))

            sig_33_new = sig_33 + lam_s * self.alpha_1 * np.tan(self.psi)
            tau_new = tau - lam_s * self.alpha_2

            # Поправка на вершину конуса текучести (Apex correction)
            apex_stress = self.c / np.tan(self.phi) if self.phi > 0 else float('inf')
            if sig_33_new > apex_stress:
                sig_33_new = apex_stress
                tau_new = 0.0
                sig_13_new = 0.0
                sig_23_new = 0.0
            else:
                sig_13_new = sig_13 * (tau_new / tau) if tau > 0 else 0.0
                sig_23_new = sig_23 * (tau_new / tau) if tau > 0 else 0.0

            sig_tensor[2, 2] = sig_33_new
            sig_tensor[0, 2] = sig_tensor[2, 0] = sig_13_new
            sig_tensor[1, 2] = sig_tensor[2, 1] = sig_23_new

        # Сохраняем обновленные напряжения
        self.stress = self._tensor_to_voigt(sig_tensor)

        # Возвращаем упругую матрицу (Initial Stiffness) для стабильности решателя
        return self.stress, self.D_eq

    def commit(self):
        self.stress_old = self.stress.copy()
        self.strain_old = self.strain.copy()
