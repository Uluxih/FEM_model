import numpy as np
from FEM.Abstract.Integration_Point_Level import ConstitutiveModel


class UbiquitousJointModel3D(ConstitutiveModel):
    """
    Строгая модель эквивалентной сплошной среды (Последовательное соединение).
    Реализует C_eq = C_rock + C_joint с точным тензорным вращением
    и ТОЧНЫМ алгоритмическим согласованным тензором жесткости (Consistent Tangent Operator)
    для обеспечения квадратичной сходимости метода Ньютона-Рафсона.
    """

    def __init__(self, material):
        super().__init__(material)

        E = self.material.E
        nu = self.material.nu
        joint_params = self.material.joint_params

        # 1. Параметры трещины
        self.kn = joint_params['kn']
        self.ks = joint_params['ks']
        self.kt = joint_params['kt']
        self.S = joint_params['spacing']

        self.c = joint_params.get('c', 0.0)
        self.phi = np.radians(joint_params.get('phi', 0.0))
        self.psi = np.radians(joint_params.get('psi', 0.0))
        self.t_limit = joint_params.get('t', 0.0)

        # Ограничение на растяжение (Apex correction)
        if self.phi > 0:
            t_max = self.c / np.tan(self.phi)
            self.t_limit = min(self.t_limit, t_max)

        # 2. Построение матриц в ЛОКАЛЬНЫХ осях трещины (Нормаль = Z')
        self.C_rock = self._build_isotropic_compliance(E, nu)
        self.C_joint_local = self._build_joint_compliance(self.kn, self.ks, self.kt, self.S)

        # Строгое последовательное соединение (сумма податливостей)
        self.C_eq_local = self.C_rock + self.C_joint_local

        # Эквивалентная упругая жесткость в локальных осях
        self.D_eq_local = np.linalg.inv(self.C_eq_local)

        # 3. Извлечение СТРОГИХ модулей для Return Mapping
        self.alpha_1 = self.D_eq_local[2, 2]  # Реакция нормального напряжения
        self.alpha_2 = (self.D_eq_local[4, 4] + self.D_eq_local[5, 5]) / 2.0  # Усредненная сдвиговая жесткость

        # 4. Построение матрицы поворота и глобальной матрицы жесткости
        normal = np.array(joint_params.get('normal', [0.0, 0.0, 1.0]))
        self.R = self._build_rotation_matrix(normal)
        self.D_eq_global = self._rotate_stiffness_matrix(self.D_eq_local, self.R)

        # Переменные состояния
        self.stress_old = np.zeros(6)
        self.strain_old = np.zeros(6)
        self.stress = np.zeros(6)
        self.strain = np.zeros(6)

    def _build_isotropic_compliance(self, E, nu):
        """Матрица податливости изотропной породы (6x6)"""
        C = np.zeros((6, 6))
        G = E / (2.0 * (1.0 + nu))
        C[0, 0] = C[1, 1] = C[2, 2] = 1.0 / E
        C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = -nu / E
        C[3, 3] = C[4, 4] = C[5, 5] = 1.0 / G
        return C

    def _build_joint_compliance(self, kn, ks, kt, S):
        """Матрица податливости трещин в локальных осях (6x6)"""
        C = np.zeros((6, 6))
        C[2, 2] = 1.0 / (kn * S)  # Нормаль к трещине (ось Z')
        C[4, 4] = 1.0 / (kt * S)  # Сдвиг Y'Z'
        C[5, 5] = 1.0 / (ks * S)  # Сдвиг X'Z'
        return C

    def _build_rotation_matrix(self, normal):
        """Тензор поворота R (столбцы - локальные базисные векторы)"""
        nz = np.array(normal, dtype=float)
        nz = nz / np.linalg.norm(nz)
        if abs(nz[2]) > 0.9999:
            nx = np.array([1.0, 0.0, 0.0])
            ny = np.cross(nz, nx)
        else:
            ny = np.cross(nz, np.array([0.0, 0.0, 1.0]))
            ny = ny / np.linalg.norm(ny)
            nx = np.cross(ny, nz)
        nx = nx / np.linalg.norm(nx)
        return np.column_stack((nx, ny, nz))

    def _voigt_to_stress_tensor(self, v):
        return np.array([[v[0], v[3], v[5]], [v[3], v[1], v[4]], [v[5], v[4], v[2]]])

    def _stress_tensor_to_voigt(self, t):
        return np.array([t[0, 0], t[1, 1], t[2, 2], t[0, 1], t[1, 2], t[0, 2]])

    def _voigt_to_strain_tensor(self, v):
        return np.array(
            [[v[0], v[3] / 2.0, v[5] / 2.0], [v[3] / 2.0, v[1], v[4] / 2.0], [v[5] / 2.0, v[4] / 2.0, v[2]]])

    def _strain_tensor_to_voigt(self, t):
        return np.array([t[0, 0], t[1, 1], t[2, 2], 2.0 * t[0, 1], 2.0 * t[1, 2], 2.0 * t[0, 2]])

    def _rotate_stiffness_matrix(self, D_local, R):
        """Строгое вращение матрицы жесткости 6x6 через тензорные преобразования."""
        D_global = np.zeros((6, 6))
        for j in range(6):
            eps_voigt_global = np.zeros(6)
            eps_voigt_global[j] = 1.0
            eps_tensor_global = self._voigt_to_strain_tensor(eps_voigt_global)
            eps_tensor_local = R.T @ eps_tensor_global @ R
            eps_voigt_local = self._strain_tensor_to_voigt(eps_tensor_local)

            sig_voigt_local = D_local @ eps_voigt_local

            sig_tensor_local = self._voigt_to_stress_tensor(sig_voigt_local)
            sig_tensor_global = R @ sig_tensor_local @ R.T
            sig_voigt_global = self._stress_tensor_to_voigt(sig_tensor_global)

            D_global[:, j] = sig_voigt_global
        return D_global

    def get_tangent_matrix(self):
        return self.D_eq_global

    def get_stress(self, strain):
        return self.stress

    def update_state(self, current_strain):
        self.strain = current_strain
        d_strain = current_strain - self.strain_old

        # 1. Упругий предиктор
        stress_trial_voigt = self.stress_old + self.D_eq_global @ d_strain
        stress_trial_tensor = self._voigt_to_stress_tensor(stress_trial_voigt)

        # 2. Перевод напряжений в локальные оси трещины
        local_stress = self.R.T @ stress_trial_tensor @ self.R

        sig_33 = local_stress[2, 2]
        sig_13 = local_stress[0, 2]
        sig_23 = local_stress[1, 2]
        tau = np.sqrt(sig_13 ** 2 + sig_23 ** 2)

        # 3. Проверка критериев
        f_s = tau + sig_33 * np.tan(self.phi) - self.c
        f_t = sig_33 - self.t_limit

        if f_s <= 0 and f_t <= 0:
            self.stress = stress_trial_voigt
            # Если упругость — возвращаем начальную эквивалентную матрицу
            return self.stress, self.D_eq_global

        # 4. Пластическая коррекция и расчет Согласованной Матрицы
        is_tension = f_t > 0

        if f_s > 0 and not is_tension:
            lam_s = f_s / (self.alpha_2 - self.alpha_1 * np.tan(self.psi) * np.tan(self.phi))
            sig_33_test = sig_33 + lam_s * self.alpha_1 * np.tan(self.psi)
            if sig_33_test > self.t_limit:
                is_tension = True

        # Инициализируем локальную касательную матрицу упругой жесткостью
        D_tan_local = self.D_eq_local.copy()

        if is_tension:
            # --- Разрушение при растяжении ---
            local_stress[2, 2] = self.t_limit
            local_stress[0, 2] = local_stress[2, 0] = 0.0
            local_stress[1, 2] = local_stress[2, 1] = 0.0

            # Точная касательная: жесткость по нормали и сдвигам обнуляется
            D_tan_local[2, :] = 0.0;
            D_tan_local[:, 2] = 0.0
            D_tan_local[4, :] = 0.0;
            D_tan_local[:, 4] = 0.0
            D_tan_local[5, :] = 0.0;
            D_tan_local[:, 5] = 0.0

        else:
            # --- Сдвиговое разрушение ---
            lam_s = f_s / (self.alpha_2 - self.alpha_1 * np.tan(self.psi) * np.tan(self.phi))

            sig_33_new = sig_33 + lam_s * self.alpha_1 * np.tan(self.psi)
            tau_new = tau - lam_s * self.alpha_2

            apex_stress = self.c / np.tan(self.phi) if self.phi > 0 else float('inf')

            if sig_33_new > apex_stress:
                # Попадание в вершину конуса (Apex)
                local_stress[2, 2] = apex_stress
                local_stress[0, 2] = local_stress[2, 0] = 0.0
                local_stress[1, 2] = local_stress[2, 1] = 0.0

                # Точная касательная: полная потеря несущей способности на площадке
                D_tan_local[2, :] = 0.0;
                D_tan_local[:, 2] = 0.0
                D_tan_local[4, :] = 0.0;
                D_tan_local[:, 4] = 0.0
                D_tan_local[5, :] = 0.0;
                D_tan_local[:, 5] = 0.0
            else:
                # Гладкая часть конуса
                local_stress[2, 2] = sig_33_new
                if tau > 0:
                    local_stress[0, 2] = local_stress[2, 0] = sig_13 * (tau_new / tau)
                    local_stress[1, 2] = local_stress[2, 1] = sig_23 * (tau_new / tau)

                # Вычисление континуальной упругопластической матрицы (Continuum Elastoplastic Tangent)
                n_vec = np.zeros(6)  # Нормаль к поверхности текучести (df/d_sigma)
                m_vec = np.zeros(6)  # Направление пластического течения (dg/d_sigma)

                n_vec[2] = np.tan(self.phi)
                m_vec[2] = np.tan(self.psi)

                if tau > 0:
                    n_vec[4] = m_vec[4] = sig_23 / tau  # Компонента yz
                    n_vec[5] = m_vec[5] = sig_13 / tau  # Компонента xz

                D_m = self.D_eq_local @ m_vec
                n_D = n_vec @ self.D_eq_local
                denom = np.dot(n_vec, D_m)

                if abs(denom) > 1e-12:
                    D_ep = self.D_eq_local - np.outer(D_m, n_D) / denom

                    # ТОЧНАЯ АЛГОРИТМИЧЕСКАЯ КОРРЕКЦИЯ (Algorithmic Tangent Operator)
                    # Учитывает кривизну конуса при радиальном возврате.
                    # Сдвиговая жесткость в направлении, перпендикулярном радиальному вектору,
                    # должна быть домножена на коэффициент beta = tau_new / tau_trial.
                    if tau > 0:
                        beta = tau_new / tau
                        n_s4 = sig_23 / tau
                        n_s5 = sig_13 / tau

                        # Матрица снижения жесткости из-за кривизны (delta_D)
                        delta_D = np.zeros((6, 6))
                        delta_D[4, 4] = self.alpha_2 * (1.0 - n_s4 * n_s4)
                        delta_D[5, 5] = self.alpha_2 * (1.0 - n_s5 * n_s5)
                        delta_D[4, 5] = delta_D[5, 4] = -self.alpha_2 * n_s4 * n_s5

                        # Итоговая алгоритмическая матрица
                        D_tan_local = D_ep - (1.0 - beta) * delta_D
                    else:
                        D_tan_local = D_ep

        # 5. Обратный перевод скорректированных напряжений и КАСАТЕЛЬНОЙ МАТРИЦЫ в глобальные оси
        global_stress_tensor = self.R @ local_stress @ self.R.T
        self.stress = self._stress_tensor_to_voigt(global_stress_tensor)

        # Вращение согласованной матрицы
        D_tan_global = self._rotate_stiffness_matrix(D_tan_local, self.R)

        return self.stress, D_tan_global

    def commit(self):
        self.stress_old = self.stress.copy()
        self.strain_old = self.strain.copy()
