import numpy as np


class PlanePlasticityModel:
    def __init__(self, hardening_type='cohesion'):
        self.hardening_type = hardening_type

        # Параметры прочности
        self.c_fixed = 2.0
        self.mu_c = 0.6
        self.A_mu = 0.005

        self.mu_fixed = 0.4
        self.c_0 = 1.0
        self.c_u = 4.0
        self.A_c = 0.005

        self.sigma_0 = 1.0
        self.eta_c = 0.7

        self.kappa = 0.0  # Накопленная пластическая деформация

    def get_friction_and_hardening(self):
        mu = self.mu_c * self.kappa / (self.A_mu + self.kappa)
        dmu_dkappa = self.mu_c * self.A_mu / ((self.A_mu + self.kappa) ** 2)
        return mu, dmu_dkappa

    def get_cohesion_and_hardening(self):
        c = self.c_0 + (self.c_u - self.c_0) * self.kappa / (self.A_c + self.kappa)
        dc_dkappa = (self.c_u - self.c_0) * self.A_c / ((self.A_c + self.kappa) ** 2)
        return c, dc_dkappa

    def get_tangent_stiffness_matrix(self, sigma_tensor, normal_vector, D_elastic):
        """Вычисление упруго-пластической матрицы жесткости 6x6"""
        n = np.array(normal_vector, dtype=float)
        n /= np.linalg.norm(n)

        t_vec = sigma_tensor @ n
        sigma_n = np.dot(t_vec, n)

        t_s_vec = t_vec - sigma_n * n
        tau = np.linalg.norm(t_s_vec)

        if tau > 1e-12:
            s_vec = t_s_vec / tau
        else:
            s_vec = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            s_vec = s_vec - np.dot(s_vec, n) * n
            s_vec /= np.linalg.norm(s_vec)

        if self.hardening_type == 'friction':
            mu, dmu_dkappa = self.get_friction_and_hardening()
            c = self.c_fixed
            H = - (dmu_dkappa * sigma_n)
        elif self.hardening_type == 'cohesion':
            mu = self.mu_fixed
            c, dc_dkappa = self.get_cohesion_and_hardening()
            H = dc_dkappa
        else:
            raise ValueError("Неизвестный тип упрочнения")

        # Проверка текучести
        f_yield = tau + mu * sigma_n - c

        if f_yield <= 1e-8:
            print(" -> Площадка работает упруго (Критерий f <= 0)")
            return D_elastic.copy()

        print(f" -> Площадка ТЕЧЕТ! (Критерий f = {f_yield:.3f} > 0)")

        # Производная пластического потенциала
        denom = self.sigma_0 - sigma_n
        if abs(denom) < 1e-8:
            dpsi_dsigma = -self.eta_c
        else:
            dpsi_dsigma = (tau / denom) - self.eta_c

        N_tens = np.outer(n, n)
        S_tens = 0.5 * (np.outer(s_vec, n) + np.outer(n, s_vec))

        df_dsigma_tens = mu * N_tens + S_tens
        dpsi_dsigma_tens = dpsi_dsigma * N_tens + S_tens

        def to_voigt(tensor):
            return np.array([
                tensor[0, 0], tensor[1, 1], tensor[2, 2],
                2.0 * tensor[0, 1], 2.0 * tensor[1, 2], 2.0 * tensor[0, 2]
            ])

        n_voigt = to_voigt(df_dsigma_tens)
        m_voigt = to_voigt(dpsi_dsigma_tens)

        D_e_m = D_elastic @ m_voigt
        n_D_e = n_voigt @ D_elastic

        denominator = np.dot(n_voigt, D_e_m) + H
        if denominator <= 0:
            denominator = 1e-8

        D_plastic = np.outer(D_e_m, n_D_e) / denominator
        D_ep = D_elastic - D_plastic

        return D_ep


# ==========================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ==========================================

# # 1. Функция для создания 3D упругой матрицы (изотропный материал)
# def create_elastic_matrix_3d(E, nu):
#     C = E / ((1 + nu) * (1 - 2 * nu))
#     D = np.zeros((6, 6))
#     D[0:3, 0:3] = C * nu
#     D[0, 0] = D[1, 1] = D[2, 2] = C * (1 - nu)
#     G = E / (2 * (1 + nu))
#     D[3, 3] = D[4, 4] = D[5, 5] = G
#     return D
#
#
# # Задаем упругие свойства (например, бетон/порода)
# E = 20000.0  # МПа
# nu = 0.2
# D_elastic = create_elastic_matrix_3d(E, nu)
#
# # 2. Задаем состояние
# # Допустим, площадка снижения жесткости горизонтальна (нормаль смотрит по оси Y)
# normal_vector = np.array([0.0, 1.0, 0.0])
#
# # Создаем тензор напряжений, который гарантированно вызовет пластику.
# # Сжимаем по Y (sigma_yy = -5 МПа) и даем сильный сдвиг (tau_xy = 4 МПа)
# sigma_tensor = np.array([
#     [0.0, 4.0, 0.0],
#     [4.0, -5.0, 0.0],
#     [0.0, 0.0, 0.0]
# ])
#
# # 3. Инициализируем модель
# model = PlanePlasticityModel(hardening_type='cohesion')
#
# # 4. Вычисляем касательную матрицу жесткости
# np.set_printoptions(precision=1, suppress=True, linewidth=100)
#
# print("=== Исходная упругая матрица D_elastic (D^co) ===")
# print(D_elastic)
# print("-" * 50)
#
# print("=== Проверка состояния на площадке ===")
# D_ep = model.get_tangent_stiffness_matrix(sigma_tensor, normal_vector, D_elastic)
# print("-" * 50)
#
# print("=== Упруго-пластическая матрица D_ep (после снижения жесткости) ===")
# print(D_ep)
# print("-" * 50)
#
# print("=== Разница (Матрица падения жесткости D_plastic) ===")
# print(D_elastic - D_ep)
