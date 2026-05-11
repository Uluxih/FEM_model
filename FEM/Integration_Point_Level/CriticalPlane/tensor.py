import numpy as np
from scipy.optimize import minimize, minimize_scalar


# ==========================================
# КЛАСС ТЕНЗОРА НАПРЯЖЕНИЙ
# ==========================================
class StressTensor:
    """
    Тензор напряжений в трёхмерном пространстве.
    Хранит матрицу 3x3 и предоставляет методы для вычисления напряжений на заданной площадке.
    """

    def __init__(self, sigma_xx, sigma_yy, sigma_zz, tau_xy, tau_yz, tau_xz):
        """
        Инициализация тензора компонентами напряжений.
        """
        self.matrix = np.array([
            [sigma_xx, tau_xy, tau_xz],
            [tau_xy, sigma_yy, tau_yz],
            [tau_xz, tau_yz, sigma_zz]
        ])

    @classmethod
    def from_matrix(cls, matrix):
        """
        Альтернативный конструктор: принимает готовую матрицу 3x3.
        """
        if matrix.shape != (3, 3):
            raise ValueError("Матрица должна быть размером 3x3")
        if not np.allclose(matrix, matrix.T):
            raise ValueError("Матрица должна быть симметричной")
        # Извлекаем компоненты
        sigma_xx = matrix[0, 0]
        sigma_yy = matrix[1, 1]
        sigma_zz = matrix[2, 2]
        tau_xy = matrix[0, 1]
        tau_yz = matrix[1, 2]
        tau_xz = matrix[0, 2]
        return cls(sigma_xx, sigma_yy, sigma_zz, tau_xy, tau_yz, tau_xz)

    @classmethod
    def from_string(cls, tensor_str):
        """
        Загрузка тензора из строки (формат: три строки по три числа).
        """
        lines = tensor_str.strip().split('\n')
        data_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
        matrix = []
        for line in data_lines:
            row = [float(x) for x in line.split()]
            matrix.append(row)
        matrix = np.array(matrix)
        return cls.from_matrix(matrix)

    @classmethod
    def from_principal_2d(cls, sigma_1, sigma_2, alpha):
        """
        Создает тензор напряжений в осях анизотропии для плоского напряженного состояния (в плоскости XZ).
        sigma_1, sigma_2 - главные напряжения
        alpha - угол между осью 1 и осью X анизотропии (в радианах).
        """
        sigma_xx = (sigma_1 + sigma_2) / 2 + (sigma_1 - sigma_2) / 2 * np.cos(2 * alpha)
        sigma_zz = (sigma_1 + sigma_2) / 2 - (sigma_1 - sigma_2) / 2 * np.cos(2 * alpha)
        tau_xz = (sigma_1 - sigma_2) / 2 * np.sin(2 * alpha)
        return cls(sigma_xx, 0.0, sigma_zz, 0.0, 0.0, tau_xz)

    def to_principal_2d(self):
        """
        Возвращает два главных напряжения и угол анизотропии для плоского напряженного состояния (в плоскости XZ).
        Возвращает: (sigma_1, sigma_2, alpha)
        """
        sigma_xx = self.matrix[0, 0]
        sigma_zz = self.matrix[2, 2]
        tau_xz = self.matrix[0, 2]

        center = (sigma_xx + sigma_zz) / 2
        radius = np.sqrt(((sigma_xx - sigma_zz) / 2) ** 2 + tau_xz ** 2)

        sigma_1 = center + radius
        sigma_2 = center - radius

        alpha = 0.5 * np.arctan2(2 * tau_xz, sigma_xx - sigma_zz)
        return sigma_1, sigma_2, alpha

    def traction(self, n):
        """
        Вектор напряжений на площадке с нормалью n (3‑элементный массив).
        t = sigma · n
        """
        return self.matrix @ n

    def normal_stress(self, n):
        """
        Нормальное напряжение на площадке: σ_n = n·(σ·n)
        """
        t = self.traction(n)
        return np.dot(n, t)

    def shear_stress_vector(self, n):
        """
        Вектор касательного напряжения: τ = t - σ_n·n
        """
        t = self.traction(n)
        sigma_n = self.normal_stress(n)
        return t - sigma_n * n

    def shear_stress_magnitude(self, n):
        """
        Модуль касательного напряжения: |τ|
        """
        tau_vec = self.shear_stress_vector(n)
        return np.linalg.norm(tau_vec)


# ==========================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================
def get_normal_vector(params, mode):
    """
    Возвращает вектор нормали n в зависимости от режима.
    Для '2D_XZ': params = [alpha], угол в плоскости XZ.
    Для '3D': params = [theta, phi].
    """
    if mode == '2D_XZ':
        alpha = params[0] if isinstance(params, (list, tuple, np.ndarray)) else params
        nx = np.cos(alpha)
        ny = 0.0
        nz = np.sin(alpha)
    else:  # 3D
        theta, phi = params[0], params[1]
        nx = np.sin(theta) * np.cos(phi)
        ny = np.sin(theta) * np.sin(phi)
        nz = np.cos(theta)
    return np.array([nx, ny, nz])


