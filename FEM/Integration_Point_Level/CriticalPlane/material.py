import numpy as np
from scipy.optimize import minimize, minimize_scalar


# ==========================================
# КЛАСС МАТЕРИАЛА
# ==========================================
class Material:
    """
    Класс, представляющий ортотропный материал с анизотропным законом трения.
    Содержит свойства прочности и методы оценки критических плоскостей.
    """

    def __init__(self, mu=0.3, A_tensor=None, Rpx=None, Rpy=None, Rpz=None,
                 Rcx=None, Rcy=None, Rcz=None):
        """
        Инициализация свойств материала.

        Параметры:
        ----------
        mu : float
            Коэффициент трения (при сжатии, sigma_n < 0)
        A_tensor : np.ndarray, shape (9,9), optional
            Тензор анизотропии для вычисления параметра C (сопротивление сдвигу).
            Если None, принимается единичная матрица.
        Rpx, Rpy, Rpz : float, optional
            Пределы прочности на растяжение вдоль осей X, Y, Z.
        Rcx, Rcy, Rcz : float, optional
            Пределы прочности на сжатие вдоль осей X, Y, Z.
        """
        self.mu = mu
        self.A_tensor = A_tensor if A_tensor is not None else np.eye(9)

        # Прочности на растяжение (положительные значения)
        self.Rpx = Rpx if Rpx is not None else 1e9  # по умолчанию очень высокие
        self.Rpy = Rpy if Rpy is not None else 1e9
        self.Rpz = Rpz if Rpz is not None else 1e9

        # Прочности на сжатие (положительные значения)
        self.Rcx = Rcx if Rcx is not None else 1e9
        self.Rcy = Rcy if Rcy is not None else 1e9
        self.Rcz = Rcz if Rcz is not None else 1e9

        # Внутренние константы (понижение трения при растяжении)
        self.mu_tensile = -0.4



    def get_cohesion(self, n, s):
        """
        Вычисляет начальное сцепление на площадке с нормалью n и направлением сдвига s.

        Параметры:
        n : array_like (3,) — единичный вектор нормали.
        s : array_like (3,) — единичный вектор направления сдвига в плоскости площадки.

        Возвращает:
        C : float — начальное сцепление.
        """
        n = np.asarray(n, dtype=float)
        s = np.asarray(s, dtype=float)
        # Нормализация на всякий случай
        n = n / np.linalg.norm(n)
        s = s / np.linalg.norm(s)

        v = np.outer(n, s).reshape(9)
        C_sq = v @ self.A_tensor @ v
        C_sq = max(0.0, C_sq)
        return np.sqrt(C_sq)

def load_tensor_from_string(tensor_str):
    lines = tensor_str.strip().split('\n')
    data_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
    matrix = []
    for line in data_lines:
        row = [float(x) for x in line.split()]
        matrix.append(row)
    return np.array(matrix)

