import numpy as np
from FEM.Abstract.Element_Level import Shape, AnalysisModel, ElementFactory
from FEM.Abstract.Integration_Point_Level import IntegrationPoint
# Для 2D обычно используется 4-узловой четырехугольник или 3-узловой треугольник
# from FEM.Element_Level.Shape4NodeQuad import *

class Solid2DModel(AnalysisModel):
    """
    Модель сплошной 2D среды (плоская деформация или плоское напряженное состояние).
    3 компоненты деформации, 2 перемещения на узел.
    """

    def __init__(self, thickness=1.0):
        """
        Инициализация 2D модели.
        :param thickness: Толщина элемента (t). Для плоской деформации обычно t=1.0.
        """
        # Если в базовом AnalysisModel есть __init__, вызываем его: super().__init__()
        self.thickness = thickness

    def get_B_matrix(self, shape_derivatives):
        """
        Собирает матрицу B (размер 3 x 2*num_nodes).
        Порядок деформаций: [xx, yy, xy]
        """
        num_nodes = shape_derivatives.shape[1]  # Например, 4 для четырехугольника
        B = np.zeros((3, num_nodes * 2))

        for i in range(num_nodes):
            dN_dx = shape_derivatives[0, i]
            dN_dy = shape_derivatives[1, i]

            # Индексы столбцов для текущего узла (u, v)
            col_x = 2 * i
            col_y = 2 * i + 1

            # Нормальные деформации (xx, yy)
            B[0, col_x] = dN_dx
            B[1, col_y] = dN_dy

            # Сдвиговая деформация (инженерная гамма xy)
            # gamma_xy = du/dy + dv/dx
            B[2, col_x] = dN_dy
            B[2, col_y] = dN_dx

        return B

    def get_h_coefficient(self):
        """
        В 2D интегрирование идет по площади (dA = detJ * dxi * deta),
        поэтому для получения интеграла по объему необходимо умножить на толщину (t).
        """
        return self.thickness
