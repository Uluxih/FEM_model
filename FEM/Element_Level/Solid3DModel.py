import numpy as np
from FEM.Abstract.Element_Level import Shape, AnalysisModel, ElementFactory
from FEM.Abstract.Integration_Point_Level import IntegrationPoint
from FEM.Element_Level.Shape8NodeHexahedron import *


class Solid3DModel(AnalysisModel):
    """Модель сплошной 3D среды (6 компонент деформации, 3 перемещения на узел)"""

    def get_B_matrix(self, shape_derivatives):
        """
        Собирает матрицу B (размер 6 x 24).
        Порядок деформаций: [xx, yy, zz, xy, yz, xz]
        """
        num_nodes = shape_derivatives.shape[1]  # 8 узлов
        B = np.zeros((6, num_nodes * 3))

        for i in range(num_nodes):
            dN_dx = shape_derivatives[0, i]
            dN_dy = shape_derivatives[1, i]
            dN_dz = shape_derivatives[2, i]

            # Индексы столбцов для текущего узла (u, v, w)
            col_x = 3 * i
            col_y = 3 * i + 1
            col_z = 3 * i + 2

            # Нормальные деформации
            B[0, col_x] = dN_dx
            B[1, col_y] = dN_dy
            B[2, col_z] = dN_dz

            # Сдвиговые деформации (инженерные гамма)
            B[3, col_x] = dN_dy
            B[3, col_y] = dN_dx

            B[4, col_y] = dN_dz
            B[4, col_z] = dN_dy

            B[5, col_x] = dN_dz
            B[5, col_z] = dN_dx

        return B

    def get_h_coefficient(self):
        # В 3D интегрирование идет по объему (dV = detJ * dxi * deta * dzeta),
        # поэтому дополнительная толщина не нужна.
        return 1.0


