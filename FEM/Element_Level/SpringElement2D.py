import numpy as np
from FEM.Abstract.Element_Level import Element

class SpringElement2D(Element):
    """
    Специальный 2D пружинный элемент для стабилизации ниспадающей ветви (уравнение 71).
    """

    def __init__(self, nodes, kx, ky):
        """
        :param nodes: Список из ровно 2-х узлов [NodeA, NodeB]
        :param kx: Жесткость вдоль оси X
        :param ky: Жесткость вдоль оси Y
        """
        # Инициализируем базовый класс пустыми значениями для континуальной физики
        super().__init__(nodes=nodes, shape=None, analysis_model=None, integration_points=[])
        self.kx = kx
        self.ky = ky
        self.is_spring = True  # Флаг для быстрой идентификации в решателе

    def compute_stiffness(self):
        """
        Аналитическая матрица жесткости K_spr (Размер 4x4 для 2D).
        Порядок dof: [u1_x, u1_y, u2_x, u2_y]
        """
        K = np.array([
            [ self.kx,  0.0,     -self.kx,  0.0    ],
            [ 0.0,      self.ky,  0.0,     -self.ky],
            [-self.kx,  0.0,      self.kx,  0.0    ],
            [ 0.0,     -self.ky,  0.0,      self.ky]
        ], dtype=float)
        return K

    def compute_internal_force(self, U_el):
        """
        Вектор внутренних сил элемента F_int = K_spr * U_el
        """
        K = self.compute_stiffness()
        return K @ U_el
