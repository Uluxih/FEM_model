import numpy as np
from FEM.Abstract.Element_Level import Shape, AnalysisModel, ElementFactory
from FEM.Abstract.Integration_Point_Level import IntegrationPoint

# Предполагается наличие 2D модели анализа по аналогии с Solid3DModel
# Если класс называется иначе (например, PlaneStress2DModel), замените импорт и вызов фабрики
from FEM.Element_Level.Solid2DModel import Solid2DModel


class Shape4NodeQuadrilateral(Shape):
    """Геометрия 4-узлового плоского изопараметрического элемента (QUAD4)"""

    def __init__(self):
        # Локальные координаты узлов в стандартном квадрате [-1, 1]
        # Порядок обхода: против часовой стрелки, начиная с нижнего левого угла
        # Узел 1: (-1, -1)
        # Узел 2: ( 1, -1)
        # Узел 3: ( 1,  1)
        # Узел 4: (-1,  1)
        self.xi_i = np.array([-1, 1, 1, -1])
        self.eta_i = np.array([-1, -1, 1, 1])

    def get_shape_functions(self, local_coords):
        xi, eta = local_coords
        # N_i = 1/4 * (1 + xi*xi_i) * (1 + eta*eta_i)
        N = 0.25 * (1 + xi * self.xi_i) * (1 + eta * self.eta_i)
        return N

    def _get_local_derivatives(self, local_coords):
        """Производные dN/dxi, dN/deta"""
        xi, eta = local_coords
        dN_dxi = 0.25 * self.xi_i * (1 + eta * self.eta_i)
        dN_deta = 0.25 * self.eta_i * (1 + xi * self.xi_i)
        return np.array([dN_dxi, dN_deta])  # Размерность (2, 4)

    def get_jacobian(self, local_coords, node_coords):
        local_derivs = self._get_local_derivatives(local_coords)
        # Матрица Якоби: J = [dN/dxi, dN/deta] * [X, Y]
        # node_coords имеет размерность (4, 2)
        J = local_derivs @ node_coords  # (2, 4) @ (4, 2) -> (2, 2)
        detJ = np.linalg.det(J)
        return J, detJ

    def get_shape_derivatives_cartesian(self, local_coords, node_coords):
        local_derivs = self._get_local_derivatives(local_coords)
        J, _ = self.get_jacobian(local_coords, node_coords)
        J_inv = np.linalg.inv(J)

        # [dN/dx, dN/dy]^T = J^-1 * [dN/dxi, dN/deta]^T
        dN_dx = J_inv @ local_derivs  # (2, 2) @ (2, 4) -> (2, 4)
        return dN_dx


# ==========================================
# ФАБРИКА ЭЛЕМЕНТОВ (FACTORY)
# ==========================================
class QUAD4Factory(ElementFactory):
    """Фабрика для сборки 2D элементов QUAD4"""

    def _get_shape(self) -> Shape:
        return Shape4NodeQuadrilateral()

    def _get_analysis_model(self, **kwargs) -> AnalysisModel:
        # Укажите здесь вашу актуальную 2D модель (Solid2DModel, PlaneStress2DModel и т.д.)
        return Solid2DModel()



    def _create_integration_points(self, material, **kwargs) -> list:
        """
        Создает 4 точки интегрирования по правилу Гаусса (2x2).
        Ожидает в kwargs параметр `constitutive_class` (класс физической модели).
        """
        # Правило Гаусса для 2 точек
        g = 1.0 / np.sqrt(3.0)
        gauss_coords = [-g, g]
        gauss_weight = 1.0  # Вес каждой точки в 1D равен 1.0

        # Получаем класс физической модели из аргументов
        constitutive_class = kwargs.get('constitutive_class')
        if constitutive_class is None:
            raise ValueError("Для создания элемента необходимо передать 'constitutive_class' в kwargs")

        integration_points = []

        # Двойной цикл для 2D
        for xi in gauss_coords:
            for eta in gauss_coords:
                # В 2D вес: W = w_xi * w_eta = 1.0 * 1.0 = 1.0
                weight = gauss_weight * gauss_weight

                # Создаем индивидуальный экземпляр модели материала для этой точки
                c_model = constitutive_class(material)

                ip = IntegrationPoint(
                    coords=[xi, eta],
                    weight=weight,
                    constitutive_model=c_model
                )
                integration_points.append(ip)

        return integration_points
