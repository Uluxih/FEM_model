import numpy as np
from FEM.Abstract.Element_Level import Shape, AnalysisModel, ElementFactory
from FEM.Abstract.Integration_Point_Level import IntegrationPoint
from FEM.Element_Level.Solid3DModel import *

class Shape8NodeHexahedron(Shape):
    """Геометрия 8-узлового объемного изопараметрического элемента (HEX8)"""

    def __init__(self):
        # Локальные координаты узлов в стандартном кубе [-1, 1]
        # Порядок обхода: нижняя грань (z=-1) против часовой, затем верхняя грань (z=1)
        self.xi_i = np.array([-1, 1, 1, -1, -1, 1, 1, -1])
        self.eta_i = np.array([-1, -1, 1, 1, -1, -1, 1, 1])
        self.zeta_i = np.array([-1, -1, -1, -1, 1, 1, 1, 1])

    def get_shape_functions(self, local_coords):
        xi, eta, zeta = local_coords
        # N_i = 1/8 * (1 + xi*xi_i) * (1 + eta*eta_i) * (1 + zeta*zeta_i)
        N = 0.125 * (1 + xi * self.xi_i) * (1 + eta * self.eta_i) * (1 + zeta * self.zeta_i)
        return N

    def _get_local_derivatives(self, local_coords):
        """Производные dN/dxi, dN/deta, dN/dzeta"""
        xi, eta, zeta = local_coords
        dN_dxi = 0.125 * self.xi_i * (1 + eta * self.eta_i) * (1 + zeta * self.zeta_i)
        dN_deta = 0.125 * self.eta_i * (1 + xi * self.xi_i) * (1 + zeta * self.zeta_i)
        dN_dzeta = 0.125 * self.zeta_i * (1 + xi * self.xi_i) * (1 + eta * self.eta_i)
        return np.array([dN_dxi, dN_deta, dN_dzeta])  # Размерность (3, 8)

    def get_jacobian(self, local_coords, node_coords):
        local_derivs = self._get_local_derivatives(local_coords)
        # Матрица Якоби: J = [dN/dxi, dN/deta, dN/dzeta] * [X, Y, Z]
        J = local_derivs @ node_coords  # (3, 8) @ (8, 3) -> (3, 3)
        detJ = np.linalg.det(J)
        return J, detJ

    def get_shape_derivatives_cartesian(self, local_coords, node_coords):
        local_derivs = self._get_local_derivatives(local_coords)
        J, _ = self.get_jacobian(local_coords, node_coords)
        J_inv = np.linalg.inv(J)

        # [dN/dx, dN/dy, dN/dz]^T = J^-1 * [dN/dxi, dN/deta, dN/dzeta]^T
        dN_dx = J_inv @ local_derivs  # (3, 3) @ (3, 8) -> (3, 8)
        return dN_dx

# ==========================================
# 3. ФАБРИКА ЭЛЕМЕНТОВ (FACTORY)
# ==========================================
class HEX8Factory(ElementFactory):
    """Фабрика для сборки 3D элементов HEX8"""

    def _get_shape(self) -> Shape:
        return Shape8NodeHexahedron()

    def _get_analysis_model(self, **kwargs) -> AnalysisModel:
        return Solid3DModel()

    def _create_integration_points(self, material, **kwargs) -> list:
        """
        Создает 8 точек интегрирования по правилу Гаусса (2x2x2).
        Ожидает в kwargs параметр `constitutive_class` (класс физической модели).
        """
        # Правило Гаусса для 2 точек
        g = 1.0 / np.sqrt(3.0)
        gauss_coords = [-g, g]
        gauss_weight = 1.0  # Вес каждой точки в 1D равен 1.0

        # Получаем класс физической модели из аргументов (упругость, пластичность и т.д.)
        constitutive_class = kwargs.get('constitutive_class')
        if constitutive_class is None:
            raise ValueError("Для создания элемента необходимо передать 'constitutive_class' в kwargs")

        integration_points = []

        # Тройной цикл для 3D
        for xi in gauss_coords:
            for eta in gauss_coords:
                for zeta in gauss_coords:
                    # В 3D вес: W = w_xi * w_eta * w_zeta = 1.0 * 1.0 * 1.0 = 1.0
                    weight = gauss_weight * gauss_weight * gauss_weight

                    # Создаем индивидуальный экземпляр модели материала для этой точки
                    # (Крайне важно для нелинейных расчетов, где каждая точка хранит свою историю деформаций)
                    c_model = constitutive_class(material)

                    ip = IntegrationPoint(
                        coords=[xi, eta, zeta],
                        weight=weight,
                        constitutive_model=c_model
                    )
                    integration_points.append(ip)

        return integration_points