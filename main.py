from Abstract.Element_Level import *
from Abstract.Structure_Level import *
from Integration_Point_Level import *
from abc import ABC, abstractmethod
import numpy as np
from FEM.Integration_Point_Level.LinElastPlaStr import *
from FEM.Structure_Level.LinearStaticControl import *


class Quadrilateral4Node(Shape):
    """Билинейный 4-узловой четырехугольник (Q4)"""

    def get_shape_functions(self, xi, eta):
        # N1, N2, N3, N4
        return 0.25 * np.array([
            (1 - xi) * (1 - eta),
            (1 + xi) * (1 - eta),
            (1 + xi) * (1 + eta),
            (1 - xi) * (1 + eta)
        ])

    def get_jacobian(self, xi, eta, node_coords):
        # Производные функций формы по локальным координатам (xi, eta)
        dN_dxi_eta = 0.25 * np.array([
            [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
            [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]
        ])
        # Якобиан J = dN/dxi_eta * Координаты
        J = dN_dxi_eta @ node_coords
        detJ = np.linalg.det(J)
        return J, detJ, dN_dxi_eta

    def get_shape_derivatives_cartesian(self, xi, eta, node_coords):
        J, _, dN_dxi_eta = self.get_jacobian(xi, eta, node_coords)
        J_inv = np.linalg.inv(J)
        # Производные по глобальным координатам (x, y)
        dN_dx_dy = J_inv @ dN_dxi_eta
        return dN_dx_dy


# --- КОНКРЕТНАЯ МОДЕЛЬ АНАЛИЗА (Analysis Model) ---
class PlaneStress2DModel(AnalysisModel):
    """Модель плоского напряженного состояния"""

    def __init__(self, thickness):
        self.thickness = thickness

    def get_B_matrix(self, dN_dx_dy):
        """
        Собирает матрицу градиентов B.
        dN_dx_dy имеет размерность (2, 4): строка 0 - по x, строка 1 - по y
        """
        num_nodes = dN_dx_dy.shape[1]
        B = np.zeros((3, num_nodes * 2))

        for i in range(num_nodes):
            dN_dx = dN_dx_dy[0, i]
            dN_dy = dN_dx_dy[1, i]

            # Матрица B для 2D упругости
            B[0, 2 * i] = dN_dx
            B[1, 2 * i + 1] = dN_dy
            B[2, 2 * i] = dN_dy
            B[2, 2 * i + 1] = dN_dx

        return B

    def get_h_coefficient(self):
        return self.thickness

class Q4PlaneStressFactory(ElementFactory):
    """Конкретная фабрика для создания 2D элементов Q4 (Плоское напряженное состояние)"""

    def _get_shape(self):
        # Возвращаем наш шаблон геометрии
        return Quadrilateral4Node()

    def _get_analysis_model(self, **kwargs):
        # Извлекаем толщину из аргументов (по умолчанию 1.0)
        thickness = kwargs.get('thickness', 1.0)
        return PlaneStress2DModel(thickness=thickness)

    def _create_integration_points(self, material, **kwargs):
        # Проверяем, запросил ли пользователь редуцированное интегрирование
        reduced_integration = kwargs.get('reduced_integration', False)

        integration_points = []

        if reduced_integration:
            # Редуцированное интегрирование: 1 точка Гаусса в центре
            # Создаем уникальную физическую модель для этой точки
            physics = LinearElasticPlaneStress(material)
            # Вес для 1 точки в квадрате [-1, 1]x[-1, 1] равен 4.0
            integration_points.append(IntegrationPoint(0.0, 0.0, 4.0, physics))

        else:
            # Полное интегрирование: 4 точки Гаусса (2x2)
            g = 1.0 / np.sqrt(3.0)
            gauss_coords = [(-g, -g), (g, -g), (g, g), (-g, g)]

            for xi, eta in gauss_coords:
                # ВАЖНО: Для КАЖДОЙ точки создаем СВОЙ экземпляр модели материала.
                # Если в будущем мы добавим пластичность, каждая точка будет
                # независимо хранить свои пластические деформации.
                physics = LinearElasticPlaneStress(material)

                # Вес каждой точки равен 1.0
                integration_points.append(IntegrationPoint(xi, eta, 1.0, physics))

        return integration_points


def main():
    # 1. Задаем узлы и материал
    nodes = [Node(1, [0, 0]), Node(2, [1, 0]), Node(3, [1, 1]), Node(4, [0, 1])]
    steel = Material(E=2e11, nu=0.3)

    # 2. Создаем нужную фабрику
    factory = Q4PlaneStressFactory()

    # 3. Просим фабрику создать элемент
    # Полное интегрирование (по умолчанию)
    element_full = factory.create_element(nodes, material=steel, thickness=0.01)

    # Или редуцированное интегрирование
    element_reduced = factory.create_element(nodes, material=steel, thickness=0.01, reduced_integration=True)

    # 4. Добавляем в модель и решаем...
    model = FEModel()
    model.elements = [element_full]
    model.nodes = nodes
    model.materials = [steel]

    print("7. Запуск алгоритма расчета (Control)...")
    solver = LinearStaticControl(model)
    solver.solve()

    print("\n--- РЕЗУЛЬТАТ ---")
    print("Локальная матрица жесткости элемента (размер 8x8):")
    K_local = element_full.compute_stiffness()

    # Выведем матрицу красиво
    np.set_printoptions(precision=2, linewidth=100)
    print(K_local)

if __name__ == "__main__":
    main()
    # ...
