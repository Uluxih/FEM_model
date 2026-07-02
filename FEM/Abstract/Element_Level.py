from abc import ABC, abstractmethod
import numpy as np


class Shape(ABC):
    """Отвечает за геометрию и интерполяцию поля (функции формы, Якобиан)"""

    @abstractmethod
    def get_shape_functions(self, local_coords):
        """
        local_coords: массив или список локальных координат.
        Для 1D: [xi], для 2D: [xi, eta], для 3D: [xi, eta, zeta]
        """
        pass

    @abstractmethod
    def get_jacobian(self, local_coords, node_coords):
        """Возвращает кортеж (Матрица Якоби J, определитель |J|)"""
        pass

    @abstractmethod
    def get_shape_derivatives_cartesian(self, local_coords, node_coords):
        """Возвращает производные функций формы по глобальным осям (x, y, z)"""
        pass


class AnalysisModel(ABC):
    """Отвечает за дифференциальное уравнение (Truss, Plane Stress, 3D Solid и т.д.)"""

    @abstractmethod
    def get_B_matrix(self, shape_derivatives):
        """Собирает матрицу связи деформаций и перемещений B"""
        pass

    @abstractmethod
    def get_h_coefficient(self):
        """Возвращает коэффициент объема h (например, толщину t для 2D, 1.0 для 3D)"""
        pass


class Element(ABC):
    """Базовый класс конечного элемента"""

    def __init__(self, nodes, shape: Shape, analysis_model: AnalysisModel, integration_points: list):
        self.nodes = nodes
        self.shape = shape
        self.analysis_model = analysis_model
        self.integration_points = integration_points

    def compute_stiffness(self):
        """
        Вычисление локальной матрицы жесткости.
        """
        # Получаем координаты узлов элемента
        node_coords = np.array([node.coords for node in self.nodes])

        # Универсальное вычисление количества степеней свободы (ndof)
        # Работает для любых элементов (2D, 3D, балки с 6 dof на узел и т.д.)
        ndof = sum(node.ndof for node in self.nodes)

        K = np.zeros((ndof, ndof))

        for ip in self.integration_points:
            # 1. Геометрия (Shape)
            # Передаем универсальные координаты точки интегрирования (ip.coords)
            _, detJ = self.shape.get_jacobian(ip.coords, node_coords)
            dN_dx = self.shape.get_shape_derivatives_cartesian(ip.coords, node_coords)

            # 2. Модель анализа (Analysis Model)
            B = self.analysis_model.get_B_matrix(dN_dx)
            h = self.analysis_model.get_h_coefficient()

            # 3. Физика (Constitutive Model из точки интегрирования)
            D = ip.constitutive_model.get_tangent_matrix()

            # 4. Численное интегрирование
            # K += B^T * D * B * |J| * h * w
            K += B.T @ D @ B * detJ * h * ip.weight

        return K

class ElementFactory(ABC):
    """
    Абстрактная фабрика конечных элементов.
    Скрывает логику связывания геометрии, математической модели и точек интегрирования.
    """

    def create_element(self, nodes, material, **kwargs):
        """
        Шаблонный метод (Template Method).
        Определяет строгий алгоритм создания элемента, но делегирует
        конкретные шаги абстрактным методам.
        """
        # 1. Запрашиваем геометрию (Shape)
        shape = self._get_shape()

        # 2. Запрашиваем модель анализа (Analysis Model)
        analysis_model = self._get_analysis_model(**kwargs)

        # 3. Генерируем точки интегрирования (каждая со своей копией физической модели)
        integration_points = self._create_integration_points(material, **kwargs)

        # 4. Собираем элемент
        return Element(
            nodes=nodes,
            shape=shape,
            analysis_model=analysis_model,
            integration_points=integration_points
        )

    @abstractmethod
    def _get_shape(self) -> 'Shape':
        """Возвращает объект геометрии (например, Quadrilateral4Node или Hexahedron8Node)"""
        pass

    @abstractmethod
    def _get_analysis_model(self, **kwargs) -> 'AnalysisModel':
        """Возвращает объект дифференциальной модели (например, PlaneStress2DModel или Solid3DModel)"""
        pass

    @abstractmethod
    def _create_integration_points(self, material, **kwargs) -> list:
        """
        Генерирует список точек интегрирования.
        Именно здесь создаются экземпляры ConstitutiveModel для каждой точки!
        """
        pass
