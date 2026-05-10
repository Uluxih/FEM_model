from abc import ABC, abstractmethod
import numpy as np

class Shape(ABC):
    """Отвечает за геометрию и интерполяцию поля (функции формы, Якобиан)"""

    @abstractmethod
    def get_shape_functions(self, xi, eta):
        pass

    @abstractmethod
    def get_jacobian(self, xi, eta, node_coords):
        """Возвращает матрицу Якоби и ее определитель |J|"""
        pass

    @abstractmethod
    def get_shape_derivatives_cartesian(self, xi, eta, node_coords):
        """Возвращает производные функций формы по x и y"""
        pass


class AnalysisModel(ABC):
    """Отвечает за дифференциальное уравнение (Truss, Plane Stress, 3D Solid)"""

    @abstractmethod
    def get_B_matrix(self, shape_derivatives):
        """Собирает матрицу связи деформаций и перемещений B"""
        pass

    @abstractmethod
    def get_h_coefficient(self):
        """Возвращает коэффициент объема h (например, толщину для 2D)"""
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
        ndof = len(self.nodes) * 2  # Для 2D задач (u, v)
        K = np.zeros((ndof, ndof))

        for ip in self.integration_points:
            # 1. Геометрия (Shape)
            detJ = self.shape.get_jacobian(ip.xi, ip.eta, node_coords)[1]
            dN_dx = self.shape.get_shape_derivatives_cartesian(ip.xi, ip.eta, node_coords)

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
        """Возвращает объект геометрии (например, Quadrilateral4Node)"""
        pass

    @abstractmethod
    def _get_analysis_model(self, **kwargs) -> 'AnalysisModel':
        """Возвращает объект дифференциальной модели (например, PlaneStress2DModel)"""
        pass

    @abstractmethod
    def _create_integration_points(self, material, **kwargs) -> list:
        """
        Генерирует список точек интегрирования.
        Именно здесь создаются экземпляры ConstitutiveModel для каждой точки!
        """
        pass
