from abc import ABC, abstractmethod
import numpy as np

class Node:
    """Узел КЭ-сетки"""

    def __init__(self, id, coords):
        self.id = id
        self.coords = coords
        self.dofs = []  # Глобальные номера степеней свободы
        self.displacements = np.zeros(2)


class FEModel:
    """Глобальная модель, хранящая списки узлов, элементов, материалов"""

    def __init__(self):
        self.nodes = []
        self.elements = []
        self.materials = []
        self.load_elements = []  # Для естественных граничных условий (нагрузок)
        self.K_global = None
        self.F_global = None

    def initialize(self):
        """Нумерация степеней свободы, выделение памяти"""
        total_dofs = len(self.nodes) * 2
        self.K_global = np.zeros((total_dofs, total_dofs))
        self.F_global = np.zeros(total_dofs)
        # В реальном коде здесь задаются индексы степеней свободы для сборки

    def assemble(self):
        """Сборка глобальной матрицы жесткости (consumer запрашивает у supplier)"""
        for element in self.elements:
            # Инкапсуляция: глобальный алгоритм не знает, какой это элемент (Q4, T3 и т.д.)
            # Он просто вызывает generic метод compute_stiffness()
            K_e = element.compute_stiffness()

            # Псевдокод сборки:
            # global_indices = self.get_global_indices(element)
            # self.K_global[np.ix_(global_indices, global_indices)] += K_e


class Control(ABC):
    """Базовый класс для алгоритмов расчета (Linear Static, Nonlinear и т.д.)"""

    def __init__(self, model: FEModel):
        self.model = model

    @abstractmethod
    def solve(self):
        pass

