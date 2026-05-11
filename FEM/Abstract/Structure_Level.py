# ====================================
# File: .\FEM\Abstract\Structure_Level.py
# ====================================
from abc import ABC, abstractmethod
import numpy as np


class Node:
    """Узел КЭ-сетки"""

    def __init__(self, id, coords, ndof=None):
        self.id = id
        self.coords = np.array(coords, dtype=float)
        self.ndof = ndof if ndof is not None else len(self.coords)
        self.dofs = []
        self.displacements = np.zeros(self.ndof)


class BoundaryCondition:
    """Кинематическое граничное условие (закрепление / заданное перемещение)"""

    def __init__(self, node, dof_axis, value=0.0):
        self.node = node
        self.dof_axis = dof_axis  # 0 для X, 1 для Y, 2 для Z
        self.value = value


class NodalLoad:
    """Узловая сила"""

    def __init__(self, node, dof_axis, value):
        self.node = node
        self.dof_axis = dof_axis  # 0 для X, 1 для Y, 2 для Z
        self.value = value


class FEModel:
    """Глобальная модель, хранящая списки узлов, элементов, материалов и ГУ"""

    def __init__(self):
        self.nodes = []
        self.elements = []
        self.materials = []

        # Списки для граничных условий и нагрузок
        self.bcs = []
        self.nodal_loads = []

        self.K_global = None
        self.F_global = None
        self.total_dofs = 0

    def initialize(self):
        """Нумерация степеней свободы и выделение памяти"""
        current_eq_num = 0
        for node in self.nodes:
            node.dofs = list(range(current_eq_num, current_eq_num + node.ndof))
            current_eq_num += node.ndof

        self.total_dofs = current_eq_num
        self.K_global = np.zeros((self.total_dofs, self.total_dofs))
        self.F_global = np.zeros(self.total_dofs)

    def assemble(self):
        """Сборка глобальной матрицы жесткости"""
        for element in self.elements:
            K_e = element.compute_stiffness()
            global_indices = []
            for node in element.nodes:
                global_indices.extend(node.dofs)
            self.K_global[np.ix_(global_indices, global_indices)] += K_e

    # --- Новые вспомогательные методы ---
    def add_bc(self, node, dof_axis, value=0.0):
        self.bcs.append(BoundaryCondition(node, dof_axis, value))

    def add_load(self, node, dof_axis, value):
        self.nodal_loads.append(NodalLoad(node, dof_axis, value))


class Control(ABC):
    """Базовый класс для алгоритмов расчета"""

    def __init__(self, model: FEModel):
        self.model = model

    @abstractmethod
    def solve(self):
        pass
