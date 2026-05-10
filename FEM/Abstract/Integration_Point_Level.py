from abc import ABC, abstractmethod
import numpy as np


class Material(ABC):
    """Базовый класс для хранения свойств материала (E, nu и т.д.)"""

    def __init__(self, E, nu):
        self.E = E
        self.nu = nu


class ConstitutiveModel(ABC):
    """Базовый класс для определяющих соотношений (упругость, пластичность и т.д.)"""

    def __init__(self, material):
        self.material = material

    @abstractmethod
    def get_tangent_matrix(self):
        """Возвращает матрицу D (матрицу упругости/касательной жесткости)"""
        pass

    @abstractmethod
    def get_stress(self, strain):
        """Возвращает вектор напряжений (формула 2 в статье)"""
        pass


class IntegrationPoint:
    """Точка интегрирования (хранит локальные координаты, вес и модель материала)"""

    def __init__(self, xi, eta, weight, constitutive_model):
        self.xi = xi
        self.eta = eta
        self.weight = weight
        self.constitutive_model = constitutive_model  # Ссылка на Constitutive Model
