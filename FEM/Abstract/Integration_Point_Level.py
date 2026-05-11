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
    """Универсальная точка интегрирования"""
    def __init__(self, coords, weight, constitutive_model):
        self.coords = coords  # Универсальные координаты: [xi, eta, zeta]
        self.weight = weight
        self.constitutive_model = constitutive_model