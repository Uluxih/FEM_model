from FEM.Abstract.Integration_Point_Level import ConstitutiveModel
from FEM.Abstract.Structure_Level import Node
import numpy as np
from FEM.Abstract.Element_Level import Shape, AnalysisModel, ElementFactory
from FEM.Abstract.Integration_Point_Level import IntegrationPoint
from FEM.Element_Level.Solid3DModel import *


# 1. Простая 3D упругая модель для теста
class ElasticModel3D(ConstitutiveModel):
    def get_tangent_matrix(self):
        E = self.material.E
        nu = self.material.nu
        C = E / ((1 + nu) * (1 - 2 * nu))

        D = np.zeros((6, 6))
        D[0:3, 0:3] = C * nu
        D[0, 0] = D[1, 1] = D[2, 2] = C * (1 - nu)

        G = E / (2 * (1 + nu))
        D[3, 3] = D[4, 4] = D[5, 5] = G
        return D

    def get_stress(self, strain):
        return self.get_tangent_matrix() @ strain

