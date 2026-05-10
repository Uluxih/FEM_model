from Abstract.Element_Level import *
from Abstract.Structure_Level import *
from Abstract.Integration_Point_Level import *

class LinearElasticPlaneStress(ConstitutiveModel):
    """Пример: Линейно-упругая модель для плоского напряженного состояния"""

    def get_tangent_matrix(self):
        E, nu = self.material.E, self.material.nu
        factor = E / (1 - nu ** 2)
        return factor * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1 - nu) / 2]
        ])

    def get_stress(self, strain):
        D = self.get_tangent_matrix()
        return D @ strain