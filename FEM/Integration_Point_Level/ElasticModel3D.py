from FEM.Abstract.Integration_Point_Level import ConstitutiveModel
import numpy as np

# 1. Простая 3D упругая модель
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

    # --- ДОБАВЛЕНО ДЛЯ СОВМЕСТИМОСТИ С НЕЛИНЕЙНЫМ РЕШАТЕЛЕМ ---
    def update_state(self, current_strain):
        """
        Возвращает текущие напряжения и касательную матрицу жесткости.
        Для линейной упругости D - константа, а stress = D * strain.
        """
        D_ep = self.get_tangent_matrix()
        stress = D_ep @ current_strain
        return stress, D_ep

    def commit(self):
        """
        Фиксация внутренних переменных состояния.
        Для упругости история не нужна, поэтому просто pass.
        """
        pass
