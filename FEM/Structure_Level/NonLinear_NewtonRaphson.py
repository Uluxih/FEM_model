import numpy as np
from Abstract.Element_Level import *
from Abstract.Structure_Level import *
from Abstract.Integration_Point_Level import *


class NewtonRaphsonControl(Control):
    def __init__(self, model: FEModel, tolerance=1e-6, max_iter=20):
        super().__init__(model)
        self.tolerance = tolerance
        self.max_iter = max_iter

    def solve(self):
        self.model.initialize()

        # Вектор внешних сил F_ext (постоянный для данного шага нагружения)
        F_ext = self.model.assemble_external_forces()

        for iteration in range(self.max_iter):
            # 1. Сборка вектора внутренних сил F_int(U) и касательной матрицы K_T(U)
            F_int = self.model.assemble_internal_forces()
            K_T = self.model.assemble_tangent_stiffness()

            # 2. Вычисление вектора невязки (Residual)
            R = F_ext - F_int

            # Учет граничных условий в K_T и R...
            self.apply_bcs(K_T, R)

            # 3. Проверка сходимости
            if np.linalg.norm(R) < self.tolerance:
                print(f"Сходимость достигнута за {iteration} итераций!")
                # Обновляем внутренние переменные материала (например, пластичность)
                self.model.commit_material_state()
                break

            # 4. Решение системы для приращений перемещений
            delta_U = np.linalg.solve(K_T, R)

            # 5. Обновление перемещений
            self.model.update_displacements(delta_U)
