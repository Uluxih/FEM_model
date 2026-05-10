import numpy as np
from Abstract.Element_Level import *
from Abstract.Structure_Level import *
from Abstract.Integration_Point_Level import *

class LinearStaticControl(Control):
    """Линейный статический расчет (наследник Control -> Equilibrium Path)"""

    def solve(self):
        print("Инициализация модели...")
        self.model.initialize()

        print("Сборка глобальных матриц...")
        self.model.assemble()

        print("Учет граничных условий и решение СЛАУ...")
        # Применение закреплений
        # U = np.linalg.solve(self.model.K_global, self.model.F_global)

        print("Обновление перемещений в узлах...")
        # self.model.update_displacements(U)
