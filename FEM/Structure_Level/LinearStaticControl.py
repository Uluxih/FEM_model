# ====================================
# File: .\FEM\Structure_Level\LinearStaticControl.py
# ====================================
import numpy as np
from FEM.Abstract.Structure_Level import Control


class LinearStaticControl(Control):
    """Линейный статический расчет"""

    def _apply_bcs(self):
        """Применяет нагрузки и кинематические граничные условия к глобальным матрицам"""

        # 1. Применяем узловые силы к вектору F_global
        for load in self.model.nodal_loads:
            global_dof = load.node.dofs[load.dof_axis]
            self.model.F_global[global_dof] += load.value

        # 2. Корректируем правую часть (вектор F) для всех кинематических ГУ
        # Это необходимо, если заданы ненулевые перемещения (осадка опор и т.д.)
        for bc in self.model.bcs:
            global_dof = bc.node.dofs[bc.dof_axis]
            prescribed_value = bc.value
            if prescribed_value != 0.0:
                # F = F - K_col * U_prescribed
                self.model.F_global -= self.model.K_global[:, global_dof] * prescribed_value

        # 3. Модифицируем матрицу K_global (зануляем строки/столбцы, ставим 1 на диагональ)
        for bc in self.model.bcs:
            global_dof = bc.node.dofs[bc.dof_axis]
            prescribed_value = bc.value

            # Зануляем строку и столбец
            self.model.K_global[global_dof, :] = 0.0
            self.model.K_global[:, global_dof] = 0.0

            # Ставим 1.0 на главную диагональ и заданное значение в вектор сил
            self.model.K_global[global_dof, global_dof] = 1.0
            self.model.F_global[global_dof] = prescribed_value

    def solve(self):
        print("Инициализация модели...")
        self.model.initialize()

        print("Сборка глобальной матрицы жесткости...")
        self.model.assemble()

        print("Применение граничных условий и нагрузок...")
        self._apply_bcs()

        print("Решение СЛАУ...")
        # Решаем систему уравнений [K] {U} = {F}
        U = np.linalg.solve(self.model.K_global, self.model.F_global)

        print("Обновление перемещений в узлах...")
        for node in self.model.nodes:
            # Извлекаем перемещения для конкретного узла по его глобальным dofs
            node.displacements = U[node.dofs]

        print("Расчет успешно завершен!")
