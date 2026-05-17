import numpy as np
from FEM.Abstract.Structure_Level import Control


class NonLinearNewtonRaphsonControl(Control):
    """
    Нелинейный решатель методом Ньютона-Рафсона (Newton-Raphson).
    Поддерживает как силовое, так и кинематическое (заданные перемещения) нагружение.
    """

    def __init__(self, model, num_steps=10, tol=1e-4, max_iter=50):
        super().__init__(model)
        self.num_steps = num_steps
        self.tol = tol
        self.max_iter = max_iter

    def solve(self):
        print("Инициализация модели...")
        self.model.initialize()

        total_dofs = self.model.total_dofs

        # Глобальный вектор полных перемещений делаем атрибутом класса
        self.U_global = np.zeros(total_dofs)

        # Собираем базовый вектор внешних сил
        F_ext_ref = np.zeros(total_dofs)
        for load in self.model.nodal_loads:
            F_ext_ref[load.node.dofs[load.dof_axis]] += load.value

        print(f"Начало нелинейного расчета: {self.num_steps} шагов.")

        for step in range(1, self.num_steps + 1):
            load_factor = step / self.num_steps
            F_ext_current = F_ext_ref * load_factor

            print(f"\n--- Шаг нагрузки {step}/{self.num_steps} (Load Factor = {load_factor:.2f}) ---")

            for iteration in range(self.max_iter):
                # 1. Сборка глобальной касательной матрицы жесткости и вектора внутренних сил
                K_t = np.zeros((total_dofs, total_dofs))
                F_int = np.zeros(total_dofs)

                for element in self.model.elements:
                    el_dofs = []
                    for node in element.nodes:
                        el_dofs.extend(node.dofs)

                    # Извлекаем текущие полные перемещения узлов элемента
                    U_el = self.U_global[el_dofs]

                    # Вычисляем матрицу и внутренние силы элемента
                    K_e, F_int_e = self._compute_element_nonlinear(element, U_el)

                    # Ассемблирование
                    K_t[np.ix_(el_dofs, el_dofs)] += K_e
                    F_int[el_dofs] += F_int_e

                # 2. Вычисление вектора невязки (Residual)
                Residual = F_ext_current - F_int

                # 3. Применение граничных условий
                free_dofs = np.ones(total_dofs, dtype=bool)

                # Сначала корректируем правую часть для свободных узлов
                for bc in self.model.bcs:
                    dof = bc.node.dofs[bc.dof_axis]
                    free_dofs[dof] = False  # Отмечаем, что этот DOF закреплен

                    # Приращение заданного перемещения применяется ТОЛЬКО на 0-й итерации шага
                    delta_u = (bc.value / self.num_steps) if iteration == 0 else 0.0

                    if delta_u != 0.0:
                        # Переносим влияние заданного перемещения на остальные узлы
                        Residual -= K_t[:, dof] * delta_u

                # Затем модифицируем матрицу и невязку для закрепленных узлов
                for bc in self.model.bcs:
                    dof = bc.node.dofs[bc.dof_axis]
                    delta_u = (bc.value / self.num_steps) if iteration == 0 else 0.0

                    K_t[dof, :] = 0.0
                    K_t[:, dof] = 0.0
                    K_t[dof, dof] = 1.0
                    Residual[dof] = delta_u

                # 4. Проверка сходимости (ТОЛЬКО по свободным степеням свободы)
                free_dofs_indices = np.where(free_dofs)[0]
                if len(free_dofs_indices) > 0:
                    error = np.linalg.norm(Residual[free_dofs_indices])
                    # Нормируем ошибку относительно текущей внешней нагрузки (если она есть)
                    f_ext_norm = np.linalg.norm(F_ext_current[free_dofs_indices])
                    if f_ext_norm > 1e-6:
                        error_rel = error / f_ext_norm
                    else:
                        error_rel = error
                else:
                    error_rel = 0.0
                    error = 0.0

                print(f"  Итерация {iteration}: Невязка = {error:.6e} (Отн. = {error_rel:.6e})")

                if np.isnan(error) or np.isinf(error):
                    print("КРИТИЧЕСКАЯ ОШИБКА: Решение разошлось (NaN/Inf). Матрица стала сингулярной.")
                    return

                # Критерий выхода (сходимость)
                if error_rel < self.tol and error < self.tol and iteration > 0:
                    print(f"  Сходимость достигнута за {iteration} итераций!")

                    # ОБНОВЛЕНО: Записываем перемещения в узлы сразу после сходимости шага
                    for node in self.model.nodes:
                        node.displacements = self.U_global[node.dofs]

                    self._commit_state()
                    break

                # 5. Решение системы уравнений K_t * dU = Residual
                try:
                    dU = np.linalg.solve(K_t, Residual)
                except np.linalg.LinAlgError:
                    print("ОШИБКА: Матрица жесткости вырождена (Singular matrix).")
                    return

                # 6. Обновление вектора перемещений
                self.U_global += dU

            else:
                print("ВНИМАНИЕ: Сходимость не достигнута за максимальное число итераций!")
                # Даже если не сошлись, фиксируем состояние, чтобы попытаться пройти дальше
                for node in self.model.nodes:
                    node.displacements = self.U_global[node.dofs]
                self._commit_state()

        print("\nРасчет завершен!")

    def _commit_state(self):
        """Фиксирует историю деформаций во всех точках Гаусса после успешного шага"""
        for element in self.model.elements:
            for ip in element.integration_points:
                if hasattr(ip.constitutive_model, 'commit'):
                    ip.constitutive_model.commit()

    def _compute_element_nonlinear(self, element, U_el):
        """Вычисляет матрицу жесткости и вектор внутренних сил для одного элемента"""
        ndof = len(U_el)
        K_e = np.zeros((ndof, ndof))
        F_int_e = np.zeros(ndof)

        node_coords = np.array([node.coords for node in element.nodes])

        for ip in element.integration_points:
            # Геометрия
            _, detJ = element.shape.get_jacobian(ip.coords, node_coords)
            dN_dx = element.shape.get_shape_derivatives_cartesian(ip.coords, node_coords)

            # Кинематика
            B = element.analysis_model.get_B_matrix(dN_dx)
            h = element.analysis_model.get_h_coefficient()
            dV = detJ * h * ip.weight

            # Вычисление полной деформации в точке Гаусса
            current_strain = B @ U_el

            # Запрос напряжений и касательной матрицы у модели материала
            stress, D_ep = ip.constitutive_model.update_state(current_strain)

            # Численное интегрирование
            K_e += B.T @ D_ep @ B * dV
            F_int_e += B.T @ stress * dV

        return K_e, F_int_e
