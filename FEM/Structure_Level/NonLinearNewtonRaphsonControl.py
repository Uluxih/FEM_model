import numpy as np
from FEM.Abstract.Structure_Level import Control


class NonLinearNewtonRaphsonControl(Control):
    def __init__(self, model, num_steps=10, tol=1e-4, max_iter=20):
        super().__init__(model)
        self.num_steps = num_steps
        self.tol = tol
        self.max_iter = max_iter

    def solve(self):
        print("Инициализация модели...")
        self.model.initialize()

        total_dofs = self.model.total_dofs
        U_global = np.zeros(total_dofs)

        F_ext_ref = np.zeros(total_dofs)
        for load in self.model.nodal_loads:
            F_ext_ref[load.node.dofs[load.dof_axis]] += load.value

        print(f"Начало нелинейного расчета: {self.num_steps} шагов.")

        for step in range(1, self.num_steps + 1):
            load_factor = step / self.num_steps
            F_ext_current = F_ext_ref * load_factor

            print(f"\n--- Шаг нагрузки {step}/{self.num_steps} (Load Factor = {load_factor:.2f}) ---")

            for iteration in range(self.max_iter):
                K_t = np.zeros((total_dofs, total_dofs))
                F_int = np.zeros(total_dofs)

                for element in self.model.elements:
                    el_dofs = []
                    for node in element.nodes:
                        el_dofs.extend(node.dofs)
                    U_el = U_global[el_dofs]

                    K_e, F_int_e = self._compute_element_nonlinear(element, U_el)

                    K_t[np.ix_(el_dofs, el_dofs)] += K_e
                    F_int[el_dofs] += F_int_e

                Residual = F_ext_current - F_int

                for bc in self.model.bcs:
                    dof = bc.node.dofs[bc.dof_axis]
                    val = bc.value * load_factor if iteration == 0 else 0.0

                    K_t[dof, :] = 0.0
                    K_t[:, dof] = 0.0
                    K_t[dof, dof] = 1.0
                    Residual[dof] = val

                free_dofs = [i for i in range(total_dofs) if K_t[i, i] != 1.0 or Residual[i] != 0.0]
                error = np.linalg.norm(Residual[free_dofs]) if len(free_dofs) > 0 else 0.0

                print(f"  Итерация {iteration}: Ошибка = {error:.6e}")

                if np.isnan(error) or np.isinf(error):
                    print("КРИТИЧЕСКАЯ ОШИБКА: Решение разошлось (NaN/Inf).")
                    return

                if error < self.tol and iteration > 0:
                    print("  Сходимость достигнута!")
                    self._commit_state()
                    break

                dU = np.linalg.solve(K_t, Residual)
                U_global += dU

            else:
                print("ВНИМАНИЕ: Сходимость не достигнута за максимальное число итераций! Сохраняем как есть.")
                self._commit_state()

        for node in self.model.nodes:
            node.displacements = U_global[node.dofs]
        print("\nРасчет успешно завершен!")

    def _commit_state(self):
        """Фиксирует историю деформаций во всех точках Гаусса"""
        for element in self.model.elements:
            for ip in element.integration_points:
                if hasattr(ip.constitutive_model, 'commit'):
                    ip.constitutive_model.commit()

    def _compute_element_nonlinear(self, element, U_el):
        ndof = len(U_el)
        K_e = np.zeros((ndof, ndof))
        F_int_e = np.zeros(ndof)
        node_coords = np.array([node.coords for node in element.nodes])

        for ip in element.integration_points:
            _, detJ = element.shape.get_jacobian(ip.coords, node_coords)
            dN_dx = element.shape.get_shape_derivatives_cartesian(ip.coords, node_coords)
            B = element.analysis_model.get_B_matrix(dN_dx)
            h = element.analysis_model.get_h_coefficient()
            dV = detJ * h * ip.weight

            # Передаем ПОЛНУЮ текущую деформацию узлов.
            # Модель сама найдет разницу с сохраненным шагом.
            current_strain = B @ U_el
            stress, D_ep = ip.constitutive_model.update_state(current_strain)

            K_e += B.T @ D_ep @ B * dV
            F_int_e += B.T @ stress * dV

        return K_e, F_int_e
