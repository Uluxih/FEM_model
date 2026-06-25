import numpy as np
from FEM.Abstract.Structure_Level import Control
import scipy.sparse.linalg as spla
import scipy.sparse as sp


class DisplacementControlNR2D(Control):
    """
    Решатель на основе метода непрямого контроля перемещений (Batoz-Dhatt).
    Позволяет стабильно проходить ниспадающие ветви (softening) диаграммы деформирования
    с использованием точной касательной матрицы жесткости, даже при потере положительной определенности.
    """

    def __init__(self, model, control_node, control_dof, target_displacements, track_nodes=None, max_iter=300, tol=1e-3):
        super().__init__(model)
        self.control_node = control_node          # Узел, перемещением которого мы управляем
        self.control_dof = control_dof            # Ось перемещения (0 - X, 1 - Y)
        self.target_displacements = target_displacements # Список целевых перемещений для каждого шага
        self.num_steps = len(target_displacements)
        self.max_iter = max_iter
        self.tol = tol
        self.track_nodes = track_nodes if track_nodes else []

        self.history_U = []
        self.history_F = []
        self.history_lambda = [] # История множителя нагрузки

        self.PENALTY = 1e15
        self.free_dofs = None

    def _precompute_topology_and_kinematics(self):
        total_dofs = self.model.total_dofs

        self.el_dofs_map = []
        self.B_map = []
        self.dV_map = []
        self.is_discrete_map = []

        num_k_entries = sum((len(el.nodes) * 2) ** 2 for el in self.model.elements) + len(self.model.bcs)

        self.I_idx = np.zeros(num_k_entries, dtype=int)
        self.J_idx = np.zeros(num_k_entries, dtype=int)

        ptr = 0
        for e_idx, element in enumerate(self.model.elements):
            el_dofs = []
            for node in element.nodes:
                el_dofs.extend(node.dofs)
            self.el_dofs_map.append(np.array(el_dofs, dtype=int))
            ndof_e = len(el_dofs)

            if getattr(element, 'is_spring', False):
                self.is_discrete_map.append(True)
                self.B_map.append([])
                self.dV_map.append([])
            else:
                self.is_discrete_map.append(False)
                B_el, dV_el = [], []
                node_coords = np.array([node.coords for node in element.nodes])
                for ip in element.integration_points:
                    _, detJ = element.shape.get_jacobian(ip.coords, node_coords)
                    dN_dx = element.shape.get_shape_derivatives_cartesian(ip.coords, node_coords)
                    B = element.analysis_model.get_B_matrix(dN_dx)
                    h = element.analysis_model.get_h_coefficient()
                    dV = detJ * h * ip.weight
                    B_el.append(B)
                    dV_el.append(dV)
                self.B_map.append(B_el)
                self.dV_map.append(dV_el)

            grid = np.meshgrid(el_dofs, el_dofs, indexing='ij')
            self.I_idx[ptr:ptr + ndof_e ** 2] = grid[0].ravel()
            self.J_idx[ptr:ptr + ndof_e ** 2] = grid[1].ravel()
            ptr += ndof_e ** 2

        self.free_dofs = np.ones(total_dofs, dtype=bool)
        for bc in self.model.bcs:
            dof = bc.node.dofs[bc.dof_axis]
            self.I_idx[ptr] = dof
            self.J_idx[ptr] = dof
            self.free_dofs[dof] = False
            ptr += 1

    def solve(self):
        print("\nИнициализация модели и предрасчет кинематики...")
        self.model.initialize()
        self._precompute_topology_and_kinematics()

        total_dofs = self.model.total_dofs
        U_global = np.zeros(total_dofs)
        V_data = np.zeros(len(self.I_idx))

        # 1. Формируем базовый вектор внешней нагрузки (P_ref)
        # В этом методе узловые нагрузки задают только "форму" распределения усилий
        P_ref = np.zeros(total_dofs)
        for load in self.model.nodal_loads:
            dof = load.node.dofs[load.dof_axis]
            P_ref[dof] += load.value

        norm_pref = np.linalg.norm(P_ref[self.free_dofs])
        if norm_pref < 1e-12:
            raise ValueError("Для метода контроля перемещений необходимо задать базовую нагрузку (nodal_loads)!")

        lambda_factor = 0.0 # Текущий множитель нагрузки
        c_idx = self.control_node.dofs[self.control_dof]

        for step, target_u in enumerate(self.target_displacements, 1):
            print(f"\n=== Шаг {step}/{self.num_steps} | Целевое перемещение: {target_u:.5f} ===")

            # Приращение перемещения, которое нужно достичь на этом шаге
            delta_u_target = target_u - U_global[c_idx]

            for iteration in range(self.max_iter):
                V_data.fill(0.0)
                F_int = np.zeros(total_dofs)
                ptr = 0

                # Сборка K_t и F_int
                for e_idx, element in enumerate(self.model.elements):
                    el_dofs = self.el_dofs_map[e_idx]
                    U_el = U_global[el_dofs]
                    ndof_e = len(el_dofs)

                    if self.is_discrete_map[e_idx]:
                        K_e = element.compute_stiffness()
                        F_int_e = element.compute_internal_force(U_el)
                    else:
                        K_e = np.zeros((ndof_e, ndof_e))
                        F_int_e = np.zeros(ndof_e)
                        for ip_idx, ip in enumerate(element.integration_points):
                            B = self.B_map[e_idx][ip_idx]
                            dV = self.dV_map[e_idx][ip_idx]

                            current_strain = B @ U_el
                            stress, D_ep = ip.constitutive_model.update_state(current_strain)

                            K_e += B.T @ D_ep @ B * dV
                            F_int_e += B.T @ stress * dV

                    V_data[ptr:ptr + ndof_e ** 2] = K_e.ravel()
                    ptr += ndof_e ** 2
                    F_int[el_dofs] += F_int_e

                # Вектор невязки
                Residual = lambda_factor * P_ref - F_int
                P_ref_mod = P_ref.copy()

                # Применение граничных условий (опор)
                for bc in self.model.bcs:
                    dof = bc.node.dofs[bc.dof_axis]
                    # В этом методе опоры обычно фиксированы (bc.value = 0 или const)
                    delta_u_bc = bc.value - U_global[dof]

                    V_data[ptr] = self.PENALTY
                    Residual[dof] = self.PENALTY * delta_u_bc
                    P_ref_mod[dof] = 0.0  # Опоры не несут базовой нагрузки в P_ref
                    ptr += 1

                # Проверка сходимости
                norm_res = np.linalg.norm(Residual[self.free_dofs])
                norm_fint = np.linalg.norm(F_int[self.free_dofs])
                F_ref_val = max(norm_pref * abs(lambda_factor), norm_fint)
                if F_ref_val < 1e-6: F_ref_val = 1.0
                error = norm_res / F_ref_val

                if error < self.tol and iteration > 0:
                    print(f"  -> Сходимость за {iteration} итераций. (Ошибка: {error:.2e}, Множитель нагрузки: {lambda_factor:.4f})")
                    break

                # Решение СЛАУ
                K_t_sparse = sp.coo_matrix((V_data, (self.I_idx, self.J_idx)), shape=(total_dofs, total_dofs)).tocsr()

                try:
                    # Решаем ДВЕ системы: одну для базовой нагрузки, вторую для невязки
                    dU_I = spla.spsolve(K_t_sparse, P_ref_mod)
                    dU_II = spla.spsolve(K_t_sparse, Residual)
                except RuntimeError:
                    # Срабатывает только если матрица стала идеально сингулярной (ровно на пике)
                    reg_matrix = sp.eye(total_dofs) * 1e-8
                    dU_I = spla.spsolve(K_t_sparse + reg_matrix, P_ref_mod)
                    dU_II = spla.spsolve(K_t_sparse + reg_matrix, Residual)

                # Вычисление приращения множителя нагрузки (формулы Batoz-Dhatt)
                u_I_c = dU_I[c_idx]
                u_II_c = dU_II[c_idx]

                # Защита от деления на ноль (на случай snap-back поведения)
                if abs(u_I_c) < 1e-12:
                    u_I_c = 1e-12 if u_I_c >= 0 else -1e-12

                if iteration == 0:
                    # На первой итерации шага заставляем узел сместиться на нужную дельту
                    d_lambda = (delta_u_target - u_II_c) / u_I_c
                else:
                    # На последующих итерациях запрещаем узлу смещаться (он уже на нужном месте)
                    d_lambda = - u_II_c / u_I_c

                # Обновление глобальных векторов
                lambda_factor += d_lambda
                dU = d_lambda * dU_I + dU_II
                U_global += dU

            else:
                print(f"  !!! ВНИМАНИЕ: Сходимость не достигнута за {self.max_iter} итераций !!! (Ошибка: {error:.2e})")

            # Фиксация состояния (Commit)
            for element in self.model.elements:
                if not getattr(element, 'is_spring', False):
                    for ip in element.integration_points:
                        ip.constitutive_model.commit()

            for node in self.model.nodes:
                node.displacements = U_global[node.dofs]

            # Запись истории
            if self.track_nodes:
                rx_force = sum(F_int[n.dofs[self.control_dof]] for n in self.track_nodes)
                self.history_U.append(U_global[c_idx])
                self.history_F.append(rx_force)
            self.history_lambda.append(lambda_factor)