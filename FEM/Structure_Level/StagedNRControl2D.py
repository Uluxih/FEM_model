import numpy as np
from FEM.Abstract.Structure_Level import Control
import scipy.sparse.linalg as spla
import scipy.sparse as sp
from FEM.Structure_Level.MaterialLogger import MaterialLogger


class StagedNRControl2D(Control):
    """
    Решатель Ньютона-Рафсона с поддержкой стадийного нагружения для 2D задач.
    Поддерживает мульти-элементные сетки (например, сплошные QUAD4 + дискретные SpringElement2D).
    Включает алгоритм Line Search для стабилизации сходимости.
    """


    def __init__(self, model, track_nodes, load_factors, track_dof=1, max_iter=300, tol=1e-3):
        super().__init__(model)
        self.load_factors = load_factors
        self.num_steps = len(load_factors)
        self.max_iter = max_iter
        self.tol = tol
        self.track_nodes = track_nodes
        self.track_dof = track_dof

        self.history_U = []
        self.history_F = []

        self.PENALTY = 1e15
        self.free_dofs = None
        # ДОБАВЛЕНО: Инициализация логгера
        self.mat_logger = MaterialLogger()


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

        target_max_factor = self.load_factors[-1] if isinstance(self.load_factors, (list, np.ndarray)) else 1.0
        current_factor = 0.0

        d_factor = self.load_factors[0] if len(self.load_factors) > 0 else 0.1
        min_d_factor = 1e-6

        step = 1

        while current_factor < target_max_factor:

            if current_factor + d_factor > target_max_factor:
                d_factor = target_max_factor - current_factor

            attempt_factor = current_factor + d_factor
            print(f"\n{'=' * 50}")
            print(
                f"ШАГ {step} | Текущий фактор: {current_factor:.5f} -> Попытка: {attempt_factor:.5f} (Инкремент: {d_factor:.5f})")
            print(f"{'=' * 50}")

            U_prev = U_global.copy()

            F_ext = np.zeros(total_dofs)
            for load in self.model.nodal_loads:
                dof = load.node.dofs[load.dof_axis]
                is_prop = getattr(load, 'is_proportional', True)
                factor = attempt_factor if is_prop else 1.0
                F_ext[dof] += load.value * factor

            norm_fext = np.linalg.norm(F_ext[self.free_dofs])
            F_ref = norm_fext if norm_fext > 1e-12 else 1e-6

            converged = False

            for iteration in range(self.max_iter):
                V_data.fill(0.0)
                F_int = np.zeros(total_dofs)
                ptr = 0

                # 1. Сборка K и F_int (ОБНОВЛЕНИЕ МАТРИЦЫ НА КАЖДОЙ ИТЕРАЦИИ)
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
                            # Важно: constitutive_model должна возвращать актуальную касательную матрицу D_ep
                            stress, D_ep = ip.constitutive_model.update_state(current_strain)

                            K_e += B.T @ D_ep @ B * dV
                            F_int_e += B.T @ stress * dV

                    V_data[ptr:ptr + ndof_e ** 2] = K_e.ravel()
                    ptr += ndof_e ** 2
                    F_int[el_dofs] += F_int_e

                Residual = F_ext - F_int

                # 2. Граничные условия
                for bc_idx, bc in enumerate(self.model.bcs):
                    dof = bc.node.dofs[bc.dof_axis]
                    is_prop = getattr(bc, 'is_proportional', True)
                    target_u = (bc.value * attempt_factor) if is_prop else bc.value
                    current_u = U_global[dof]
                    delta_u = target_u - current_u

                    V_data[ptr] = self.PENALTY
                    Residual[dof] = self.PENALTY * delta_u
                    ptr += 1

                # 3. Проверка сходимости
                norm_res = np.linalg.norm(Residual[self.free_dofs])
                norm_fint_total = np.linalg.norm(F_int)
                F_ref = max(F_ref, norm_fint_total, 1.0)
                error = norm_res / F_ref

                print(
                    f"  [Итерация {iteration + 1:2d}] Невязка: {norm_res:.3e} | Ошибка: {error:.3e} (Допуск: {self.tol:.1e})")

                if (error < self.tol or norm_res < 1e-7) and iteration > 0:
                    print(f"  -> [+] СХОДИМОСТЬ ДОСТИГНУТА за {iteration + 1} итераций.")
                    converged = True
                    break

                # 4. Решение СЛАУ
                diag_values = V_data[self.I_idx == self.J_idx]
                valid_diag = diag_values[diag_values < self.PENALTY * 0.1]
                max_stiff = np.max(valid_diag) if len(valid_diag) > 0 else 1.0
                reg_value = max_stiff * 1e-9

                K_t_sparse = sp.coo_matrix((V_data, (self.I_idx, self.J_idx)), shape=(total_dofs, total_dofs)).tocsr()
                reg_matrix = sp.eye(total_dofs) * reg_value

                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", spla.MatrixRankWarning)
                        dU = spla.spsolve(K_t_sparse, Residual)
                except Exception as e:
                    diag = K_t_sparse.diagonal()
                    valid_diag = diag[diag < self.PENALTY * 0.1]
                    max_stiff = np.max(valid_diag) if len(valid_diag) > 0 else 1.0
                    reg_matrix = sp.eye(total_dofs) * (max_stiff * 1e-9)
                    dU = spla.spsolve(K_t_sparse + reg_matrix, Residual)

                # --- 5. УСИЛЕННЫЙ LINE SEARCH ---
                eta = 1.0
                ls_success = True  # По умолчанию считаем шаг успешным

                # Запускаем Line Search ТОЛЬКО если невязка выросла или упала меньше чем на 10%
                needs_line_search = False
                if iteration > 0:
                    if norm_res > norm_res_prev * 0.9:
                        needs_line_search = True

                if needs_line_search:
                    ls_success = False
                    ls_max_iter = 5  # 10 - это слишком много для мелкой сетки, 5 достаточно

                    for ls_iter in range(ls_max_iter):
                        U_trial = U_global + eta * dU
                        F_int_trial = np.zeros(total_dofs)

                        for e_idx, element in enumerate(self.model.elements):
                            el_dofs = self.el_dofs_map[e_idx]
                            U_el = U_trial[el_dofs]

                            if self.is_discrete_map[e_idx]:
                                F_int_e = element.compute_internal_force(U_el)
                            else:
                                F_int_e = np.zeros(len(el_dofs))
                                for ip_idx, ip in enumerate(element.integration_points):
                                    B = self.B_map[e_idx][ip_idx]
                                    dV = self.dV_map[e_idx][ip_idx]
                                    current_strain = B @ U_el
                                    # ВАЖНО: здесь мы запрашиваем только напряжения, матрица D_ep не нужна
                                    stress, _ = ip.constitutive_model.update_state(current_strain)
                                    F_int_e += B.T @ stress * dV

                            F_int_trial[el_dofs] += F_int_e

                        Residual_trial = F_ext - F_int_trial

                        # Учет граничных условий для trial-невязки
                        for bc_idx, bc in enumerate(self.model.bcs):
                            dof = bc.node.dofs[bc.dof_axis]
                            is_prop = getattr(bc, 'is_proportional', True)
                            target_u = (bc.value * attempt_factor) if is_prop else bc.value
                            Residual_trial[dof] = self.PENALTY * (target_u - U_trial[dof])

                        norm_res_trial = np.linalg.norm(Residual_trial[self.free_dofs])

                        if norm_res_trial < norm_res:
                            ls_success = True
                            print(
                                f"    [LS] Шаг скорректирован: eta = {eta:.3f} (невязка: {norm_res:.2e} -> {norm_res_trial:.2e})")
                            break

                        eta *= 0.5

                    if not ls_success:
                        print("    [!] Line Search не смог уменьшить невязку. Откат.")
                        break

                        # Сохраняем текущую невязку для следующей итерации
                norm_res_prev = norm_res

                # 6. Обновление перемещений
                if iteration == 0 or ls_success:
                    U_global += eta * dU

            # --- ОБРАБОТКА РЕЗУЛЬТАТОВ ШАГА ---
            if converged:
                current_factor = attempt_factor

                # ИЗМЕНЕНО: Коммит и логирование
                for e_idx, element in enumerate(self.model.elements):
                    if not getattr(element, 'is_spring', False):
                        for ip_idx, ip in enumerate(element.integration_points):
                            ip.constitutive_model.commit()
                            # Записываем состояние в лог
                            self.mat_logger.log_state(step, current_factor, e_idx, ip_idx, ip.constitutive_model)

                step += 1

                for element in self.model.elements:
                    if not getattr(element, 'is_spring', False):
                        for ip in element.integration_points:
                            ip.constitutive_model.commit()

                for node in self.model.nodes:
                    node.displacements = U_global[node.dofs]

                if self.track_nodes:
                    rx_force = sum(F_int[n.dofs[self.track_dof]] for n in self.track_nodes)
                    current_u = U_global[self.track_nodes[0].dofs[self.track_dof]]
                    self.history_U.append(current_u)
                    self.history_F.append(rx_force)

                if iteration < 6 and current_factor < target_max_factor:
                    old_d_factor = d_factor
                    d_factor = min(d_factor * 1.5, self.load_factors[0])
                    if d_factor > old_d_factor:
                        print(f"  -> [^] Быстрая сходимость. Увеличиваем шаг нагрузки до {d_factor:.5f}")

            else:
                print(f"  -> [-] РАСХОДИМОСТЬ.")
                print(f"  -> [v] Откат состояния. Уменьшаем шаг нагрузки в 2 раза: {d_factor} -> {d_factor / 2.0}")
                d_factor /= 2.0
                U_global = U_prev

                if d_factor < min_d_factor:
                    print(f"\n[!!!] КРИТИЧЕСКАЯ ОШИБКА: Шаг нагрузки стал меньше минимального ({min_d_factor}).")
                    print("Дальнейшее решение невозможно. Остановка.")
                    break
        # ДОБАВЛЕНО: Генерация Excel файла после завершения всего расчета
        self.mat_logger.save_to_excel("material_history.xlsx")

