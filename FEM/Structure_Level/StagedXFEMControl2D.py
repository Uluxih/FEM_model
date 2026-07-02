import numpy as np
from FEM.Abstract.Structure_Level import Control
import scipy.sparse.linalg as spla
import scipy.sparse as sp
from FEM.Structure_Level.MaterialLogger import MaterialLogger


class StagedXFEMControl2D(Control):
    """
    Решатель Ньютона-Рафсона, адаптированный для XFEM.
    Поддерживает мульти-элементные сетки (XFEMQUAD4 + SpringElement2D).
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
        self.mat_logger = MaterialLogger()
        self.U_global = None

    def _update_topology_and_kinematics(self):
        # 1. ДИНАМИЧЕСКИЙ ПОДСЧЕТ СТЕПЕНЕЙ СВОБОДЫ
        # Находим максимальный индекс DOF среди всех элементов
        max_dof_idx = -1
        for element in self.model.elements:
            if getattr(element, 'is_spring', False):
                for node in element.nodes:
                    if len(node.dofs) > 0:
                        max_dof_idx = max(max_dof_idx, max(node.dofs))
            else:
                dofs = element.get_active_dofs()
                if len(dofs) > 0:
                    max_dof_idx = max(max_dof_idx, max(dofs))

        # Обновляем глобальное количество степеней свободы в модели
        total_dofs = max_dof_idx + 1
        self.model.total_dofs = total_dofs

        # 2. РАСШИРЕНИЕ ВЕКТОРА ПЕРЕМЕЩЕНИЙ
        if self.U_global is None:
            self.U_global = np.zeros(total_dofs)
        elif len(self.U_global) < total_dofs:
            added_dofs = total_dofs - len(self.U_global)
            self.U_global = np.pad(self.U_global, (0, added_dofs), 'constant', constant_values=0.0)

        # 3. ПОСТРОЕНИЕ КАРТЫ ИНДЕКСОВ ДЛЯ РАЗРЕЖЕННОЙ МАТРИЦЫ
        self.el_dofs_map = []
        num_k_entries = 0

        for element in self.model.elements:
            if getattr(element, 'is_spring', False):
                el_dofs = []
                for node in element.nodes:
                    el_dofs.extend(node.dofs)
                el_dofs = np.array(el_dofs, dtype=int)
            else:
                el_dofs = element.get_active_dofs()

            self.el_dofs_map.append(el_dofs)
            num_k_entries += len(el_dofs) ** 2

        num_k_entries += len(self.model.bcs)

        self.I_idx = np.zeros(num_k_entries, dtype=int)
        self.J_idx = np.zeros(num_k_entries, dtype=int)

        ptr = 0
        for e_idx, el_dofs in enumerate(self.el_dofs_map):
            ndof_e = len(el_dofs)
            grid = np.meshgrid(el_dofs, el_dofs, indexing='ij')
            self.I_idx[ptr:ptr + ndof_e ** 2] = grid[0].ravel()
            self.J_idx[ptr:ptr + ndof_e ** 2] = grid[1].ravel()
            ptr += ndof_e ** 2

        # 4. ОБНОВЛЕНИЕ МАССИВА СВОБОДНЫХ СТЕПЕНЕЙ СВОБОДЫ
        self.free_dofs = np.ones(total_dofs, dtype=bool)
        for bc in self.model.bcs:
            dof = bc.node.dofs[bc.dof_axis]
            self.I_idx[ptr] = dof
            self.J_idx[ptr] = dof
            self.free_dofs[dof] = False
            ptr += 1

    def solve(self):
        print("\nИнициализация XFEM модели...")
        self.model.initialize()
        self._update_topology_and_kinematics()

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
            print(f"ШАГ {step} | Фактор: {current_factor:.5f} -> {attempt_factor:.5f}")

            U_prev = self.U_global.copy()
            total_dofs = self.model.total_dofs

            F_ext = np.zeros(total_dofs)
            for load in self.model.nodal_loads:
                dof = load.node.dofs[load.dof_axis]
                factor = attempt_factor if getattr(load, 'is_proportional', True) else 1.0
                F_ext[dof] += load.value * factor

            norm_fext = np.linalg.norm(F_ext[self.free_dofs])
            F_ref = norm_fext if norm_fext > 1e-12 else 1e-6
            converged = False
            V_data = np.zeros(len(self.I_idx))

            for iteration in range(self.max_iter):
                V_data.fill(0.0)
                F_int = np.zeros(total_dofs)
                ptr = 0

                for e_idx, element in enumerate(self.model.elements):
                    el_dofs = self.el_dofs_map[e_idx]
                    U_el = self.U_global[el_dofs]
                    ndof_e = len(el_dofs)

                    # ПРОВЕРКА НА ПРУЖИНУ
                    if getattr(element, 'is_spring', False):
                        K_e = element.compute_stiffness()
                        F_int_e = element.compute_internal_force(U_el)
                    else:
                        K_e = np.zeros((ndof_e, ndof_e))
                        F_int_e = np.zeros(ndof_e)

                        # Объемное интегрирование
                        for ip in element.get_bulk_integration_points():
                            B = element.get_B_matrix_enriched(ip.coords)
                            dV = ip.weight * element.get_detJ(ip.coords)

                            current_strain = B @ U_el
                            stress, D_ep = ip.constitutive_model.update_state(current_strain)

                            K_e += B.T @ D_ep @ B * dV
                            F_int_e += B.T @ stress * dV

                        # Поверхностное интегрирование
                        if element.is_enriched:
                            for ip_coh in element.get_cohesive_integration_points():
                                N_jump = element.get_jump_operator(ip_coh.coords)
                                dA = ip_coh.weight * element.get_crack_segment_length()

                                current_jump = N_jump @ U_el
                                traction, K_coh = ip_coh.constitutive_model.update_state(current_jump)

                                K_e += N_jump.T @ K_coh @ N_jump * dA
                                F_int_e += N_jump.T @ traction * dA

                    V_data[ptr:ptr + ndof_e ** 2] = K_e.ravel()
                    ptr += ndof_e ** 2
                    F_int[el_dofs] += F_int_e

                Residual = F_ext - F_int

                for bc in self.model.bcs:
                    dof = bc.node.dofs[bc.dof_axis]
                    target_u = (bc.value * attempt_factor) if getattr(bc, 'is_proportional', True) else bc.value
                    V_data[ptr] = self.PENALTY
                    Residual[dof] = self.PENALTY * (target_u - self.U_global[dof])
                    ptr += 1

                norm_res = np.linalg.norm(Residual[self.free_dofs])
                F_ref = max(F_ref, np.linalg.norm(F_int), 1.0)
                error = norm_res / F_ref

                print(f"  [Итер {iteration + 1:2d}] Невязка: {norm_res:.3e} | Ошибка: {error:.3e}")

                if (error < self.tol or norm_res < 1e-7) and iteration > 0:
                    converged = True
                    break

                K_t_sparse = sp.coo_matrix((V_data, (self.I_idx, self.J_idx)), shape=(total_dofs, total_dofs)).tocsr()

                # Добавлена базовая регуляризация, как было в вашем StagedNRControl2D
                try:
                    dU = spla.spsolve(K_t_sparse, Residual)
                except Exception:
                    reg_matrix = sp.eye(total_dofs) * 1e-9
                    dU = spla.spsolve(K_t_sparse + reg_matrix, Residual)

                # --- УМНАЯ СТАБИЛИЗАЦИЯ ---
                eta = 1.0
                max_du = np.max(np.abs(dU))

                # Если перемещения за итерацию превышают 5 мм, гасим их
                if max_du > 0.005:
                    eta = 0.005 / max_du
                # Дополнительное гашение на первых итерациях после разрыва
                elif iteration < 3 and norm_res > 1e4:
                    eta = 0.5
                self.U_global += eta * dU

            if converged:
                current_factor = attempt_factor
                step += 1

                # Коммит состояний
                for element in self.model.elements:
                    if getattr(element, 'is_spring', False):
                        continue
                    for ip in element.get_bulk_integration_points():
                        ip.constitutive_model.commit()
                    if element.is_enriched:
                        for ip_coh in element.get_cohesive_integration_points():
                            ip_coh.constitutive_model.commit()

                # Сохранение истории для графиков
                if self.track_nodes:
                    rx_force = sum(F_int[n.dofs[self.track_dof]] for n in self.track_nodes)
                    current_u = self.U_global[self.track_nodes[0].dofs[self.track_dof]]
                    self.history_U.append(current_u)
                    self.history_F.append(rx_force)

                # Проверка роста трещины (если реализовано в FEModel)
                if hasattr(self.model, 'check_and_propagate_crack'):
                    crack_propagated = self.model.check_and_propagate_crack()
                    if crack_propagated:
                        print("  [XFEM] Обнаружен рост трещины! Обновление топологии...")
                        self._update_topology_and_kinematics()

                if iteration < 6 and current_factor < target_max_factor:
                    d_factor = min(d_factor * 1.5, self.load_factors[0])

            else:
                print("  -> [-] РАСХОДИМОСТЬ. Откат.")
                d_factor /= 2.0
                self.U_global = U_prev
                if d_factor < min_d_factor:
                    print("Критическая ошибка: минимальный шаг.")
                    break