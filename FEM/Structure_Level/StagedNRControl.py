import numpy as np
from FEM.Abstract.Structure_Level import Control
import scipy.sparse.linalg as spla
import scipy.sparse as sp


class StagedNRControl(Control):
    """
    Решатель Ньютона-Рафсона с поддержкой стадийного нагружения.
    Позволяет разделять граничные условия на пропорциональные (нарастают с load_factor)
    и постоянные (прикладываются на первом шаге и удерживаются).
    """

    def __init__(self, model, track_nodes, load_factors, track_dof=2, max_iter=300, tol=1e-3):
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

    def _precompute_topology_and_kinematics(self):
        num_elements = len(self.model.elements)
        ndof_per_el = len(self.model.elements[0].nodes) * 3
        total_dofs = self.model.total_dofs

        self.el_dofs_map = np.zeros((num_elements, ndof_per_el), dtype=int)
        self.B_map = []
        self.dV_map = []

        num_k_entries = num_elements * (ndof_per_el ** 2) + len(self.model.bcs)
        self.I_idx = np.zeros(num_k_entries, dtype=int)
        self.J_idx = np.zeros(num_k_entries, dtype=int)

        ptr = 0
        for e_idx, element in enumerate(self.model.elements):
            el_dofs = []
            for node in element.nodes:
                el_dofs.extend(node.dofs)
            self.el_dofs_map[e_idx] = el_dofs

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
            self.I_idx[ptr:ptr + ndof_per_el ** 2] = grid[0].ravel()
            self.J_idx[ptr:ptr + ndof_per_el ** 2] = grid[1].ravel()
            ptr += ndof_per_el ** 2

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

        ndof_per_el = self.el_dofs_map.shape[1]

        # Массив для отслеживания уже приложенных перемещений по каждому ГУ
        applied_bc_u = [0.0] * len(self.model.bcs)

        for step, current_factor in enumerate(self.load_factors, 1):
            print(f"\n=== Шаг нагрузки {step}/{self.num_steps} | Фактор: {current_factor:.5f} ===")

            F_ext = np.zeros(total_dofs)
            for load in self.model.nodal_loads:
                dof = load.node.dofs[load.dof_axis]
                is_prop = getattr(load, 'is_proportional', True)
                factor = current_factor if is_prop else 1.0
                F_ext[dof] += load.value * factor

            norm_fext = np.linalg.norm(F_ext[self.free_dofs])
            F_ref = norm_fext if norm_fext > 1e-12 else 1e-6

            for iteration in range(self.max_iter):
                V_data.fill(0.0)
                F_int = np.zeros(total_dofs)
                ptr = 0

                # 1. Сборка K и F_int
                for e_idx, element in enumerate(self.model.elements):
                    el_dofs = self.el_dofs_map[e_idx]
                    U_el = U_global[el_dofs]

                    K_e = np.zeros((ndof_per_el, ndof_per_el))
                    F_int_e = np.zeros(ndof_per_el)

                    for ip_idx, ip in enumerate(element.integration_points):
                        B = self.B_map[e_idx][ip_idx]
                        dV = self.dV_map[e_idx][ip_idx]

                        current_strain = B @ U_el
                        stress, D_ep = ip.constitutive_model.update_state(current_strain)

                        K_e += B.T @ D_ep @ B * dV
                        F_int_e += B.T @ stress * dV

                    V_data[ptr:ptr + ndof_per_el ** 2] = K_e.ravel()
                    ptr += ndof_per_el ** 2
                    F_int[el_dofs] += F_int_e

                Residual = F_ext - F_int

                # 2. Граничные условия
                for bc_idx, bc in enumerate(self.model.bcs):
                    dof = bc.node.dofs[bc.dof_axis]

                    if iteration == 0:
                        is_prop = getattr(bc, 'is_proportional', True)
                        target_u = (bc.value * current_factor) if is_prop else bc.value
                        delta_u = target_u - applied_bc_u[bc_idx]
                        applied_bc_u[bc_idx] = target_u
                    else:
                        delta_u = 0.0

                    V_data[ptr] = self.PENALTY
                    Residual[dof] = self.PENALTY * delta_u
                    ptr += 1

                # 3. Сходимость
                norm_res = np.linalg.norm(Residual[self.free_dofs])
                norm_fint = np.linalg.norm(F_int[self.free_dofs])
                F_ref = max(F_ref, norm_fint)
                error = norm_res / F_ref

                if error < self.tol and iteration > 0:
                    print(f"  -> Сходимость за {iteration} итераций. (Ошибка: {error:.2e})")
                    break

                # 4. Решение СЛАУ
                K_t_sparse = sp.coo_matrix((V_data, (self.I_idx, self.J_idx)), shape=(total_dofs, total_dofs)).tocsr()

                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", spla.MatrixRankWarning)
                        dU = spla.spsolve(K_t_sparse, Residual)
                except Exception as e:
                    diag = K_t_sparse.diagonal()
                    valid_diag = diag[diag < self.PENALTY * 0.1]
                    max_stiff = np.max(valid_diag) if len(valid_diag) > 0 else 1.0
                    reg_matrix = sp.eye(total_dofs) * (max_stiff * 1e-4)
                    dU = spla.spsolve(K_t_sparse + reg_matrix, Residual)

                # 5. Обновление
                U_global += dU

            else:
                print(f"  !!! ВНИМАНИЕ: Сходимость не достигнута за {self.max_iter} итераций !!! (Ошибка: {error:.2e})")

            # 6. Фиксация состояния
            for element in self.model.elements:
                for ip in element.integration_points:
                    ip.constitutive_model.commit()

            for node in self.model.nodes:
                node.displacements = U_global[node.dofs]

            if self.track_nodes:
                rx_force = sum(F_int[n.dofs[self.track_dof]] for n in self.track_nodes)
                current_u = U_global[self.track_nodes[0].dofs[self.track_dof]]
                self.history_U.append(current_u)
                self.history_F.append(rx_force)
