import warnings
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from traits.trait_types import true

from FEM.Abstract.Structure_Level import Control
from FEM.Structure_Level.MaterialLogger import MaterialLogger


class StagedNRControl2D(Control):
    """
    Решатель Ньютона-Рафсона со стадийным нагружением для 2D задач.
    Оптимизации:
      - батч-сборка K и F_int через einsum (группировка элементов по типу);
      - фиксированный портрет CSR-матрицы (сборка без пересоздания coo/csr);
      - переиспользование LU-факторизации (Modified Newton, параметр refactor_every);
      - line search без пересборки касательной матрицы.
    """

    def __init__(self, model, track_nodes, load_factors, track_dof=1,
                 max_iter=300, tol=1e-3, refactor_every=1, enable_logging= True):
        super().__init__(model)
        self.load_factors = load_factors
        self.num_steps = len(load_factors)
        self.max_iter = max_iter
        self.tol = tol
        self.track_nodes = track_nodes
        self.track_dof = track_dof
        # Логирование состояния точек интегрирования после каждого шага
        self.enable_logging = enable_logging
        # 1 = классический NR (K на каждой итерации),
        # 2-4 = Modified Newton (K и LU обновляются реже — итераций больше, но каждая дешевле)
        self.refactor_every = max(1, refactor_every)

        self.history_U = []
        self.history_F = []

        self.PENALTY = 1e15
        self.free_dofs = None
        self.mat_logger = MaterialLogger()

    # ------------------------------------------------------------------ #
    #  Предрасчёт: топология, кинематика, портрет матрицы
    # ------------------------------------------------------------------ #

    def _precompute_topology_and_kinematics(self):
        total_dofs = self.model.total_dofs

        # --- разделение элементов: сплошные (по группам) и дискретные ---
        self.groups = []  # батчи однотипных сплошных элементов
        self.springs = []  # дискретные элементы (пружины)

        grouped = {}  # (n_nodes, n_ip) -> список (element, el_dofs)
        for element in self.model.elements:
            el_dofs = np.array([d for node in element.nodes for d in node.dofs], dtype=int)
            if getattr(element, 'is_spring', False):
                self.springs.append({'element': element, 'dofs': el_dofs})
            else:
                key = (len(element.nodes), len(element.integration_points))
                grouped.setdefault(key, []).append((element, el_dofs))

        I_parts, J_parts = [], []
        ptr = 0

        for (n_nodes, n_ip), items in grouped.items():
            ne = len(items)
            ndof = 2 * n_nodes
            dofs = np.empty((ne, ndof), dtype=int)
            B = None
            dV = np.empty((ne, n_ip))
            cms = []  # материальные модели [элемент][точка интегрирования]
            elems = []

            for e, (element, el_dofs) in enumerate(items):
                dofs[e] = el_dofs
                elems.append(element)
                node_coords = np.array([node.coords for node in element.nodes])
                cm_row = []
                for q, ip in enumerate(element.integration_points):
                    _, detJ = element.shape.get_jacobian(ip.coords, node_coords)
                    dN_dx = element.shape.get_shape_derivatives_cartesian(ip.coords, node_coords)
                    B_ip = element.analysis_model.get_B_matrix(dN_dx)
                    h = element.analysis_model.get_h_coefficient()
                    if B is None:
                        B = np.empty((ne, n_ip) + B_ip.shape)
                    B[e, q] = B_ip
                    dV[e, q] = detJ * h * ip.weight
                    cm_row.append(ip.constitutive_model)
                cms.append(cm_row)

            n_entries = ne * ndof * ndof
            # индексы I/J для всех элементов группы разом
            I_parts.append(np.repeat(dofs, ndof, axis=1).ravel())
            J_parts.append(np.tile(dofs, (1, ndof)).ravel())

            self.groups.append({
                'elements': elems,
                'dofs': dofs,
                'B': B,
                'dV': dV,
                'cms': cms,
                'slice': slice(ptr, ptr + n_entries),
                'ndof': ndof,
            })
            ptr += n_entries

        for spring in self.springs:
            d = spring['dofs']
            n = len(d)
            I_parts.append(np.repeat(d, n))
            J_parts.append(np.tile(d, n))
            spring['slice'] = slice(ptr, ptr + n * n)
            ptr += n * n

        # --- граничные условия (penalty на диагонали) ---
        self.free_dofs = np.ones(total_dofs, dtype=bool)
        bc_dofs = []
        for bc in self.model.bcs:
            dof = bc.node.dofs[bc.dof_axis]
            bc_dofs.append(dof)
            self.free_dofs[dof] = False
        bc_dofs = np.array(bc_dofs, dtype=int)
        I_parts.append(bc_dofs)
        J_parts.append(bc_dofs)
        self.bc_slice = slice(ptr, ptr + len(bc_dofs))
        ptr += len(bc_dofs)

        self.I_idx = np.concatenate(I_parts) if I_parts else np.zeros(0, dtype=int)
        self.J_idx = np.concatenate(J_parts) if J_parts else np.zeros(0, dtype=int)
        self.V_data = np.zeros(ptr)
        # penalty-члены постоянны — записываем один раз
        self.V_data[self.bc_slice] = self.PENALTY

        self._build_csr_pattern(total_dofs)

    def _build_csr_pattern(self, total_dofs):
        """Фиксированный портрет CSR + отображение coo-записей в csr.data."""
        order = np.lexsort((self.J_idx, self.I_idx))
        I_s, J_s = self.I_idx[order], self.J_idx[order]
        new_entry = np.ones(len(I_s), dtype=bool)
        new_entry[1:] = (I_s[1:] != I_s[:-1]) | (J_s[1:] != J_s[:-1])
        group_id = np.cumsum(new_entry) - 1

        self.coo_to_csr = np.empty(len(order), dtype=np.int64)
        self.coo_to_csr[order] = group_id
        self.nnz = int(group_id[-1]) + 1 if len(group_id) else 0

        rows, cols = I_s[new_entry], J_s[new_entry]
        indptr = np.zeros(total_dofs + 1, dtype=np.int64)
        np.add.at(indptr, rows + 1, 1)
        np.cumsum(indptr, out=indptr)
        self.K_csr = sp.csr_matrix(
            (np.zeros(self.nnz), cols.astype(np.int64), indptr),
            shape=(total_dofs, total_dofs))

    # ------------------------------------------------------------------ #
    #  Батч-вычисление внутренних сил (и, опционально, касательной K)
    # ------------------------------------------------------------------ #

    def _internal_forces(self, U, need_K):
        """Возвращает F_int; при need_K=True дополнительно заполняет self.V_data."""
        F_int = np.zeros(self.model.total_dofs)

        for grp in self.groups:
            B, dV, dofs = grp['B'], grp['dV'], grp['dofs']
            U_els = U[dofs]  # (ne, ndof)
            strains = np.einsum('eqij,ej->eqi', B, U_els)  # (ne, nip, nstr)

            stresses = np.empty_like(strains)
            nstr = strains.shape[-1]
            D = np.empty(strains.shape[:2] + (nstr, nstr)) if need_K else None

            # единственный оставшийся Python-цикл — вызов материальной модели
            for e, cm_row in enumerate(grp['cms']):
                for q, cm in enumerate(cm_row):
                    stress, D_ep = cm.update_state(strains[e, q])
                    stresses[e, q] = stress
                    if need_K:
                        D[e, q] = D_ep

            F_els = np.einsum('eqji,eqj,eq->ei', B, stresses, dV, optimize=True)
            np.add.at(F_int, dofs.ravel(), F_els.ravel())

            if need_K:
                K_els = np.einsum('eqji,eqjk,eqkl,eq->eil', B, D, B, dV, optimize=True)
                self.V_data[grp['slice']] = K_els.reshape(-1)

        for spring in self.springs:
            d = spring['dofs']
            el = spring['element']
            F_int[d] += el.compute_internal_force(U[d])
            if need_K:
                self.V_data[spring['slice']] = el.compute_stiffness().ravel()

        return F_int

    def _assemble_K(self):
        """Суммирование V_data в фиксированный портрет CSR."""
        self.K_csr.data[:] = np.bincount(self.coo_to_csr, weights=self.V_data,
                                         minlength=self.nnz)
        return self.K_csr

    def _apply_bc_residual(self, Residual, U, attempt_factor):
        for bc in self.model.bcs:
            dof = bc.node.dofs[bc.dof_axis]
            is_prop = getattr(bc, 'is_proportional', True)
            target_u = (bc.value * attempt_factor) if is_prop else bc.value
            Residual[dof] = self.PENALTY * (target_u - U[dof])

    def _factorize(self, K):
        try:
            return spla.splu(K.tocsc())
        except RuntimeError:
            diag = K.diagonal()
            valid = diag[diag < self.PENALTY * 0.1]
            max_stiff = np.max(valid) if len(valid) > 0 else 1.0
            K_reg = K + sp.eye(K.shape[0], format='csr') * (max_stiff * 1e-9)
            return spla.splu(K_reg.tocsc())

    # ------------------------------------------------------------------ #
    #  Экспорт результатов в VTK
    # ------------------------------------------------------------------ #
    def export_vtk(self, filename="results.vtk"):
        """Запись сетки и перемещений в формат Legacy VTK (ASCII)."""
        print(f"\nЭкспорт результатов в файл {filename}...")

        # Отфильтруем только сплошные элементы (без пружин)
        solid_elements = [el for el in self.model.elements if not getattr(el, 'is_spring', False)]

        with open(filename, "w") as f:
            # Заголовок
            f.write("# vtk DataFile Version 3.0\n")
            f.write("FEM 2D Results\n")
            f.write("ASCII\n")
            f.write("DATASET UNSTRUCTURED_GRID\n")

            # Узлы
            f.write(f"POINTS {len(self.model.nodes)} float\n")
            node_map = {}
            for i, node in enumerate(self.model.nodes):
                node_map[node] = i
                x, y = node.coords[0], node.coords[1]
                f.write(f"{x:.6e} {y:.6e} 0.0\n")

            # Элементы
            total_nodes_in_cells = sum(len(el.nodes) + 1 for el in solid_elements)
            f.write(f"\nCELLS {len(solid_elements)} {total_nodes_in_cells}\n")
            for el in solid_elements:
                n_nodes = len(el.nodes)
                indices = " ".join(str(node_map[n]) for n in el.nodes)
                f.write(f"{n_nodes} {indices}\n")

            # Типы элементов (VTK_TRIANGLE=5, VTK_QUAD=9)
            f.write(f"\nCELL_TYPES {len(solid_elements)}\n")
            for el in solid_elements:
                n_nodes = len(el.nodes)
                if n_nodes == 3:
                    f.write("5\n")
                elif n_nodes == 4:
                    f.write("9\n")
                else:
                    f.write("7\n")  # VTK_POLYGON как запасной вариант

            # Данные в узлах (перемещения)
            f.write(f"\nPOINT_DATA {len(self.model.nodes)}\n")
            f.write("VECTORS Displacements float\n")
            for node in self.model.nodes:
                ux = node.displacements[0] if len(node.displacements) > 0 else 0.0
                uy = node.displacements[1] if len(node.displacements) > 1 else 0.0
                f.write(f"{ux:.6e} {uy:.6e} 0.0\n")

        print("Экспорт завершен.")

    # ------------------------------------------------------------------ #
    #  Основной цикл
    # ------------------------------------------------------------------ #

    def solve(self, initial_displacements=None):
        """Выполняет одну стадию нагружения.

        initial_displacements позволяет продолжить расчёт от равновесного
        состояния предыдущей стадии, сохраняя закоммиченные состояния материала.
        """
        print("\nИнициализация модели и предрасчет кинематики...")
        self.model.initialize()
        self._precompute_topology_and_kinematics()

        total_dofs = self.model.total_dofs
        if initial_displacements is None:
            U_global = np.zeros(total_dofs)
        else:
            U_global = np.asarray(initial_displacements, dtype=float).copy()
            if U_global.shape != (total_dofs,):
                raise ValueError(
                    "Размер initial_displacements не соответствует числу степеней свободы модели."
                )

        target_max_factor = (self.load_factors[-1]
                             if isinstance(self.load_factors, (list, np.ndarray)) else 1.0)
        current_factor = 0.0
        d_factor = self.load_factors[0] if len(self.load_factors) > 0 else 0.1
        min_d_factor = 1e-6
        step = 1

        while current_factor < target_max_factor:
            if current_factor + d_factor > target_max_factor:
                d_factor = target_max_factor - current_factor

            attempt_factor = current_factor + d_factor
            print(f"\n{'=' * 50}")
            print(f"ШАГ {step} | Текущий фактор: {current_factor:.5f} -> "
                  f"Попытка: {attempt_factor:.5f} (Инкремент: {d_factor:.5f})")
            print(f"{'=' * 50}")

            U_prev = U_global.copy()

            F_ext = np.zeros(total_dofs)
            for load in self.model.nodal_loads:
                dof = load.node.dofs[load.dof_axis]
                is_prop = getattr(load, 'is_proportional', True)
                F_ext[dof] += load.value * (attempt_factor if is_prop else 1.0)

            norm_fext = np.linalg.norm(F_ext[self.free_dofs])
            F_ref = norm_fext if norm_fext > 1e-12 else 1e-6

            converged = False
            lu = None
            norm_res_prev = np.inf
            F_int = None
            iteration = 0

            for iteration in range(self.max_iter):
                # 1. Внутренние силы; касательная K — по расписанию Modified Newton
                need_K = (lu is None) or (iteration % self.refactor_every == 0)
                F_int = self._internal_forces(U_global, need_K)

                Residual = F_ext - F_int
                self._apply_bc_residual(Residual, U_global, attempt_factor)

                # 2. Проверка сходимости
                norm_res = np.linalg.norm(Residual[self.free_dofs])
                F_ref = max(F_ref, np.linalg.norm(F_int), 1.0)
                error = norm_res / F_ref

                # Невязка от penalty-ГУ (перемещение-драйвер сидит на закреплённых dof
                # и не попадает в norm_res по свободным dof)
                bc_res = Residual[~self.free_dofs]
                bc_du = np.max(np.abs(bc_res)) / self.PENALTY if bc_res.size else 0.0

                print(f"  [Итерация {iteration + 1:2d}] Невязка: {norm_res:.3e} | "
                      f"Ошибка: {error:.3e} | dU_гу: {bc_du:.2e} (Допуск: {self.tol:.1e})")

                # ВАЖНО: iteration > 0 обязательно — при нагружении через ГУ шаг
                # без единого решения СЛАУ «сходится» мгновенно, и перемещения не растут.
                if (error < self.tol or norm_res < 1e-7) and iteration > 0:
                    print(f"  -> [+] СХОДИМОСТЬ ДОСТИГНУТА за {iteration + 1} итераций.")
                    converged = True
                    break

                # если сходимость замедлилась при замороженной K — форсируем пересборку
                if not need_K and norm_res > norm_res_prev * 0.99:
                    F_int = self._internal_forces(U_global, need_K=True)
                    need_K = True

                # 3. Решение СЛАУ (LU переиспользуется, пока K не обновлялась)
                if need_K:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        lu = self._factorize(self._assemble_K())
                dU = lu.solve(Residual)

                if not np.all(np.isfinite(dU)):
                    print("    [!] Решение СЛАУ содержит NaN/Inf. Откат.")
                    break

                # 4. Line search (только при стагнации невязки; пересчитывает лишь F_int)
                eta = 1.0
                ls_success = True
                if iteration > 0 and norm_res > norm_res_prev * 0.9:
                    ls_success = False
                    for _ in range(10):
                        U_trial = U_global + eta * dU
                        Residual_trial = F_ext - self._internal_forces(U_trial, need_K=False)
                        self._apply_bc_residual(Residual_trial, U_trial, attempt_factor)
                        norm_res_trial = np.linalg.norm(Residual_trial[self.free_dofs])
                        if norm_res_trial < norm_res:
                            ls_success = True
                            print(f"    [LS] Шаг скорректирован: eta = {eta:.3f} "
                                  f"(невязка: {norm_res:.2e} -> {norm_res_trial:.2e})")
                            break
                        eta *= 0.5
                    if not ls_success:
                        print("    [!] Line Search не смог уменьшить невязку. Откат.")
                        break

                norm_res_prev = norm_res

                # 5. Обновление перемещений
                U_global += eta * dU

            # --- обработка результатов шага ---
            if converged:
                current_factor = attempt_factor

                # фиксация состояния материала (ровно один commit на точку)
                # и логирование: после commit в лог попадают только
                # подтверждённые значения (*_old), trial-состояния не пишутся
                el_id = 0
                for grp in self.groups:
                    for element, cm_row in zip(grp['elements'], grp['cms']):
                        cx = cy = 0.0
                        if self.enable_logging:
                            cx = sum(n.coords[0] for n in element.nodes) / len(element.nodes)
                            cy = sum(n.coords[1] for n in element.nodes) / len(element.nodes)
                        for ip_id, cm in enumerate(cm_row):
                            cm.commit()
                            if self.enable_logging:
                                self.mat_logger.log_state(step, current_factor,
                                                          el_id, ip_id, cx, cy, cm)
                        el_id += 1

                for node in self.model.nodes:
                    node.displacements = U_global[node.dofs]

                if self.track_nodes:
                    rx_force = sum(F_int[n.dofs[self.track_dof]] for n in self.track_nodes)
                    self.history_U.append(U_global[self.track_nodes[0].dofs[self.track_dof]])
                    self.history_F.append(rx_force)

                if iteration < 6 and current_factor < target_max_factor:
                    old_d_factor = d_factor
                    d_factor = min(d_factor * 1.5, self.load_factors[0])
                    if d_factor > old_d_factor:
                        print(f"  -> [^] Быстрая сходимость. Увеличиваем шаг нагрузки до {d_factor:.5f}")
                step += 1
            else:
                print(f"  -> [-] РАСХОДИМОСТЬ.")
                print(f"  -> [v] Откат состояния. Уменьшаем шаг нагрузки в 2 раза: "
                      f"{d_factor} -> {d_factor / 2.0}")
                d_factor /= 2.0
                U_global = U_prev
                # ВАЖНО: сброс незакоммиченного состояния материала к последнему commit,
                # иначе неявные внутренние переменные останутся от неудачной попытки
                for grp in self.groups:
                    for cm_row in grp['cms']:
                        for cm in cm_row:
                            if hasattr(cm, 'revert'):
                                cm.revert()

                if d_factor < min_d_factor:
                    print(f"\n[!!!] КРИТИЧЕСКАЯ ОШИБКА: Шаг нагрузки стал меньше минимального ({min_d_factor}).")
                    print("Дальнейшее решение невозможно. Остановка.")
                    break

        # Вызов экспорта в VTK в самом конце расчета
        self.export_vtk("results.vtk")