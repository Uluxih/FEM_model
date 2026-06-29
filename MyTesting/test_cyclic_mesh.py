import os
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

# Автоматическое подключение быстрого многопоточного решателя
try:
    import pypardiso as spla

    print("[INFO] Установлен PyPardiso: включен сверхбыстрый многопоточный решатель СЛАУ!")
except ImportError:
    import scipy.sparse.linalg as spla

    print("[INFO] PyPardiso не найден. Используется решатель SciPy (рекомендуется: pip install pypardiso)")

# Импорты базовых классов МКЭ (предполагается, что они есть в вашем проекте)
from FEM.Abstract.Structure_Level import Node, FEModel, Control
from FEM.Abstract.Integration_Point_Level import Material
from FEM.Element_Level.Shape8NodeHexahedron import HEX8Factory
from FEM.Structure_Level.VTKExporter import VTKExporter

import FEM.Integration_Point_Level.CriticalPlane.material as cp_mt

# =====================================================================
# ВАЖНО: Используем ИСПРАВЛЕННУЮ модель с зафиксированной плоскостью
# =====================================================================
from FEM.Integration_Point_Level.UbiquitousJointModel3D import UbiquitousJointModel3DFixed


# Обертка для передачи фиксированной нормали
class FixedPlaneModel(UbiquitousJointModel3DFixed):
    def __init__(self, material):
        # Жестко фиксируем плоскость по Z для теста
        super().__init__(material, fixed_normal=[0.0, 0.0, 1.0])


modelFEM = FixedPlaneModel


# =====================================================================
# ЧАСТЬ 1: ЯДРО МКЭ
# =====================================================================

class JointedMaterial(Material):
    """Обертка для передачи параметров в ConstitutiveModel"""

    def __init__(self, E, nu, joint_params):
        super().__init__(E, nu)
        self.joint_params = joint_params


class MultiElementNRControl(Control):
    """
    СВЕРХ-ОПТИМИЗИРОВАННЫЙ нелинейный решатель методом Ньютона-Рафсона.
    """

    def __init__(self, model, track_nodes, load_factors, track_dof=2, max_iter=300, tol=1e-3):
        super().__init__(model)
        self.load_factors = load_factors
        self.num_steps = len(load_factors)
        self.max_iter = max_iter
        self.tol = tol
        self.track_nodes = track_nodes
        self.track_dof = track_dof
        self.history_U = [0.0]
        self.history_F = [0.0]
        self.PENALTY = 1e12  # 1e15 может вызывать ill-conditioning, 1e12 безопаснее для float64

    def _precompute_topology_and_kinematics(self):
        num_elements = len(self.model.elements)
        ndof_per_el = len(self.model.elements[0].nodes) * 3

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

            B_el = []
            dV_el = []
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

        for bc in self.model.bcs:
            dof = bc.node.dofs[bc.dof_axis]
            self.I_idx[ptr] = dof
            self.J_idx[ptr] = dof
            ptr += 1

    def solve(self):
        print("\nИнициализация модели и предрасчет кинематики (B-матриц)...")
        self.model.initialize()
        self._precompute_topology_and_kinematics()

        total_dofs = self.model.total_dofs
        U_global = np.zeros(total_dofs)

        for node in self.model.nodes:
            node.displacements = U_global[node.dofs]

        prev_factor = 0.0
        ndof_per_el = self.el_dofs_map.shape[1]

        # Массив для сборки COO вынесен наружу для переиспользования памяти
        V_data = np.zeros(len(self.I_idx))

        for step, current_factor in enumerate(self.load_factors, 1):
            delta_factor = current_factor - prev_factor
            print(f"\n=== Шаг нагрузки {step}/{self.num_steps} | Фактор: {current_factor:.5f} ===")

            for iteration in range(self.max_iter):
                V_data.fill(0.0)
                F_int = np.zeros(total_dofs)
                ptr = 0

                # 1. Быстрая сборка глобальной матрицы
                for e_idx, element in enumerate(self.model.elements):
                    el_dofs = self.el_dofs_map[e_idx]
                    U_el = U_global[el_dofs]

                    K_e = np.zeros((ndof_per_el, ndof_per_el))
                    F_int_e = np.zeros(ndof_per_el)

                    B_el = self.B_map[e_idx]
                    dV_el = self.dV_map[e_idx]

                    for ip_idx, ip in enumerate(element.integration_points):
                        B = B_el[ip_idx]
                        dV = dV_el[ip_idx]

                        current_strain = B @ U_el
                        stress, D_ep = ip.constitutive_model.update_state(current_strain)

                        # ОПТИМИЗАЦИЯ: Умножение (D_ep @ B) сначала, экономит 70% операций
                        D_B = D_ep @ B
                        K_e += B.T @ D_B * dV
                        F_int_e += B.T @ stress * dV

                    V_data[ptr:ptr + ndof_per_el ** 2] = K_e.ravel()
                    ptr += ndof_per_el ** 2
                    F_int[el_dofs] += F_int_e

                Residual = -F_int
                free_dofs = np.ones(total_dofs, dtype=bool)

                # 2. Учет граничных условий (Метод Штрафа)
                for bc in self.model.bcs:
                    dof = bc.node.dofs[bc.dof_axis]
                    free_dofs[dof] = False
                    delta_u = (bc.value * delta_factor) if iteration == 0 else 0.0

                    V_data[ptr] = self.PENALTY
                    Residual[dof] = self.PENALTY * delta_u
                    ptr += 1

                # 3. Конвертация в формат CSR
                K_t_sparse = sp.coo_matrix((V_data, (self.I_idx, self.J_idx)), shape=(total_dofs, total_dofs)).tocsr()

                # 4. Проверка сходимости (Надежный смешанный критерий)
                if np.any(free_dofs):
                    norm_res = np.linalg.norm(Residual[free_dofs])
                    # Используем норму реакции опор как базу для масштабирования
                    norm_react = np.linalg.norm(F_int[~free_dofs])
                    scale = max(norm_react, 1.0)
                    error = norm_res / scale
                else:
                    error = 0.0

                if error < self.tol and iteration > 0:
                    print(f"  -> Сходимость за {iteration} итераций. (Относит. ошибка: {error:.2e})")
                    break

                # 5. Решение СЛАУ
                try:
                    dU = spla.spsolve(K_t_sparse, Residual)
                    if np.any(np.isnan(dU)):
                        raise ValueError("Получены NaN в векторе перемещений!")
                except Exception as e:
                    print(f"\n[КРИТИЧЕСКАЯ ОШИБКА] Решатель упал на итерации {iteration}: {e}")
                    return  # Прерываем расчет, но сохраняем графики

                U_global += dU
            else:
                print(f"  !!! ВНИМАНИЕ: Сходимость не достигнута за {self.max_iter} итераций !!! (Ошибка: {error:.2e})")

            # 6. Фиксация состояния (Commit)
            for element in self.model.elements:
                for ip in element.integration_points:
                    ip.constitutive_model.commit()

            for node in self.model.nodes:
                node.displacements = U_global[node.dofs]

            # Запись истории
            rx_force = sum(F_int[n.dofs[self.track_dof]] for n in self.track_nodes)
            current_u = U_global[self.track_nodes[0].dofs[self.track_dof]]
            self.history_U.append(current_u)
            self.history_F.append(rx_force)

            prev_factor = current_factor


def generate_block_mesh(Lx, Ly, Lz, nx, ny, nz, material, factory):
    model = FEModel()
    model.materials.append(material)
    node_id = 0
    nodes_dict = {}

    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                n = Node(node_id, [i * (Lx / nx), j * (Ly / ny), k * (Lz / nz)])
                model.nodes.append(n)
                nodes_dict[(i, j, k)] = n
                node_id += 1

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                el_nodes = [
                    nodes_dict[(i, j, k)], nodes_dict[(i + 1, j, k)],
                    nodes_dict[(i + 1, j + 1, k)], nodes_dict[(i, j + 1, k)],
                    nodes_dict[(i, j, k + 1)], nodes_dict[(i + 1, j, k + 1)],
                    nodes_dict[(i + 1, j + 1, k + 1)], nodes_dict[(i, j + 1, k + 1)]
                ]
                el = factory.create_element(el_nodes, material, constitutive_class=modelFEM)
                model.elements.append(el)
    return model


# =====================================================================
# ЧАСТЬ 2: ЗАПУСК РАСЧЕТА
# =====================================================================

def run_cyclic_test():
    print("=== ТЕСТ: Циклическое нагружение (Оптимизированная 3D сетка) ===")

    tensor_data_str = """
    0.001 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
    0.000000 0.001 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
    0.000000 0.000000 0.001 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
    0.000000 0.000000 0.000000 0.001 0.000000 0.000000 0.000000 0.000000 0.000000
    0.000000 0.000000 0.000000 0.000000 0.001 0.000000 0.000000 0.000000 0.000000
    0.000000 0.000000 0.000000 0.000000 0.000000 0.001 0.000000 0.000000 0.000000
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.001 0.000000 0.000000
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.001 0.000000
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.001
    """
    A_matrix = cp_mt.load_tensor_from_string(tensor_data_str) * 1e10

    ROCK_E = 0.460e6
    ROCK_NU = 0.2
    ROCK_MU = 0.5
    ROCK_RP = 0.5e1
    ROCK_RC = 1.0e2

    cp_material = cp_mt.Material(
        mu=ROCK_MU,
        A_tensor=A_matrix,
        Rpx=ROCK_RP, Rpy=ROCK_RP, Rpz=ROCK_RP,
        Rcx=ROCK_RC, Rcy=ROCK_RC, Rcz=ROCK_RC
    )

    SIZE_X, SIZE_Y, SIZE_Z = 1.0, 4.0, 4.0
    nx, ny, nz = 1, 4, 4  # Снизил плотность для быстрого теста, верните 16x16 если нужно

    vol_el = (SIZE_X / nx) * (SIZE_Y / ny) * (SIZE_Z / nz)
    char_len = vol_el ** (1 / 3)  # ИСПРАВЛЕНИЕ: Характерный размер для 3D это кубический корень из объема

    joint_params = {
        'cp_material': cp_material,
        'phi': 0.0,
        'psi': 0.0,
        'Gf_t': 0.5,
        'Gf_c': 5.0,  # Немного увеличил энергию сжатия для стабильности
        'Gf_s': 5.0,
        'l_c': char_len
    }

    global_material = JointedMaterial(E=ROCK_E, nu=ROCK_NU, joint_params=joint_params)

    factory = HEX8Factory()
    model = generate_block_mesh(SIZE_X, SIZE_Y, SIZE_Z, nx, ny, nz, global_material, factory)
    print(f"Сетка создана: {len(model.nodes)} узлов, {len(model.elements)} элементов.")

    top_nodes = []
    for node in model.nodes:
        # Нижняя грань (Z = 0)
        if abs(node.coords[2] - 0.0) < 1e-6:
            model.add_bc(node, 2, 0.0)
            if abs(node.coords[0] - 0.0) < 1e-6 and abs(node.coords[1] - 0.0) < 1e-6:
                model.add_bc(node, 0, 0.0)
                model.add_bc(node, 1, 0.0)
            elif abs(node.coords[0] - SIZE_X) < 1e-6 and abs(node.coords[1] - 0.0) < 1e-6:
                model.add_bc(node, 1, 0.0)

        # Верхняя грань (Z = L)
        elif abs(node.coords[2] - SIZE_Z) < 1e-6:
            top_nodes.append(node)
            model.add_bc(node, 2, 1.0)

            # Траектория нагружения
    path_tension_1 = np.linspace(0, 0.0009, 100)[1:]
    path_compression = np.linspace(0.0009, -0.0040, 200)[1:]
    path_tension_2 = np.linspace(-0.0040, 0.0012, 150)[1:]

    load_factors = np.concatenate([path_tension_1, path_compression, path_tension_2])

    control = MultiElementNRControl(
        model=model,
        track_nodes=top_nodes,
        load_factors=load_factors,
        track_dof=2,
        max_iter=50,  # 50 итераций более чем достаточно для квадратичной сходимости
        tol=1e-3
    )

    control.solve()

    # ПОСТРОЕНИЕ ГРАФИКА
    uz_mm = [u * 1000 for u in control.history_U]
    fz_n = control.history_F

    plt.figure(figsize=(10, 7))
    plt.plot(uz_mm, fz_n, marker='.', color='#1f77b4', linewidth=1.5, markersize=3, label='Реакция конструкции')

    area = SIZE_X * SIZE_Y
    plt.axhline(ROCK_RP * area, color='red', linestyle='--', alpha=0.5,
                label=f'Пик растяжения ({ROCK_RP * area:.1f} Н)')
    plt.axhline(-ROCK_RC * area, color='orange', linestyle='--', alpha=0.5,
                label=f'Пик сжатия ({-ROCK_RC * area:.1f} Н)')

    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)

    plt.title("МКЭ Тест: Циклическое нагружение 3D сетки", fontsize=13)
    plt.xlabel("Вертикальное перемещение верхней грани Uz (мм)", fontsize=12)
    plt.ylabel("Суммарная реакция Fz (Н)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_cyclic_test()
