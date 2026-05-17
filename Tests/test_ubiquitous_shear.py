import os
import numpy as np
import matplotlib.pyplot as plt

# Импорты базовых классов МКЭ
from FEM.Abstract.Structure_Level import Node, FEModel, Control
from FEM.Abstract.Integration_Point_Level import Material
from FEM.Element_Level.Shape8NodeHexahedron import HEX8Factory
from FEM.Structure_Level.VTKExporter import VTKExporter

# Импорты конститутивной модели и поиска критической плоскости
from FEM.Integration_Point_Level.UbiquitousJointModel3D import UbiquitousJointModel3D
import FEM.Integration_Point_Level.CriticalPlane.material as cp_mt


# =====================================================================
# ЧАСТЬ 1: ЯДРО МКЭ (РЕШАТЕЛЬ, ГЕНЕРАТОР СЕТКИ, ОБЕРТКА МАТЕРИАЛА)
# =====================================================================

class JointedMaterial(Material):
    """Обертка для передачи параметров в ConstitutiveModel"""
    def __init__(self, E, nu, joint_params):
        super().__init__(E, nu)
        self.joint_params = joint_params


class MultiElementNRControl(Control):
    """Нелинейный решатель методом Ньютона-Рафсона"""
    def __init__(self, model, track_nodes, num_steps=20, max_iter=100, tol=1e-4):
        super().__init__(model)
        self.num_steps = num_steps
        self.max_iter = max_iter
        self.tol = tol
        self.track_nodes = track_nodes
        self.history_Ux = [0.0]
        self.history_Fx = [0.0]

    def solve(self):
        print("\nИнициализация модели...")
        self.model.initialize()
        total_dofs = self.model.total_dofs
        U_global = np.zeros(total_dofs)

        for node in self.model.nodes: node.displacements = U_global[node.dofs]
        VTKExporter.export(self.model, "results_step_000.vtk")

        for step in range(1, self.num_steps + 1):
            print(f"\n=== Шаг нагрузки {step}/{self.num_steps} ===")

            for iteration in range(self.max_iter):
                K_t = np.zeros((total_dofs, total_dofs))
                F_int = np.zeros(total_dofs)

                for element in self.model.elements:
                    el_dofs = []
                    for node in element.nodes: el_dofs.extend(node.dofs)
                    U_el = U_global[el_dofs]
                    K_e, F_int_e = self._compute_element(element, U_el)
                    K_t[np.ix_(el_dofs, el_dofs)] += K_e
                    F_int[el_dofs] += F_int_e

                Residual = -F_int
                free_dofs = np.ones(total_dofs, dtype=bool)

                # Применение граничных условий
                for bc in self.model.bcs:
                    dof = bc.node.dofs[bc.dof_axis]
                    free_dofs[dof] = False
                    delta_u = (bc.value / self.num_steps) if iteration == 0 else 0.0
                    if delta_u != 0.0: Residual -= K_t[:, dof] * delta_u

                for bc in self.model.bcs:
                    dof = bc.node.dofs[bc.dof_axis]
                    delta_u = (bc.value / self.num_steps) if iteration == 0 else 0.0
                    K_t[dof, :] = 0.0
                    K_t[:, dof] = 0.0
                    K_t[dof, dof] = 1.0
                    Residual[dof] = delta_u

                free_dofs_indices = np.where(free_dofs)[0]
                error = np.linalg.norm(Residual[free_dofs_indices]) if len(free_dofs_indices) > 0 else 0.0

                if error < self.tol and iteration > 0:
                    print(f"  -> Сходимость достигнута за {iteration} итераций.")
                    break

                dU = np.linalg.solve(K_t, Residual)
                U_global += dU
            else:
                print("  !!! ВНИМАНИЕ: Сходимость не достигнута !!!")

            # Фиксация состояния (Commit)
            F_int_final = np.zeros(total_dofs)
            for element in self.model.elements:
                el_dofs = []
                for node in element.nodes: el_dofs.extend(node.dofs)
                _, F_int_e = self._compute_element(element, U_global[el_dofs])
                F_int_final[el_dofs] += F_int_e
                for ip in element.integration_points:
                    ip.constitutive_model.commit()

            # Экспорт в VTK
            for node in self.model.nodes: node.displacements = U_global[node.dofs]
            VTKExporter.export(self.model, f"results_step_{step:03d}.vtk")

            # Запись истории
            rx_force = sum(F_int_final[n.dofs[0]] for n in self.track_nodes)
            current_ux = U_global[self.track_nodes[0].dofs[0]]
            self.history_Ux.append(current_ux)
            self.history_Fx.append(rx_force)

    def _compute_element(self, element, U_el):
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

            current_strain = B @ U_el
            stress, D_ep = ip.constitutive_model.update_state(current_strain)

            K_e += B.T @ D_ep @ B * dV
            F_int_e += B.T @ stress * dV

        return K_e, F_int_e


def generate_block_mesh(Lx, Ly, Lz, nx, ny, nz, material, factory):
    """Генератор сетки HEX8 элементов в виде параллелепипеда"""
    model = FEModel()
    model.materials.append(material)
    node_id = 0
    nodes_dict = {}

    # Создание узлов
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                n = Node(node_id, [i * (Lx / nx), j * (Ly / ny), k * (Lz / nz)])
                model.nodes.append(n)
                nodes_dict[(i, j, k)] = n
                node_id += 1

    # Создание элементов
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                el_nodes = [
                    nodes_dict[(i, j, k)], nodes_dict[(i + 1, j, k)],
                    nodes_dict[(i + 1, j + 1, k)], nodes_dict[(i, j + 1, k)],
                    nodes_dict[(i, j, k + 1)], nodes_dict[(i + 1, j, k + 1)],
                    nodes_dict[(i + 1, j + 1, k + 1)], nodes_dict[(i, j + 1, k + 1)]
                ]
                el = factory.create_element(el_nodes, material, constitutive_class=UbiquitousJointModel3D)
                model.elements.append(el)
    return model


# =====================================================================
# ЧАСТЬ 2: ПОЛЬЗОВАТЕЛЬСКИЙ ВВОД И ЗАПУСК РАСЧЕТА
# =====================================================================

if __name__ == "__main__":
    print("=== ТЕСТ: Адаптивная модель на ОДНОМ элементе ===")

    # ---------------------------------------------------------
    # ПАРАМЕТР 1: ТЕНЗОР АНИЗОТРОПИИ СЦЕПЛЕНИЯ (A_tensor) 9x9
    # ---------------------------------------------------------
    tensor_data_str = """
    0.900000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
    0.000000 0.399714 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
    0.000000 0.000000 0.900000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
    0.000000 0.000000 0.000000 0.399994 0.000000 0.000000 0.000000 0.000000 0.000000
    0.000000 0.000000 0.000000 0.000000 0.396589 0.000000 0.000000 0.000000 0.000000
    0.000000 0.000000 0.000000 0.000000 0.000000 0.396613 0.000000 0.000000 0.000000
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.544270 0.000000 0.000000
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.400018 0.000000
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.900000
    """
    A_matrix = cp_mt.load_tensor_from_string(tensor_data_str)
    A_matrix=A_matrix*1000000000000000000

    # ---------------------------------------------------------
    # ПАРАМЕТР 2: СВОЙСТВА ЦЕЛОЙ ПОРОДЫ (До разрушения)
    # ---------------------------------------------------------
    ROCK_E = 20.0e10   # Модуль Юнга (Па)
    ROCK_NU = 0.2     # Коэффициент Пуассона

    # Прочность матрицы (для поиска критической плоскости)
    ROCK_MU = 0.5     # Коэффициент трения матрицы (поиск)
    ROCK_RP = 15.5e10   # Предел на растяжение (Па)
    ROCK_RC = 15.0e10  # Предел на сжатие (Па)

    cp_material = cp_mt.Material(
        mu=ROCK_MU,
        A_tensor=A_matrix,
        Rpx=ROCK_RP, Rpy=ROCK_RP, Rpz=ROCK_RP,
        Rcx=ROCK_RC, Rcy=ROCK_RC, Rcz=ROCK_RC
    )

    # ---------------------------------------------------------
    # ПАРАМЕТР 3: СВОЙСТВА ТРЕЩИНЫ (После разрушения)
    # ---------------------------------------------------------
    joint_params = {
        'cp_material': cp_material,  # Передаем материал поиска

        # Податливость (жесткость) образующейся трещины
        'kn': 5.0,      # Нормальная жесткость (Па/м)
        'ks': 1.0,      # Сдвиговая жесткость (Па/м)
        'kt': 1.0,      # Сдвиговая жесткость (Па/м)
        'spacing': 0.2,  # Расстояние между трещинами (м)

        # Прочность скольжения по образовавшейся трещине
        # ВНИМАНИЕ: Очень низкие значения приведут к резкому сбросу силы (хрупкое разрушение)
        'c': 15.0e6,       # Сцепление трещины (Па)
        'phi': 0.0,       # Угол внутреннего трения трещины (град)
        'psi': 0.0,       # Угол дилатансии трещины (град)
        't': 15.0e6         # Прочность на отрыв по трещине (Па)
    }

    global_material = JointedMaterial(E=ROCK_E, nu=ROCK_NU, joint_params=joint_params)

    # ---------------------------------------------------------
    # ПАРАМЕТР 4: ГЕОМЕТРИЯ И СЕТКА (ИЗМЕНЕНО НА 1 ЭЛЕМЕНТ)
    # ---------------------------------------------------------
    SIZE_X, SIZE_Y, SIZE_Z = 1.0, 1.0, 1.0  # Единичный куб (м)
    ELEM_X, ELEM_Y, ELEM_Z = 1, 1, 1       # Всего 1 элемент (8 узлов)

    factory = HEX8Factory()
    model = generate_block_mesh(SIZE_X, SIZE_Y, SIZE_Z, ELEM_X, ELEM_Y, ELEM_Z, global_material, factory)
    print(f"Сетка создана: {len(model.nodes)} узлов, {len(model.elements)} элементов.")

    # ---------------------------------------------------------
    # ПАРАМЕТР 5: ГРАНИЧНЫЕ УСЛОВИЯ И НАГРУЗКА
    # ---------------------------------------------------------
    # Сдвиг 50 мм (5% деформации - достаточно, чтобы увидеть разрушение, но не вывернуть элемент)
    DISPLACEMENT_X = 0.050

    top_nodes = []
    for node in model.nodes:
        # Жесткая заделка основания (Z = 0)
        if abs(node.coords[2] - 0.0) < 1e-6:
            model.add_bc(node, 0, 0.0)  # Блок X
            model.add_bc(node, 1, 0.0)  # Блок Y
            model.add_bc(node, 2, 0.0)  # Блок Z

        # Нагружение верхней грани (Z = Lz)
        elif abs(node.coords[2] - SIZE_Z) < 1e-6:
            top_nodes.append(node)
            # model.add_bc(node, 1, 0.0)  # Блок Y (чтобы не уезжала вбок)
            model.add_bc(node, 0, DISPLACEMENT_X)  # Сдвиг по X

    # ---------------------------------------------------------
    # ПАРАМЕТР 6: НАСТРОЙКИ РЕШАТЕЛЯ
    # ---------------------------------------------------------
    NUM_STEPS = 40    # Количество шагов нагружения
    MAX_ITER = 200    # Макс. итераций Ньютона-Рафсона на шаг
    TOLERANCE = 1e-4  # Допуск сходимости по невязке

    control = MultiElementNRControl(
        model=model,
        track_nodes=top_nodes,
        num_steps=NUM_STEPS,
        max_iter=MAX_ITER,
        tol=TOLERANCE
    )

    # ЗАПУСК РАСЧЕТА
    control.solve()

    # ---------------------------------------------------------
    # ПАРАМЕТР 7: ПОСТРОЕНИЕ ГРАФИКА
    # ---------------------------------------------------------
    ux_mm = [u * 1000 for u in control.history_Ux]
    fx_mn = [abs(f) / 1e6 for f in control.history_Fx]

    plt.figure(figsize=(9, 6))
    plt.plot(ux_mm, fx_mn, marker='o', color='blue', linewidth=2, markersize=6)
    plt.title("Сдвиг ОДНОГО элемента (Адаптивный поиск критической плоскости)", fontsize=14)
    plt.xlabel("Горизонтальное перемещение верхней грани Ux (мм)", fontsize=12)
    plt.ylabel("Суммарная сдвигающая реакция Fx (МН)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(max(fx_mn), color='black', linestyle=':', label=f'Пиковая несущая способность ({max(fx_mn):.2f} МН)')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_multi_element_test()
