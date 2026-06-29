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

modelFEM = UbiquitousJointModel3D


# =====================================================================
# ЧАСТЬ 1: ЯДРО МКЭ (РЕШАТЕЛЬ, ГЕНЕРАТОР СЕТКИ, ОБЕРТКА МАТЕРИАЛА)
# =====================================================================

class JointedMaterial(Material):
    """Обертка для передачи параметров в ConstitutiveModel"""

    def __init__(self, E, nu, joint_params):
        super().__init__(E, nu)
        self.joint_params = joint_params


class MultiElementNRControl(Control):
    """Нелинейный решатель методом Ньютона-Рафсона со встроенной диагностикой сингулярности"""

    def __init__(self, model, track_nodes, load_factors, track_dof=2, max_iter=100, tol=1e-4):
        super().__init__(model)
        self.load_factors = load_factors
        self.num_steps = len(load_factors)
        self.max_iter = max_iter
        self.tol = tol
        self.track_nodes = track_nodes
        self.track_dof = track_dof
        self.history_U = [0.0]
        self.history_F = [0.0]

    def solve(self):
        print("\nИнициализация модели...")
        self.model.initialize()
        total_dofs = self.model.total_dofs
        U_global = np.zeros(total_dofs)

        for node in self.model.nodes:
            node.displacements = U_global[node.dofs]

        try:
            VTKExporter.export(self.model, "results_step_000.vtk")
        except:
            pass

        prev_factor = 0.0

        for step, current_factor in enumerate(self.load_factors, 1):
            delta_factor = current_factor - prev_factor
            print(f"\n=== Шаг нагрузки {step}/{self.num_steps} | Фактор: {current_factor:.5f} ===")

            for iteration in range(self.max_iter):
                K_t = np.zeros((total_dofs, total_dofs))
                F_int = np.zeros(total_dofs)

                # Сборка глобальной матрицы
                for element in self.model.elements:
                    el_dofs = []
                    for node in element.nodes:
                        el_dofs.extend(node.dofs)
                    U_el = U_global[el_dofs]
                    K_e, F_int_e = self._compute_element(element, U_el)
                    K_t[np.ix_(el_dofs, el_dofs)] += K_e
                    F_int[el_dofs] += F_int_e

                Residual = -F_int
                free_dofs = np.ones(total_dofs, dtype=bool)

                # 1. Модификация вектора невязки для заданных перемещений
                for bc in self.model.bcs:
                    dof = bc.node.dofs[bc.dof_axis]
                    free_dofs[dof] = False
                    delta_u = (bc.value * delta_factor) if iteration == 0 else 0.0
                    if delta_u != 0.0:
                        Residual -= K_t[:, dof] * delta_u

                # 2. Модификация матрицы жесткости (зануление строк/столбцов)
                for bc in self.model.bcs:
                    dof = bc.node.dofs[bc.dof_axis]
                    delta_u = (bc.value * delta_factor) if iteration == 0 else 0.0
                    K_t[dof, :] = 0.0
                    K_t[:, dof] = 0.0
                    K_t[dof, dof] = 1.0
                    Residual[dof] = delta_u

                free_dofs_indices = np.where(free_dofs)[0]
                error = np.linalg.norm(Residual[free_dofs_indices]) if len(free_dofs_indices) > 0 else 0.0

                if error < self.tol and iteration > 0:
                    print(f"  -> Сходимость достигнута за {iteration} итераций. (Невязка: {error:.2e})")
                    break

                # === БЛОК ДИАГНОСТИКИ СИНГУЛЯРНОСТИ ===
                try:
                    dU = np.linalg.solve(K_t, Residual)
                except np.linalg.LinAlgError as e:
                    print("\n" + "=" * 60)
                    print("[КРИТИЧЕСКАЯ ОШИБКА] Глобальная матрица K_t сингулярна!")
                    diag_Kt = np.diag(K_t)
                    zero_diag_idx = np.where(np.abs(diag_Kt) < 1e-12)[0]
                    if len(zero_diag_idx) > 0:
                        print(f"-> Найдено {len(zero_diag_idx)} степеней свободы с НУЛЕВОЙ жесткостью на диагонали!")
                        print(f"-> Индексы проблемных DOF: {zero_diag_idx[:20]}")
                        print("-> ПРИЧИНА: Недостаточные граничные условия (Rigid Body Motion).")
                    else:
                        print("-> Нулей на диагонали нет. Матрица выродилась из-за полного разрушения материала.")
                    print("=" * 60 + "\n")
                    raise e
                # =======================================

                U_global += dU
            else:
                print(
                    f"  !!! ВНИМАНИЕ: Сходимость не достигнута за {self.max_iter} итераций !!! (Невязка: {error:.2e})")

            # Фиксация состояния (Commit)
            F_int_final = np.zeros(total_dofs)
            for element in self.model.elements:
                el_dofs = []
                for node in element.nodes:
                    el_dofs.extend(node.dofs)
                _, F_int_e = self._compute_element(element, U_global[el_dofs])
                F_int_final[el_dofs] += F_int_e
                for ip in element.integration_points:
                    ip.constitutive_model.commit()

            # Экспорт результатов
            for node in self.model.nodes:
                node.displacements = U_global[node.dofs]
            try:
                VTKExporter.export(self.model, f"results_step_{step:03d}.vtk")
            except:
                pass

            # Запись истории для графика
            rx_force = sum(F_int_final[n.dofs[self.track_dof]] for n in self.track_nodes)
            current_u = U_global[self.track_nodes[0].dofs[self.track_dof]]
            self.history_U.append(current_u)
            self.history_F.append(rx_force)

            prev_factor = current_factor

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
# ЧАСТЬ 2: ПОЛЬЗОВАТЕЛЬСКИЙ ВВОД И ЗАПУСК РАСЧЕТА
# =====================================================================

def run_cyclic_test():
    print("=== ТЕСТ: Циклическое нагружение (Упрочнение -> Растяжение -> Сжатие) ===")

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

    ROCK_E = 0.460e7
    ROCK_NU = 0.2
    ROCK_MU = 0.5
    ROCK_RP = 3.5e1
    ROCK_RC = 2.0e2

    cp_material = cp_mt.Material(
        mu=ROCK_MU,
        A_tensor=A_matrix,
        Rpx=ROCK_RP, Rpy=ROCK_RP, Rpz=ROCK_RP,
        Rcx=ROCK_RC, Rcy=ROCK_RC, Rcz=ROCK_RC
    )

    joint_params = {
        'cp_material': cp_material,
        'phi': 0.0,
        'psi': 0.0,
        'R_inf': 1.0,
        'b_param': 500.0,
        'Gf_t': 0.5,
        'Gf_c': 0.5,
        'l_c': 1.0
    }

    global_material = JointedMaterial(E=ROCK_E, nu=ROCK_NU, joint_params=joint_params)

    SIZE_X, SIZE_Y, SIZE_Z = 1.0, 1.0, 1.0
    factory = HEX8Factory()
    model = generate_block_mesh(SIZE_X, SIZE_Y, SIZE_Z, 2, 2, 2, global_material, factory)
    print(f"Сетка создана: {len(model.nodes)} узлов, {len(model.elements)} элементов.")

    # =====================================================================
    # ИСПРАВЛЕНИЕ: ГРАНИЧНЫЕ УСЛОВИЯ (Полная защита от Rigid Body Motion)
    # =====================================================================
    top_nodes = []
    for node in model.nodes:
        # Нижняя грань (Z = 0)
        if abs(node.coords[2] - 0.0) < 1e-6:
            model.add_bc(node, 2, 0.0)  # Жестко фиксируем по Z

            # Для исключения вращения и сдвига, но сохранения эффекта Пуассона,
            # фиксируем оси симметрии:
            if abs(node.coords[0] - 0.0) < 1e-6:
                model.add_bc(node, 0, 0.0)  # Запрет движения по X на левой грани
            if abs(node.coords[1] - 0.0) < 1e-6:
                model.add_bc(node, 1, 0.0)  # Запрет движения по Y на передней грани

        # Верхняя грань (Z = L)
        elif abs(node.coords[2] - SIZE_Z) < 1e-6:
            top_nodes.append(node)
            model.add_bc(node, 2, 1.0)  # Тянем по Z (X и Y свободны для сужения)

    # =====================================================================
    # ИСПРАВЛЕНИЕ: ПУТЬ НАГРУЖЕНИЯ (Убрали [0] чтобы решатель не делал пустой шаг)
    # =====================================================================
    # Используем [1:], чтобы первый фактор был > 0
    path_tension_1 = np.linspace(0, 0.0019, 20)[1:]
    path_compression = np.linspace(0.0009, -0.0040, 20)[1:]
    path_tension_2 = np.linspace(-0.0040, 0.0035, 20)[1:]

    load_factors = np.concatenate([path_tension_1, path_compression, path_tension_2])

    control = MultiElementNRControl(
        model=model,
        track_nodes=top_nodes,
        load_factors=load_factors,
        track_dof=2,
        max_iter=100,
        tol=1e-4
    )

    control.solve()

    # ПОСТРОЕНИЕ ГРАФИКА
    uz_mm = [u * 1000 for u in control.history_U]
    fz_n = control.history_F

    plt.figure(figsize=(10, 7))
    plt.plot(uz_mm, fz_n, marker='o', color='#1f77b4', linewidth=2, markersize=4, label='Реакция элемента')

    plt.axhline(ROCK_RP, color='red', linestyle='--', alpha=0.5, label=f'Пик растяжения ({ROCK_RP} Н)')
    plt.axhline(-ROCK_RC, color='orange', linestyle='--', alpha=0.5, label=f'Пик сжатия ({-ROCK_RC} Н)')
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)

    plt.text(0.1, 20, 'Пластичность\n(Упрочнение)', color='green', fontsize=9, ha='center')
    plt.text(0.25, 10, 'Damage\n(Растяжение)', color='red', fontsize=9, ha='left')
    plt.text(0.1, -50, 'Разгрузка\n(Деградированная\nжесткость)', color='blue', fontsize=9, ha='center')
    plt.text(-0.05, -100, 'Закрытие\nтрещины\n(Восстановление E)', color='purple', fontsize=9, ha='right')
    plt.text(-0.35, -200, 'Damage\n(Сжатие)', color='darkorange', fontsize=9, ha='center')

    plt.title("МКЭ Тест: Циклическое нагружение 3D элемента", fontsize=13)
    plt.xlabel("Вертикальное перемещение верхней грани Uz (мм)", fontsize=12)
    plt.ylabel("Суммарная реакция Fz (Н)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_cyclic_test()
