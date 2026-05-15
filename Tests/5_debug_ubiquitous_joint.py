import numpy as np
import matplotlib.pyplot as plt

from FEM.Abstract.Structure_Level import Node, FEModel, Control
from FEM.Abstract.Integration_Point_Level import Material
from FEM.Element_Level.Shape8NodeHexahedron import HEX8Factory
from FEM.Integration_Point_Level.UbiquitousJointModel3D import UbiquitousJointModel3D

# Импортируем ваш экспортер
from FEM.Structure_Level.VTKExporter import VTKExporter


# ==========================================
# 1. Глобальный материал
# ==========================================
class JointedMaterial(Material):
    def __init__(self, E, nu, joint_params):
        super().__init__(E, nu)
        self.joint_params = joint_params


# ==========================================
# 2. Нелинейный решатель с отслеживанием невязки
# ==========================================
class MultiElementNRControl(Control):
    """
    Полноценный метод Ньютона-Рафсона.
    Выводит невязку по свободным узлам на каждой итерации и экспортирует КЭ-сетку.
    """

    def __init__(self, model, track_nodes, num_steps=20, max_iter=100, tol=1e-4):
        super().__init__(model)
        self.num_steps = num_steps
        self.max_iter = max_iter
        self.tol = tol
        self.track_nodes = track_nodes

        self.history_Ux = [0.0]
        self.history_Fx = [0.0]

    def solve(self):
        print("Инициализация модели...")
        self.model.initialize()
        total_dofs = self.model.total_dofs
        U_global = np.zeros(total_dofs)

        # Выгрузим начальное (нулевое) состояние сетки
        for node in self.model.nodes:
            node.displacements = U_global[node.dofs]
        VTKExporter.export(self.model, "results_step_000.vtk")

        for step in range(1, self.num_steps + 1):
            print(f"\n=== Шаг нагрузки {step}/{self.num_steps} ===")

            for iteration in range(self.max_iter):
                # 1. Сборка глобальной матрицы жесткости и вектора внутренних сил
                K_t = np.zeros((total_dofs, total_dofs))
                F_int = np.zeros(total_dofs)

                for element in self.model.elements:
                    el_dofs = []
                    for node in element.nodes:
                        el_dofs.extend(node.dofs)

                    U_el = U_global[el_dofs]
                    K_e, F_int_e = self._compute_element(element, U_el)

                    K_t[np.ix_(el_dofs, el_dofs)] += K_e
                    F_int[el_dofs] += F_int_e

                # 2. Вектор невязки (Внешних сил нет, поэтому R = -F_int)
                Residual = -F_int

                # 3. Применение граничных условий
                free_dofs = np.ones(total_dofs, dtype=bool)

                # Корректировка правой части для свободных узлов
                for bc in self.model.bcs:
                    dof = bc.node.dofs[bc.dof_axis]
                    free_dofs[dof] = False  # Узел закреплен

                    # Приращение перемещения задается ТОЛЬКО на нулевой итерации шага
                    delta_u = (bc.value / self.num_steps) if iteration == 0 else 0.0

                    if delta_u != 0.0:
                        Residual -= K_t[:, dof] * delta_u

                # Модификация матрицы для закрепленных узлов
                for bc in self.model.bcs:
                    dof = bc.node.dofs[bc.dof_axis]
                    delta_u = (bc.value / self.num_steps) if iteration == 0 else 0.0

                    K_t[dof, :] = 0.0
                    K_t[:, dof] = 0.0
                    K_t[dof, dof] = 1.0
                    Residual[dof] = delta_u

                # 4. Вычисление ошибки (Невязка ТОЛЬКО по свободным узлам!)
                free_dofs_indices = np.where(free_dofs)[0]
                error = np.linalg.norm(Residual[free_dofs_indices]) if len(free_dofs_indices) > 0 else 0.0

                # print(f"  Итерация {iteration}: Невязка = {error:.6e}")

                # 5. Проверка сходимости
                if error < self.tol and iteration > 0:
                    print(f"  -> Сходимость достигнута за {iteration} итераций.")
                    break

                # 6. Решение СЛАУ и обновление перемещений
                dU = np.linalg.solve(K_t, Residual)
                U_global += dU

            else:
                print("  !!! ВНИМАНИЕ: Сходимость не достигнута !!!")

            # 7. Фиксация состояния (Commit)
            F_int_final = np.zeros(total_dofs)
            for element in self.model.elements:
                el_dofs = []
                for node in element.nodes: el_dofs.extend(node.dofs)
                _, F_int_e = self._compute_element(element, U_global[el_dofs])
                F_int_final[el_dofs] += F_int_e

                for ip in element.integration_points:
                    ip.constitutive_model.commit()

            # --- ОБНОВЛЕНИЕ ПЕРЕМЕЩЕНИЙ В УЗЛАХ И ЭКСПОРТ В VTK ---
            # Записываем перемещения обратно в модель, чтобы их увидел VTKExporter
            for node in self.model.nodes:
                node.displacements = U_global[node.dofs]

            vtk_filename = f"results_step_{step:03d}.vtk"
            VTKExporter.export(self.model, vtk_filename)

            # Сохранение истории для графика (Сумма реакций по оси X на верхней грани)
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


# ==========================================
# 3. ГЕНЕРАТОР 3D СЕТКИ
# ==========================================
def generate_block_mesh(Lx, Ly, Lz, nx, ny, nz, material, factory):
    """Генерирует сетку HEX8 элементов в виде параллелепипеда"""
    model = FEModel()
    model.materials.append(material)

    # Создание узлов
    node_id = 0
    nodes_dict = {}
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                x = i * (Lx / nx)
                y = j * (Ly / ny)
                z = k * (Lz / nz)
                n = Node(node_id, [x, y, z])
                model.nodes.append(n)
                nodes_dict[(i, j, k)] = n
                node_id += 1

    # Создание элементов
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                # Индексы 8 узлов для HEX8 (против часовой снизу, затем сверху)
                n1 = nodes_dict[(i, j, k)]
                n2 = nodes_dict[(i + 1, j, k)]
                n3 = nodes_dict[(i + 1, j + 1, k)]
                n4 = nodes_dict[(i, j + 1, k)]
                n5 = nodes_dict[(i, j, k + 1)]
                n6 = nodes_dict[(i + 1, j, k + 1)]
                n7 = nodes_dict[(i + 1, j + 1, k + 1)]
                n8 = nodes_dict[(i, j + 1, k + 1)]

                el_nodes = [n1, n2, n3, n4, n5, n6, n7, n8]
                el = factory.create_element(el_nodes, material, constitutive_class=UbiquitousJointModel3D)
                model.elements.append(el)

    return model


# ==========================================
# 4. ЗАПУСК ТЕСТА
# ==========================================
def run_multi_element_test():
    print("=== ТЕСТ: Сдвиг колонны (Ubiquitous Joint, Много элементов) ===")

    # 1. Материал
    joint_params = {
        'kn': 500.0, 'ks': 10000000.0, 'kt': 1000000.0,
        'spacing': 0.2,  # Трещины каждые 20 см
        'c': 1.5,  # Сцепление 1.5 МПа
        'phi': 25.0,  # Угол трения 25 град
        'psi': 0.0,
        't': 10.5  # Предел на растяжение 0.5 МПа
    }
    material = JointedMaterial(E=20000.0, nu=0.2, joint_params=joint_params)
    factory = HEX8Factory()

    # 2. Генерация сетки: Колонна 1м x 1м x 2м (разбиение 2x2x4 = 16 элементов)
    model = generate_block_mesh(Lx=1.0, Ly=1.0, Lz=2.0, nx=2, ny=2, nz=4, material=material, factory=factory)
    print(f"Модель создана: {len(model.nodes)} узлов, {len(model.elements)} элементов.")

    # 3. Граничные условия
    top_nodes = []

    for node in model.nodes:
        # Жесткая заделка основания (Z = 0)
        if abs(node.coords[2] - 0.0) < 1e-6:
            model.add_bc(node, 0, 0.0)
            model.add_bc(node, 1, 0.0)
            model.add_bc(node, 2, 0.0)

        # Нагружение верхней грани (Z = 2.0)
        elif abs(node.coords[2] - 2.0) < 1e-6:
            top_nodes.append(node)
            model.add_bc(node, 1, 0.0)  # Блокируем Y
            model.add_bc(node, 2, -0.002)  # Обжатие по Z (2 мм)
            model.add_bc(node, 0, 0.010)  # Сдвиг по X (10 мм)

    # 4. Решение
    # 20 шагов нагрузки. Макс. итераций = 100
    control = MultiElementNRControl(model=model, track_nodes=top_nodes, num_steps=20, max_iter=100)
    control.solve()

    # 5. Построение графика
    ux_mm = [u * 1000 for u in control.history_Ux]
    fx_mn = [abs(f) for f in control.history_Fx]

    plt.figure(figsize=(9, 6))
    plt.plot(ux_mm, fx_mn, marker='s', color='#d62728', linewidth=2, markersize=5)

    plt.title("Сдвиг колонны 2x2x4 элементов (Ubiquitous Joint)", fontsize=14)
    plt.xlabel("Горизонтальное перемещение верхней грани Ux (мм)", fontsize=12)
    plt.ylabel("Суммарная сдвигающая реакция Fx (МН)", fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(max(fx_mn), color='black', linestyle=':', label=f'Макс. несущая способность ({max(fx_mn):.2f} МН)')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_multi_element_test()
