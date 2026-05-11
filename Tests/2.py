import numpy as np
import matplotlib.pyplot as plt

import FEM.Integration_Point_Level.CriticalPlane.material as mt
from FEM.Abstract.Structure_Level import Node, FEModel
from FEM.Abstract.Integration_Point_Level import Material
from FEM.Element_Level.Shape8NodeHexahedron import HEX8Factory
from FEM.Integration_Point_Level.CriticalPlane.CriticalPlanePlasticity3D import CriticalPlanePlasticity3D
from FEM.Structure_Level.NonLinearNewtonRaphsonControl import NonLinearNewtonRaphsonControl


# ==========================================
# 1. Глобальный материал
# ==========================================
class GlobalMaterial(Material):
    def __init__(self, E, nu, crit_mat):
        super().__init__(E, nu)
        self.crit_mat = crit_mat


# ==========================================
# 2. Кастомный VTK Экспортер с записью нормалей площадок
# ==========================================
class CustomVTKExporter:
    @staticmethod
    def export(model, filename="results.vtk"):
        print(f"Экспорт результатов в {filename}...")
        num_nodes = len(model.nodes)
        num_elements = len(model.elements)

        if num_nodes == 0 or num_elements == 0:
            return

        node_to_index = {node.id: idx for idx, node in enumerate(model.nodes)}

        with open(filename, "w", encoding="utf-8") as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("FEM 3D Results with Critical Planes\n")
            f.write("ASCII\n")
            f.write("DATASET UNSTRUCTURED_GRID\n\n")

            # POINTS
            f.write(f"POINTS {num_nodes} float\n")
            for node in model.nodes:
                c = node.coords
                f.write(f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")

            # CELLS
            total_list_size = sum(len(el.nodes) + 1 for el in model.elements)
            f.write(f"\nCELLS {num_elements} {total_list_size}\n")
            for el in model.elements:
                indices = [str(node_to_index[node.id]) for node in el.nodes]
                f.write(f"{len(el.nodes)} " + " ".join(indices) + "\n")

            # CELL_TYPES
            f.write(f"\nCELL_TYPES {num_elements}\n")
            for el in model.elements:
                f.write("12\n" if len(el.nodes) == 8 else "9\n")

            # POINT_DATA (Перемещения)
            f.write(f"\nPOINT_DATA {num_nodes}\n")
            f.write("VECTORS Displacements float\n")
            for node in model.nodes:
                d = node.displacements
                f.write(f"{d[0]:.6e} {d[1]:.6e} {d[2]:.6e}\n")

            # CELL_DATA (Данные интегрирования: Нормали и Статус)
            f.write(f"\nCELL_DATA {num_elements}\n")

            # 1. Статус текучести (1 - потек, 0 - упругий)
            f.write("SCALARS Is_Yielded int 1\n")
            f.write("LOOKUP_TABLE default\n")
            for el in model.elements:
                yielded = 0
                for ip in el.integration_points:
                    if hasattr(ip.constitutive_model, 'is_cracked_old') and ip.constitutive_model.is_cracked_old:
                        yielded = 1
                        break
                f.write(f"{yielded}\n")

            # 2. Накопленная пластическая деформация (Kappa)
            f.write("\nSCALARS Kappa float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for el in model.elements:
                kappa_avg = sum(ip.constitutive_model.kappa_old for ip in el.integration_points if
                                hasattr(ip.constitutive_model, 'kappa_old'))
                kappa_avg /= len(el.integration_points)
                f.write(f"{kappa_avg:.6e}\n")

            # 3. Вектор нормали критической площадки (Yield_Normal)
            f.write("\nVECTORS Yield_Normal float\n")
            for el in model.elements:
                nx, ny, nz = 0.0, 0.0, 0.0
                for ip in el.integration_points:
                    c_model = ip.constitutive_model
                    if hasattr(c_model, 'is_cracked_old') and c_model.is_cracked_old:
                        if c_model.fixed_normal_old is not None:
                            nx, ny, nz = c_model.fixed_normal_old
                            break  # Берем нормаль первой потекшей точки в элементе
                f.write(f"{nx:.6f} {ny:.6f} {nz:.6f}\n")

        print("Готово!")


# ==========================================
# 3. Решатель с записью истории (Обновлен для 2 осей)
# ==========================================
class TrackingNewtonRaphsonControl(NonLinearNewtonRaphsonControl):
    def __init__(self, model, track_node, num_steps=10, tol=1e-4, max_iter=20):
        super().__init__(model, num_steps, tol, max_iter)
        self.track_node = track_node
        # Отслеживаем смещения и силы по X (боковая) и Z (сжатие)
        self.history_Ux = [0.0]
        self.history_Uz = [0.0]
        self.history_Fx = [0.0]
        self.history_Fz = [0.0]

    def solve(self):
        print("Инициализация модели...")
        self.model.initialize()

        total_dofs = self.model.total_dofs
        U_global = np.zeros(total_dofs)

        F_ext_ref = np.zeros(total_dofs)
        total_force_x = 0.0
        total_force_z = 0.0

        for load in self.model.nodal_loads:
            F_ext_ref[load.node.dofs[load.dof_axis]] += load.value
            if load.dof_axis == 0:
                total_force_x += load.value
            elif load.dof_axis == 2:
                total_force_z += load.value

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
                    self.history_Ux.append(U_global[self.track_node.dofs[0]])
                    self.history_Uz.append(U_global[self.track_node.dofs[2]])
                    self.history_Fx.append(total_force_x * load_factor)
                    self.history_Fz.append(total_force_z * load_factor)
                    break

                dU = np.linalg.solve(K_t, Residual)
                U_global += dU
            else:
                self._commit_state()
                self.history_Ux.append(U_global[self.track_node.dofs[0]])
                self.history_Uz.append(U_global[self.track_node.dofs[2]])
                self.history_Fx.append(total_force_x * load_factor)
                self.history_Fz.append(total_force_z * load_factor)

        for node in self.model.nodes:
            node.displacements = U_global[node.dofs]


# ==========================================
# 4. Генератор сетки
# ==========================================
def create_wall_mesh(Lx, Ly, Lz, nx, ny, nz):
    nodes = []
    node_dict = {}
    id_counter = 0
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz

    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                n = Node(id_counter, [i * dx, j * dy, k * dz])
                nodes.append(n)
                node_dict[(i, j, k)] = n
                id_counter += 1

    elements_topology = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n1, n2 = node_dict[(i, j, k)], node_dict[(i + 1, j, k)]
                n3, n4 = node_dict[(i + 1, j + 1, k)], node_dict[(i, j + 1, k)]
                n5, n6 = node_dict[(i, j, k + 1)], node_dict[(i + 1, j, k + 1)]
                n7, n8 = node_dict[(i + 1, j + 1, k + 1)], node_dict[(i, j + 1, k + 1)]
                elements_topology.append([n1, n2, n3, n4, n5, n6, n7, n8])

    return nodes, elements_topology


# ==========================================
# 5. Основной сценарий расчета
# ==========================================
def run_wall_compression_test():
    tensor_data = """
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
    A_matrix = mt.load_tensor_from_string(tensor_data)
    crit_mat = mt.Material(mu=-0.4, A_tensor=A_matrix)
    material = GlobalMaterial(E=20000.0, nu=0.2, crit_mat=crit_mat)
    factory = HEX8Factory()

    nodes, elements_topology = create_wall_mesh(Lx=1.0, Ly=0.5, Lz=3.0, nx=2, ny=1, nz=6)

    model = FEModel()
    model.nodes = nodes
    model.materials = [material]

    for el_nodes in elements_topology:
        el = factory.create_element(el_nodes, material, constitutive_class=CriticalPlanePlasticity3D)
        model.elements.append(el)

    bottom_nodes = [n for n in model.nodes if abs(n.coords[2] - 0.0) < 1e-6]
    top_nodes = [n for n in model.nodes if abs(n.coords[2] - 3.0) < 1e-6]

    # Жесткое защемление основания
    for node in bottom_nodes:
        model.add_bc(node, 0, 0.0)
        model.add_bc(node, 1, 0.0)
        model.add_bc(node, 2, 0.0)

    # ---------------------------------------------
    # ПРИЛОЖЕНИЕ НАГРУЗОК
    # ---------------------------------------------
    total_force_z = -0.0  # Сжатие по оси Z
    total_force_x = 0.1  # Боковая нагрузка (сдвиг) по оси X

    force_per_node_z = total_force_z / len(top_nodes)
    force_per_node_x = total_force_x / len(top_nodes)

    for node in top_nodes:
        model.add_load(node, 2, force_per_node_z)  # Вертикальная сила
        model.add_load(node, 0, force_per_node_x)  # Горизонтальная (боковая) сила

    # Запускаем решатель
    control = TrackingNewtonRaphsonControl(
        model=model, track_node=top_nodes[0], num_steps=5, tol=1e-4
    )
    control.solve()

    CustomVTKExporter.export(model, "wall_compression_shear_results.vtk")

    # ---------------------------------------------
    # ПОСТРОЕНИЕ ГРАФИКОВ (Два окна: Сжатие и Сдвиг)
    # ---------------------------------------------
    displacements_z = [abs(u) * 1000 for u in control.history_Uz]
    forces_z = [abs(f) for f in control.history_Fz]

    displacements_x = [abs(u) * 1000 for u in control.history_Ux]
    forces_x = [abs(f) for f in control.history_Fx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # График по оси Z (Сжатие)
    ax1.plot(displacements_z, forces_z, marker='o', linestyle='-', color='r', linewidth=2)
    ax1.set_title("Сжатие стенки (Ось Z)")
    ax1.set_xlabel("Осадка верхнего узла $|U_z|$ (мм)")
    ax1.set_ylabel("Суммарная сила сжатия $F_z$ (МН)")
    ax1.grid(True, linestyle='--', alpha=0.7)

    # График по оси X (Сдвиг / Боковая нагрузка)
    ax2.plot(displacements_x, forces_x, marker='s', linestyle='-', color='b', linewidth=2)
    ax2.set_title("Сдвиг стенки (Ось X)")
    ax2.set_xlabel("Боковое смещение узла $|U_x|$ (мм)")
    ax2.set_ylabel("Суммарная боковая сила $F_x$ (МН)")
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("load_displacement_curve.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    run_wall_compression_test()
