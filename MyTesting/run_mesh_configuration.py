import os
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

# --- Попытка импорта решателя ---
try:
    import pypardiso as spla

    print("[INFO] Установлен PyPardiso: включен сверхбыстрый многопоточный решатель СЛАУ!")
except ImportError:
    import scipy.sparse.linalg as spla

    print("[INFO] PyPardiso не найден. Используется однопоточный решатель SciPy.")

# --- Импорты МКЭ ---
from FEM.Abstract.Structure_Level import Node, FEModel, Control
from FEM.Abstract.Integration_Point_Level import Material
from FEM.Element_Level.Shape8NodeHexahedron import HEX8Factory
from FEM.Structure_Level.VTKExporter import VTKExporter
from FEM.Integration_Point_Level.UbiquitousJointModel3D import UbiquitousJointModel3D
import FEM.Integration_Point_Level.CriticalPlane.material as cp_mt
from FEM.Structure_Level.NonLinearNewtonRaphsonControl import MultiElementNRControl

modelFEM = UbiquitousJointModel3D


# =====================================================================
# ЧАСТЬ 1: ЯДРО МКЭ
# =====================================================================

class JointedMaterial(Material):
    def __init__(self, E, nu, joint_params):
        super().__init__(E, nu)
        self.joint_params = joint_params


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
# ЧАСТЬ 2: ФУНКЦИЯ ДЛЯ ЗАПУСКА ОДНОГО ТЕСТА С ЗАДАННОЙ СЕТКОЙ
# =====================================================================

def run_mesh_configuration(Lx, Ly, Lz, nx, ny, nz, max_disp, steps):
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

    ROCK_E, ROCK_NU, ROCK_MU = 0.460e6, 0.2, 0.5
    ROCK_RP, ROCK_RC = 1.0e1, 1.50e2

    cp_material = cp_mt.Material(
        mu=ROCK_MU, A_tensor=A_matrix,
        Rpx=ROCK_RP, Rpy=ROCK_RP, Rpz=ROCK_RP,
        Rcx=ROCK_RC, Rcy=ROCK_RC, Rcz=ROCK_RC
    )

    # Расчет характеристической длины элемента
    vol_el = (Lx / nx) * (Ly / ny) * (Lz / nz)
    char_len = vol_el ** (1.0 / 3.0)

    joint_params = {
        'cp_material': cp_material,
        'phi': 0.0, 'psi': 0.0,
        'R_inf': 150.0, 'b_param': 5.0,
        'Gf_t': 0.05, 'Gf_c': 0.5,
        'l_c': char_len  # Этот параметр отвечает за регуляризацию
    }

    global_material = JointedMaterial(E=ROCK_E, nu=ROCK_NU, joint_params=joint_params)
    factory = HEX8Factory()
    model = generate_block_mesh(Lx, Ly, Lz, nx, ny, nz, global_material, factory)

    top_nodes = []
    for node in model.nodes:
        if abs(node.coords[2] - 0.0) < 1e-6:
            model.add_bc(node, 2, 0.0)
            if abs(node.coords[0] - 0.0) < 1e-6 and abs(node.coords[1] - 0.0) < 1e-6:
                model.add_bc(node, 0, 0.0)
                model.add_bc(node, 1, 0.0)
            elif abs(node.coords[0] - Lx) < 1e-6 and abs(node.coords[1] - 0.0) < 1e-6:
                model.add_bc(node, 1, 0.0)
        elif abs(node.coords[2] - Lz) < 1e-6:
            top_nodes.append(node)
            model.add_bc(node, 2, 1.0)

            # Монотонное растяжение
    load_factors = np.linspace(0, max_disp, steps)[1:]

    control = MultiElementNRControl(
        model=model,
        track_nodes=top_nodes,
        load_factors=load_factors,
        track_dof=2,
        max_iter=100,
        tol=1e-3
    )

    print(f"Запуск расчета: Сетка {nx}x{ny}x{nz}, l_c = {char_len:.4f}")
    control.solve()

    return control.history_U, control.history_F


# =====================================================================
# ЧАСТЬ 3: ДЕМОНСТРАЦИЯ РЕГУЛЯРИЗАЦИИ И ПОСТРОЕНИЕ 2Х ГРАФИКОВ
# =====================================================================

def demonstrate_regularization():
    print("=== ДЕМОНСТРАЦИЯ РЕГУЛЯРИЗАЦИИ СЕТКИ (Mesh Objectivity) ===")

    # Геометрия образца
    Lx, Ly, Lz = 1.0, 1.0, 1.0
    Area = Lx * Ly

    # Настройки теста
    max_displacement = 0.0015  # Общее удлинение 1.5 мм
    steps = 40

    # Сетки разной плотности по оси приложения нагрузки
    meshes = [
        {"nx": 1, "ny": 1, "nz": 5, "color": "blue", "label": "Грубая (1x1x5)"},
        {"nx": 1, "ny": 1, "nz": 10, "color": "green", "label": "Средняя (1x1x10)"},
        {"nx": 1, "ny": 1, "nz": 20, "color": "red", "label": "Мелкая (1x1x20)"}
    ]

    results = []

    for mesh in meshes:
        U, F = run_mesh_configuration(Lx, Ly, Lz, mesh["nx"], mesh["ny"], mesh["nz"], max_displacement, steps)

        # Перевод в напряжения и деформации
        # Деформация (Strain) = Перемещение / Начальная длина
        Strain = [u / Lz for u in U]

        # Напряжение (Stress) = Сила / Площадь сечения
        Stress = [f / Area for f in F]

        results.append({
            "U_mm": [u * 1000 for u in U],
            "F": F,
            "Strain": Strain,
            "Stress": Stress,
            "label": mesh["label"],
            "color": mesh["color"]
        })

    # === ПОСТРОЕНИЕ ГРАФИКОВ ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ROCK_RP = 1.0e1  # Прочность на растяжение из материала

    # --- График 1: Сила - Перемещение (Конструктивный отклик) ---
    for res in results:
        ax1.plot(res["U_mm"], res["F"], marker='o', markersize=3,
                 color=res["color"], linewidth=2, label=res["label"])

    ax1.axhline(ROCK_RP * Area, color='black', linestyle='--', alpha=0.5, label=f'Пик ({ROCK_RP * Area} Н)')
    ax1.set_title("Диаграмма: Сила - Перемещение (Реакция конструкции)", fontsize=12)
    ax1.set_xlabel("Вертикальное перемещение $U_z$, мм", fontsize=11)
    ax1.set_ylabel("Суммарная реакция $F_z$, Н", fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend(loc='upper right')

    # Текст-пояснение для F-U
    ax1.text(0.05, 0.05,
             "ЗДЕСЬ РАБОТАЕТ РЕГУЛЯРИЗАЦИЯ:\nГрафики совпадают. Диссипированная\nэнергия (площадь под кривой)\nне зависит от размера сетки.",
             transform=ax1.transAxes, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

    # --- График 2: Напряжение - Деформация (Материальный отклик) ---
    for res in results:
        ax2.plot(res["Strain"], res["Stress"], marker='s', markersize=3,
                 color=res["color"], linewidth=2, label=res["label"])

    ax2.axhline(ROCK_RP, color='black', linestyle='--', alpha=0.5, label=f'Прочность ({ROCK_RP} Па)')
    ax2.set_title("Диаграмма деформирования: Напряжение - Деформация", fontsize=12)
    ax2.set_xlabel("Номинальная деформация $\\varepsilon_z$ (ед.)", fontsize=11)
    ax2.set_ylabel("Номинальное напряжение $\\sigma_z$, Па", fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend(loc='upper right')

    # Текст-пояснение для Sigma-Epsilon
    ax2.text(0.05, 0.05,
             "ЛОКАЛИЗАЦИЯ ДЕФОРМАЦИЙ:\nВетви расходятся. Чем мельче сетка,\nтем круче спад напряжений,\nтак как разрушение локализуется\nв узкой полосе элементов.",
             transform=ax2.transAxes, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.suptitle("Демонстрация работы Crack Band Theory (Регуляризация энергии разрушения)", fontsize=14,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demonstrate_regularization()
