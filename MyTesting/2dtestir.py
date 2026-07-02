import numpy as np
import matplotlib.pyplot as plt

# Импорт базовых классов МКЭ и генератора сетки
from FEM.Abstract.Structure_Level import Node, FEModel
from FEM.Abstract.Integration_Point_Level import Material
from FEM.Element_Level.Shape4NodeQuadrilateral import QUAD4Factory  # Используем 2D элемент

# Импорт пружинного элемента
from FEM.Element_Level.SpringElement2D import SpringElement2D

# Импорт моделей материала
from FEM.Integration_Point_Level.UbiquitousJointModel2D import UbiquitousJointModel2D
import FEM.Integration_Point_Level.CriticalPlane.material as cp_mt

# Импорт решателя
from FEM.Structure_Level.StagedNRControl2D import StagedNRControl2D

modelFEM = UbiquitousJointModel2D


class JointedMaterial(Material):
    def __init__(self, E, nu, joint_params, matrix_params=None):
        super().__init__(E, nu)
        self.joint_params = joint_params
        # Сохраняем параметры матрицы для новой логики Друкера-Прагера
        self.matrix_params = matrix_params if matrix_params is not None else {}


def generate_rect_mesh(Lx, Ly, nx, ny, material, factory):
    """Генерация 2D прямоугольной сетки в плоскости X-Y"""
    model = FEModel()
    model.materials.append(material)
    node_id = 0
    nodes_dict = {}

    for j in range(ny + 1):
        for i in range(nx + 1):
            # Координаты [x, y], где y - это индекс 1 в 2D пространстве
            n = Node(node_id, [i * (Lx / nx), j * (Ly / ny)])
            model.nodes.append(n)
            nodes_dict[(i, j)] = n
            node_id += 1

    for j in range(ny):
        for i in range(nx):
            el_nodes = [
                nodes_dict[(i, j)],
                nodes_dict[(i + 1, j)],
                nodes_dict[(i + 1, j + 1)],
                nodes_dict[(i, j + 1)]
            ]
            el = factory.create_element(el_nodes, material, constitutive_class=modelFEM)
            model.elements.append(el)
    return model


def run_masonry_wall_test(sample_name="J4D"):
    print(f"=== ТЕСТ: Сдвиговая стена 2D в осях X-Y (Валидация по образцу {sample_name}) ===")

    # Словарь экспериментальных данных (Fz0 переименовано в Fy0)
    exp_data = {
        "J4D": {"Fy0": 30_000, "peak_Fx": 51},
        "J6D": {"Fy0": 120_000, "peak_Fx": 72},
        "J7D": {"Fy0": 210_000, "peak_Fx": 97}
    }

    if sample_name not in exp_data:
        raise ValueError(f"Неизвестный образец: {sample_name}. Выберите J4D, J6D или J7D.")

    Fy0 = exp_data[sample_name]["Fy0"]
    print('Вертикальная нагрузка (Fy0):', Fy0)
    expected_peak = exp_data[sample_name]["peak_Fx"]

    tensor_data = """
0.266944 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
0.000000 0.202500 -0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
0.000000 -0.000000 0.202500 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
0.000000 0.000000 0.000000 0.239842 0.000000 -0.000000 0.000000 0.000000 0.000000
0.000000 0.000000 0.000000 0.000000 0.266944 0.000000 0.000000 0.000000 0.000000
0.000000 0.000000 0.000000 -0.000000 0.000000 0.059960 0.000000 0.000000 0.000000
0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.525625 -0.000000 0.000000
0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 -0.000000 0.525625 0.000000
0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.266944
                    """
    A_matrix = cp_mt.load_tensor_from_string(tensor_data) * 1e12 * 4.0

    # Параметры материала
    MASONRY_E = 3500e6
    MASONRY_NU = 0.2
    MASONRY_MU = 0.577
    MASONRY_RP = 00.75e6
    MASONRY_RC = 12.000e6

    cp_material = cp_mt.Material(
        mu=MASONRY_MU,
        A_tensor=A_matrix,
        Rpx=MASONRY_RP*1.0, Rpy=MASONRY_RP * 0.7, Rpz=MASONRY_RP * 1.0,
        Rcx=MASONRY_RC*0.75, Rcy=MASONRY_RC * 1.0, Rcz=MASONRY_RC * 1.0
    )

    nq = 2
    SIZE_X, SIZE_Y = 1.0, 1.0
    THICKNESS = 0.10
    nx, ny = nq, nq
    area_el = (SIZE_X / nx) * (SIZE_Y / ny)
    char_len = (area_el) ** 0.5
    n = 1

    # Параметры ослабленной плоскости (трещины)
    joint_params = {
        'phi': 030.0,
        'psi': 0.0,
        'phi_r': 0.0,
        'cp_material': cp_material,
        'l_c': char_len,
        'Gf_t': 400*n,
        'Gf_c': 1000*n,
        'Gf_s': 1000*n,
        'a_t': 01.00,
        'a_s': 01.00,
        'mu': 0.10,
        'fcr_over_fc': 0.10
    }

    # Параметры целого материала (матрицы) для Друкера-Прагера
    # Вы можете корректировать их в зависимости от прочности вашей кладки на сжатие/срез
    matrix_params = {
        'c': 000.700e6,  # Сцепление матрицы (Па)
        'phi': 30.0,  # Угол внутреннего трения матрицы (градусы)
        'psi': 0.0  # Угол дилатансии матрицы (градусы)
    }

    global_material = JointedMaterial(
        E=MASONRY_E,
        nu=MASONRY_NU,
        joint_params=joint_params,
        matrix_params=matrix_params
    )

    factory = QUAD4Factory()
    model = generate_rect_mesh(SIZE_X, SIZE_Y, nx, ny, global_material, factory)

    # =========================================================================
    # ДОБАВЛЕНИЕ ПРУЖИННЫХ ЭЛЕМЕНТОВ И НАГРУЗОК
    # =========================================================================

    control_nodes = []

    # Находим максимальный ID узла, чтобы создавать новые без конфликтов
    max_node_id = max(node.id for node in model.nodes)

    # Жесткость пружины
    k_stiffness = 1e12
    k_stiffnessY = 1e7

    TARGET_DISP_X = 0.005

    # Расчет общей вертикальной силы на 2D модель (модель единичной толщины)
    # Знак минус, так как сила направлена вниз (сжатие вдоль Y)
    Fy_2D_total = -Fy0 / THICKNESS

    # Итерируемся по КОПИИ списка узлов
    for node in list(model.nodes):
        # Нижняя грань (y = 0) - жестко защемлена
        if abs(node.coords[1] - 0.0) < 1e-6:
            model.add_bc(node, 0, 0.0)  # X
            model.add_bc(node, 1, 0.0)  # Y (индекс 1 в 2D)

        # Верхняя грань (y = SIZE_Y) - передаем нагрузку
        elif abs(node.coords[1] - SIZE_Y) < 1e-6:

            # --- 1. ПРИЛОЖЕНИЕ ВЕРТИКАЛЬНОЙ СИЛЫ К УЗЛУ КЛАДКИ ---
            # Распределяем вертикальную силу: на крайние узлы половина, на внутренние полная доля
            if abs(node.coords[0] - 0.0) < 1e-6 or abs(node.coords[0] - SIZE_X) < 1e-6:
                nodal_force = Fy_2D_total / (2 * nx)
            else:
                nodal_force = Fy_2D_total / nx

            # Вертикальное обжатие по Y через СИЛУ в узел кладки (Не пропорциональное)
            model.add_load(node, 1, nodal_force)
            model.nodal_loads[-1].is_proportional = False

            # --- 2. ДОБАВЛЕНИЕ ПРУЖИНЫ И ГОРИЗОНТАЛЬНОГО ПЕРЕМЕЩЕНИЯ ---
            max_node_id += 1
            ctrl_node = Node(max_node_id, node.coords.copy())
            model.nodes.append(ctrl_node)
            control_nodes.append(ctrl_node)

            # Создаем пружинный элемент
            spring = SpringElement2D(nodes=[node, ctrl_node], kx=k_stiffness, ky=k_stiffnessY)
            model.elements.append(spring)

            # Горизонтальный сдвиг по X через пружину (Пропорциональный)
            model.add_bc(ctrl_node, 0, TARGET_DISP_X)
            model.bcs[-1].is_proportional = True

    print(f"Добавлено {len(control_nodes)} пружинных элементов.")
    print(f"Сетка обновлена: {len(model.nodes)} узлов, {len(model.elements)} элементов.")

    # 50 шагов нагружения до 1.0 (100% от TARGET_DISP_X)
    load_factors = np.linspace(0, 1.0, 50)[1:]

    control = StagedNRControl2D(
        model=model,
        track_nodes=control_nodes,
        load_factors=load_factors,
        track_dof=0,  # Отслеживаем реакцию по оси X
        max_iter=20,
        tol=5e-3
    )

    control.solve()

    # Извлечение данных для графика
    ux_mm = [u * 1000 for u in control.history_U]

    # Умножаем силу на фактическую толщину стены
    fx_kN = [f * THICKNESS / 1000 for f in control.history_F]

    # Построение графиков
    plt.figure(figsize=(10, 6))
    plt.plot(ux_mm, fx_kN, color='#1f77b4', linewidth=2, label=f'Численное мод. 2D Fx ({sample_name})')

    plt.axhline(expected_peak, color='red', linestyle='--', alpha=0.7,
                label=f'Эксперимент {sample_name} (макс ~{expected_peak} кН)')

    plt.title(f"МКЭ 2D Тест: Сдвиг кирпичной стены (Образец {sample_name}, Fy0 = {Fy0 / 1000} кН)", fontsize=13)
    plt.xlabel("Горизонтальное смещение верха стены (мм)", fontsize=12)
    plt.ylabel("Горизонтальная сдвигающая сила Fx (кН)", fontsize=12)

    plt.xlim(0, 10.0)
    plt.ylim(0, max(max(fx_kN) * 1.2, expected_peak * 1.2) if fx_kN else expected_peak * 1.2)

    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_masonry_wall_test("J4D")