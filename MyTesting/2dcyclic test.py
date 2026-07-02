import numpy as np
import matplotlib.pyplot as plt

# Импорт базовых классов МКЭ и генератора сетки
from FEM.Abstract.Structure_Level import Node, FEModel
from FEM.Abstract.Integration_Point_Level import Material
from FEM.Element_Level.Shape4NodeQuadrilateral import QUAD4Factory

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
        self.matrix_params = matrix_params if matrix_params is not None else {}


def generate_rect_mesh(Lx, Ly, nx, ny, material, factory):
    """Генерация 2D прямоугольной сетки в плоскости X-Y"""
    model = FEModel()
    model.materials.append(material)
    node_id = 0
    nodes_dict = {}

    for j in range(ny + 1):
        for i in range(nx + 1):
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


def generate_exact_cyclic_load_factors(amplitudes_mm, max_disp_mm, steps_per_ramp=10, cycles_per_amp=2):
    """
    Генерация множителей нагрузки по конкретным шагам амплитуд (в мм).
    Выполняет заданное количество циклов (по умолчанию 2) для каждой амплитуды,
    как описано в оригинальной статье 1994 г. (раздел 3.1).
    """
    factors = []
    current_ratio = 0.0

    for amp in amplitudes_mm:
        amp_ratio = amp / max_disp_mm

        # Повторяем цикл заданное количество раз (2 раза по статье)
        for _ in range(cycles_per_amp):
            # 1. Нагружение от 0 до +amp
            factors.extend(np.linspace(current_ratio, amp_ratio, steps_per_ramp + 1)[1:])
            # 2. Разгрузка и нагружение в обратную сторону от +amp до -amp
            factors.extend(np.linspace(amp_ratio, -amp_ratio, 2 * steps_per_ramp + 1)[1:])
            # 3. Возврат от -amp к 0
            factors.extend(np.linspace(-amp_ratio, 0, steps_per_ramp + 1)[1:])

            # После первого прохода текущее положение = 0, следующий цикл начнется с нуля
            current_ratio = 0.0

    return factors


def run_cyclic_masonry_wall_test(wall_type="Low"):
    print(f"=== ТЕСТ: Циклический сдвиг 2D (Оригинальная статья 1994 г., стена: {wall_type}) ===")

    # Параметры из статьи (раздел 2 и 3.1)
    THICKNESS = 0.25
    COMPRESSIVE_PRESSURE = 0.6e6  # 0.6 MPa (150 kN / (1.0m * 0.25m) = 0.6 MPa)

    # Словарь с точными данными из оригинальной статьи
    wall_data = {
        "Low": {
            "SIZE_X": 1.0,
            "SIZE_Y": 1.35,
            "TARGET_DISP_X": 0.006,
            "amplitudes_mm": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "expected_peak": 84  # F_max = 84 kN (из текста статьи)
        },
        "High": {
            "SIZE_X": 1.0,
            "SIZE_Y": 2.00,
            "TARGET_DISP_X": 0.012,
            "amplitudes_mm": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
            "expected_peak": 72  # F_max = 72 kN (из текста статьи)
        }
    }

    if wall_type not in wall_data:
        raise ValueError("Неизвестный тип стены. Выберите 'Low' или 'High'.")

    data = wall_data[wall_type]
    SIZE_X = data["SIZE_X"]
    SIZE_Y = data["SIZE_Y"]
    TARGET_DISP_X = data["TARGET_DISP_X"]
    expected_peak = data["expected_peak"]
    amplitudes_mm = data["amplitudes_mm"]

    # Вычисление общей вертикальной силы: Давление * Площадь (Ширина * Толщина)
    Fy0 = COMPRESSIVE_PRESSURE * SIZE_X * THICKNESS
    print(f"Вертикальная нагрузка (Fy0): {Fy0} Н ({Fy0 / 1000} кН)")

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
    A_matrix = cp_mt.load_tensor_from_string(tensor_data) * 1e12 * 8.0

    # Параметры материала
    MASONRY_E = 3000e6
    MASONRY_NU = 0.15
    MASONRY_MU = 0.58
    MASONRY_RP = 1.75e6
    MASONRY_RC = 6.2e6

    cp_material = cp_mt.Material(
        mu=MASONRY_MU,
        A_tensor=A_matrix,
        Rpx=MASONRY_RP * 1.0, Rpy=MASONRY_RP * 0.7, Rpz=MASONRY_RP * 1.0,
        Rcx=MASONRY_RC * 0.75, Rcy=MASONRY_RC * 1.0, Rcz=MASONRY_RC * 1.0
    )

    # Генерация сетки пропорционально размерам
    nx = 10
    ny = int(nx * (SIZE_Y / SIZE_X))
    area_el = (SIZE_X / nx) * (SIZE_Y / ny)
    char_len = (area_el) ** 0.5
    n = 0.25

    joint_params = {
        'phi': 30.0,
        'psi': 0.0,
        'phi_r': 1.0,
        'cp_material': cp_material,
        'l_c': char_len,
        'Gf_t': 1000 * n,
        'Gf_c': 4000 * n,
        'Gf_s': 2500 * n,
        'a_t': 1.00,
        'a_s': 1.00,
        'mu': 0.10,
        'fcr_over_fc': 0.10
    }

    matrix_params = {
        'c': 1.2000e6,
        'phi': 30.0,
        'psi': 0.0
    }

    global_material = JointedMaterial(
        E=MASONRY_E,
        nu=MASONRY_NU,
        joint_params=joint_params,
        matrix_params=matrix_params
    )

    factory = QUAD4Factory()
    model = generate_rect_mesh(SIZE_X, SIZE_Y, nx, ny, global_material, factory)

    control_nodes = []
    max_node_id = max(node.id for node in model.nodes)

    k_stiffness = 1e12
    k_stiffnessY = 1e7

    # Расчет общей вертикальной силы на 2D модель (модель единичной толщины)
    Fy_2D_total = -Fy0 / THICKNESS

    for node in list(model.nodes):
        # Нижняя грань - жестко защемлена
        if abs(node.coords[1] - 0.0) < 1e-6:
            model.add_bc(node, 0, 0.0)
            model.add_bc(node, 1, 0.0)

        # Верхняя грань - передаем нагрузку
        elif abs(node.coords[1] - SIZE_Y) < 1e-6:
            if abs(node.coords[0] - 0.0) < 1e-6 or abs(node.coords[0] - SIZE_X) < 1e-6:
                nodal_force = Fy_2D_total / (2 * nx)
            else:
                nodal_force = Fy_2D_total / nx

            # Вертикальное обжатие (Не пропорциональное)
            model.add_load(node, 1, nodal_force)
            model.nodal_loads[-1].is_proportional = False

            # Добавление пружины и горизонтального перемещения
            max_node_id += 1
            ctrl_node = Node(max_node_id, node.coords.copy())
            model.nodes.append(ctrl_node)
            control_nodes.append(ctrl_node)

            spring = SpringElement2D(nodes=[node, ctrl_node], kx=k_stiffness, ky=k_stiffnessY)
            model.elements.append(spring)

            # Горизонтальный сдвиг (Пропорциональный целевому перемещению)
            model.add_bc(ctrl_node, 0, TARGET_DISP_X)
            model.bcs[-1].is_proportional = True

    print(f"Добавлено {len(control_nodes)} пружинных элементов.")
    print(f"Сетка: {len(model.nodes)} узлов, {len(model.elements)} элементов.")

    # ГЕНЕРАЦИЯ ТОЧНОЙ ЦИКЛИЧЕСКОЙ ИСТОРИИ НАГРУЖЕНИЯ
    max_disp_mm = TARGET_DISP_X * 1000
    load_factors = generate_exact_cyclic_load_factors(
        amplitudes_mm=amplitudes_mm,
        max_disp_mm=max_disp_mm,
        steps_per_ramp=10,
        cycles_per_amp=2
    )

    control = StagedNRControl2D(
        model=model,
        track_nodes=control_nodes,
        load_factors=load_factors,
        track_dof=0,
        max_iter=15,
        tol=5e-4
    )

    print(f"Запуск решателя. Будет выполнено шагов: {len(load_factors)}")
    control.solve()

    # Извлечение данных для графика
    ux_mm = [u * 1000 for u in control.history_U]
    fx_kN = [f * THICKNESS / 1000 for f in control.history_F]

    # Построение графика гистерезиса
    plt.figure(figsize=(10, 8))
    plt.plot(ux_mm, fx_kN, color='black', linewidth=1.2, label=f'Численный цикл. ({wall_type} Wall)')

    # Добавление ориентировочных линий пиковой нагрузки
    plt.axhline(expected_peak, color='red', linestyle='--', alpha=0.5,
                label=f'Пик из эксперимента (~{expected_peak} кН)')
    plt.axhline(-expected_peak, color='red', linestyle='--', alpha=0.5)

    plt.title(f"МКЭ 2D: Циклический сдвиг (Стена: {wall_type}, Сжатие = {COMPRESSIVE_PRESSURE / 1e6} МПа)", fontsize=13)
    plt.xlabel("Горизонтальное смещение (мм)", fontsize=12)
    plt.ylabel("Горизонтальная сила (кН)", fontsize=12)

    limit_x = TARGET_DISP_X * 1000 * 1.1
    limit_y = expected_peak * 1.3
    plt.xlim(-limit_x, limit_x)
    plt.ylim(-limit_y, limit_y)

    plt.axhline(0, color='gray', linewidth=0.8)
    plt.axvline(0, color='gray', linewidth=0.8)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Вы можете выбрать "Low" (короткая стена, Fig. 4) или "High" (высокая стена, Fig. 5)
    run_cyclic_masonry_wall_test("Low")