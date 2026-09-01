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


def generate_rect_mesh_with_opening(Lx, Ly, nx, ny, material, factory,
                                     opening_x_range, opening_y_range):
    """
    Генерация 2D прямоугольной сетки в плоскости X-Y с прямоугольным проемом
    в произвольной позиции (аналог стены с проемом, см. Рисунок 33).

    opening_x_range, opening_y_range: кортежи (x0, x1) и (y0, y1) в метрах, задающие
    границы проема в пределах стены [0, Lx] x [0, Ly]. Границы округляются
    до ближайших линий сетки nx x ny; ширина/высота проема в ячейках
    выдерживаются точно (а не теряется из-за округления обеих границ независимо).

    Оставшиеся по краям элементы формируют простенки/перемычку и пилястры.
    Узлы, не используемые ни одним элементом (полностью внутри проема),
    в модель не добавляются, чтобы не оставлять "висящих" вырожденных
    степеней свободы.
    """
    model = FEModel()
    model.materials.append(material)

    dx = Lx / nx
    dy = Ly / ny

    x0, x1 = opening_x_range
    y0, y1 = opening_y_range

    # Индексы элементов, образующих проем (левый/нижний край округляется до сетки,
    # ширина/высота в ячейках выдерживается точно)
    i_lo = round(x0 / dx)
    i_hi = i_lo + round((x1 - x0) / dx) - 1
    j_lo = round(y0 / dy)
    j_hi = j_lo + round((y1 - y0) / dy) - 1

    def is_opening_cell(i, j):
        return i_lo <= i <= i_hi and j_lo <= j <= j_hi

    # Определяем активные элементы (все, кроме проема) и узлы, которые они используют
    active_elements_ij = []
    active_nodes_ij = set()
    for j in range(ny):
        for i in range(nx):
            if is_opening_cell(i, j):
                continue
            active_elements_ij.append((i, j))
            active_nodes_ij.update([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])

    node_id = 0
    nodes_dict = {}
    for (i, j) in sorted(active_nodes_ij, key=lambda t: (t[1], t[0])):
        n = Node(node_id, [i * dx, j * dy])
        model.nodes.append(n)
        nodes_dict[(i, j)] = n
        node_id += 1

    for (i, j) in active_elements_ij:
        el_nodes = [
            nodes_dict[(i, j)],
            nodes_dict[(i + 1, j)],
            nodes_dict[(i + 1, j + 1)],
            nodes_dict[(i, j + 1)]
        ]
        el = factory.create_element(el_nodes, material, constitutive_class=modelFEM)
        model.elements.append(el)
    return model


def run_masonry_wall_opening_test(sample_name="J4D", nx=15, ny=15, k_stiffness=1e10,
                                  k_beam=1e13, scale_kx_by_nodes=False, target_disp=0.025,
                                  n_shear=50, n_preload=21):
    print(f"=== ТЕСТ: Сдвиговая стена 2D с проемом в осях X-Y (Валидация по образцу {sample_name}) ===")
    print(f"Сетка: {nx}x{ny}, kx={k_stiffness:.2e}, k_beam={k_beam:.2e}, scale_kx_by_nodes={scale_kx_by_nodes}")

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
1777777777777.777588 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
0.000000 1959999987937.882080 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
0.000000 0.000000 1959999987917.281494 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
0.000000 0.000000 0.000000 1439999991091.858643 0.000000 0.000000 0.000000 0.000000 0.000000
0.000000 0.000000 0.000000 0.000000 1777777777777.777588 0.000000 0.000000 0.000000 0.000000
0.000000 0.000000 0.000000 0.000000 0.000000 1439999990978.519775 0.000000 0.000000 0.000000
0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 1959999987877.562744 0.000000 0.000000
0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 1959999987877.562988 0.000000
0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 1777777777777.777588
                    """
    A_matrix = cp_mt.load_tensor_from_string(tensor_data)

    # Параметры материала
    MASONRY_E = 3500e6
    MASONRY_NU = 0.2
    MASONRY_MU = 0.577
    MASONRY_RP = 001.00e6
    MASONRY_RC = 12.0e6

    cp_material = cp_mt.Material(
        mu=MASONRY_MU,
        A_tensor=A_matrix,
        Rpx=MASONRY_RP*01.0, Rpy=MASONRY_RP * 0.5, Rpz=MASONRY_RP * 01.0,
        Rcx=MASONRY_RC*0.75, Rcy=MASONRY_RC * 1.0, Rcz=MASONRY_RC * 1.0
    )

    SIZE_X, SIZE_Y = 1.0, 1.0
    THICKNESS = 0.10

    # Геометрия проема: квадрат 0.25х0.25 м, расположенный на расстоянии 0.35 м
    # от правого края стены и по центру высоты стены.
    OPENING_WIDTH = 0.25
    OPENING_HEIGHT = 0.35
    OPENING_DIST_FROM_RIGHT = 0.35

    opening_x1 = SIZE_X - OPENING_DIST_FROM_RIGHT
    opening_x0 = opening_x1 - OPENING_WIDTH
    opening_y0 = (SIZE_Y - OPENING_HEIGHT) / 2.0
    opening_y1 = opening_y0 + OPENING_HEIGHT

    # Сетка выбрана прямоугольной (nx x ny), чтобы границы проема точно попадали на линии
    # сетки: dx = 0.05 м (кратно 0.35 и 0.25 м), dy = 0.0625 м (кратно 0.375 и 0.25 м),
    # что обеспечивает точное горизонтальное положение проема и точное центрирование по высоте.
    area_el = (SIZE_X / nx) * (SIZE_Y / ny)
    char_len = (area_el) ** 0.5
    n = 1

    # Параметры ослабленной плоскости (трещины)
    joint_params = {
        # 'tangent_mode': 'numerical',
        'tangent_mode': 'secant',
        'phi': 030.0,
        'psi': 0.0,
        'phi_r': 0.0,
        'cp_material': cp_material,
        'l_c': char_len,
        'Gf_t': 400,
        'Gf_c': 3000,
        'Gf_s': 800,
        'a_t': 0.100,
        'a_s': 0.00,
        'mu': 0.20,  # доля остаточной деформации трещины (damage-plasticity split), НЕ трение
        'fcr_over_fc': 0.10
    }

    # Параметры целого материала (матрицы) для Друкера-Прагера
    # Вы можете корректировать их в зависимости от прочности вашей кладки на сжатие/срез
    matrix_params = {
        'c': 001.400e6,  # Сцепление матрицы (Па)
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
    # Сетка стены с прямоугольным проемом 0.25x0.25 м, смещенным к правому краю стены
    model = generate_rect_mesh_with_opening(SIZE_X, SIZE_Y, nx, ny, global_material, factory,
                                             opening_x_range=(opening_x0, opening_x1),
                                             opening_y_range=(opening_y0, opening_y1))

    # =========================================================================
    # ДОБАВЛЕНИЕ ПРУЖИННЫХ ЭЛЕМЕНТОВ И НАГРУЗОК
    # =========================================================================

    control_nodes = []
    preload_loads = []
    shear_bcs = []

    # Находим максимальный ID узла, чтобы создавать новые без конфликтов
    max_node_id = max(node.id for node in model.nodes)

    # Жесткость пружины
    k_stiffnessY = 1e8

    TARGET_DISP_X = target_disp

    # Расчет общей вертикальной силы на 2D модель (модель единичной толщины)
    # Знак минус, так как сила направлена вниз (сжатие вдоль Y)
    Fy_2D_total = -Fy0 / THICKNESS

    # Число верхних узлов (каждому будет своя пружина) — для опционального масштабирования kx
    n_top = nx + 1
    kx_spring = k_stiffness / n_top if scale_kx_by_nodes else k_stiffness

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
                k_node_y = k_stiffnessY / (2 * nx)
            else:
                nodal_force = Fy_2D_total / nx
                k_node_y = k_stiffnessY / nx

            # Вертикальное обжатие: первая, отдельная стадия нагружения.
            model.add_load(node, 1, nodal_force)
            model.nodal_loads[-1].is_proportional = True
            preload_loads.append(model.nodal_loads[-1])

            # --- 2. ДОБАВЛЕНИЕ ПРУЖИНЫ И ГОРИЗОНТАЛЬНОГО ПЕРЕМЕЩЕНИЯ ---
            max_node_id += 1
            ctrl_node = Node(max_node_id, node.coords.copy())
            model.nodes.append(ctrl_node)
            control_nodes.append(ctrl_node)

            # Пружина передает сдвиг и обеспечивает распределенное обжатие.
            spring = SpringElement2D(nodes=[node, ctrl_node], kx=kx_spring, ky=k_node_y)
            model.elements.append(spring)

            # Заземляем вертикальную степень свободы управляющего узла,
            # иначе вертикальная ветвь пружины кинематически не работает.
            model.add_bc(ctrl_node, 1, 0.0)

            # Горизонтальный сдвиг: вторая стадия, поэтому в первой он выключен.
            model.add_bc(ctrl_node, 0, TARGET_DISP_X)
            model.bcs[-1].is_proportional = True
            shear_bcs.append(model.bcs[-1])

    n_top = len(control_nodes)

    # Жесткая балка
    # одинаковое вертикальное перемещение, как при кинематическом закреплении.
    control_ids = {node.id for node in control_nodes}
    top_nodes = sorted(
        [node for node in model.nodes
         if abs(node.coords[1] - SIZE_Y) < 1e-6 and node.id not in control_ids],
        key=lambda node: node.coords[0]
    )
    for n1, n2 in zip(top_nodes[:-1], top_nodes[1:]):
        model.elements.append(SpringElement2D(nodes=[n1, n2], kx=k_beam, ky=k_beam))

    print(f"Добавлено {len(control_nodes)} пружинных элементов.")
    print(f"Добавлена жесткая балка: {len(top_nodes) - 1} связей между {len(top_nodes)} верхними узлами.")
    print(f"Сетка обновлена: {len(model.nodes)} узлов, {len(model.elements)} элементов.")

    # =====================================================================
    # СТАДИЯ 1. Плавное вертикальное предсжатие без горизонтального сдвига.
    # =====================================================================
    for bc in shear_bcs:
        bc.value = 0.0

    preload_factors = np.linspace(0.0, 1.0, n_preload)[1:]
    preload_control = StagedNRControl2D(
        model=model,
        track_nodes=[],
        load_factors=preload_factors,
        max_iter=50,
        tol=1e-2,
        enable_logging=False
    )
    preload_control.solve()

    # =====================================================================
    # СТАДИЯ 2. Горизонтальный сдвиг при уже приложенном постоянном Fy0.
    # Нагрузку Fy0 переводим в непропорциональную, чтобы она сохранялась
    # полной на каждом шаге второй стадии.
    # =====================================================================
    for load in preload_loads:
        load.is_proportional = False
    for bc in shear_bcs:
        bc.value = TARGET_DISP_X

    shear_factors = np.linspace(0.0, 1.0, n_shear)[1:]
    control = StagedNRControl2D(
        model=model,
        track_nodes=control_nodes,
        load_factors=shear_factors,
        track_dof=0,  # Отслеживаем реакцию по оси X
        max_iter=50,
        tol=5e-2,
        enable_logging=False
    )
    control.solve(initial_displacements=np.concatenate(
        [node.displacements for node in model.nodes]
    ))

    # Извлечение данных для графика второй стадии — горизонтального сдвига.
    ux_mm = [u * 1000 for u in control.history_U]

    # Умножаем силу на фактическую толщину стены
    fx_kN = [f * THICKNESS / 1000 for f in control.history_F]

    # Построение графиков
    plt.figure(figsize=(10, 6))
    plt.plot(ux_mm, fx_kN, color='#1f77b4', linewidth=2, label=f'Численное мод. 2D Fx ({sample_name}, с проемом)')

    plt.axhline(expected_peak, color='red', linestyle='--', alpha=0.7,
                label=f'Эксперимент {sample_name} (макс ~{expected_peak} кН)')

    plt.title(f"МКЭ 2D Тест: Сдвиг кирпичной стены с проемом (Образец {sample_name}, Fy0 = {Fy0 / 1000} кН)",
              fontsize=13)
    plt.xlabel("Горизонтальное смещение верха стены (мм)", fontsize=12)
    plt.ylabel("Горизонтальная сдвигающая сила Fx (кН)", fontsize=12)

    plt.xlim(0, 20.0)
    plt.ylim(0, max(max(fx_kN) * 1.2, expected_peak * 1.2) if fx_kN else expected_peak * 1.2)

    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    return ux_mm, fx_kN


if __name__ == "__main__":
    run_masonry_wall_opening_test("J4D")
