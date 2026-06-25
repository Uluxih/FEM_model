import numpy as np
import matplotlib.pyplot as plt

# Импорт базовых классов МКЭ и генератора сетки
from FEM.Abstract.Structure_Level import Node, FEModel
from FEM.Abstract.Integration_Point_Level import Material
from FEM.Element_Level.Shape4NodeQuadrilateral import QUAD4Factory  # Используем 2D элемент

# Импорт пружинного элемента (убедитесь, что файл создан согласно предыдущему шагу)
from FEM.Element_Level.SpringElement2D import SpringElement2D

# Импорт моделей материала
from FEM.Integration_Point_Level.UbiquitousJointModel2D import UbiquitousJointModel2D
import FEM.Integration_Point_Level.CriticalPlane.material as cp_mt

# Импорт решателя
from FEM.Structure_Level.StagedNRControl2D import StagedNRControl2D
from FEM.Structure_Level.DisplacementControlNR2D import DisplacementControlNR2D

modelFEM = UbiquitousJointModel2D


class JointedMaterial(Material):
    def __init__(self, E, nu, joint_params):
        super().__init__(E, nu)
        self.joint_params = joint_params


def generate_rect_mesh(Lx, Ly, nx, ny, material, factory):
    """Генерация 2D прямоугольной сетки"""
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


def run_masonry_wall_test(sample_name="J4D"):
    print(f"=== ТЕСТ: Сдвиговая стена 2D (Валидация по образцу {sample_name}) ===")

    # Словарь экспериментальных данных
    exp_data = {
        "J4D": {"Fy0": 30_000, "peak_Fx": 51},
        "J6D": {"Fy0": 120_000, "peak_Fx": 72},
        "J7D": {"Fy0": 210_000, "peak_Fx": 97}
    }

    if sample_name not in exp_data:
        raise ValueError(f"Неизвестный образец: {sample_name}. Выберите J4D, J6D или J7D.")

    Fy0 = exp_data[sample_name]["Fy0"]
    expected_peak = exp_data[sample_name]["peak_Fx"]

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
    A_matrix = cp_mt.load_tensor_from_string(tensor_data) * 1e12

    # Параметры материала
    MASONRY_E = 3500e6
    MASONRY_NU = 0.2
    MASONRY_MU = 0.6
    MASONRY_RP = 0.5e6
    MASONRY_RC = 12.0e6

    cp_material = cp_mt.Material(
        mu=MASONRY_MU,
        A_tensor=A_matrix,
        Rpx=MASONRY_RP, Rpy=MASONRY_RP, Rpz=MASONRY_RP,
        Rcx=MASONRY_RC, Rcy=MASONRY_RC, Rcz=MASONRY_RC
    )

    nq = 20
    SIZE_X, SIZE_Y = 1.0, 1.0
    THICKNESS = 0.1
    nx, ny = nq, nq
    area_el = (SIZE_X / nx) * (SIZE_Y / ny)
    char_len = (area_el) ** 0.5
    # char_len=1
    n=100
    joint_params = {
        'phi': 0030.0,
        'psi': 0.0,
        'phi_r': 001.0,
        'cp_material': cp_material,
        'l_c': char_len,
        'Gf_t': 0.05*n,
        'Gf_c': 00.80*n*10,
        'Gf_s': 00.80*n*10,
        'a_t': 000.00,
        'a_s': 00.00,
        'mu': 0.10,
        'fcr_over_fc': 0.10,
        'force_horizontal': True
    }

    global_material = JointedMaterial(E=MASONRY_E, nu=MASONRY_NU, joint_params=joint_params)
    factory = QUAD4Factory()
    model = generate_rect_mesh(SIZE_X, SIZE_Y, nx, ny, global_material, factory)

    # =========================================================================
    # ДОБАВЛЕНИЕ ПРУЖИННЫХ ЭЛЕМЕНТОВ
    # =========================================================================

    control_nodes = []

    # Находим максимальный ID узла, чтобы создавать новые без конфликтов
    max_node_id = max(n.id for n in model.nodes)

    # Жесткость пружины. Должна быть достаточно большой, чтобы не вносить
    # паразитных упругих деформаций, но служить стабилизатором.
    # Значение 1e10 Н/м обычно хорошо подходит для бетона/кладки.
    k_stiffness = 1e12
    k_stiffnessY = 1e10

    # Расчет вертикального смещения
    area = SIZE_X * THICKNESS
    stress_y = Fy0 / area
    strain_y = stress_y / MASONRY_E
    PRE_COMPRESSION_DISP_Y = -strain_y * SIZE_Y*30
    PRE_COMPRESSION_DISP_Y = -0.0005
    TARGET_DISP_X = 0.004


    # Итерируемся по КОПИИ списка узлов, так как мы будем добавлять новые узлы в цикле
    for node in list(model.nodes):
        # Нижняя грань (y = 0) - жестко защемлена
        if abs(node.coords[1] - 0.0) < 1e-6:
            model.add_bc(node, 0, 0.0)  # X
            model.add_bc(node, 1, 0.0)  # Y

        # Верхняя грань (y = SIZE_Y) - передаем нагрузку через пружину
        elif abs(node.coords[1] - SIZE_Y) < 1e-6:
            # 1. Создаем контрольный узел в тех же координатах
            max_node_id += 1
            ctrl_node = Node(max_node_id, node.coords.copy())
            model.nodes.append(ctrl_node)
            control_nodes.append(ctrl_node)

            # 2. Создаем пружинный элемент, связывающий верх стены и контрольный узел
            spring = SpringElement2D(nodes=[node, ctrl_node], kx=k_stiffness, ky=k_stiffnessY)
            model.elements.append(spring)

            # 3. Прикладываем граничные условия к КОНТРОЛЬНОМУ узлу
            # Горизонтальный сдвиг (Пропорциональный)
            model.add_bc(ctrl_node, 0, TARGET_DISP_X)
            model.bcs[-1].is_proportional = True

            # Вертикальное обжатие (Не пропорциональное, прикладывается сразу)
            model.add_bc(ctrl_node, 1, PRE_COMPRESSION_DISP_Y)
            model.bcs[-1].is_proportional = False

            # # СДЕЛАЙТЕ ТАК:
            # # Распределяем общую силу Fy0 поровну между всеми верхними контрольными узлами
            # num_top_nodes = nx + 1
            # force_per_node = -Fy0 / num_top_nodes

            # Прикладываем силу (is_proportional = False означает, что она действует на 100% с первого шага)
            # model.add_load(ctrl_node, 1, force_per_node)

            # model.nodal_loads[-1].is_proportional = False

    print(f"Добавлено {len(control_nodes)} пружинных элементов.")
    print(f"Сетка обновлена: {len(model.nodes)} узлов, {len(model.elements)} элементов.")

    # 50 шагов нагружения до 1.0 (100% от TARGET_DISP_X)
    load_factors = np.linspace(0, 1.0, 50)[1:]

    control = StagedNRControl2D(
        model=model,
        track_nodes=control_nodes,  # Отслеживаем реакцию на контрольных узлах пружин
        load_factors=load_factors,
        track_dof=0,  # Отслеживаем реакцию по оси X
        max_iter=200,
        tol=1e-4
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

    plt.xlim(0, 6.0)
    plt.ylim(0, max(max(fx_kN) * 1.2, expected_peak * 1.2) if fx_kN else expected_peak * 1.2)

    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_masonry_wall_test("J4D")
