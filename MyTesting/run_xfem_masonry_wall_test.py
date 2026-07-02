import numpy as np
import matplotlib.pyplot as plt

# Импорт базовых классов МКЭ и генератора сетки
from FEM.Abstract.Structure_Level import Node, FEModel
from FEM.Abstract.Integration_Point_Level import Material

# ПРЕДПОЛАГАЕТСЯ: У вас есть фабрика элементов XFEM, которая создает элементы с поддержкой обогащения
from FEM.Element_Level.XFEMQUAD4 import XFEMQUAD4Factory
from FEM.Element_Level.SpringElement2D import SpringElement2D

# Импорт новых XFEM моделей (которые мы написали ранее)
from FEM.Integration_Point_Level.XFEM_Models.DruckerPragerMatrix2D import DruckerPragerMatrix2D
from FEM.Integration_Point_Level.XFEM_Models.XFEM_CohesiveDamagePlasticity2D import XFEM_CohesiveDamagePlasticity2D
import FEM.Integration_Point_Level.CriticalPlane.material as cp_mt
from FEM.Integration_Point_Level.CriticalPlane.tensor import StressTensor
from FEM.Integration_Point_Level.CriticalPlane.criterion import find_critical_plane_shear, find_critical_plane_tensile

# Импорт нового XFEM решателя
from FEM.Structure_Level.StagedXFEMControl2D import StagedXFEMControl2D


class XFEM_JointedMaterial(Material):
    """
    Контейнер материала для XFEM.
    Хранит параметры как для сплошной среды (Bulk), так и для трещины (Cohesive).
    Фабрика элементов XFEM будет использовать этот класс для инициализации
    DruckerPragerMatrix2D в объемных точках Гаусса и XFEM_CohesiveDamagePlasticity2D на трещинах.
    """

    def __init__(self, E, nu, matrix_params, cohesive_params):
        super().__init__(E, nu)
        self.matrix_params = matrix_params
        self.cohesive_params = cohesive_params


def generate_xfem_rect_mesh(Lx, Ly, nx, ny, material, factory):
    """Генерация 2D прямоугольной сетки для XFEM"""
    # Предполагается, что FEModelXFEM умеет работать с Level Set и функциями обогащения
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
            # Фабрика должна сама инициализировать объемные точки Гаусса с DruckerPragerMatrix2D
            el = factory.create_element(el_nodes, material)
            model.elements.append(el)
    return model


def run_xfem_masonry_wall_test(sample_name="J4D"):
    print(f"=== ТЕСТ XFEM: Сдвиговая стена 2D в осях X-Y (Валидация по образцу {sample_name}) ===")

    # Словарь экспериментальных данных
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

    # Параметры материала
    MASONRY_E = 3500e6
    MASONRY_NU = 0.2
    MASONRY_RP = 0.0175e6
    MASONRY_RC = 12.000e6

    # 1. Параметры сплошной среды (для DruckerPragerMatrix2D)
    matrix_params = {
        'c': 1110.05e6,  # Сцепление матрицы (Па)
        'phi': 30.0,  # Угол внутреннего трения матрицы (градусы)
        'psi': 0.0  # Угол дилатансии матрицы (градусы)
    }

    n = 500.25  # Масштабный коэффициент для энергии

    # 2. Параметры интерфейса (для XFEM_CohesiveDamagePlasticity2D)
    # ВАЖНО: l_c больше не используется. Gf задаются строго на единицу площади (Дж/м2).
    # Добавлены штрафные жесткости K_n и K_s.
    cohesive_params = {
        'K_n': MASONRY_E * 2e2,  # Штрафная нормальная жесткость (очень большая)
        'K_s': MASONRY_E * 2e2,  # Штрафная сдвиговая жесткость
        'f_t': MASONRY_RP,
        'f_c': MASONRY_RC,
        'c': 50e6,  # Когезия трещины
        'phi': 30.0,
        'psi': 0.0,
        'phi_r': 0.0,
        'Gf_t': 200 * n,
        'Gf_c': 5000 * n,
        'Gf_s': 18000 * n,
        'a_t': 1.00,
        'a_s': 1.00,
        'fcr_over_fc': 0.10
    }


    global_material = XFEM_JointedMaterial(
        E=MASONRY_E,
        nu=MASONRY_NU,
        matrix_params=matrix_params,
        cohesive_params=cohesive_params
    )

    nq = 2
    SIZE_X, SIZE_Y = 1.0, 1.0
    THICKNESS = 0.10
    nx, ny = nq, nq

    factory = XFEMQUAD4Factory()
    model = generate_xfem_rect_mesh(SIZE_X, SIZE_Y, nx, ny, global_material, factory)

    # =========================================================================
    # ДОБАВЛЕНИЕ ПРУЖИННЫХ ЭЛЕМЕНТОВ И НАГРУЗОК
    # =========================================================================

    control_nodes = []
    max_node_id = max(node.id for node in model.nodes)

    k_stiffness = 1e12
    k_stiffnessY = 1e7
    TARGET_DISP_X = 0.001

    Fy_2D_total = -Fy0 / THICKNESS

    for node in list(model.nodes):
        if abs(node.coords[1] - 0.0) < 1e-6:
            model.add_bc(node, 0, 0.0)
            model.add_bc(node, 1, 0.0)

        elif abs(node.coords[1] - SIZE_Y) < 1e-6:
            if abs(node.coords[0] - 0.0) < 1e-6 or abs(node.coords[0] - SIZE_X) < 1e-6:
                nodal_force = Fy_2D_total / (2 * nx)
            else:
                nodal_force = Fy_2D_total / nx

            model.add_load(node, 1, nodal_force)
            model.nodal_loads[-1].is_proportional = False

            max_node_id += 1
            ctrl_node = Node(max_node_id, node.coords.copy())
            model.nodes.append(ctrl_node)
            control_nodes.append(ctrl_node)

            spring = SpringElement2D(nodes=[node, ctrl_node], kx=k_stiffness, ky=k_stiffnessY)
            model.elements.append(spring)

            model.add_bc(ctrl_node, 0, TARGET_DISP_X)
            model.bcs[-1].is_proportional = True

    print(f"Добавлено {len(control_nodes)} пружинных элементов.")
    print(f"Сетка обновлена: {len(model.nodes)} узлов, {len(model.elements)} элементов.")

    # 50 шагов нагружения
    load_factors = np.linspace(0, 1.0, 50)[1:]

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
    A_matrix = cp_mt.load_tensor_from_string(tensor_data) * 1e12 * 300.0

    # Параметры материала
    MASONRY_E = 3500e6
    MASONRY_NU = 0.2
    MASONRY_MU = 0.577
    MASONRY_RP = 0.09175e6
    MASONRY_RC = 7.000e6

    cp_material = cp_mt.Material(
        mu=MASONRY_MU,
        A_tensor=A_matrix,
        Rpx=MASONRY_RP * 1.0, Rpy=MASONRY_RP * 0.7, Rpz=MASONRY_RP * 1.0,
        Rcx=MASONRY_RC * 0.75, Rcy=MASONRY_RC * 1.0, Rcz=MASONRY_RC * 1.0
    )

    # =========================================================================
    # ЛОГИКА ЗАРОЖДЕНИЯ ТРЕЩИНЫ ДЛЯ XFEM
    # =========================================================================
    def custom_crack_check():
        crack_propagated = False
        max_dof = 0

        # Находим максимальный текущий номер степени свободы (DOF)
        for n in model.nodes:
            max_dof = max(max_dof, max(n.dofs))
            if hasattr(n, 'xfem_heaviside_dofs'):
                max_dof = max(max_dof, max(n.xfem_heaviside_dofs))

        global_max_f = -1e9  # Для отладки

        for el in model.elements:
            if getattr(el, 'is_spring', False) or el.is_enriched:
                continue

            el_max_f = -1e9
            best_n = None

            # Проверяем напряжения во ВСЕХ точках Гаусса элемента
            for ip in el.get_bulk_integration_points():
                stress = ip.constitutive_model.stress
                st = StressTensor(stress[0], stress[1], 0.0, stress[2], 0.0, 0.0)

                f_t, n_t, u_t = find_critical_plane_tensile(st, cp_material)
                f_s, n_s, u_s = find_critical_plane_shear(st, cp_material)

                current_max = max(f_t, f_s)
                global_max_f = max(global_max_f, current_max)

                if current_max > el_max_f:
                    el_max_f = current_max
                    best_n = n_t if f_t > f_s else n_s

            # Если предел прочности превышен
            if el_max_f > 1e-4:
                # ВАЖНО: Раз мы для теста режем элемент строго горизонтально,
                # его физическая нормаль должна быть строго вертикальна!
                forced_normal = np.array([0.0, 1.0])

                max_dof = el.cut_element(max_dof, (-1.0, 0.0), (1.0, 0.0), forced_normal)
                crack_propagated = True
                print(f"  [!!!] ЭЛЕМЕНТ РАЗРУШЕН! Макс. критерий: {el_max_f:.2e}")

                break

        print(f"  [Инфо] Максимальное значение критерия трещины на шаге: {global_max_f:.2e} (нужно > 0)")
        return crack_propagated

    # Привязываем функцию к модели
    model.check_and_propagate_crack = custom_crack_check
    # ИСПОЛЬЗУЕМ XFEM РЕШАТЕЛЬ
    control = StagedXFEMControl2D(
        model=model,
        track_nodes=control_nodes,
        load_factors=load_factors,
        track_dof=0,
        max_iter=2000,
        tol=5e-1
    )

    control.solve()

    # Извлечение данных для графика
    ux_mm = [u * 1000 for u in control.history_U]
    fx_kN = [f * THICKNESS / 1000 for f in control.history_F]

    # Построение графиков
    plt.figure(figsize=(10, 6))
    plt.plot(ux_mm, fx_kN, color='#ff7f0e', linewidth=2, label=f'Численное XFEM мод. Fx ({sample_name})')

    plt.axhline(expected_peak, color='red', linestyle='--', alpha=0.7,
                label=f'Эксперимент {sample_name} (макс ~{expected_peak} кН)')

    plt.title(f"XFEM 2D Тест: Сдвиг кирпичной стены (Образец {sample_name}, Fy0 = {Fy0 / 1000} кН)", fontsize=13)
    plt.xlabel("Горизонтальное смещение верха стены (мм)", fontsize=12)
    plt.ylabel("Горизонтальная сдвигающая сила Fx (кН)", fontsize=12)

    plt.xlim(0, 10.0)
    plt.ylim(0, max(max(fx_kN) * 1.2, expected_peak * 1.2) if fx_kN else expected_peak * 1.2)

    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_xfem_masonry_wall_test("J7D")