import numpy as np
import matplotlib.pyplot as plt

# Импорт базовых классов МКЭ и генератора сетки
from FEM.Abstract.Structure_Level import Node, FEModel
from FEM.Abstract.Integration_Point_Level import Material
from FEM.Element_Level.Shape8NodeHexahedron import HEX8Factory

# Импорт моделей материала
from FEM.Integration_Point_Level.UbiquitousJointModel3D import UbiquitousJointModel3D
import FEM.Integration_Point_Level.CriticalPlane.material as cp_mt

# Импорт нового решателя!
from FEM.Structure_Level.StagedNRControl import StagedNRControl

modelFEM = UbiquitousJointModel3D


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


def run_masonry_wall_test(sample_name="J4D"):
    print(f"=== ТЕСТ: Сдвиговая стена (Валидация по образцу {sample_name}) ===")

    # Словарь экспериментальных данных (Нагрузка обжатия в Ньютонах, Ожидаемый пик сдвига в кН)
    exp_data = {
        "J4D": {"Fy0": 30_000, "peak_Fx": 51},
        "J6D": {"Fy0": 120_000, "peak_Fx": 72},
        "J7D": {"Fy0": 210_000, "peak_Fx": 97}
    }

    if sample_name not in exp_data:
        raise ValueError(f"Неизвестный образец: {sample_name}. Выберите J4D, J6D или J7D.")

    Fy0 = exp_data[sample_name]["Fy0"]
    expected_peak = exp_data[sample_name]["peak_Fx"]

    tensor_data_str = """
    01.001 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
    0.000000 01.001 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
    0.000000 0.000000 01.001 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
    0.000000 0.000000 0.000000 01.001 0.000000 0.000000 0.000000 0.000000 0.000000
    0.000000 0.000000 0.000000 0.000000 01.001 0.000000 0.000000 0.000000 0.000000
    0.000000 0.000000 0.000000 0.000000 0.000000 01.001 0.000000 0.000000 0.000000
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 01.001 0.000000 0.000000
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 01.001 0.000000
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.0001000 01.001
    """
    A_matrix = cp_mt.load_tensor_from_string(tensor_data_str) * 3e15

    # Параметры из Таблицы 13 (Строго в СИ)555
    MASONRY_E = 3500e6  # 3500 МПа
    MASONRY_NU = 0.2
    MASONRY_MU = 0.6  # Трение
    MASONRY_RP = 0.5e6  # Rtn = 0.5 МПа
    MASONRY_RC = 12.0e6  # Rcn = 12 МПа

    cp_material = cp_mt.Material(
        mu=MASONRY_MU,
        A_tensor=A_matrix,
        Rpx=MASONRY_RP, Rpy=MASONRY_RP, Rpz=MASONRY_RP,
        Rcx=MASONRY_RC, Rcy=MASONRY_RC, Rcz=MASONRY_RC
    )
    nq=1
    SIZE_X, SIZE_Y, SIZE_Z = 1.0, 0.1, 1.0
    nx, ny, nz = nq, 1, nq

    vol_el = (SIZE_X / nx) * (SIZE_Y / ny) * (SIZE_Z / nz)
    char_len = vol_el ** (1 / nq)

    joint_params = {
        'phi': 20.0,
        'psi': 0.0,
        'phi_r': 05.0,
        'cp_material': cp_material,
        'l_c': char_len,
        'Gf_t': 20.0*1e0,  # Gtn = 200 Дж/м2
        'Gf_c': 200.0*1e0,  # Gcn = 2000 Дж/м2
        'Gf_s': 200.0*1e0,
        'a_t': 00.0,
        'a_s': 00.0,
        'mu': 0.0,
        'fcr_over_fc': 0.0,
    }

    global_material = JointedMaterial(E=MASONRY_E, nu=MASONRY_NU, joint_params=joint_params)
    factory = HEX8Factory()
    model = generate_block_mesh(SIZE_X, SIZE_Y, SIZE_Z, nx, ny, nz, global_material, factory)
    print(f"Сетка создана: {len(model.nodes)} узлов, {len(model.elements)} элементов.")

    top_nodes = []

    # Расчет вертикального смещения для достижения нужного Fy0
    area = SIZE_X * SIZE_Y
    stress_z = Fy0 / area
    strain_z = stress_z / MASONRY_E
    PRE_COMPRESSION_DISP_Z = -strain_z * SIZE_Z

    TARGET_DISP_X = 0.004  # 4 мм сдвига

    for node in model.nodes:
        if abs(node.coords[2] - 0.0) < 1e-6:
            model.add_bc(node, 0, 0.0)
            model.add_bc(node, 1, 0.0)
            model.add_bc(node, 2, 0.0)

        elif abs(node.coords[2] - SIZE_Z) < 1e-6:
            top_nodes.append(node)

            # 1. Горизонтальный сдвиг (Пропорциональный, нарастает от 0 до TARGET_DISP_X)
            model.add_bc(node, 0, TARGET_DISP_X)
            # По умолчанию is_proportional = True, но можем указать явно для надежности
            model.bcs[-1].is_proportional = True

            # 2. Запрет смещения из плоскости
            model.add_bc(node, 1, 0.0)

            # 3. Вертикальное обжатие (НЕ пропорциональное, прикладывается сразу полностью)
            model.add_bc(node, 2, PRE_COMPRESSION_DISP_Z)
            model.bcs[-1].is_proportional = False  # <--- Ключевое отличие для нового решателя!

    # 100 шагов нагружения до 1.0 (100% от TARGET_DISP_X)
    load_factors = np.linspace(0, 1.0, 20)[1:]

    # Используем новый решатель
    control = StagedNRControl(
        model=model,
        track_nodes=top_nodes,
        load_factors=load_factors,
        track_dof=0,  # Отслеживаем реакцию по оси X
        max_iter=150,
        tol=5e-3
    )

    control.solve()

    # Извлечение данных для графика
    ux_mm = [u * 1000 for u in control.history_U]
    fx_kN = [f / 1000 for f in control.history_F]

    # Построение графиков
    plt.figure(figsize=(10, 6))
    plt.plot(ux_mm, fx_kN, color='#1f77b4', linewidth=2, label=f'Численное мод. Fx ({sample_name})')

    plt.axhline(expected_peak, color='red', linestyle='--', alpha=0.7,
                label=f'Эксперимент {sample_name} (макс ~{expected_peak} кН)')

    plt.title(f"МКЭ Тест: Сдвиг кирпичной стены (Образец {sample_name}, Fy0 = {Fy0 / 1000} кН)", fontsize=13)
    plt.xlabel("Горизонтальное смещение верха стены (мм)", fontsize=12)
    plt.ylabel("Горизонтальная сдвигающая сила Fx (кН)", fontsize=12)

    plt.xlim(0, 4.0)
    plt.ylim(0, max(max(fx_kN) * 1.2, expected_peak * 1.2) if fx_kN else expected_peak * 1.2)

    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Вы можете раскомментировать нужный образец
    run_masonry_wall_test("J4D")
    # run_masonry_wall_test("J6D")
    # run_masonry_wall_test("J7D")
