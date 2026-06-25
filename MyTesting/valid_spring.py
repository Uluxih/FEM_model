import numpy as np
import matplotlib.pyplot as plt

# Импорт базовых классов МКЭ и генератора сетки
from FEM.Abstract.Structure_Level import Node, FEModel
from FEM.Abstract.Integration_Point_Level import Material
from FEM.Element_Level.Shape8NodeHexahedron import HEX8Factory

# Импорт моделей материала
from FEM.Integration_Point_Level.UbiquitousJointModel3D import UbiquitousJointModel3D
from FEM.Integration_Point_Level.ElasticModel3D import ElasticModel3D
import FEM.Integration_Point_Level.CriticalPlane.material as cp_mt

# Импорт решателя и экспортера
from FEM.Structure_Level.StagedNRControl import StagedNRControl
from FEM.Structure_Level.VTKExporter import VTKExporter


class JointedMaterial(Material):
    def __init__(self, E, nu, joint_params):
        super().__init__(E, nu)
        self.joint_params = joint_params


def generate_block_mesh_with_beam(Lx, Ly, Lz, h_beam, nx, ny, nz, masonry_mat, beam_mat, factory):
    """
    Генерирует сетку стены (nz слоев) и добавляет сверху 1 слой упругой балки.
    """
    model = FEModel()
    model.materials.extend([masonry_mat, beam_mat])
    node_id = 0
    nodes_dict = {}

    # Узлы: nz слоев кладки + 1 слой балки = nz + 1 слоев элементов -> nz + 2 слоев узлов
    for k in range(nz + 2):
        for j in range(ny + 1):
            for i in range(nx + 1):
                # Если k <= nz, это стена. Если k == nz + 1, это верх балки.
                z_coord = k * (Lz / nz) if k <= nz else Lz + h_beam
                n = Node(node_id, [i * (Lx / nx), j * (Ly / ny), z_coord])
                model.nodes.append(n)
                nodes_dict[(i, j, k)] = n
                node_id += 1

    # Элементы
    for k in range(nz + 1):
        # Если это самый верхний слой (k == nz) -> используем упругую балку
        is_beam = (k == nz)
        current_mat = beam_mat if is_beam else masonry_mat
        current_model = ElasticModel3D if is_beam else UbiquitousJointModel3D

        for j in range(ny):
            for i in range(nx):
                el_nodes = [
                    nodes_dict[(i, j, k)], nodes_dict[(i + 1, j, k)],
                    nodes_dict[(i + 1, j + 1, k)], nodes_dict[(i, j + 1, k)],
                    nodes_dict[(i, j, k + 1)], nodes_dict[(i + 1, j, k + 1)],
                    nodes_dict[(i + 1, j + 1, k + 1)], nodes_dict[(i, j + 1, k + 1)]
                ]
                el = factory.create_element(el_nodes, current_mat, constitutive_class=current_model)
                model.elements.append(el)

    return model


def run_masonry_wall_test(sample_name="J4D"):
    print(f"=== ТЕСТ: Сдвиговая стена с упругой балкой (Валидация по образцу {sample_name}) ===")

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
    1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0
    0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0
    0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0
    0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0
    0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0
    """
    A_matrix = cp_mt.load_tensor_from_string(tensor_data_str) * 3e20

    # 1. ПАРАМЕТРЫ КЛАДКИ
    MASONRY_E = 3500e6  # 3500 МПа
    MASONRY_NU = 0.2
    MASONRY_MU = 0.6  # Трение
    MASONRY_RP = 0.5e6  # Rtn = 0.5 МПа
    MASONRY_RC = 12.0e6  # Rcn = 12 МПа

    # 2. ПАРАМЕТРЫ БАЛКИ (Сталь или жесткий бетон)
    BEAM_E = 200e9  # 200 ГПа
    BEAM_NU = 0.3
    BEAM_HEIGHT = 0.1  # Высота балки 10 см
    beam_material = Material(E=BEAM_E, nu=BEAM_NU)

    cp_material = cp_mt.Material(
        mu=MASONRY_MU,
        A_tensor=A_matrix,
        Rpx=MASONRY_RP, Rpy=MASONRY_RP, Rpz=MASONRY_RP,
        Rcx=MASONRY_RC, Rcy=MASONRY_RC, Rcz=MASONRY_RC
    )

    # Геометрия и сетка
    SIZE_X, SIZE_Y, SIZE_Z = 1.0, 0.1, 1.0

    # СГУЩЕНИЕ СЕТКИ: 4x1x4 элементов по стене (+ 1 слой балки сверху)
    nq = 1
    nx, ny, nz = nq, 1, nq

    vol_el = (SIZE_X / nx) * (SIZE_Y / ny) * (SIZE_Z / nz)
    char_len = vol_el ** (1.0 / 3.0)  # ИСПРАВЛЕННЫЙ КОРЕНЬ ДЛЯ 3D

    joint_params = {
        'phi': 20.0,
        'psi': 0.0,
        'phi_r': 0.0,
        'cp_material': cp_material,
        'l_c': char_len,
        'Gf_t': 200.0,  # УБРАН множитель 1e-1 (200 Дж/м2)
        'Gf_c': 2000.0,  # УВЕЛИЧЕНО для стабильности сжатия
        'Gf_s': 2000.0,  # УБРАН множитель 1e-1
        'a_t': 1.0,
        'a_s': 1.0,
        'mu': 0.0,
        'fcr_over_fc': 0.0,
    }

    masonry_material = JointedMaterial(E=MASONRY_E, nu=MASONRY_NU, joint_params=joint_params)
    factory = HEX8Factory()

    # Генерация сетки с балкой
    model = generate_block_mesh_with_beam(
        SIZE_X, SIZE_Y, SIZE_Z, BEAM_HEIGHT,
        nx, ny, nz, masonry_material, beam_material, factory
    )
    print(f"Сетка создана: {len(model.nodes)} узлов, {len(model.elements)} элементов.")

    top_nodes = []

    # Расчет вертикального смещения для достижения нужного Fy0
    area = SIZE_X * SIZE_Y
    stress_z = Fy0 / area
    strain_z = stress_z / MASONRY_E
    PRE_COMPRESSION_DISP_Z = -strain_z * SIZE_Z

    TARGET_DISP_X = 0.0043  # 4 мм сдвига

    for node in model.nodes:
        # Низ стены жестко закреплен
        if abs(node.coords[2] - 0.0) < 1e-6:
            model.add_bc(node, 0, 0.0)
            model.add_bc(node, 1, 0.0)
            model.add_bc(node, 2, 0.0)

        # Граничные условия прикладываем К ВЕРХУ БАЛКИ
        elif abs(node.coords[2] - (SIZE_Z + BEAM_HEIGHT)) < 1e-6:
            top_nodes.append(node)

            # 1. Горизонтальный сдвиг балки (Пропорциональный)
            model.add_bc(node, 0, TARGET_DISP_X)
            model.bcs[-1].is_proportional = True

            # 2. Запрет смещения из плоскости
            model.add_bc(node, 1, 0.0)

            # 3. Вертикальное обжатие через балку (прикладывается на первом шаге)
            model.add_bc(node, 2, PRE_COMPRESSION_DISP_Z)
            model.bcs[-1].is_proportional = False

    # УВЕЛИЧЕНО КОЛИЧЕСТВО ШАГОВ для стабильности разупрочнения
    load_factors = np.linspace(0, 1.0, 100)[1:]

    # Решатель
    control = StagedNRControl(
        model=model,
        track_nodes=top_nodes,
        load_factors=load_factors,
        track_dof=0,  # Отслеживаем реакцию по оси X
        max_iter=50,
        tol=1e-2  # Смягченный допуск для стабильности первых тестов
    )

    control.solve()

    # Экспорт результатов для визуализации в ParaView
    vtk_filename = f"Wall_{sample_name}_Results.vtu"
    VTKExporter.export(model, vtk_filename)
    print(f"Результаты сохранены в файл: {vtk_filename}")

    # Извлечение данных для графика
    ux_mm = [u * 1000 for u in control.history_U]
    fx_kN = [f / 1000 for f in control.history_F]

    # Построение графиков
    plt.figure(figsize=(10, 6))
    plt.plot(ux_mm, fx_kN, color='#1f77b4', linewidth=2, label=f'Численное мод. Fx ({sample_name})')

    plt.axhline(expected_peak, color='red', linestyle='--', alpha=0.7,
                label=f'Эксперимент {sample_name} (макс ~{expected_peak} кН)')

    plt.title(f"МКЭ Тест: Сдвиг кирпичной стены с балкой (Образец {sample_name}, Fy0 = {Fy0 / 1000} кН)", fontsize=13)
    plt.xlabel("Горизонтальное смещение верха балки (мм)", fontsize=12)
    plt.ylabel("Горизонтальная сдвигающая сила Fx (кН)", fontsize=12)

    plt.xlim(0, 4.0)
    plt.ylim(0, max(max(fx_kN) * 1.2, expected_peak * 1.2) if fx_kN else expected_peak * 1.2)

    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Запуск теста
    run_masonry_wall_test("J4D")
