import os
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

# Автоматическое подключение быстрого многопоточного решателя (если установлен)
try:
    import pypardiso as spla

    print("[INFO] Установлен PyPardiso: включен сверхбыстрый многопоточный решатель СЛАУ!")
except ImportError:
    import scipy.sparse.linalg as spla

    print("[INFO] PyPardiso не найден. Используется однопоточный решатель SciPy (рекомендуется: pip install pypardiso)")

# Импорты базовых классов МКЭ
from FEM.Abstract.Structure_Level import Node, FEModel, Control
from FEM.Abstract.Integration_Point_Level import Material
from FEM.Element_Level.Shape8NodeHexahedron import HEX8Factory
from FEM.Structure_Level.VTKExporter import VTKExporter

# Импорты конститутивной модели и поиска критической плоскости
from FEM.Integration_Point_Level.UbiquitousJointModel3D import UbiquitousJointModel3D


import FEM.Integration_Point_Level.CriticalPlane.material as cp_mt
from FEM.Structure_Level.NonLinearNewtonRaphsonControl import MultiElementNRControl

modelFEM = UbiquitousJointModel3D

class JointedMaterial(Material):
    """Обертка для передачи параметров в ConstitutiveModel"""

    def __init__(self, E, nu, joint_params):
        super().__init__(E, nu)
        self.joint_params = joint_params

# Предполагается, что UbiquitousJointModel3D и JointedMaterial импортированы
# from your_module import UbiquitousJointModel3D, JointedMaterial, generate_block_mesh
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


import numpy as np
import matplotlib.pyplot as plt

# Импорты базовых классов МКЭ (согласно вашей структуре)
from FEM.Abstract.Structure_Level import Node, FEModel
from FEM.Element_Level.Shape8NodeHexahedron import HEX8Factory
from FEM.Structure_Level.NonLinearNewtonRaphsonControl import MultiElementNRControl
import FEM.Integration_Point_Level.CriticalPlane.material as cp_mt


# Предполагается, что классы JointedMaterial и generate_block_mesh
# определены в вашем основном скрипте.

def run_minga_short_wall_with_tensor():
    print("=== ВАЛИДАЦИЯ: Short Wall (Minga 2017) с использованием A_tensor ===")

    # =====================================================================
    # 1. ТЕНЗОР АНИЗОТРОПИИ И МАТЕРИАЛ CRITICAL PLANE
    # =====================================================================
    # Используем ваш формат задания тензора.
    # В статье Minga кладка рассматривается как ортотропная/изотропная на макроуровне.
    # Вы можете настроить коэффициенты тензора так, чтобы get_cohesion_limit
    # выдавал сцепление c ≈ 0.23 МПа при заданных R_p и R_c.
    tensor_data_str = """
    1.000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
    0.000000 1.000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
    0.000000 0.000000 1.000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
    0.000000 0.000000 0.000000 1.000 0.000000 0.000000 0.000000 0.000000 0.000000
    0.000000 0.000000 0.000000 0.000000 1.000 0.000000 0.000000 0.000000 0.000000
    0.000000 0.000000 0.000000 0.000000 0.000000 1.000 0.000000 0.000000 0.000000
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 1.000 0.000000 0.000000
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 1.000 0.000000
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 1.000
    """
    # Масштабирующий множитель для тензора (подберите для калибровки сцепления)
    TENSOR_SCALE = 0500.1
    A_matrix = cp_mt.load_tensor_from_string(tensor_data_str) * TENSOR_SCALE

    # Макро-свойства по статье (Table 2 и 3)
    ROCK_E = 3000.0  # Модуль упругости кладки, МПа
    ROCK_NU = 0.15  # Коэффициент Пуассона
    ROCK_MU = 0.58  # Тангенс угла трения (tan phi)
    ROCK_RP = 0.04  # Предел на растяжение (f_t), МПа
    ROCK_RC = 6.20  # Предел на сжатие (f_c), МПа

    cp_material = cp_mt.Material(
        mu=ROCK_MU,
        A_tensor=A_matrix,
        Rpx=ROCK_RP, Rpy=ROCK_RP, Rpz=ROCK_RP,
        Rcx=ROCK_RC, Rcy=ROCK_RC, Rcz=ROCK_RC
    )

    # =====================================================================
    # 2. ГЕОМЕТРИЯ СТЕНЫ И СЕТКА
    # =====================================================================
    SIZE_X = 1000.0  # Ширина (мм)
    SIZE_Y = 250.0  # Толщина (мм)
    SIZE_Z = 1350.0  # Высота (мм)

    # Сетка: 4x1x5 элементов для баланса скорости расчета и точности
    nx, ny, nz = 4, 1, 5
    vol_el = (SIZE_X / nx) * (SIZE_Y / ny) * (SIZE_Z / nz)
    char_len = vol_el ** (1 / 3)

    joint_params = {
        'cp_material': cp_material,
        'phi': np.degrees(np.arctan(ROCK_MU)),
        'psi': 0.0,
        'phi_r': np.degrees(np.arctan(ROCK_MU)),
        'l_c': char_len,
        'Gf_t': 0.05,  # Энергия растяжения (N/mm)
        'Gf_c': 1.00,  # Энергия сжатия (N/mm)
        'Gf_s': 0.10,  # Энергия сдвига (N/mm)
        'a_t': 1.0,
        'a_s': 1.0,
        'mu': 0.1,  # Параметр остаточных деформаций
        'fcr_over_fc': 0.0
        # ВАЖНО: cohesion здесь не задаем, ваша модель вычислит его сама
        # через get_cohesion_limit(normal, stress_tensor, cp_material)
    }

    global_material = JointedMaterial(E=ROCK_E, nu=ROCK_NU, joint_params=joint_params)

    factory = HEX8Factory()
    model = generate_block_mesh(SIZE_X, SIZE_Y, SIZE_Z, nx, ny, nz, global_material, factory)
    print(f"Сетка создана: {len(model.nodes)} узлов, {len(model.elements)} элементов.")

    # =====================================================================
    # 3. ГРАНИЧНЫЕ УСЛОВИЯ (Сдвиг с пред-сжатием 0.6 МПа)
    # =====================================================================
    # Вертикальная деформация для создания начального обжатия 0.6 МПа
    # Uz = (Sigma / E) * Lz = (-0.6 / 3000) * 1350 = -0.27 мм
    PRE_COMP_Z = -0.27

    top_nodes = []
    for node in model.nodes:
        # Жесткая заделка основания (Z = 0)
        if abs(node.coords[2] - 0.0) < 1e-6:
            model.add_bc(node, 0, 0.0)
            model.add_bc(node, 1, 0.0)
            model.add_bc(node, 2, 0.0)

        # Верхняя грань (Z = Lz) - стальная балка
        elif abs(node.coords[2] - SIZE_Z) < 1e-6:
            top_nodes.append(node)
            model.add_bc(node, 0, 1.0)  # Степень свободы 0 (X) будет управляемой
            model.add_bc(node, 1, 0.0)  # Запрет смещения из плоскости
            model.add_bc(node, 2, PRE_COMP_Z)  # Постоянное обжатие по Z

    # =====================================================================
    # 4. ПУТЬ НАГРУЖЕНИЯ (Циклический сдвиг из статьи)
    # =====================================================================
    # Пиковые значения смещений по X (в мм)
    peaks = [0, 1.0, -1.0, 2.0, -2.0, 3.5, -3.5, 5.0]

    load_factors = []
    steps_per_branch = 100
    for i in range(len(peaks) - 1):
        branch = np.linspace(peaks[i], peaks[i + 1], steps_per_branch)[1:]
        load_factors.extend(branch)

    control = MultiElementNRControl(
        model=model,
        track_nodes=top_nodes,
        load_factors=load_factors,
        track_dof=0,  # Отслеживаем ось X
        max_iter=200,
        tol=1e-3
    )

    print("Начат расчет (решатель ищет критические плоскости по тензору)...")
    control.solve()

    # =====================================================================
    # 5. ПОСТРОЕНИЕ ГРАФИКА И СРАВНЕНИЕ С ЭКСПЕРИМЕНТОМ
    # =====================================================================
    ux_num = np.array(control.history_U)
    fx_num = np.array(control.history_F) / 1000.0  # Перевод из Ньютонов в кН

    # Оцифрованная огибающая эксперимента (Minga 2017, Fig 13 - Short Wall)
    exp_disp = np.array([0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    exp_force_pos = np.array([0, 60, 95, 115, 125, 130, 120, 105, 90, 75])
    exp_force_neg = -np.array([0, 55, 90, 110, 120, 125, 115, 100, 85, 70])

    plt.figure(figsize=(10, 7))

    # Реакция вашей модели
    plt.plot(ux_num, fx_num, 'b-', linewidth=2, label='Ваша модель (A_tensor + Critical Plane)')

    # Экспериментальные данные
    plt.plot(exp_disp, exp_force_pos, 'r--o', markersize=5, label='Эксперимент Minga (+)')
    plt.plot(-exp_disp, exp_force_neg, 'r--o', markersize=5, label='Эксперимент Minga (-)')

    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)

    plt.title("Валидация: Short Wall in-plane shear (Minga 2017)", fontsize=13)
    plt.xlabel("Горизонтальное смещение верха стены Ux (мм)", fontsize=12)
    plt.ylabel("Сдвигающая реакция Fx (кН)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_minga_short_wall_with_tensor()


