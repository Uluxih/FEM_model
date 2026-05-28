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
from FEM.Structure_Level.NonLinearNewtonRaphsonControl import MultiElementNRControl

# Импорт новой конститутивной модели Minga
from FEM.Integration_Point_Level.MingaMasonryJointModel3DFixed import MingaMasonryJointModel3DFixed

modelFEM = MingaMasonryJointModel3DFixed


# =====================================================================
# ЧАСТЬ 1: ЯДРО МКЭ (РЕШАТЕЛЬ, ГЕНЕРАТОР СЕТКИ, ОБЕРТКА МАТЕРИАЛА)
# =====================================================================

class JointedMaterial(Material):
    """Обертка для передачи параметров в ConstitutiveModel"""
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
# ЧАСТЬ 2: ПОЛЬЗОВАТЕЛЬСКИЙ ВВОД И ЗАПУСК РАСЧЕТА
# =====================================================================

def run_cyclic_test_minga():
    print("=== ТЕСТ: Циклическое нагружение модели Minga (Оптимизированная 3D сетка) ===")

    # Базовые упругие параметры (например, кладка)
    ROCK_E = 2000.0  # MPa
    ROCK_NU = 0.2

    # Параметры сетки
    SIZE_X, SIZE_Y, SIZE_Z = 1.0, 1.0, 4.0
    nx, ny, nz = 1, 1, 135

    # Параметры модели Minga et al.
    joint_params = {
        'ft': 0.25,        # Предел прочности на растяжение (MPa)
        'fc': 1.0,        # Предел прочности на сжатие (MPa)
        'c': 0.5,          # Сцепление (MPa)
        'phi': 30.0,       # Угол внутреннего трения (градусы)
        'phi_g': 0.0,      # Угол дилатансии (градусы)
        'Gf_1': 0.005/100,      # Энергия разрушения при отрыве (N/mm)
        'Gf_2': 0.010/100,      # Энергия разрушения при сдвиге (N/mm)
        'Gf_3': 0.100/100,      # Энергия разрушения при сжатии (N/mm)
        'l': 0.01,          # Параметр остаточных деформаций при разгрузке
        'alpha_t': 1.0,    # Связь повреждений
        'alpha_s': 1.0
    }

    global_material = JointedMaterial(E=ROCK_E, nu=ROCK_NU, joint_params=joint_params)

    factory = HEX8Factory()
    model = generate_block_mesh(SIZE_X, SIZE_Y, SIZE_Z, nx, ny, nz, global_material, factory)
    print(f"Сетка создана: {len(model.nodes)} узлов, {len(model.elements)} элементов.")

    top_nodes = []
    for node in model.nodes:
        # --- ПРАВИЛЬНЫЕ ГРАНИЧНЫЕ УСЛОВИЯ (Не блокируют эффект Пуассона) ---

        # Нижняя грань (Z = 0)
        if abs(node.coords[2] - 0.0) < 1e-6:
            model.add_bc(node, 2, 0.0)  # Жестко фиксируем ВСЕ узлы низа по Z

            # Фиксируем ТОЛЬКО ОДИН узел от смещения по X и Y
            if abs(node.coords[0] - 0.0) < 1e-6 and abs(node.coords[1] - 0.0) < 1e-6:
                model.add_bc(node, 0, 0.0)
                model.add_bc(node, 1, 0.0)

            # Фиксируем ЕЩЕ ОДИН узел по Y (предотвращение вращения вокруг оси Z)
            elif abs(node.coords[0] - SIZE_X) < 1e-6 and abs(node.coords[1] - 0.0) < 1e-6:
                model.add_bc(node, 1, 0.0)

        # Верхняя грань (Z = L)
        elif abs(node.coords[2] - SIZE_Z) < 1e-6:
            top_nodes.append(node)
            model.add_bc(node, 2, 1.0)  # Тянем по Z

    # Траектория нагружения (Растяжение -> Сжатие -> Растяжение)
    # Упругое смещение до пика отрыва: ~0.0005 мм
    # Упругое смещение до пика сжатия: ~-0.020 мм
    path_tension_1 = np.linspace(0, 0.006, 10)[1:]          # 1. Тянем до разрушения на отрыв
    path_compression = np.linspace(0.006, -0.04, 10)[1:]  # 2. Жмем вниз (закрытие трещины и раздавливание)
    path_tension_2 = np.linspace(-0.04, 0.0012, 10)[1:]     # 3. Снова тянем (проверка сниженной жесткости)

    load_factors = np.concatenate([path_tension_1, path_compression, path_tension_2])

    control = MultiElementNRControl(
        model=model,
        track_nodes=top_nodes,
        load_factors=load_factors,
        track_dof=2,
        max_iter=150,
        tol=1e-3  # Относительная точность 0.1%
    )

    control.solve()

    # ПОСТРОЕНИЕ ГРАФИКА
    uz_mm = [u for u in control.history_U] # Оставляем в мм
    fz_n = control.history_F

    plt.figure(figsize=(10, 7))
    plt.plot(uz_mm, fz_n, marker='o', color='#1f77b4', linewidth=2, markersize=4, label='Реакция конструкции (Minga)')

    # Теоретические пики с учетом площади сечения
    area = SIZE_X * SIZE_Y
    plt.axhline(joint_params['ft'] * area, color='red', linestyle='--', alpha=0.5, label=f"Пик растяжения ({joint_params['ft'] * area} Н)")
    plt.axhline(-joint_params['fc'] * area, color='orange', linestyle='--', alpha=0.5, label=f"Пик сжатия ({-joint_params['fc'] * area} Н)")

    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)

    plt.title("МКЭ Тест: Циклическое нагружение модели Minga 3D", fontsize=13)
    plt.xlabel("Вертикальное перемещение верхней грани Uz (мм)", fontsize=12)
    plt.ylabel("Суммарная реакция Fz (Н)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_cyclic_test_minga()
