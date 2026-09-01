import numpy as np
import matplotlib.pyplot as plt

# Импорт базовых классов МКЭ
from FEM.Abstract.Structure_Level import Node, FEModel
from FEM.Abstract.Integration_Point_Level import Material
from FEM.Element_Level.Shape4NodeQuadrilateral import QUAD4Factory

# Импорт решателя
from FEM.Structure_Level.StagedNRControl2D import StagedNRControl2D

# Импорт вашей новой модели
from FEM.Integration_Point_Level.OrthotropicMasonryDamageModel2D import OrthotropicMasonryDamageModel2D


class OrthotropicMaterial(Material):
    def __init__(self, matrix_params):
        # Базовые E и nu передаем фиктивно, основная логика берет их из matrix_params
        super().__init__(E=matrix_params.get('Ex', 2000.0), nu=matrix_params.get('nu_xy', 0.1))
        self.matrix_params = matrix_params


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
            el = factory.create_element(el_nodes, material, constitutive_class=OrthotropicMasonryDamageModel2D)
            model.elements.append(el)
    return model


def run_ortho_test(test_type="compression_y"):
    print(f"=== ТЕСТ Orthotropic Damage: {test_type.upper()} ===")

    # Геометрия 1x1 метр
    SIZE_X, SIZE_Y = 1.0, 1.0
    nx, ny = 10, 10

    area_el = (SIZE_X / nx) * (SIZE_Y / ny)
    char_len = area_el ** 0.5

    # Параметры материала (в МПа, как в статье Berto et al.)
    matrix_params = {
        'Ex': 2000.0,
        'Ey': 2990.0,
        'Gxy': 900.0,
        'nu_xy': 0.10,
        'fcx0': 1.80,
        'fcy0': 3.81,
        'ftx': 0.50,
        'fty': 0.18,
        'fv': 0.40,
        'Gfx': 0.4,  # Н/мм или МПа*мм
        'Gfy': 0.05,
        'l_c': char_len * 1000.0,  # перевод в мм, если Gf задано в Н/мм
        'Ax_minus': 0.3,
        'Ay_minus': 0.34,
        'Bx_minus': 1.0,
        'By_minus': 1.3
    }

    material = OrthotropicMaterial(matrix_params)
    factory = QUAD4Factory()

    model = generate_rect_mesh(SIZE_X, SIZE_Y, nx, ny, material, factory)

    track_nodes = []

    top_nodes = [n for n in model.nodes if abs(n.coords[1] - SIZE_Y) < 1e-6]
    bottom_nodes = [n for n in model.nodes if abs(n.coords[1] - 0.0) < 1e-6]
    right_nodes = [n for n in model.nodes if abs(n.coords[0] - SIZE_X) < 1e-6]
    left_nodes = [n for n in model.nodes if abs(n.coords[0] - 0.0) < 1e-6]

    if test_type == "compression_y":
        TARGET_DISP = -0.005
        track_dof = 1
        for n in bottom_nodes:
            model.add_bc(n, 1, 0.0)
            if n.coords[0] == 0.0: model.add_bc(n, 0, 0.0)
        for n in top_nodes:
            model.add_bc(n, 1, TARGET_DISP)
            model.bcs[-1].is_proportional = True
            track_nodes.append(n)

    elif test_type == "tension_y":
        TARGET_DISP = 0.0005
        track_dof = 1
        for n in bottom_nodes:
            model.add_bc(n, 1, 0.0)
            if n.coords[0] == 0.0: model.add_bc(n, 0, 0.0)
        for n in top_nodes:
            model.add_bc(n, 1, TARGET_DISP)
            model.bcs[-1].is_proportional = True
            track_nodes.append(n)

    elif test_type == "compression_x":
        TARGET_DISP = -0.005
        track_dof = 0
        for n in left_nodes:
            model.add_bc(n, 0, 0.0)
            if n.coords[1] == 0.0: model.add_bc(n, 1, 0.0)
        for n in right_nodes:
            model.add_bc(n, 0, TARGET_DISP)
            model.bcs[-1].is_proportional = True
            track_nodes.append(n)

    elif test_type == "tension_x":
        TARGET_DISP = 0.001
        track_dof = 0
        for n in left_nodes:
            model.add_bc(n, 0, 0.0)
            if n.coords[1] == 0.0: model.add_bc(n, 1, 0.0)
        for n in right_nodes:
            model.add_bc(n, 0, TARGET_DISP)
            model.bcs[-1].is_proportional = True
            track_nodes.append(n)

    elif test_type == "shear":
        TARGET_DISP = 0.002
        track_dof = 0
        for n in bottom_nodes:
            model.add_bc(n, 0, 0.0)
            model.add_bc(n, 1, 0.0)
        for n in top_nodes:
            model.add_bc(n, 1, 0.0)
            model.add_bc(n, 0, TARGET_DISP)
            model.bcs[-1].is_proportional = True
            track_nodes.append(n)
    else:
        raise ValueError("Неизвестный тип теста")

    load_factors = np.linspace(0, 1.0, 40)[1:]

    control = StagedNRControl2D(
        model=model,
        track_nodes=track_nodes,
        load_factors=load_factors,
        track_dof=track_dof,
        max_iter=100,
        tol=5e-2
    )

    control.solve()

    # Извлечение данных (напряжения уже в МПа из-за настроек модели)
    strains = [abs(u) / SIZE_Y if track_dof == 1 else abs(u) / SIZE_X for u in control.history_U]
    stresses_MPa = [abs(f) / (SIZE_X * 1.0) for f in control.history_F]

    return strains, stresses_MPa


if __name__ == "__main__":
    tests = {
        "tension_x": "Растяжение X",
        "compression_x": "Сжатие X",
        "tension_y": "Растяжение Y",
        "compression_y": "Сжатие Y",
        "shear": "Сдвиг (X-Y)"
    }

    results = {}
    for t_key in tests.keys():
        strains, stresses = run_ortho_test(t_key)
        results[t_key] = (strains, stresses)

    # Построение графиков
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Валидация Orthotropic Masonry Damage Model 2D", fontsize=16)
    axes = axes.flatten()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (t_key, t_name) in enumerate(tests.items()):
        ax = axes[i]
        eps, sig = results[t_key]

        ax.plot([e * 1000 for e in eps], sig, color=colors[i], linewidth=2)
        ax.set_title(t_name)
        ax.set_xlabel("Деформация (x10^-3)")
        ax.set_ylabel("Напряжение (МПа)")
        ax.grid(True, linestyle=':', alpha=0.7)

    # Удаляем пустой 6-й график
    fig.delaxes(axes[5])

    plt.tight_layout()
    plt.show()