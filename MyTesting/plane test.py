import numpy as np
import matplotlib.pyplot as plt

# Импорты из вашей архитектуры
from FEM.Abstract.Structure_Level import FEModel, Node
from FEM.Abstract.Integration_Point_Level import Material as BaseMaterial
from FEM.Element_Level.Shape8NodeHexahedron import HEX8Factory
from FEM.Integration_Point_Level.UbiquitousJointModel3D import UbiquitousJointModel3D
from FEM.Integration_Point_Level.CriticalPlane.material import Material as CPMaterial
from FEM.Structure_Level.NonLinearNewtonRaphsonControl import MultiElementNRControl
from FEM.Structure_Level.VTKExporter import VTKExporter


# Вспомогательный класс для обхода абстрактности базового материала
class SolidMaterial(BaseMaterial):
    def __init__(self, E, nu):
        super().__init__(E, nu)


def run_test():
    print("=== ПОДГОТОВКА ТЕСТА UBIQUITOUS JOINT MODEL ===")

    # 1. Настройка свойств материала
    # 1.1 Материал для поиска критической плоскости (прочности)
    # Задаем прочность на сжатие 20 МПа, растяжение 2 МПа
    cp_mat = CPMaterial(
        mu=0.5,
        Rpx=2e6, Rpy=2e6, Rpz=2e6,
        Rcx=20e6, Rcy=20e6, Rcz=20e6
    )

    # 1.2 Основной FEM материал (Упругость + параметры трещины/разупрочнения)
    mat = SolidMaterial(E=20e9, nu=0.2)  # E = 20 ГПа
    mat.joint_params = {
        'phi': 30.0,  # Угол внутреннего трения (градусы)
        'psi': 10.0,  # Угол дилатансии (градусы)
        'l_c': 1.0,  # Характеристическая длина элемента (для куба 1х1х1 равна 1)
        'Gf_t': 100.0,  # Энергия разрушения при отрыве (Дж/м2)
        'Gf_c': 5000.0,  # Энергия разрушения при сжатии (Дж/м2)
        'Gf_s': 500.0,  # Энергия разрушения при сдвиге (Дж/м2)
        'cp_material': cp_mat  # Передаем объект прочности
    }

    # 2. Создание геометрии (1 кубический элемент HEX8 1x1x1 метр)
    model = FEModel()

    # Порядок узлов важен для HEX8 (Нижняя грань против часовой, затем верхняя)
    nodes = [
        Node(0, [0.0, 0.0, 0.0]), Node(1, [1.0, 0.0, 0.0]),
        Node(2, [1.0, 1.0, 0.0]), Node(3, [0.0, 1.0, 0.0]),
        Node(4, [0.0, 0.0, 1.0]), Node(5, [1.0, 0.0, 1.0]),
        Node(6, [1.0, 1.0, 1.0]), Node(7, [0.0, 1.0, 1.0])
    ]
    model.nodes.extend(nodes)

    # 3. Сборка элемента через фабрику
    factory = HEX8Factory()
    # Обязательно передаем constitutive_class
    elem = factory.create_element(nodes, mat, constitutive_class=UbiquitousJointModel3D)
    model.elements.append(elem)

    # 4. Граничные условия
    # 4.1 Жестко фиксируем нижнюю грань (Z=0) по всем осям
    bottom_nodes = [nodes[0], nodes[1], nodes[2], nodes[3]]
    for node in bottom_nodes:
        model.add_bc(node, dof_axis=0, value=0.0)  # X
        model.add_bc(node, dof_axis=1, value=0.0)  # Y
        model.add_bc(node, dof_axis=2, value=0.0)  # Z

    # 4.2 Задаем перемещение верхней грани (Z=1) вниз (Сжатие)
    # Пиковая деформация примерно Rc/E = 20e6 / 20e9 = 0.001.
    # Зададим -0.003 м, чтобы увидеть спад (softening).
    target_displacement = -0.003
    top_nodes = [nodes[4], nodes[5], nodes[6], nodes[7]]
    for node in top_nodes:
        # X и Y оставляем свободными (свободное поперечное расширение - эффект Пуассона)
        model.add_bc(node, dof_axis=2, value=target_displacement)  # Z

    # 5. Настройка нелинейного решателя
    # Разбиваем нагрузку (перемещение) на 60 шагов
    load_factors = np.linspace(0.0, -0.050, 100)

    # Отслеживаем узлы верхней грани по оси Z (dof=2) для графика реакций
    solver = MultiElementNRControl(
        model=model,
        track_nodes=top_nodes,
        load_factors=load_factors,
        track_dof=2,
        max_iter=550,
        tol=1e-4
    )

    # 6. Запуск расчета
    solver.solve()

    # 7. Постпроцессинг
    # Экспорт в Paraview/PyVista
    VTKExporter.export(model, "ubiquitous_compression_test.vtu")

    # Построение графика Сила - Перемещение
    # Переводим перемещения в мм, а силу в МегаНьютоны для красоты
    U_history = np.array(solver.history_U) * 1000
    F_history = np.array(solver.history_F) / 1e6

    plt.figure(figsize=(8, 5))
    plt.plot(U_history, F_history, marker='o', markersize=4, linestyle='-', color='b')
    plt.title('Uniaxial Compression Test (Ubiquitous Joint Model 3D)')
    plt.xlabel('Displacement Z (mm)')
    plt.ylabel('Reaction Force Z (MN)')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Инвертируем оси, так как сжатие идет в минус
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_test()
