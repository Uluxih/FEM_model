import numpy as np
import FEM.Integration_Point_Level.CriticalPlane.material as mt

from FEM.Abstract.Structure_Level import Node, FEModel
from FEM.Abstract.Integration_Point_Level import Material
from FEM.Element_Level.Shape8NodeHexahedron import HEX8Factory
from FEM.Integration_Point_Level.CriticalPlane.CriticalPlanePlasticity3D import CriticalPlanePlasticity3D
from FEM.Structure_Level.NonLinearNewtonRaphsonControl import NonLinearNewtonRaphsonControl
from FEM.Structure_Level.VTKExporter import VTKExporter


class GlobalMaterial(Material):
    def __init__(self, E, nu, crit_mat):
        super().__init__(E, nu)
        self.crit_mat = crit_mat


def run_nonlinear_fem():
    print("=== Нелинейный расчет (Единицы: Метры, Меганьютоны, Мегапаскали) ===")

    nodes = [
        Node(0, [0.0, 0.0, 0.0]), Node(1, [1.0, 0.0, 0.0]), Node(2, [1.0, 1.0, 0.0]), Node(3, [0.0, 1.0, 0.0]),
        Node(4, [0.0, 0.0, 1.0]), Node(5, [1.0, 0.0, 1.0]), Node(6, [1.0, 1.0, 1.0]), Node(7, [0.0, 1.0, 1.0]),
        Node(8, [2.0, 0.0, 0.0]), Node(9, [2.0, 1.0, 0.0]), Node(10, [2.0, 0.0, 1.0]), Node(11, [2.0, 1.0, 1.0])
    ]

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

    # C_val из A_matrix получается около 1.0 (что означает 1.0 МПа)
    A_matrix = mt.load_tensor_from_string(tensor_data)
    crit_mat = mt.Material(mu=-0.7, A_tensor=A_matrix)

    # E = 20 ГПа = 20000 МПа
    material = GlobalMaterial(E=20000.0, nu=0.2, crit_mat=crit_mat)
    factory = HEX8Factory()

    el1 = factory.create_element(nodes[0:8], material, constitutive_class=CriticalPlanePlasticity3D)
    el2_nodes = [nodes[1], nodes[8], nodes[9], nodes[2], nodes[5], nodes[10], nodes[11], nodes[6]]
    el2 = factory.create_element(el2_nodes, material, constitutive_class=CriticalPlanePlasticity3D)

    model = FEModel()
    model.nodes = nodes
    model.elements = [el1, el2]
    model.materials = [material]

    for node in [nodes[0], nodes[3], nodes[4], nodes[7]]:
        model.add_bc(node, 0, 0.0)
        model.add_bc(node, 1, 0.0)
        model.add_bc(node, 2, 0.0)

    # Сила 100 кН = 0.1 МН. Делим на 4 узла = 0.025 МН на узел.
    for node in [nodes[8], nodes[9], nodes[10], nodes[11]]:
        model.add_load(node, 2, -0.025)

    # Запуск
    control = NonLinearNewtonRaphsonControl(model, num_steps=10, tol=1e-4)
    control.solve()

    print("\nПеремещения на свободном конце (X=2.0) после нелинейного расчета:")
    for node in [nodes[8], nodes[9], nodes[10], nodes[11]]:
        print(f"Узел {node.id}: dZ = {node.displacements[2]:.5e} м")

    VTKExporter.export(model, "nonlinear_beam_results.vtk")


if __name__ == "__main__":
    run_nonlinear_fem()
