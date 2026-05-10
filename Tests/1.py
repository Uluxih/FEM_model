import numpy as np
from abc import ABC, abstractmethod


# =====================================================================
# 1. ФИЗИКА (НЕЛИНЕЙНЫЙ МАТЕРИАЛ)
# =====================================================================
class Material:
    def __init__(self, E, nu, eps_0=None):
        self.E = E
        self.nu = nu
        self.eps_0 = eps_0  # Предел упругости (для нелинейной модели)


class ConstitutiveModel(ABC):
    def __init__(self, material):
        self.material = material

    @abstractmethod
    def get_tangent_matrix(self, strain): pass

    @abstractmethod
    def get_stress(self, strain): pass


class IsotropicDamageModel(ConstitutiveModel):
    """Нелинейная модель: жесткость падает при превышении деформации eps_0"""

    def _get_elastic_matrix(self):
        E, nu = self.material.E, self.material.nu
        factor = E / (1 - nu ** 2)
        return factor * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1 - nu) / 2]
        ])

    def get_stress(self, strain):
        D0 = self._get_elastic_matrix()
        eps_eq = np.linalg.norm(strain)  # Эквивалентная деформация

        if eps_eq <= self.material.eps_0 or eps_eq == 0:
            # Линейно-упругая зона
            return D0 @ strain
        else:
            # Зона повреждения (напряжения растут медленнее)
            damage_factor = self.material.eps_0 / eps_eq
            return damage_factor * (D0 @ strain)

    def get_tangent_matrix(self, strain):
        D0 = self._get_elastic_matrix()
        eps_eq = np.linalg.norm(strain)

        if eps_eq <= self.material.eps_0 or eps_eq == 0:
            # Касательная матрица равна начальной
            return D0
        else:
            # Аналитическая касательная матрица для метода Ньютона-Рафсона (D_t = d_sigma / d_epsilon)
            I = np.eye(3)
            outer_strain = np.outer(strain, strain)
            damage_factor = self.material.eps_0 / eps_eq
            return damage_factor * D0 @ (I - outer_strain / (eps_eq ** 2))


class IntegrationPoint:
    def __init__(self, xi, eta, weight, constitutive_model):
        self.xi, self.eta, self.weight = xi, eta, weight
        self.constitutive_model = constitutive_model


# =====================================================================
# 2. ГЕОМЕТРИЯ И АНАЛИЗ (БЕЗ ИЗМЕНЕНИЙ!)
# =====================================================================
class Shape(ABC):
    @abstractmethod
    def get_jacobian(self, xi, eta, node_coords): pass

    @abstractmethod
    def get_shape_derivatives_cartesian(self, xi, eta, node_coords): pass


class Quadrilateral4Node(Shape):
    def get_jacobian(self, xi, eta, node_coords):
        dN_dxi_eta = 0.25 * np.array([[-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                                      [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]])
        J = dN_dxi_eta @ node_coords
        return J, np.linalg.det(J), dN_dxi_eta

    def get_shape_derivatives_cartesian(self, xi, eta, node_coords):
        J, _, dN_dxi_eta = self.get_jacobian(xi, eta, node_coords)
        return np.linalg.inv(J) @ dN_dxi_eta


class AnalysisModel(ABC):
    @abstractmethod
    def get_B_matrix(self, shape_derivatives): pass

    @abstractmethod
    def get_h_coefficient(self): pass


class PlaneStress2DModel(AnalysisModel):
    def __init__(self, thickness): self.thickness = thickness

    def get_B_matrix(self, dN_dx_dy):
        n = dN_dx_dy.shape[1]
        B = np.zeros((3, n * 2))
        for i in range(n):
            B[0, 2 * i], B[1, 2 * i + 1] = dN_dx_dy[0, i], dN_dx_dy[1, i]
            B[2, 2 * i], B[2, 2 * i + 1] = dN_dx_dy[1, i], dN_dx_dy[0, i]
        return B

    def get_h_coefficient(self): return self.thickness


# =====================================================================
# 3. ЭЛЕМЕНТ (ДОБАВЛЕНЫ НЕЛИНЕЙНЫЕ МЕТОДЫ)
# =====================================================================
class Element:
    def __init__(self, nodes, shape, analysis_model, integration_points):
        self.nodes, self.shape = nodes, shape
        self.analysis_model, self.integration_points = analysis_model, integration_points

    def _get_u_e(self):
        """Возвращает текущий вектор перемещений узлов элемента"""
        u_e = np.zeros(len(self.nodes) * 2)
        for i, node in enumerate(self.nodes):
            u_e[2 * i], u_e[2 * i + 1] = node.displacements[0], node.displacements[1]
        return u_e

    def compute_internal_forces(self):
        """Интегрирование напряжений: F_int = integral( B^T * sigma * |J| * h * W )"""
        node_coords = np.array([n.coords for n in self.nodes])
        u_e = self._get_u_e()
        F_int = np.zeros(len(u_e))

        for ip in self.integration_points:
            _, detJ, _ = self.shape.get_jacobian(ip.xi, ip.eta, node_coords)
            dN_dx = self.shape.get_shape_derivatives_cartesian(ip.xi, ip.eta, node_coords)
            B = self.analysis_model.get_B_matrix(dN_dx)

            strain = B @ u_e
            stress = ip.constitutive_model.get_stress(strain)

            F_int += B.T @ stress * detJ * self.analysis_model.get_h_coefficient() * ip.weight
        return F_int

    def compute_tangent_stiffness(self):
        """Касательная матрица жесткости: K_T = integral( B^T * D_T * B * |J| * h * W )"""
        node_coords = np.array([n.coords for n in self.nodes])
        u_e = self._get_u_e()
        ndof = len(u_e)
        K_T = np.zeros((ndof, ndof))

        for ip in self.integration_points:
            _, detJ, _ = self.shape.get_jacobian(ip.xi, ip.eta, node_coords)
            dN_dx = self.shape.get_shape_derivatives_cartesian(ip.xi, ip.eta, node_coords)
            B = self.analysis_model.get_B_matrix(dN_dx)

            strain = B @ u_e
            D_t = ip.constitutive_model.get_tangent_matrix(strain)

            K_T += B.T @ D_t @ B * detJ * self.analysis_model.get_h_coefficient() * ip.weight
        return K_T


# =====================================================================
# 4. ГЛОБАЛЬНАЯ МОДЕЛЬ И НЕЛИНЕЙНЫЙ РЕШАТЕЛЬ (NEWTON-RAPHSON)
# =====================================================================
class Node:
    def __init__(self, id, coords):
        self.id, self.coords = id, np.array(coords, dtype=float)
        self.dofs, self.forces, self.is_fixed = [], np.zeros(2), [False, False]
        self.displacements = np.zeros(2)  # Хранит текущие полные перемещения!


class FEModel:
    def __init__(self):
        self.nodes, self.elements = [], []
        self.total_dofs = 0

    def initialize(self):
        self.total_dofs = len(self.nodes) * 2
        dof_counter = 0
        for node in self.nodes:
            node.dofs = [dof_counter, dof_counter + 1]
            dof_counter += 2

    def assemble_external_forces(self):
        F_ext = np.zeros(self.total_dofs)
        for node in self.nodes:
            F_ext[node.dofs[0]] += node.forces[0]
            F_ext[node.dofs[1]] += node.forces[1]
        return F_ext

    def assemble_internal_forces(self):
        F_int = np.zeros(self.total_dofs)
        for el in self.elements:
            f_e = el.compute_internal_forces()
            dofs = [d for node in el.nodes for d in node.dofs]
            for i, gdof in enumerate(dofs):
                F_int[gdof] += f_e[i]
        return F_int

    def assemble_tangent_stiffness(self):
        K_T = np.zeros((self.total_dofs, self.total_dofs))
        for el in self.elements:
            k_e = el.compute_tangent_stiffness()
            dofs = [d for node in el.nodes for d in node.dofs]
            for i, gi in enumerate(dofs):
                for j, gj in enumerate(dofs):
                    K_T[gi, gj] += k_e[i, j]
        return K_T

    def update_displacements(self, delta_U):
        for node in self.nodes:
            node.displacements[0] += delta_U[node.dofs[0]]
            node.displacements[1] += delta_U[node.dofs[1]]


class NewtonRaphsonControl:
    def __init__(self, model: FEModel, tol=1e-6, max_iter=15):
        self.model, self.tol, self.max_iter = model, tol, max_iter

    def solve(self):
        self.model.initialize()
        F_ext = self.model.assemble_external_forces()

        print(f"{'Итерация':<10} | {'Норма невязки ||R||':<20}")
        print("-" * 35)

        for iteration in range(1, self.max_iter + 1):
            F_int = self.model.assemble_internal_forces()
            K_T = self.model.assemble_tangent_stiffness()

            # Вектор невязки (Residual) R = F_ext - F_int
            R = F_ext - F_int

            # Учет граничных условий
            for node in self.model.nodes:
                for i in range(2):
                    if node.is_fixed[i]:
                        gdof = node.dofs[i]
                        K_T[gdof, :] = 0
                        K_T[:, gdof] = 0
                        K_T[gdof, gdof] = 1.0
                        R[gdof] = 0.0  # Невязка в закреплениях равна 0

            # Проверка сходимости
            residual_norm = np.linalg.norm(R)
            print(f"{iteration:<10} | {residual_norm:.6e}")

            if residual_norm < self.tol:
                print("-" * 35)
                print(f"УСПЕХ: Сходимость достигнута за {iteration - 1} итераций!\n")
                break

            # Решение СЛАУ для приращений: K_T * delta_U = R
            delta_U = np.linalg.solve(K_T, R)

            # Обновление перемещений
            self.model.update_displacements(delta_U)
        else:
            print("ОШИБКА: Решатель не сошелся!")


# =====================================================================
# 5. ФАБРИКА ЭЛЕМЕНТОВ
# =====================================================================
class ElementFactory:
    def create_q4_damage_element(self, nodes, material, thickness):
        shape = Quadrilateral4Node()
        analysis = PlaneStress2DModel(thickness)

        g = 1.0 / np.sqrt(3.0)
        points = []
        for xi, eta in [(-g, -g), (g, -g), (g, g), (-g, g)]:
            physics = IsotropicDamageModel(material)
            points.append(IntegrationPoint(xi, eta, 1.0, physics))

        return Element(nodes, shape, analysis, points)


# =====================================================================
# 6. MAIN (ЗАПУСК)
# =====================================================================
def main():
    # 1. Узлы (Квадрат 1х1 м)
    nodes = [Node(1, [0, 0]), Node(2, [1, 0]), Node(3, [1, 1]), Node(4, [0, 1])]
    nodes[0].is_fixed, nodes[3].is_fixed = [True, True], [True, True]  # Левый край закреплен

    # Прикладываем ОЧЕНЬ большую силу (100 000 Н на правый край)
    # Это вызовет деформации, сильно превышающие предел упругости eps_0
    nodes[1].forces[0] = 50000.0
    nodes[2].forces[0] = 50000.0

    # 2. Нелинейный материал
    # Сталь: E = 200 ГПа, nu = 0.3. Предел упругой деформации eps_0 = 0.0001 (0.01%)
    steel_damage = Material(E=2e11, nu=0.3, eps_0=1e-4)

    # 3. Сборка
    factory = ElementFactory()
    element = factory.create_q4_damage_element(nodes, steel_damage, thickness=0.01)

    model = FEModel()
    model.nodes, model.elements = nodes, [element]

    # 4. Запуск МЕТОДА НЬЮТОНА-РАФСОНА
    solver = NewtonRaphsonControl(model, tol=1e-5)
    solver.solve()

    # 5. Результаты
    print("Финальные перемещения узлов (м):")
    for node in model.nodes:
        print(f"Узел {node.id}: Ux = {node.displacements[0]:.5e}, Uy = {node.displacements[1]:.5e}")


if __name__ == "__main__":
    main()
