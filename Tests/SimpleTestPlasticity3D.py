# ====================================
# File: .\Tests\4_debug_single_element.py
# ====================================
import numpy as np
import matplotlib.pyplot as plt

from FEM.Abstract.Structure_Level import Node, FEModel, Control
from FEM.Abstract.Integration_Point_Level import Material, ConstitutiveModel
from FEM.Element_Level.Shape8NodeHexahedron import HEX8Factory


# ==========================================
# 1. Глобальный материал
# ==========================================
class GlobalMaterial(Material):
    def __init__(self, E, nu, crit_mat=None):
        super().__init__(E, nu)
        self.crit_mat = crit_mat

# ==========================================
# 2. МАКСИМАЛЬНО ПРОСТАЯ МОДЕЛЬ МАТЕРИАЛА (J2 von Mises)
# ==========================================
class SimpleTestPlasticity3D(ConstitutiveModel):
    """
    Максимально простая билинейная модель (J2 von Mises).
    Служит ТОЛЬКО для проверки корректности глобальной системы (сборка, ГУ, решатель).
    Возвращает упругую касательную матрицу (Modified Newton-Raphson) для 100% стабильности.
    """

    def __init__(self, material):
        super().__init__(material)
        E = self.material.E
        nu = self.material.nu

        self.yield_stress = 10.0  # Предел текучести (МПа)
        self.H = -1000.00  # Модуль изотропного упрочнения (МПа)

        # Упругая матрица 6x6
        C = E / ((1 + nu) * (1 - 2 * nu))
        self.D_e = np.zeros((6, 6))
        self.D_e[0:3, 0:3] = C * nu
        self.D_e[0, 0] = self.D_e[1, 1] = self.D_e[2, 2] = C * (1 - nu)
        self.G = E / (2 * (1 + nu))
        self.D_e[3, 3] = self.D_e[4, 4] = self.D_e[5, 5] = self.G

        self.stress_old = np.zeros(6)
        self.strain_old = np.zeros(6)
        self.ep_eq_old = 0.0  # Накопленная пластическая деформация

        self.stress = np.zeros(6)
        self.strain = np.zeros(6)
        self.ep_eq = 0.0

    # Обязательные методы от базового класса
    def get_tangent_matrix(self):
        return self.D_e

    def get_stress(self, strain):
        return self.stress

    def update_state(self, current_strain):
        self.strain = current_strain
        d_strain = current_strain - self.strain_old

        # 1. Упругий шаг (Trial state)
        stress_trial = self.stress_old + self.D_e @ d_strain

        # 2. Интенсивность напряжений Мизеса (q)
        p_trial = np.sum(stress_trial[0:3]) / 3.0
        s_trial = stress_trial.copy()
        s_trial[0:3] -= p_trial

        J2 = 0.5 * (s_trial[0] ** 2 + s_trial[1] ** 2 + s_trial[2] ** 2) + \
             (s_trial[3] ** 2 + s_trial[4] ** 2 + s_trial[5] ** 2)
        q_trial = np.sqrt(3.0 * J2) if J2 > 0 else 0.0

        # 3. Критерий текучести
        current_yield = self.yield_stress + self.H * self.ep_eq_old
        f_yield = q_trial - current_yield

        if f_yield <= 1e-6:
            # Упругое состояние
            self.stress = stress_trial
            self.ep_eq = self.ep_eq_old
            return self.stress, self.D_e

        # 4. Пластическое состояние (Radial Return)
        d_gamma = f_yield / (3 * self.G + self.H)
        self.ep_eq = self.ep_eq_old + d_gamma

        factor = 1.0 - (3 * self.G * d_gamma) / q_trial
        s_new = s_trial * factor

        self.stress = s_new
        self.stress[0:3] += p_trial

        # Для проверки системы возвращаем упругую матрицу (метод начальной жесткости).
        return self.stress, self.D_e

    def commit(self):
        self.stress_old = self.stress.copy()
        self.strain_old = self.strain.copy()
        self.ep_eq_old = self.ep_eq

# ==========================================
# 3. ОТЛАДОЧНЫЙ РЕШАТЕЛЬ (С правильным учетом ГУ)
# ==========================================
class DebugNewtonRaphsonControl(Control):
    def __init__(self, model, track_nodes, num_steps=20, tol=1e-4, max_iter=50):
        super().__init__(model)
        self.num_steps = num_steps
        self.tol = tol
        self.max_iter = max_iter
        self.track_nodes = track_nodes
        self.history_Uz = [0.0]
        self.history_Fz = [0.0]

    def solve(self):
        print("Инициализация модели...")
        self.model.initialize()
        total_dofs = self.model.total_dofs
        U_global = np.zeros(total_dofs)

        for step in range(1, self.num_steps + 1):
            print(f"\n=== Шаг {step}/{self.num_steps} ===")

            for iteration in range(self.max_iter):
                K_t = np.zeros((total_dofs, total_dofs))
                F_int = np.zeros(total_dofs)

                for element in self.model.elements:
                    el_dofs = []
                    for node in element.nodes: el_dofs.extend(node.dofs)
                    U_el = U_global[el_dofs]
                    K_e, F_int_e = self._compute_element_nonlinear(element, U_el)
                    K_t[np.ix_(el_dofs, el_dofs)] += K_e
                    F_int[el_dofs] += F_int_e

                Residual = -F_int

                # ПРИМЕНЕНИЕ ГУ
                free_dofs = np.ones(total_dofs, dtype=bool)

                # 1. Корректировка Residual для свободных узлов
                for bc in self.model.bcs:
                    dof = bc.node.dofs[bc.dof_axis]
                    free_dofs[dof] = False
                    delta_u = (bc.value / self.num_steps) if iteration == 0 else 0.0
                    if delta_u != 0.0:
                        Residual -= K_t[:, dof] * delta_u

                # 2. Модификация матрицы
                for bc in self.model.bcs:
                    dof = bc.node.dofs[bc.dof_axis]
                    delta_u = (bc.value / self.num_steps) if iteration == 0 else 0.0
                    K_t[dof, :] = 0.0
                    K_t[:, dof] = 0.0
                    K_t[dof, dof] = 1.0
                    Residual[dof] = delta_u

                # Ошибка только по свободным узлам
                free_dofs_indices = np.where(free_dofs)[0]
                error = np.linalg.norm(Residual[free_dofs_indices]) if len(free_dofs_indices) > 0 else 0.0

                print(f"  Итер {iteration}: Невязка = {error:.6e}")

                if np.isnan(error) or np.isinf(error):
                    print("!!! КРИТИЧЕСКАЯ ОШИБКА: NaN/Inf. Решение разошлось !!!")
                    return

                if error < self.tol and iteration > 0:
                    print("  -> Сходимость достигнута.")
                    self._commit_state()

                    # Сбор реакций
                    rz_force = sum(F_int[n.dofs[2]] for n in self.track_nodes)
                    self.history_Uz.append(U_global[self.track_nodes[0].dofs[2]])
                    self.history_Fz.append(rz_force)
                    break

                dU = np.linalg.solve(K_t, Residual)
                U_global += dU
            else:
                print("!!! ВНИМАНИЕ: Сходимость не достигнута за макс. число итераций !!!")
                self._commit_state()
                rz_force = sum(F_int[n.dofs[2]] for n in self.track_nodes)
                self.history_Uz.append(U_global[self.track_nodes[0].dofs[2]])
                self.history_Fz.append(rz_force)

        for node in self.model.nodes:
            node.displacements = U_global[node.dofs]

    def _commit_state(self):
        for element in self.model.elements:
            for ip in element.integration_points:
                if hasattr(ip.constitutive_model, 'commit'): ip.constitutive_model.commit()

    def _compute_element_nonlinear(self, element, U_el):
        ndof = len(U_el)
        K_e = np.zeros((ndof, ndof))
        F_int_e = np.zeros(ndof)
        node_coords = np.array([node.coords for node in element.nodes])

        for ip in element.integration_points:
            _, detJ = element.shape.get_jacobian(ip.coords, node_coords)
            dN_dx = element.shape.get_shape_derivatives_cartesian(ip.coords, node_coords)
            B = element.analysis_model.get_B_matrix(dN_dx)
            h = element.analysis_model.get_h_coefficient()
            dV = detJ * h * ip.weight

            current_strain = B @ U_el
            stress, D_ep = ip.constitutive_model.update_state(current_strain)

            K_e += B.T @ D_ep @ B * dV
            F_int_e += B.T @ stress * dV

        return K_e, F_int_e

# ==========================================
# 4. ОСНОВНОЙ ТЕСТ (1 Элемент)
# ==========================================
def run_single_element_debug():
    print("=== ЗАПУСК ОТЛАДОЧНОГО ТЕСТА (1 ЭЛЕМЕНТ - ПРОСТАЯ ПЛАСТИЧНОСТЬ) ===")

    # 1. Материал
    material = GlobalMaterial(E=20000.0, nu=0.2, crit_mat=None)
    factory = HEX8Factory()

    # 2. Сетка: 1 кубик 1x1x1
    nodes = [
        Node(0, [0, 0, 0]), Node(1, [1, 0, 0]), Node(2, [1, 1, 0]), Node(3, [0, 1, 0]),
        Node(4, [0, 0, 1]), Node(5, [1, 0, 1]), Node(6, [1, 1, 1]), Node(7, [0, 1, 1])
    ]
    model = FEModel()
    model.nodes = nodes
    model.materials = [material]

    # Создаем элемент с ПРОСТОЙ моделью пластичности
    el = factory.create_element(nodes, material, constitutive_class=SimpleTestPlasticity3D)
    model.elements.append(el)

    # 3. Граничные условия (Идеальное одноосное сжатие)
    # Запрещаем движение нижней грани по Z
    for i in [0, 1, 2, 3]: model.add_bc(nodes[i], 2, 0.0)

    # Симметрия для предотвращения жесткого смещения (Roller supports)
    model.add_bc(nodes[0], 0, 0.0)
    model.add_bc(nodes[3], 0, 0.0)
    model.add_bc(nodes[4], 0, 0.0)
    model.add_bc(nodes[7], 0, 0.0)

    model.add_bc(nodes[0], 1, 0.0)
    model.add_bc(nodes[1], 1, 0.0)
    model.add_bc(nodes[4], 1, 0.0)
    model.add_bc(nodes[5], 1, 0.0)

    # Задаем перемещение верхней грани по Z
    top_nodes = [nodes[4], nodes[5], nodes[6], nodes[7]]
    target_disp_z = -0.005  # -5 мм (даст деформацию -0.005)
    for n in top_nodes: model.add_bc(n, 2, target_disp_z)

    # 4. Решение
    # 20 шагов дадут красивую кривую. max_iter=50 т.к. мы используем упругую касательную D_e
    control = DebugNewtonRaphsonControl(model=model, track_nodes=top_nodes, num_steps=20, tol=1e-4, max_iter=50)
    control.solve()

    # 5. Построение графика
    displacements_z = [abs(u) * 1000 for u in control.history_Uz]
    forces_z = [abs(f) for f in control.history_Fz]

    plt.figure(figsize=(8, 6))
    plt.plot(displacements_z, forces_z, marker='o', color='r', linewidth=2)
    plt.title("Отладочный тест: 1 Элемент (Модель Мизеса)")
    plt.xlabel("Осадка |Uz| (мм)")
    plt.ylabel("Сила реакции Fz (МН)")
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig("debug_1_element_mises.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    run_single_element_debug()
