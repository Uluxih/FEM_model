import numpy as np
from FEM.Integration_Point_Level.XFEM_Models.DruckerPragerMatrix2D import DruckerPragerMatrix2D
from FEM.Integration_Point_Level.XFEM_Models.XFEM_CohesiveDamagePlasticity2D import XFEM_CohesiveDamagePlasticity2D


class IntegrationPoint:
    def __init__(self, coords, weight, constitutive_model=None):
        self.coords = coords  # Локальные координаты (xi, eta) или (r) для 1D
        self.weight = weight
        self.constitutive_model = constitutive_model


class XFEMQUAD4:
    def __init__(self, nodes, material):
        self.nodes = nodes
        self.material = material

        # Флаги и данные XFEM
        self.is_enriched = False
        self.crack_line = None  # Локальные координаты начала и конца трещины: [(xi1, eta1), (xi2, eta2)]
        self.crack_normal = None  # Вектор нормали к трещине [nx, ny]
        self.node_heaviside = np.zeros(4)  # Значения функции Хевисайда в узлах (+1 или -1)

        # Точки интегрирования
        self.bulk_ips = []
        self.cohesive_ips = []

        # Инициализация стандартных точек Гаусса (2x2)
        gauss_pts = [-0.577350269, 0.577350269]
        for xi in gauss_pts:
            for eta in gauss_pts:
                # Создаем модель матрицы для каждой точки
                mat_model = DruckerPragerMatrix2D(
                    E=material.E,
                    nu=material.nu,
                    matrix_params=material.matrix_params
                )
                self.bulk_ips.append(IntegrationPoint(np.array([xi, eta]), 1.0, mat_model))

    # ==========================================
    # БАЗОВАЯ КИНЕМАТИКА QUAD4
    # ==========================================
    def _shape_functions(self, xi, eta):
        return np.array([
            0.25 * (1 - xi) * (1 - eta),
            0.25 * (1 + xi) * (1 - eta),
            0.25 * (1 + xi) * (1 + eta),
            0.25 * (1 - xi) * (1 + eta)
        ])

    def _shape_derivatives(self, xi, eta):
        # Производные по локальным координатам (dN/dxi, dN/deta)
        dN = np.zeros((2, 4))
        dN[0, :] = [-0.25 * (1 - eta), 0.25 * (1 - eta), 0.25 * (1 + eta), -0.25 * (1 + eta)]
        dN[1, :] = [-0.25 * (1 - xi), -0.25 * (1 + xi), 0.25 * (1 + xi), 0.25 * (1 - xi)]
        return dN

    def get_detJ(self, coords):
        xi, eta = coords
        dN_local = self._shape_derivatives(xi, eta)
        # Координаты узлов
        X = np.array([n.coords for n in self.nodes])
        J = dN_local @ X
        return np.linalg.det(J)

    def _get_dN_dx(self, coords):
        xi, eta = coords
        dN_local = self._shape_derivatives(xi, eta)
        X = np.array([n.coords for n in self.nodes])
        J = dN_local @ X
        J_inv = np.linalg.inv(J)
        dN_dx = J_inv @ dN_local
        return dN_dx

    # ==========================================
    # ИНТЕРФЕЙС ДЛЯ РЕШАТЕЛЯ (XFEM)
    # ==========================================
    def get_active_dofs(self):
        """Возвращает список всех активных DoF (стандартные + обогащенные)"""
        dofs = []
        for node in self.nodes:
            dofs.extend(node.dofs)
            # Если узел обогащен функцией Хевисайда, добавляем его новые степени свободы
            if hasattr(node, 'xfem_heaviside_dofs'):
                dofs.extend(node.xfem_heaviside_dofs)
        return np.array(dofs, dtype=int)

    def get_bulk_integration_points(self):
        return self.bulk_ips

    def get_cohesive_integration_points(self):
        return self.cohesive_ips

    def get_B_matrix_enriched(self, coords):
        """
        Собирает матрицу B. Если элемент разрезан, добавляет обогащенные столбцы.
        """
        xi, eta = coords
        dN_dx = self._get_dN_dx(coords)
        N = self._shape_functions(xi, eta)

        # Количество активных степеней свободы
        num_dofs = len(self.get_active_dofs())
        B = np.zeros((3, num_dofs))

        col = 0
        for i, node in enumerate(self.nodes):
            # 1. Стандартная часть (Standard B-matrix)
            B[0, col] = dN_dx[0, i]
            B[1, col + 1] = dN_dx[1, i]
            B[2, col] = dN_dx[1, i]
            B[2, col + 1] = dN_dx[0, i]
            col += 2

            # 2. Обогащенная часть (Heaviside Enrichment)
            if hasattr(node, 'xfem_heaviside_dofs'):
                # Вычисляем H(x) в точке интегрирования
                # В объеме H(x) совпадает со знаком Level Set. Для простоты берем интерполяцию
                H_x = 1.0 if np.dot(N, self.node_heaviside) > 0 else -1.0
                H_i = self.node_heaviside[i]

                # Функция обогащения: F_i = N_i * (H(x) - H_i)
                # Ее градиент (т.к. grad(H) = 0 вне трещины): grad(F_i) = grad(N_i) * (H(x) - H_i)
                enrich_factor = H_x - H_i

                B[0, col] = dN_dx[0, i] * enrich_factor
                B[1, col + 1] = dN_dx[1, i] * enrich_factor
                B[2, col] = dN_dx[1, i] * enrich_factor
                B[2, col + 1] = dN_dx[0, i] * enrich_factor
                col += 2

        return B

    def get_jump_operator(self, coords_1d):
        """
        Матрица N_jump переводит узловые перемещения в скачок на трещине [du_n, du_s].
        [[u]] = u^+ - u^- = 2 * sum(N_i * a_i), где a_i - обогащенные DoFs.
        """
        xi, eta = coords_1d  # coords_1d - это точка на линии трещины
        N = self._shape_functions(xi, eta)

        num_dofs = len(self.get_active_dofs())
        N_jump_global = np.zeros((2, num_dofs))

        col = 0
        for i, node in enumerate(self.nodes):
            col += 2  # Пропускаем стандартные DoFs (они непрерывны, скачка не дают)
            if hasattr(node, 'xfem_heaviside_dofs'):
                # Скачок от Хевисайда равен 2 * N_i
                N_jump_global[0, col] = 2.0 * N[i]
                N_jump_global[1, col + 1] = 2.0 * N[i]
                col += 2

        # Трансформация скачка из глобальных осей [X, Y] в локальные оси трещины [Normal, Shear]
        nx, ny = self.crack_normal
        T = np.array([
            [nx, ny],  # Нормаль
            [-ny, nx]  # Касательная
        ])

        return T @ N_jump_global

    def get_crack_segment_length(self):
        if not self.is_enriched:
            return 0.0
        # Перевод локальных координат трещины в глобальные
        N1 = self._shape_functions(*self.crack_line[0])
        N2 = self._shape_functions(*self.crack_line[1])
        X = np.array([n.coords for n in self.nodes])
        p1 = N1 @ X
        p2 = N2 @ X
        return np.linalg.norm(p2 - p1)

    # ==========================================
    # МЕХАНИКА РАЗРЕЗАНИЯ (ОБОГАЩЕНИЯ)
    # ==========================================
    def cut_element(self, global_max_dof, p1_loc, p2_loc, normal):
        """
        Метод вызывается решателем, когда трещина пересекает элемент.
        p1_loc, p2_loc - локальные координаты входа и выхода трещины.
        """
        self.is_enriched = True
        self.crack_line = [p1_loc, p2_loc]
        self.crack_normal = normal

        # 1. Определяем, по какую сторону от трещины находятся узлы (Level Set)
        # Уравнение прямой через 2 точки: (y-y1)/(y2-y1) = (x-x1)/(x2-x1)
        x1, y1 = p1_loc
        x2, y2 = p2_loc
        for i, node in enumerate(self.nodes):
            # Локальные координаты узлов QUAD4
            xi, eta = [(-1, -1), (1, -1), (1, 1), (-1, 1)][i]
            # Векторное произведение для определения стороны (знак Level Set)
            cross_prod = (x2 - x1) * (eta - y1) - (y2 - y1) * (xi - x1)
            self.node_heaviside[i] = 1.0 if cross_prod > 0 else -1.0

            # Добавляем обогащенные степени свободы узлу, если их еще нет
            if not hasattr(node, 'xfem_heaviside_dofs'):
                node.xfem_heaviside_dofs = [global_max_dof + 1, global_max_dof + 2]
                global_max_dof += 2

        # 2. Перестройка объемных точек интегрирования (Sub-triangulation)
        # В реальном коде здесь строится Делоне. Для примера мы просто увеличиваем
        # порядок интегрирования (4x4), чтобы лучше проинтегрировать разрыв.
        self.bulk_ips = []
        gauss_pts_4 = [-0.861136, -0.339981, 0.339981, 0.861136]
        weights_4 = [0.347854, 0.652145, 0.652145, 0.347854]
        for xi, wx in zip(gauss_pts_4, weights_4):
            for eta, wy in zip(gauss_pts_4, weights_4):
                mat_model = DruckerPragerMatrix2D(self.material.E, self.material.nu, self.material.matrix_params)
                self.bulk_ips.append(IntegrationPoint((xi, eta), wx * wy, mat_model))

        # 3. Создание интерфейсных точек интегрирования (на самой трещине)
        # Интегрируем по 1D отрезку трещины (2 точки Гаусса)
        self.cohesive_ips = []
        g_1d = [-0.57735, 0.57735]
        for r in g_1d:
            # Интерполяция координаты на отрезке трещины
            xi_c = 0.5 * (1 - r) * x1 + 0.5 * (1 + r) * x2
            eta_c = 0.5 * (1 - r) * y1 + 0.5 * (1 + r) * y2

            coh_model = XFEM_CohesiveDamagePlasticity2D(self.material.cohesive_params)
            # Вес = 1.0 (длина отрезка учитывается отдельно в get_crack_segment_length)
            self.cohesive_ips.append(IntegrationPoint((xi_c, eta_c), 1.0, coh_model))

        return global_max_dof


class XFEMQUAD4Factory:
    """Фабрика для создания элементов XFEMQUAD4"""

    def create_element(self, nodes, material, constitutive_class=None):
        # constitutive_class игнорируется, так как XFEMQUAD4 сам создает
        # нужные объемные и когезионные модели из material
        return XFEMQUAD4(nodes, material)