
class Element:
    # ... старые методы ...

    def compute_internal_force(self):
        # Получаем текущие перемещения узлов элемента
        u_e = self.get_element_displacements()
        ndof = len(self.nodes) * 2
        F_int = np.zeros(ndof)

        for ip in self.integration_points:
            _, detJ, _ = self.shape.get_jacobian(...)
            dN_dx = self.shape.get_shape_derivatives_cartesian(...)
            B = self.analysis_model.get_B_matrix(dN_dx)
            h = self.analysis_model.get_h_coefficient()

            # 1. Вычисляем текущую деформацию
            strain = B @ u_e

            # 2. Запрашиваем текущее напряжение у физической модели
            stress = ip.constitutive_model.get_stress(strain)

            # 3. Интегрируем B^T * sigma
            F_int += B.T @ stress * detJ * h * ip.weight

        return F_int
