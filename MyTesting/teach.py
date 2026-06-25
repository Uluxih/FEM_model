import numpy as np


class PlasticMaterial1D:
    """
    Одноосная пластичность с изотропным упрочнением
    """

    def __init__(self, E, sigma_y, H):
        """
        E : модуль упругости (МПа)
        sigma_y : предел текучести (МПа)
        H : модуль упрочнения (МПа)
        """
        self.E = E
        self.sigma_y = sigma_y
        self.H = H

    def residual(self, x, sigma_trial, eps_p_old):
        """
        Residual функция R(σ, λ)

        x = [sigma, lambda]
        """
        sigma, lam = x

        R1 = sigma - sigma_trial + self.E * lam
        R2 = abs(sigma) - self.sigma_y - self.H * (eps_p_old + lam)

        return np.array([R1, R2])

    def jacobian(self, x, sigma_trial, eps_p_old):
        """
        Якобиан J(σ, λ)
        """
        sigma, lam = x

        sign_sigma = np.sign(sigma) if sigma != 0 else 1.0

        J = np.array([
            [1.0, self.E],
            [sign_sigma, -self.H]
        ])

        return J

    def return_mapping(self, eps, eps_old, sigma_old, eps_p_old,
                       tol=1e-10, max_iter=20, verbose=False):
        """
        Return Mapping алгоритм

        Возвращает: sigma_new, eps_p_new, converged, iterations
        """

        # 1. Elastic predictor
        d_eps = eps - eps_old
        sigma_trial = sigma_old + self.E * d_eps

        # 2. Check yield condition
        f_trial = abs(sigma_trial) - self.sigma_y - self.H * eps_p_old

        if f_trial <= 0:
            # Elastic step
            if verbose:
                print("Упругий шаг")
            return sigma_trial, eps_p_old, True, 0

        # 3. Plastic corrector - Newton-Raphson
        if verbose:
            print(f"\nПластический шаг (f_trial = {f_trial:.3e})")
            print(f"{'Iter':<6} {'||R||':<15} {'σ':<15} {'λ':<15}")
            print("-" * 60)

        # Начальное приближение
        x = np.array([sigma_trial, 0.0])

        for iteration in range(max_iter):
            # Residual
            r = self.residual(x, sigma_trial, eps_p_old)
            r_norm = np.linalg.norm(r)

            if verbose:
                print(f"{iteration:<6} {r_norm:<15.6e} {x[0]:<15.6f} {x[1]:<15.6e}")

            # Проверка сходимости
            if r_norm < tol:
                sigma_new = x[0]
                lam = x[1]
                eps_p_new = eps_p_old + lam

                if verbose:
                    print(f"\n✓ Сошлось за {iteration} итераций")

                return sigma_new, eps_p_new, True, iteration

            # Якобиан
            jac = self.jacobian(x, sigma_trial, eps_p_old)

            # Решаем систему
            try:
                dx = np.linalg.solve(jac, -r)
            except np.linalg.LinAlgError:
                if verbose:
                    print("✗ Якобиан вырожден")
                return sigma_old, eps_p_old, False, iteration

            # Обновляем
            x = x + dx

        # Не сошлось
        if verbose:
            print(f"✗ Не сошлось за {max_iter} итераций")

        return sigma_old, eps_p_old, False, max_iter

    def get_consistent_tangent(self, sigma, eps_p):
        """
        Consistent stiffness (формула 38)
        """
        # Проверка: пластичность или упругость?
        f = abs(sigma) - self.sigma_y - self.H * eps_p

        if f < -1e-10:
            # Упругая жесткость
            return self.E
        else:
            # Пластическая жесткость (из обратного Якобиана)
            K_consistent = self.E * self.H / (self.E + self.H)
            return K_consistent


# ДЕМОНСТРАЦИЯ
if __name__ == "__main__":
    # Параметры материала (сталь)
    E = 200000.0  # МПа
    sigma_y = 300.0  # МПа
    H = 20000.0  # МПа

    material = PlasticMaterial1D(E, sigma_y, H)

    print("=" * 70)
    print("RETURN MAPPING: Детальный пример")
    print("=" * 70)
    print(f"Материал: E = {E} МПа, σ_y = {sigma_y} МПа, H = {H} МПа")
    print()

    # История нагружения
    eps_history = [0.0, 0.001, 0.002, 0.003, 0.002, 0.001]

    # Начальное состояние
    sigma = 0.0
    eps_p = 0.0
    eps_old = 0.0

    results = []

    for step, eps in enumerate(eps_history):
        print(f"\n{'=' * 70}")
        print(f"ШАГ {step}: ε = {eps:.4f}")
        print(f"{'=' * 70}")

        sigma_new, eps_p_new, converged, iters = material.return_mapping(
            eps, eps_old, sigma, eps_p, verbose=True
        )

        # Consistent tangent
        K_c = material.get_consistent_tangent(sigma_new, eps_p_new)

        print(f"\nРезультат:")
        print(f"  σ = {sigma_new:.3f} МПа")
        print(f"  ε_p = {eps_p_new:.6f}")
        print(f"  K_consistent = {K_c:.1f} МПа")

        results.append({
            'eps': eps,
            'sigma': sigma_new,
            'eps_p': eps_p_new,
            'K_c': K_c,
            'iters': iters
        })

        # Обновляем состояние
        sigma = sigma_new
        eps_p = eps_p_new
        eps_old = eps

    # Визуализация
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    eps_vals = [r['eps'] for r in results]
    sigma_vals = [r['sigma'] for r in results]
    eps_p_vals = [r['eps_p'] for r in results]
    K_vals = [r['K_c'] for r in results]
    iters_vals = [r['iters'] for r in results]

    # 1. Кривая напряжение-деформация
    ax = axes[0, 0]
    ax.plot(eps_vals, sigma_vals, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Деформация ε')
    ax.set_ylabel('Напряжение σ (МПа)')
    ax.grid(True)
    ax.set_title('Кривая σ-ε')

    # 2. Пластическая деформация
    ax = axes[0, 1]
    ax.plot(eps_vals, eps_p_vals, 'ro-', linewidth=2, markersize=8)
    ax.set_xlabel('Деформация ε')
    ax.set_ylabel('Пластическая деформация ε_p')
    ax.grid(True)
    ax.set_title('Накопление пластической деформации')

    # 3. Consistent stiffness
    ax = axes[1, 0]
    ax.plot(eps_vals, K_vals, 'go-', linewidth=2, markersize=8)
    ax.axhline(y=E, color='b', linestyle='--', label='Упругая жесткость E')
    ax.set_xlabel('Деформация ε')
    ax.set_ylabel('Жесткость K (МПа)')
    ax.grid(True)
    ax.legend()
    ax.set_title('Изменение жесткости')

    # 4. Итерации
    ax = axes[1, 1]
    ax.bar(range(len(iters_vals)), iters_vals, color='purple', alpha=0.7)
    ax.set_xlabel('Шаг нагружения')
    ax.set_ylabel('Количество итераций')
    ax.grid(True, axis='y')
    ax.set_title('Эффективность сходимости')

    plt.tight_layout()
    plt.savefig('return_mapping_detailed.png', dpi=150)
    plt.show()
