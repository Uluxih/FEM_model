import matplotlib.pyplot as plt
import numpy as np

# Параметры материала
E = 200e3      # Модуль Юнга (упругий)
H = 20e3       # Модуль упрочнения (пластический)
sigma_y = 200  # Предел текучести
eps_y = sigma_y / E

# Функция напряжения для линейного упрочнения
def stress(eps):
    if eps <= eps_y:
        return E * eps
    else:
        return sigma_y + H * (eps - eps_y)

# Точка A (начало шага) - на пределе текучести
eps_A = eps_y
sig_A = sigma_y

# Сценарий 1: Малый шаг
delta_eps_1 = 0.0005
eps_B1 = eps_A + delta_eps_1
sig_B1 = stress(eps_B1)

# Сценарий 2: Большой шаг
delta_eps_2 = 0.003
eps_B2 = eps_A + delta_eps_2
sig_B2 = stress(eps_B2)

# Создание графика
plt.figure(figsize=(10, 6))

# Рисуем кривую деформирования
eps_plot = np.linspace(0, 0.005, 500)
sig_plot = [stress(e) for e in eps_plot]
plt.plot(eps_plot, sig_plot, 'k-', label='Истинная кривая $\sigma(\epsilon)$', linewidth=2)

# Отмечаем точку A
plt.plot(eps_A, sig_A, 'ro', markersize=8)
plt.annotate('Точка A (начало шага)', xy=(eps_A, sig_A), xytext=(eps_A-0.001, sig_A+50),
             arrowprops=dict(facecolor='black', shrink=0.05))

# Сценарий 1: Малый шаг
plt.plot([eps_A, eps_B1], [sig_A, sig_B1], 'b--', linewidth=2, label='Малый шаг: $E_{alg} \approx E_{tan}$')
plt.plot(eps_B1, sig_B1, 'bo', markersize=6)
plt.annotate('$B_1$', xy=(eps_B1, sig_B1), xytext=(eps_B1+0.0001, sig_B1-30))

# Сценарий 2: Большой шаг
plt.plot([eps_A, eps_B2], [sig_A, sig_B2], 'g-', linewidth=2, label='Большой шаг: $E_{alg} = \Delta\sigma/\Delta\epsilon$ (Секущий)')
plt.plot(eps_B2, sig_B2, 'go', markersize=6)
plt.annotate('$B_2$', xy=(eps_B2, sig_B2), xytext=(eps_B2+0.0001, sig_B2-30))

# Континуальный касательный модуль в точке A (пунктирная линия)
# Продлеваем упругую линию чуть дальше, чтобы показать наклон
eps_tan = np.linspace(0, eps_A + 0.001, 100)
sig_tan = E * eps_tan
plt.plot(eps_tan, sig_tan, 'r:', linewidth=1.5, label='Континуальный $E_{tan}$ в точке A')

# Континуальный касательный модуль в точке B2
eps_tan_B2 = np.linspace(eps_B2 - 0.001, eps_B2 + 0.001, 100)
sig_tan_B2 = sig_B2 + H * (eps_tan_B2 - eps_B2)
plt.plot(eps_tan_B2, sig_tan_B2, 'm:', linewidth=1.5, label='Континуальный $E_{tan}$ в точке $B_2$')

# Оформление
plt.xlabel('Деформация $\epsilon$')
plt.ylabel('Напряжение $\sigma$')
plt.title('Зависимость согласованного модуля от длины шага')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.show()