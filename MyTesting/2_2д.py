import numpy as np
import matplotlib.pyplot as plt

# Импорты (предполагается, что они доступны в вашей структуре проекта)
from FEM.Abstract.Integration_Point_Level import Material as BaseMaterial
from FEM.Integration_Point_Level.CriticalPlane.material import Material as CPMaterial
# Импортируйте вашу модель из соответствующего файла
# from FEM.Integration_Point_Level.MultiUbiquitousJointModel2D import MultiUbiquitousJointModel2D
from FEM.Integration_Point_Level.MultiUbiquitousJointModel2D import MultiUbiquitousJointModel2D

class RockMaterial(BaseMaterial):
    def __init__(self, E, nu, joint_params):
        super().__init__(E, nu)
        self.joint_params = joint_params


# =====================================================================
# 1. ПОДГОТОВКА МАТЕРИАЛОВ
# =====================================================================

# Задаем тензор анизотропии (фиктивный для теста)
A_matrix = np.eye(9) * (0.4e6) ** 2

# Создаем материал для Critical Plane.
# Пределы прочности: Растяжение = 0.5 МПа, Сжатие = 1.0 МПа
cp_mat = CPMaterial(mu=-0.3, A_tensor=A_matrix,
                    Rpx=0.5e6, Rpy=0.5e6, Rpz=0.5e6,
                    Rcx=1.0e6, Rcy=1.0e6, Rcz=1.0e6)

# Настраиваем параметры мульти-модели
joint_parameters = {
    'cp_material': cp_mat,
    'cp_num_planes': 100,
    'phi': 30.0,       # Угол внутреннего трения (град)
    'psi': 10.0,       # Угол дилатансии (град)
    'phi_r': 15.0,     # Остаточный угол трения (град)
    'Gf_t': 100.0,     # Энергия разрушения при отрыве (Дж/м2)
    'Gf_c': 5000.0,    # Энергия разрушения при сжатии (Дж/м2)
    'Gf_s': 500.0,     # Энергия разрушения при сдвиге (Дж/м2)
    'l_c': 0.1,        # Характеристическая длина (м)
    'mu': 0.1,         # Параметр разупрочнения
    'a_t': 1.0,
    'a_s': 1.0,
    'fcr_over_fc': 0.1,# Остаточная прочность при сжатии
    'nr_tol': 1e-8,
    'nr_max_iter': 25
}

# Инициализируем материал (Модуль Юнга = 2 ГПа, Коэффициент Пуассона = 0.2)
material = RockMaterial(E=2e9, nu=0.2, joint_params=joint_parameters)

# Создаем экземпляр тестируемой модели
model = MultiUbiquitousJointModel2D(material)

# =====================================================================
# 2. ТЕСТ: СЛОЖНЫЙ ЦИКЛ НАГРУЖЕНИЯ (ОДНООСНАЯ ДЕФОРМАЦИЯ ПО X)
# =====================================================================
# Создаем непрерывный путь нагружения для компоненты X (индекс 0)
# Вектор деформаций в 2D (Voigt): [eps_xx, eps_yy, gamma_xy]
path1 = np.linspace(0, 0.001, 150)           # 1. Отрыв (первичное разрушение)
path2 = np.linspace(0.001, -0.002, 200)[1:]  # 2. Разгрузка и уход в сильное сжатие
path3 = np.linspace(-0.002, 0.0015, 250)[1:] # 3. Повторное нагружение

loading_path = np.concatenate([path1, path2, path3])

strain = np.zeros(3)
epsilon_history = []
sigma_history = []

print("\nЗапуск теста 2D: Отрыв -> Повреждение -> Сжатие -> Повторное нагружение по оси X...")
for d_eps in loading_path:
    # Задаем жесткое деформирование только по X (без поперечного расширения/сужения)
    current_strain = np.copy(strain)
    current_strain[0] = d_eps

    # Вычисляем напряжения и тангенциальную матрицу
    stress, D_tan = model.update_state(current_strain)
    model.commit()

    epsilon_history.append(d_eps)
    sigma_history.append(stress[0])

# =====================================================================
# 3. ОТРИСОВКА ГРАФИКА
# =====================================================================
# Разделяем историю для отрисовки разными цветами
l1, l2, l3 = len(path1), len(path2), len(path3)
eps1, sig1 = epsilon_history[:l1], sigma_history[:l1]
eps2, sig2 = epsilon_history[l1 - 1:l1 + l2], sigma_history[l1 - 1:l1 + l2]
eps3, sig3 = epsilon_history[l1 + l2 - 1:], sigma_history[l1 + l2 - 1:]

plt.figure(figsize=(12, 8))

# Графики фаз
plt.plot(np.array(eps1) * 1000, np.array(sig1) / 1e6,
         color='#1f77b4', linewidth=3.5, label='1. Первичное растяжение')
plt.plot(np.array(eps2) * 1000, np.array(sig2) / 1e6,
         color='#ff7f0e', linewidth=2.5, linestyle='--', label='2. Разгрузка и Сжатие')
plt.plot(np.array(eps3) * 1000, np.array(sig3) / 1e6,
         color='#2ca02c', linewidth=2.5, linestyle='-.', label='3. Повторное нагружение')

# Линии пределов (примерные, взяты из cp_mat)
plt.axhline(0.5, color='red', linestyle='--', linewidth=1.5, label='Пик растяжения (~0.5 МПа)')
plt.axhline(-1.0, color='orange', linestyle='--', linewidth=1.5, label='Пик сжатия (~-1.0 МПа)')

# Настройка осей координат
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)

# Аннотации к ключевым зонам
plt.text(0.5, 0.25, 'Разупрочнение\n(Tension Damage)', color='#1f77b4', fontsize=10, ha='center')
plt.text(0.2, 0.05, 'Разгрузка с деградацией\nжесткости', color='#ff7f0e', fontsize=10, ha='center')
plt.text(-1.0, -0.6, 'Сжатие\n(Compression Damage)', color='#ff7f0e', fontsize=10, ha='center')

# Оформление
plt.title('MultiUbiquitousJointModel2D: Одноосное деформирование по X', fontsize=14)
plt.xlabel('Деформация $\\varepsilon_{xx}$ (x $10^{-3}$)', fontsize=12)
plt.ylabel('Напряжение $\\sigma_{xx}$ (МПа)', fontsize=12)

plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
