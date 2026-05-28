import numpy as np
import matplotlib.pyplot as plt

# Импорты из вашей кодовой базы
from FEM.Abstract.Integration_Point_Level import Material as BaseMaterial
from FEM.Integration_Point_Level.UbiquitousJointModel3D import UbiquitousJointModel3D
from FEM.Integration_Point_Level.CriticalPlane.material import Material as CPMaterial
from FEM.Integration_Point_Level.CriticalPlane.tensor import StressTensor


# =====================================================================
# 1. ПОДГОТОВКА МАТЕРИАЛОВ
# =====================================================================

class RockMaterial(BaseMaterial):
    def __init__(self, E, nu, joint_params):
        super().__init__(E, nu)
        self.joint_params = joint_params


# Задаем тензор анизотропии
A_matrix = np.eye(9) * (0.4e6) ** 2

# Создаем материал для Critical Plane.
# Пределы прочности: Растяжение = 0.5 МПа, Сжатие = 1.0 МПа
cp_mat = CPMaterial(mu=-0.3, A_tensor=A_matrix,
                    Rpx=1.0e6, Rpy=01.0e6, Rpz=01.0e6,
                    Rcx=1.0e6, Rcy=1.0e6, Rcz=1.0e6)

# Настраиваем параметры породы и трещины по модели Minga (2017)
joint_parameters = {
    'cp_material': cp_mat,
    'phi': 0.0,
    'psi': 0.0,
    'R_inf': 15.0,
    'b_param': 5.0,
    'Gf_t': 150.5,
    'Gf_c': 150.5,
    'l_c': 0.1,  # Характеристическая длина (м)
}

# joint_parameters = {
#     'phi': 0.0,
#     'psi': 30.0,
#     'cp_material': cp_mat,
#
#     # Новые параметры Damage-Plasticity
#     'Gf_t': 150.0,  # Энергия разрушения при отрыве (Дж/м2)
#     'Gf_c': 150.0,  # Энергия разрушения при сжатии (Дж/м2)
#     'l_c': 0.1,  # Характеристическая длина (м)
#     'lambda_param': 0.15  # Контроль остаточной деформации (15% от макс. деформации)
# }

# Инициализируем материал (E = 2 ГПа, nu = 0.2)
material = RockMaterial(E=2e9, nu=0.2, joint_params=joint_parameters)
model = UbiquitousJointModel3D(material)

# =====================================================================
# 2. ПРИНУДИТЕЛЬНАЯ ФИКСАЦИЯ ТРЕЩИНЫ
# =====================================================================
dummy_stress = StressTensor(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
model._lock_plane(np.array([0.0, 0.0, 1.0]), dummy_stress)

# Извлекаем рассчитанные пределы для графиков (теперь они f_t и f_c)
f_t = model.f_t
f_c = model.f_c

# =====================================================================
# 3. ТЕСТ: СЛОЖНЫЙ ЦИКЛ НАГРУЖЕНИЯ
# =====================================================================
# Создаем непрерывный путь нагружения
path1 = np.linspace(0, 0.0015, 150)  # 1. Отрыв (первичное разрушение)
path2 = np.linspace(0.0015, -0.0008, 200)[1:]  # 2. Разгрузка и уход в сильное сжатие
path3 = np.linspace(-0.0008, 0.0005, 250)[1:]  # 3. Повторное нагружение (идем дальше максимума!)

loading_path = np.concatenate([path1, path2, path3])

strain = np.zeros(6)
epsilon_history = []
sigma_history = []

print("\nЗапуск теста: Отрыв -> Повреждение -> Сжатие -> Повторное нагружение...")
for d_eps in loading_path:
    current_strain = np.copy(strain)
    current_strain[2] = d_eps

    stress, D_tan = model.update_state(current_strain)
    model.commit()

    epsilon_history.append(d_eps)
    sigma_history.append(stress[2])

# =====================================================================
# 4. ОТРИСОВКА ГРАФИКА
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

# Линии пределов
plt.axhline(f_t / 1e6, color='red', linestyle='--', linewidth=1.5, label=f'Пик растяжения ({f_t / 1e6:.2f} МПа)')
plt.axhline(-f_c / 1e6, color='orange', linestyle='--', linewidth=1.5, label=f'Пик сжатия ({-f_c / 1e6:.2f} МПа)')

# Настройка осей координат
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)

# Аннотации к ключевым зонам
plt.text(0.7, 0.25, 'Полиномиальное\nразупрочнение (Tension)', color='#1f77b4', fontsize=10, ha='center')
plt.text(0.3, 0.06, 'Разгрузка в $\\varepsilon^p_n$\n(Остаточная деформация)', color='#ff7f0e', fontsize=10,
         ha='center')
plt.text(-0.05, -0.4, 'Закрытие\nтрещины (E)', color='purple', fontsize=10, ha='right')
plt.text(-0.9, -0.8, 'Синусоидальное\nразупрочнение (Compression)', color='#ff7f0e', fontsize=10, ha='center')
plt.text(1.7, 0.15, 'Продолжение\nразрушения', color='#2ca02c', fontsize=10, ha='center')

# Оформление
plt.title('Модель Minga (2017): Damage-Plasticity (Растяжение + Сжатие)', fontsize=14)
plt.xlabel('Нормальная деформация $\\varepsilon_{zz}$ (x $10^{-3}$)', fontsize=12)
plt.ylabel('Нормальное напряжение $\\sigma_{zz}$ (МПа)', fontsize=12)

plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
