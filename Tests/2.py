import numpy as np
import matplotlib.pyplot as plt

# Импорты из вашей реальной кодовой базы
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
A_matrix = np.eye(9) * (0.4e6)**2

# Создаем материал для Critical Plane.
# НОВОВВЕДЕНИЕ: Задаем пределы прочности: Растяжение = 0.5 МПа, Сжатие = 6.0 МПа
cp_mat = CPMaterial(mu=-0.3, A_tensor=A_matrix,
                    Rpx=0.5e6, Rpy=0.5e6, Rpz=0.5e6,
                    Rcx=1.0e6, Rcy=1.0e6, Rcz=1.0e6)

# Настраиваем параметры породы и трещины
joint_parameters = {
    'kn': 50e9,
    'ks': 10e9,
    'kt': 10e9,
    'spacing': 0.1,
    'phi': 0,      # Угол внутреннего трения: 0
    'psi': 0,      # Дилатансия: 0
    # Параметр 't' удален, так как модель теперь считает его сама из cp_material
    'cp_material': cp_mat
}

# Инициализируем материал (E = 20 ГПа, nu = 0.2)
material = RockMaterial(E=20e9, nu=0.2, joint_params=joint_parameters)

# Создаем экземпляр модели
model = UbiquitousJointModel3D(material)

# =====================================================================
# 2. ПРИНУДИТЕЛЬНАЯ ФИКСАЦИЯ ТРЕЩИНЫ
# =====================================================================
# Создаем "нулевой" тензор напряжений для инициализации
dummy_stress = StressTensor(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

# Фиксируем горизонтальную плоскость (0, 0, 1)
# В этот момент модель сама посчитает t_limit и c_limit
model._lock_plane(np.array([0.0, 0.0, 1.0]), dummy_stress)

# Извлекаем рассчитанные моделью пределы прочности для построения графиков
t_limit_theory = model.t_limit
c_limit_theory = model.c_limit

# =====================================================================
# 3. ТЕСТ: НОРМАЛЬНОЕ НАГРУЖЕНИЕ (ОТРЫВ И СЖАТИЕ)
# =====================================================================
strain = np.zeros(6)
epsilon_history = []
sigma_history = []

# Путь нагружения: 0 -> Растяжение -> Сильное сжатие -> Возврат к 0
loading_path = np.concatenate([
    np.linspace(0, 0.0004, 50),         # Отрыв (растяжение)
    np.linspace(0.0004, -0.0020, 100),  # Разгрузка и переход в сильное сжатие
    np.linspace(-0.0020, 0, 50)         # Возврат к нулю
])

print("\nЗапуск теста на отрыв и сжатие...")
for d_eps in loading_path:
    current_strain = np.copy(strain)
    current_strain[2] = d_eps  # Нормальная деформация Z

    # Обновляем состояние (Return Mapping срежет напряжения, если f_t > 0 или f_c > 0)
    stress, D_tan = model.update_state(current_strain)
    model.commit()

    epsilon_history.append(d_eps)
    sigma_history.append(stress[2])


# =====================================================================
# 4. ОТРИСОВКА ГРАФИКА
# =====================================================================
plt.figure(figsize=(9, 7))

# График реакции модели
plt.plot(np.array(epsilon_history) * 1000, np.array(sigma_history) / 1e6, 'g-', linewidth=2, label='Реакция модели')

# Линия предела на растяжение (сверху)
plt.axhline(t_limit_theory / 1e6, color='r', linestyle='--', label=f'Предел на отрыв (+{t_limit_theory/1e6:.2f} МПа)')

# Линия предела на сжатие (снизу)
plt.axhline(-c_limit_theory / 1e6, color='orange', linestyle='--', label=f'Предел на сжатие (-{c_limit_theory/1e6:.2f} МПа)')

# Настройка осей и оформления
plt.title('Поведение зафиксированной трещины при отрыве и сжатии\n(Динамический расчет пределов)', fontsize=14)
plt.xlabel('Нормальная деформация $\\varepsilon_{zz}$ (x $10^{-3}$)', fontsize=12)
plt.ylabel('Нормальное напряжение $\\sigma_{zz}$ (МПа)', fontsize=12)

# Добавляем оси координат для наглядности
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)

plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
