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
# Пределы прочности: Растяжение = 0.5 МПа, Сжатие = 1.0 МПа
cp_mat = CPMaterial(mu=-0.3, A_tensor=A_matrix,
                    Rpx=0.5e6, Rpy=0.5e6, Rpz=0.5e6,
                    Rcx=1.0e6, Rcy=1.0e6, Rcz=1.0e6)

# Настраиваем параметры породы и трещины
joint_parameters = {
    'phi': 0.0,      # Угол внутреннего трения
    'psi': 0.0,      # Дилатансия
    'cp_material': cp_mat,

    # --- ПАРАМЕТРЫ УПРОЧНЕНИЯ (Saksala) ---
    # Начальный предел (0.5 МПа) вырастет на 0.3 МПа (до 0.8 МПа)
    'R_inf': 0.3e6,
    'b_param': 20000.0   # Скорость выхода на насыщение
}

# Инициализируем материал (E = 2 ГПа, nu = 0.2)
material = RockMaterial(E=2e9, nu=0.2, joint_params=joint_parameters)

# Создаем экземпляр модели
model = UbiquitousJointModel3D(material)

# =====================================================================
# 2. ПРИНУДИТЕЛЬНАЯ ФИКСАЦИЯ ТРЕЩИНЫ
# =====================================================================
# Создаем "нулевой" тензор напряжений для инициализации
dummy_stress = StressTensor(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

# Фиксируем горизонтальную плоскость (0, 0, 1)
model._lock_plane(np.array([0.0, 0.0, 1.0]), dummy_stress)

# Извлекаем рассчитанные моделью пределы прочности для построения графиков
t_limit_initial = model.t_limit
c_limit_theory = model.c_limit
t_limit_saturated = t_limit_initial + joint_parameters['R_inf']

# =====================================================================
# 3. ТЕСТ: НОРМАЛЬНОЕ НАГРУЖЕНИЕ (ОТРЫВ И СЖАТИЕ)
# =====================================================================
strain = np.zeros(6)
epsilon_history = []
sigma_history = []

# Путь нагружения: 0 -> Растяжение -> Разгрузка и Сжатие -> Возврат к 0
# При E = 2 ГПа текучесть наступит при деформации 0.00025.
# Идем до 0.0006, чтобы увидеть плавный выход на насыщение.
loading_path = np.concatenate([
    np.linspace(0, 0.0006, 120),         # Отрыв (растяжение) с выходом на насыщение
    np.linspace(0.0006, -0.0015, 200),   # Разгрузка (упругая) и переход в сильное сжатие
    np.linspace(-0.0015, 0, 80)          # Разгрузка из сжатия к нулю
])

print("\nЗапуск теста на отрыв (с упрочнением) и сжатие...")
for d_eps in loading_path:
    current_strain = np.copy(strain)
    current_strain[2] = d_eps  # Нормальная деформация Z

    # Обновляем состояние (Return Mapping)
    stress, D_tan = model.update_state(current_strain)
    model.commit()

    epsilon_history.append(d_eps)
    sigma_history.append(stress[2])


# =====================================================================
# 4. ОТРИСОВКА ГРАФИКА
# =====================================================================
plt.figure(figsize=(10, 7))

# График реакции модели
plt.plot(np.array(epsilon_history) * 1000, np.array(sigma_history) / 1e6,
         color='#1f77b4', linewidth=2.5, label='Траектория напряжений модели')

# Линии пределов на растяжение
plt.axhline(t_limit_initial / 1e6, color='red', linestyle='--', linewidth=1.5,
            label=f'Начальный предел текучести (+{t_limit_initial/1e6:.2f} МПа)')
plt.axhline(t_limit_saturated / 1e6, color='darkred', linestyle=':', linewidth=2,
            label=f'Предел насыщения (+{t_limit_saturated/1e6:.2f} МПа)')

# Линия предела на сжатие
plt.axhline(-c_limit_theory / 1e6, color='orange', linestyle='--', linewidth=2,
            label=f'Предел на сжатие (-{c_limit_theory/1e6:.2f} МПа)')

# Настройка осей и оформления
plt.title('Поведение зафиксированной трещины: Упрочнение при отрыве и Срез при сжатии', fontsize=14)
plt.xlabel('Нормальная деформация $\\varepsilon_{zz}$ (x $10^{-3}$)', fontsize=12)
plt.ylabel('Нормальное напряжение $\\sigma_{zz}$ (МПа)', fontsize=12)

# Добавляем оси координат
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)

# Аннотации для наглядности фаз
plt.text(0.3, 0.65, 'Нелинейное\nупрочнение', color='darkred', fontsize=10, ha='center')
plt.text(-0.75, -1.05, 'Пластическое течение\n(Сжатие)', color='orange', fontsize=10, ha='center')

plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
