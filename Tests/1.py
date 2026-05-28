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

# Задаем тензор анизотропии так, чтобы базовое сцепление C равнялось 1 МПа
# Формула внутри: C = sqrt(v * A * v). Если A = I * (1e6)^2, то C = 1e6
A_matrix = np.eye(9) * (0.4e6)**2

# Создаем материал для Critical Plane с настроенным тензором
cp_mat = CPMaterial(mu=-0.3, A_tensor=A_matrix, Rpx=1e6, Rpy=1e6, Rpz=1e6, Rcx=2e6, Rcy=2e6, Rcz=2e6)

# Настраиваем параметры породы и трещины
joint_parameters = {
    'kn': 50e9,
    'ks': 10e9,
    'kt': 10e9,
    'spacing': 0.1,
    'phi': 0,  # Угол внутреннего трения: 0 (чистое сцепление для наглядности)
    'psi': 0,  # Дилатансия: 0
    't': 0,    # Прочность на разрыв: 0
    'cp_material': cp_mat  # Передаем объект CPMaterial


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

# НОВОВВЕДЕНИЕ: Передаем тензор напряжений в метод фиксации
# Сцепление (model.c) будет вычислено динамически внутри этого метода!
model._lock_plane(np.array([0.0, 0.0, 1.0]), dummy_stress)

# =====================================================================
# 3. ШАГ 1: НОРМАЛЬНОЕ ОБЖАТИЕ
# =====================================================================
strain = np.zeros(6)
strain[2] = -0.0005  # Сжатие по оси Z
stress, _ = model.update_state(strain)
model.commit()

sigma_n = stress[2]
print(f"Нормальное напряжение на трещине: {sigma_n / 1e6:.2f} МПа")

# НОВОВВЕДЕНИЕ: Теоретический предел сдвига теперь берется из динамически рассчитанного model.c
tau_yield_theory = model.c - sigma_n * np.tan(np.radians(material.joint_params['phi']))
print(f"Теоретический предел текучести (сдвиг): {tau_yield_theory / 1e6:.2f} МПа")

# =====================================================================
# 4. ШАГ 2: ЦИКЛИЧЕСКОЕ СДВИГОВОЕ НАГРУЖЕНИЕ
# =====================================================================
gamma_history = []
tau_history = []

# Создаем путь нагружения: 0 -> 0.002 -> -0.002 -> 0
loading_path = np.concatenate([
    np.linspace(0, 0.0001, 50),
    np.linspace(0.0001, -0.0003, 100),
    np.linspace(-0.0003, 0.0004, 50)
])

print("Запуск циклического сдвига...")
for d_gamma in loading_path:
    current_strain = np.copy(strain)
    current_strain[5] = d_gamma  # Задаем сдвиговую деформацию XZ

    stress, D_tan = model.update_state(current_strain)
    model.commit()

    gamma_history.append(d_gamma)
    tau_history.append(stress[5])

# =====================================================================
# 5. ОТРИСОВКА ГРАФИКА
# =====================================================================
plt.figure(figsize=(9, 6))
plt.plot(np.array(gamma_history) * 1000, np.array(tau_history) / 1e6, 'b-', linewidth=2,
         label='Реакция модели (Return Mapping)')

# Рисуем линии теоретического предела текучести
plt.axhline(tau_yield_theory / 1e6, color='r', linestyle='--', label='Теоретический предел текучести ($+\\tau_{max}$)')
plt.axhline(-tau_yield_theory / 1e6, color='r', linestyle='--', label='Теоретический предел текучести ($-\\tau_{max}$)')

plt.title('Поведение зафиксированной трещины при циклическом сдвиге\n(Сцепление вычислено динамически)', fontsize=14)
plt.xlabel('Сдвиговая деформация $\\gamma_{xz}$ (x $10^{-3}$)', fontsize=12)
plt.ylabel('Касательное напряжение $\\tau_{xz}$ (МПа)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
