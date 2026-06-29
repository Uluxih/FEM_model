import numpy as np
import matplotlib.pyplot as plt

from FEM.Abstract.Integration_Point_Level import Material as BaseMaterial
from FEM.Integration_Point_Level.UbiquitousJointModel2D import UbiquitousJointModel2D
from FEM.Integration_Point_Level.CriticalPlane.material import Material as CPMaterial
from FEM.Integration_Point_Level.CriticalPlane.tensor import StressTensor


# =====================================================================
# 1. ПОДГОТОВКА МАТЕРИАЛОВ
# =====================================================================

class RockMaterial(BaseMaterial):
    def __init__(self, E, nu, joint_params):
        super().__init__(E, nu)
        self.joint_params = joint_params


# Матрица анизотропии для критической плоскости
A_matrix = np.eye(9) * (0.9e6) ** 2

cp_mat = CPMaterial(
    mu=-0.3,
    A_tensor=A_matrix,
    Rpx=1.0e6, Rpy=1.0e6, Rpz=1.0e6,  # Прочность на растяжение ~1.0 МПа
    Rcx=12.0e6, Rcy=12.0e6, Rcz=12.0e6,
)
n=5
joint_parameters = {
    'phi': 30.0,
    'psi': 0.0,
    'phi_r': 0.0,

    'cp_material': cp_mat,

    'l_c': 0.5,
    'Gf_t': 50.0*n,  # Энергия разрушения при растяжении (уменьшена для наглядности спада)
    'Gf_c': 800.0*n,
    'Gf_s': 800.0*n,

    'a_t': 0.0,
    'a_s': 0.0,

    'mu': 0.0,

    'fcr_over_fc': 0.0,
}

material = RockMaterial(E=20.0e9, nu=0.2, joint_params=joint_parameters)
model = UbiquitousJointModel2D(material)

# =====================================================================
# 2. ФИКСАЦИЯ КРИТИЧЕСКОЙ ПЛОСКОСТИ В 2D
#    Горизонтальная плоскость: нормаль n = [0, 1, 0] (ось Y)
# =====================================================================
initial_stress = StressTensor(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
model._lock_plane(np.array([0.0, 1.0, 0.0]), initial_stress)

print("=" * 60)
print("  ПАРАМЕТРЫ ТРЕЩИНЫ ПОСЛЕ ФИКСАЦИИ (РАСТЯЖЕНИЕ)")
print("=" * 60)
print(f"  Предел растяжения         ft = {model.f_t / 1e6:.4f} МПа")
print(f"  Нормальная жёсткость      En = {model.E_n / 1e9:.4f} ГПа")
print(f"  Модуль харденинга          Ht = {model.H_t / 1e9:.4f} ГПа")
print(f"  Энергия разрушения раст.  Gf = {model.Gf_t :.1f} Дж/м²")
print("=" * 60)

# Оцениваем упругую деформацию до предела прочности
eps_elastic = model.f_t / model.E_n
print(f"  Упругий предел деформации ε_el = {eps_elastic * 1e3:.4f} ×10⁻³")

# =====================================================================
# 3. ПОСТРОЕНИЕ ТРАЕКТОРИИ НАГРУЖЕНИЯ
#    Управляем деформацией ε_yy (индекс 1 в Voigt 2D: [xx, yy, xy])
# =====================================================================
eps_max =  eps_elastic  # Максимальная деформация первого полуцикла

N_fwd = 100
N_back = 100
N_reload = 100

loading_path = np.concatenate([
    np.linspace(0, 30.5*eps_max, N_fwd),  # 1. Растяжение за предел прочности
    np.linspace(30.5*eps_max, -13*eps_max, N_back),  # 2. Разгрузка до нуля деформаций
    np.linspace(-13*eps_max, 20*eps_max, N_reload),  # 3. Повторное сильное растяжение
])

eps_history = []
sig_history = []
D_n_history = []
eps_p_history = []

print("\n  Запуск циклического растяжения...")

for eps_yy in loading_path:
    current_strain = np.zeros(3)
    current_strain[1] = eps_yy  # ε_yy — индекс 1 в Voigt 2D

    stress, _ = model.update_state(current_strain)
    model.commit()

    eps_history.append(eps_yy)
    sig_history.append(stress[1])  # σ_yy

    # Извлекаем текущий нормальный damage D_n
    D_nt_val, D_nc_val, _ = model._calculate_damage(
        model.W_pl_t_old, model.W_pl_c_old, model.W_pl_s_old, model.q_old, model.sig_eff_old[0]
    )
    # Так как у нас растяжение, активен растягивающий damage D_nt
    D_n_val = D_nt_val if model.sig_eff_old[0] >= 0 else D_nc_val
    D_n_history.append(D_n_val)

    # Пластическая деформация (нормальная компонента — индекс 0 в eps_p)
    eps_p_history.append(model.eps_p_old[0])

eps_arr = np.array(eps_history)
sig_arr = np.array(sig_history)
D_n_arr = np.array(D_n_history)
eps_p_arr = np.array(eps_p_history)

# Эмпирический пик прочности
sig_peak_empirical = np.max(sig_arr)

print("  Готово.")
print(f"  Заданный предел f_t       = {model.f_t / 1e6:.4f} МПа")
print(f"  Пиковое напряжение σ_yy   = {sig_peak_empirical / 1e6:.4f} МПа")
print(f"  Максимальный damage D_n   = {D_n_arr[-1]:.4f}")
print(f"  Остаточная деформация ε_p = {eps_p_arr[N_fwd + N_back - 1] * 1e3:.4f} ×10⁻³")

# =====================================================================
# 4. ОТРИСОВКА РЕЗУЛЬТАТОВ
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    'UbiquitousJointModel2D — циклическое растяжение перпендикулярно шву\n'
    f'Заданная прочность $f_t$ = {model.f_t / 1e6:.2f} МПа,  '
    f'Нормальная жесткость $E_n$ = {model.E_n / 1e9:.2f} ГПа',
    fontsize=13,
)

# ---- (a) Диаграмма σ_yy – ε_yy (Отклик) ----
ax1 = axes[0]
ax1.plot(eps_arr * 1e3, sig_arr / 1e6, 'b-', lw=2, label='Отклик шва (σ_yy)')
ax1.axhline(model.f_t / 1e6, color='r', ls='--', lw=1.5, label=f'Предел $f_t$ = {model.f_t / 1e6:.2f} МПа')
ax1.axvline(eps_elastic * 1e3, color='gray', ls=':', label='Упругая граница')

# Отрисовка секущей жесткости при разгрузке (для наглядности)
idx_unload_start = N_fwd - 1
idx_unload_end = N_fwd + N_back - 1
ax1.plot(eps_arr[idx_unload_start:idx_unload_end + 1] * 1e3,
         sig_arr[idx_unload_start:idx_unload_end + 1] / 1e6,
         'orange', lw=2.5, ls='--', label='Ветка разгрузки')

ax1.set_xlabel('Нормальная деформация $\\varepsilon_{yy}$ (×10⁻³)', fontsize=11)
ax1.set_ylabel('Нормальное напряжение $\\sigma_{yy}$ (МПа)', fontsize=11)
ax1.set_title('(a) Диаграмма σ_yy – ε_yy', fontsize=12)
ax1.grid(True, ls=':', alpha=0.7)
ax1.legend(fontsize=9)

# ---- (b) Пластическая деформация ε^p_yy vs ε_yy ----
ax2 = axes[1]
ax2.plot(eps_arr * 1e3, eps_p_arr * 1e3, 'g-', lw=2, label='Пластическая деформация $\\varepsilon^p_{yy}$')
ax2.set_xlabel('Полная деформация $\\varepsilon_{yy}$ (×10⁻³)', fontsize=11)
ax2.set_ylabel('Пластическая деформация $\\varepsilon^p_{yy}$ (×10⁻³)', fontsize=11)
ax2.set_title('(b) Накопление пластической деформации', fontsize=12)
ax2.grid(True, ls=':', alpha=0.7)
ax2.legend(fontsize=9)

# ---- (c) Накопление нормального damage D_n ----
ax3 = axes[2]
ax3.plot(eps_arr * 1e3, D_n_arr, 'm-', lw=2, label='Поврежденность $D_n$')
ax3.axhline(1.0, color='r', ls=':', lw=1, label='Полное раскрытие (D=1)')
ax3.set_xlabel('Полная деформация $\\varepsilon_{yy}$ (×10⁻³)', fontsize=11)
ax3.set_ylabel('Растягивающий damage $D_n$ (−)', fontsize=11)
ax3.set_title('(c) Накопление нормального damage', fontsize=12)
ax3.set_ylim(-0.02, 1.05)
ax3.grid(True, ls=':', alpha=0.7)
ax3.legend(fontsize=9)

plt.tight_layout()
plt.show()