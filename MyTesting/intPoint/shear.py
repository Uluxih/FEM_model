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
    Rpx=0.5e6, Rpy=0.5e6, Rpz=0.5e6,
    Rcx=12.0e6, Rcy=12.0e6, Rcz=12.0e6,
)

joint_parameters = {
    'phi': 20.0,
    'psi': 0.0,
    'phi_r': 0.0,

    'cp_material': cp_mat,

    'l_c': 0.05,
    'Gf_t': 20.0,
    'Gf_c': 800.0,
    'Gf_s': 200.0,

    'a_t': 0.0,
    'a_s': 0.0,

    'mu': 0.1,

    'fcr_over_fc': 0.0,
}

material = RockMaterial(E=3500e6, nu=0.2, joint_params=joint_parameters)
model = UbiquitousJointModel2D(material)

# =====================================================================
# 2. ФИКСАЦИЯ КРИТИЧЕСКОЙ ПЛОСКОСТИ В 2D
#    Горизонтальная плоскость: нормаль n = [0, 1] (ось Y)
# =====================================================================
initial_stress = StressTensor(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
model._lock_plane(np.array([0.0, 1.0, 0.0]), initial_stress)

# Модуль сдвига — из упругих констант
G_shear = material.E / (2.0 * (1.0 + material.nu))

print("=" * 60)
print("  ПАРАМЕТРЫ ТРЕЩИНЫ ПОСЛЕ ФИКСАЦИИ (2D)")
print("=" * 60)
print(f"  Сцепление (MC-проекция)   c  = {model.c / 1e6:.4f} МПа")
print(f"  Предел растяжения         ft = {model.f_t / 1e6:.4f} МПа")
print(f"  Предел сжатия             fc = {model.f_c / 1e6:.4f} МПа")
print(f"  Угол трения               φ  = {np.degrees(model.phi):.1f}°")
print(f"  Угол дилатансии           ψ  = {np.degrees(model.psi):.1f}°")
print(f"  Модуль харденинга          Ht = {model.H_t / 1e9:.4f} ГПа")
print(f"  Нормальная жёсткость      En = {model.E_n / 1e9:.4f} ГПа")
print(f"  Модуль сдвига (E,ν)       G  = {G_shear / 1e9:.4f} ГПа")
print(f"  Энергия разрушения сдвига Gf = {model.Gf_s :.1f} Дж/м²")
print("=" * 60)

# =====================================================================
# 3. ШАГ 1: НОРМАЛЬНОЕ ОБЖАТИЕ (вдоль оси Y)
# =====================================================================
eps_yy_comp = -0.00001
strain_comp = np.zeros(3)  # Voigt 2D: [xx, yy, xy]
strain_comp[1] = eps_yy_comp

stress_comp, _ = model.update_state(strain_comp)
model.commit()

sigma_n_comp = stress_comp[1]  # σ_yy после обжатия (отрицательное = сжатие)

tau_mc_reference = model.c - sigma_n_comp * model.tan_phi
gamma_elastic = tau_mc_reference / G_shear

print(f"\n  После обжатия:")
print(f"  σ_yy                      = {sigma_n_comp / 1e6:.4f} МПа")
print(f"  τ_ref (MC, model.c)       = {tau_mc_reference / 1e6:.4f} МПа  [справочно]")
print(f"  γ_el  (τ_ref / G)         = {gamma_elastic * 1e3:.4f} ×10⁻³")

# =====================================================================
# 4. ШАГ 2: ЦИКЛИЧЕСКОЕ СДВИГОВОЕ НАГРУЖЕНИЕ (компонента xy)
#    Полная деформация: [0, eps_yy, gamma_xy]
# =====================================================================
gamma_max = gamma_elastic
N_fwd = 80
N_back = 160
N_ret = 80

loading_path = np.concatenate([
    np.linspace(0, 1.5*gamma_max, N_fwd),  # нагрузка вперёд
    np.linspace(1.5*gamma_max, -2*gamma_max, N_back),  # разгрузка + нагрузка назад
    np.linspace(-2*gamma_max, 2*gamma_max , N_ret),  # частичный возврат
])

gamma_history = []
tau_history = []
sig_n_history = []
D_s_history = []

print(f"\n  γ_max = {gamma_max * 1e3:.4f} ×10⁻³")
print("  Запуск циклического сдвига...")

for gamma_xy in loading_path:
    current_strain = strain_comp.copy()
    current_strain[2] = gamma_xy  # γ_xy — индекс 2 в Voigt 2D

    stress, _ = model.update_state(current_strain)
    model.commit()

    gamma_history.append(gamma_xy)
    tau_history.append(stress[2])  # τ_xy
    sig_n_history.append(stress[1])  # σ_yy

    # Извлекаем текущий сдвиговый damage D_s через внутреннюю функцию расчета поврежденности
    _, _, D_s_val = model._calculate_damage(
        model.W_pl_t_old, model.W_pl_c_old, model.W_pl_s_old, model.q_old, model.sig_eff_old[0]
    )
    D_s_history.append(D_s_val)

gamma_arr = np.array(gamma_history)
tau_arr = np.array(tau_history)
sig_n_arr = np.array(sig_n_history)
D_s_arr = np.array(D_s_history)

# Эмпирический пик на первой ветви нагружения
tau_peak_empirical = np.max(np.abs(tau_arr[:N_fwd]))

# Динамический предел Мора-Кулона на каждом шаге
tau_yield_dynamic = model.c - sig_n_arr * model.tan_phi

print("  Готово.")
print(f"  model.c (MC-проекция)     = {model.c / 1e6:.4f} МПа")
print(f"  τ_peak эмпирический       = {tau_peak_empirical / 1e6:.4f} МПа")
print(f"  Отношение peak / model.c  = {tau_peak_empirical / model.c:.4f}")
print(f"  Максимальный |τ_xy|       = {np.max(np.abs(tau_arr)) / 1e6:.4f} МПа")
print(f"  Итоговый damage D_s       = {D_s_arr[-1]:.4f}")

violation = np.any(np.abs(tau_arr) > tau_yield_dynamic * 1.01)
print(f"  Нарушение MC-критерия     = {violation}  (ожидается False)")

# =====================================================================
# 5. ОТРИСОВКА
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    'UbiquitousJointModel2D — циклический сдвиг на зафиксированной плоскости\n'
    f'φ = {np.degrees(model.phi):.0f}°,  '
    f'ψ = {np.degrees(model.psi):.0f}°,  '
    f'c (MC-проекция) = {model.c / 1e6:.3f} МПа,  '
    f'σ_n = {sigma_n_comp / 1e6:.3f} МПа',
    fontsize=13,
)

# ---- (a) Гистерезис τ – γ ----
ax1 = axes[0]
ax1.plot(gamma_arr * 1e3, tau_arr / 1e6,
         'b-', lw=2, label='Отклик модели')

ax1.axhline(+tau_peak_empirical / 1e6, color='green', ls='-', lw=2,
            label=f'τ_peak (эмпир.) = {+tau_peak_empirical / 1e6:.3f} МПа')
ax1.axhline(-tau_peak_empirical / 1e6, color='green', ls='-', lw=2)

ax1.axhline(+tau_mc_reference / 1e6, color='r', ls='--', lw=1.5,
            label=f'τ_ref (MC) = {+tau_mc_reference / 1e6:.3f} МПа')
ax1.axhline(-tau_mc_reference / 1e6, color='r', ls='--', lw=1.5)

ax1.set_xlabel('$\\gamma_{xy}$ (×10⁻³)', fontsize=11)
ax1.set_ylabel('$\\tau_{xy}$ (МПа)', fontsize=11)
ax1.set_title('(a) Диаграмма τ – γ (гистерезис)', fontsize=12)
ax1.grid(True, ls=':', alpha=0.7)
ax1.legend(fontsize=9)

# ---- (b) σ_n vs γ ----
ax2 = axes[1]
ax2.plot(gamma_arr * 1e3, sig_n_arr / 1e6,
         'g-', lw=2, label='$\\sigma_{yy}$ (нормальное на трещине)')
ax2.axhline(sigma_n_comp / 1e6, color='gray', ls=':', lw=1.5,
            label=f'Обжатие = {sigma_n_comp / 1e6:.3f} МПа')
ax2.set_xlabel('$\\gamma_{xy}$ (×10⁻³)', fontsize=11)
ax2.set_ylabel('$\\sigma_{yy}$ (МПа)', fontsize=11)
ax2.set_title(
    f'(b) Нормальное напряжение\n'
    f'(дилатансия ψ = {np.degrees(model.psi):.0f}°)',
    fontsize=12,
)
ax2.grid(True, ls=':', alpha=0.7)
ax2.legend(fontsize=9)

# ---- (c) Накопление damage D_s ----
ax3 = axes[2]
ax3.plot(gamma_arr * 1e3, D_s_arr, 'm-', lw=2, label='$D_s$')
ax3.axhline(1.0, color='r', ls=':', lw=1, label='Полное разрушение')
ax3.set_xlabel('$\\gamma_{xy}$ (×10⁻³)', fontsize=11)
ax3.set_ylabel('Сдвиговый damage $D_s$ (−)', fontsize=11)
ax3.set_title('(c) Накопление сдвигового damage', fontsize=12)
ax3.set_ylim(-0.02, 1.05)
ax3.grid(True, ls=':', alpha=0.7)
ax3.legend(fontsize=9)

plt.tight_layout()
plt.show()