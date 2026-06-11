import numpy as np
import matplotlib.pyplot as plt

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


# Тензор анизотропии: задаёт c ≈ 0.5 МПа, ft ≈ 1 МПа, fc ≈ 4 МПа
# (точные значения зависят от реализации get_*_limit в вашей кодовой базе)
A_matrix = np.eye(9) * (0.2e6) ** 2

cp_mat = CPMaterial(
    mu=-0.3,
    A_tensor=A_matrix,
    Rpx=1e6, Rpy=1e6, Rpz=1e6,  # пределы растяжения по осям
    Rcx=4e6, Rcy=4e6, Rcz=4e6   # пределы сжатия (> |σ_n| при обжатии)
)

# ИСПРАВЛЕНИЕ 1: phi, psi, phi_r — в ГРАДУСАХ; модель сама делает np.radians()
# ИСПРАВЛЕНИЕ 2: kn/ks/kt/spacing/t НЕ используются в UbiquitousJointModel3D — убраны
# ИСПРАВЛЕНИЕ 3: добавлены все параметры, которые реально читает __init__
joint_parameters = {
    'phi':   20.0,    # угол трения на трещине, градусы (0 = чистое сцепление)
    'psi':   0.0,    # угол дилатансии, градусы
    'phi_r': 10.0,    # остаточный угол трения, градусы

    'cp_material': cp_mat,

    # Регуляризованные энергии разрушения (Дж/м²)
    # Выбраны большими, чтобы damage был мал на масштабе теста
    'l_c':    0.5,    # характерный размер элемента, м
    'Gf_t':   200.0,  # Дж/м² — растяжение
    'Gf_c':  8000.0,  # Дж/м² — сжатие
    'Gf_s':  2000.0,  # Дж/м² — сдвиг

    # Коэффициенты перекрёстного damage
    'a_t': 1.0,
    'a_s': 1.0,

    # Параметр остаточной нормальной деформации (Minga, Eq. 23)
    'mu': 0.1,

    # Доля остаточной прочности при сжатии
    'fcr_over_fc': 0.0,
}

material = RockMaterial(E=20.0e9, nu=0.2, joint_params=joint_parameters)
model = UbiquitousJointModel3D(material)

# =====================================================================
# 2. ФИКСАЦИЯ КРИТИЧЕСКОЙ ПЛОСКОСТИ
#    Горизонтальная плоскость: нормаль n = [0, 0, 1] (ось Z)
# =====================================================================
# ИСПРАВЛЕНИЕ 4: _lock_plane принимает StressTensor, а не словарь/массив.
#               Это уже было в исходном тесте — оставляем.
initial_stress = StressTensor(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
model._lock_plane(np.array([0.0, 0.0, 1.0]), initial_stress)

# После фиксации модель сообщает рассчитанные прочностные параметры
print("=" * 55)
print("  ПАРАМЕТРЫ ТРЕЩИНЫ ПОСЛЕ ФИКСАЦИИ")
print("=" * 55)
print(f"  Сцепление            c  = {model.c   / 1e6:.4f} МПа")
print(f"  Предел растяжения    ft = {model.f_t / 1e6:.4f} МПа")
print(f"  Предел сжатия        fc = {model.f_c / 1e6:.4f} МПа")
print(f"  Угол трения          φ  = {np.degrees(model.phi):.1f}°")
print(f"  Угол дилатансии      ψ  = {np.degrees(model.psi):.1f}°")
print(f"  Модуль хардеинга     Ht = {model.H_t / 1e9:.4f} ГПа")
print(f"  Нормальная жёсткость En = {model.E_n / 1e9:.4f} ГПа")
print(f"  Сдвиговая жёсткость  Gs = {model.G_s / 1e9:.4f} ГПа")
print("=" * 55)

# =====================================================================
# 3. ШАГ 1: НОРМАЛЬНОЕ ОБЖАТИЕ (упругое; |σ_n| < fc)
#
#    Модель использует ПОЛНУЮ деформацию (total strain formulation):
#    eps_elastic = current_strain − eps_plastic_old
#    Поэтому мы просто задаём текущее состояние eps, НЕ приращение.
# =====================================================================
eps_zz_comp  = -00.001       # деформация сжатия по нормали к трещине
strain_comp  = np.zeros(6)
# Voigt-порядок: [eps_xx, eps_yy, eps_zz, gamma_xy, gamma_yz, gamma_xz]
strain_comp[2] = eps_zz_comp  # eps_zz

stress_comp, _ = model.update_state(strain_comp)
model.commit()

sigma_n_comp = stress_comp[2]  # нормальное напряжение на трещине = σ_zz
print(f"\n  После обжатия:")
print(f"  σ_zz = {sigma_n_comp / 1e6:.4f} МПа  (σ_n < 0 — сжатие)")

# ИСПРАВЛЕНИЕ 5: используем model.tan_phi вместо пересчёта из градусов,
#               что согласовано с внутренней логикой модели.
# Критерий Мора–Кулона:  f_s = τ + σ_n·tanφ − c = 0
#   ⟹  τ_max = c − σ_n·tanφ
# При σ_n < 0 (сжатие): τ_max = c + |σ_n|·tanφ
tau_yield_theory = model.c - sigma_n_comp * model.tan_phi
gamma_elastic    = tau_yield_theory / model.G_s

print(f"  τ_max (теория M-C) = {tau_yield_theory / 1e6:.4f} МПа")
print(f"  γ_el  (теория)     = {gamma_elastic * 1e3:.4f} ×10⁻³")

# =====================================================================
# 4. ШАГ 2: ЦИКЛИЧЕСКОЕ СДВИГОВОЕ НАГРУЖЕНИЕ
#
#    На каждом шаге задаём ПОЛНЫЙ вектор деформаций:
#      strain = [0, 0, eps_zz_comp, 0, 0, gamma_xz_current]
#    Нормальная деформация фиксирована (обжатие сохранено).
# =====================================================================
# Диапазон деформаций: ±2.5 упругих предела, чтобы наглядно видеть плато
gamma_max = 05.5 * gamma_elastic

# Путь: 0 → +γ_max → −γ_max → +γ_max/2
loading_path = np.concatenate([
    np.linspace(0,          +gamma_max,  80),   # нагрузка вперёд
    np.linspace(+gamma_max, -gamma_max, 160),   # разгрузка + нагрузка назад
    np.linspace(-gamma_max, +gamma_max/2, 80),  # частичный возврат
])

gamma_history = []
tau_history   = []
sig_n_history = []
D_s_history   = []

print(f"\n  γ_max в тесте = {gamma_max * 1e3:.4f} ×10⁻³")
print("  Запуск циклического сдвига...")

for gamma_xz in loading_path:
    # ИСПРАВЛЕНИЕ 6: берём strain_comp.copy(), а не np.zeros(6),
    #               чтобы сохранить нормальную деформацию eps_zz
    current_strain    = strain_comp.copy()
    current_strain[5] = gamma_xz   # γ_xz — индекс 5 в Voigt

    stress, _ = model.update_state(current_strain)
    model.commit()

    gamma_history.append(gamma_xz)
    tau_history.append(stress[5])       # τ_xz — индекс 5
    sig_n_history.append(stress[2])     # σ_zz — индекс 2
    D_s_history.append(model.D_s_old)  # накопленный сдвиговой damage

gamma_arr = np.array(gamma_history)
tau_arr   = np.array(tau_history)
sig_n_arr = np.array(sig_n_history)
D_s_arr   = np.array(D_s_history)

print("  Готово.")
print(f"  Максимальный |τ_xz| = {np.max(np.abs(tau_arr)) / 1e6:.4f} МПа")
print(f"  Итоговый damage D_s = {D_s_arr[-1]:.4f}")

# Проверка: пиковый τ не должен превышать теорию более чем на 1%
ratio = np.max(np.abs(tau_arr)) / tau_yield_theory
print(f"  τ_peak / τ_theory   = {ratio:.4f}  (ожидается ≤ 1.01)")

# =====================================================================
# 5. ОТРИСОВКА (3 субплота)
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    'UbiquitousJointModel3D — циклический сдвиг на зафиксированной плоскости\n'
    f'φ = {np.degrees(model.phi):.0f}°,  '
    f'ψ = {np.degrees(model.psi):.0f}°,  '
    f'c = {model.c / 1e6:.3f} МПа,  '
    f'σ_n = {sigma_n_comp / 1e6:.3f} МПа',
    fontsize=13
)

# ---- (a) Гистерезис τ – γ ----
ax1 = axes[0]
ax1.plot(gamma_arr * 1e3, tau_arr / 1e6, 'b-', lw=2, label='Отклик модели')
ax1.axhline(+tau_yield_theory / 1e6, color='r', ls='--', lw=1.5,
            label=f'+τ_max = {+tau_yield_theory / 1e6:.3f} МПа')
ax1.axhline(-tau_yield_theory / 1e6, color='r', ls='--', lw=1.5,
            label=f'−τ_max = {-tau_yield_theory / 1e6:.3f} МПа')
ax1.set_xlabel('$\\gamma_{xz}$ (×10⁻³)', fontsize=11)
ax1.set_ylabel('$\\tau_{xz}$ (МПа)', fontsize=11)
ax1.set_title('(a) Диаграмма τ – γ (гистерезис)', fontsize=12)
ax1.grid(True, ls=':', alpha=0.7)
ax1.legend(fontsize=9)

# ---- (b) σ_n vs γ (эффект дилатансии) ----
ax2 = axes[1]
ax2.plot(gamma_arr * 1e3, sig_n_arr / 1e6, 'g-', lw=2,
         label='$\\sigma_{zz}$ (нормальное на трещине)')
ax2.axhline(sigma_n_comp / 1e6, color='gray', ls=':', lw=1.5,
            label=f'Обжатие = {sigma_n_comp / 1e6:.3f} МПа')
ax2.set_xlabel('$\\gamma_{xz}$ (×10⁻³)', fontsize=11)
ax2.set_ylabel('$\\sigma_{zz}$ (МПа)', fontsize=11)
ax2.set_title(
    f'(b) Нормальное напряжение\n'
    f'(дилатансия ψ = {np.degrees(model.psi):.0f}°)',
    fontsize=12
)
ax2.grid(True, ls=':', alpha=0.7)
ax2.legend(fontsize=9)

# ---- (c) Накопление damage D_s ----
ax3 = axes[2]
ax3.plot(gamma_arr * 1e3, D_s_arr, 'm-', lw=2, label='$D_s$')
ax3.axhline(1.0, color='r', ls=':', lw=1, label='Полное разрушение')
ax3.set_xlabel('$\\gamma_{xz}$ (×10⁻³)', fontsize=11)
ax3.set_ylabel('Сдвиговый damage $D_s$ (−)', fontsize=11)
ax3.set_title('(c) Накопление сдвигового damage', fontsize=12)
ax3.set_ylim(-0.02, 1.05)
ax3.grid(True, ls=':', alpha=0.7)
ax3.legend(fontsize=9)

plt.tight_layout()
plt.show()
