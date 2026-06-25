import numpy as np
import matplotlib.pyplot as plt

from FEM.Integration_Point_Level.UbiquitousJointModel3D import UbiquitousJointModel3D
from FEM.Integration_Point_Level.CriticalPlane.material import Material as CPMaterial
from FEM.Integration_Point_Level.CriticalPlane.tensor import StressTensor
from FEM.Abstract.Integration_Point_Level import Material as BaseMaterial


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательный класс
# ─────────────────────────────────────────────────────────────────────────────
class RockMaterial(BaseMaterial):
    def __init__(self, E, nu, joint_params):
        super().__init__(E, nu)
        self.joint_params = joint_params


# ─────────────────────────────────────────────────────────────────────────────
# Параметры — ТОЧЬ-В-ТОЧЬ как в Simple Shear тесте
# ─────────────────────────────────────────────────────────────────────────────
ROCK_E  = 0.460e4          # Па  (как в вашем МКЭ-тесте)
ROCK_NU = 0.2

# Характерная длина: vol_el = (0.1/1)*(1.0/5)*(1.0/5) = 0.004;  l_c = 0.004^(1/5)
SIZE_X, SIZE_Y, SIZE_Z = 0.10, 1.0, 1.0
nx, ny, nz = 1, 5, 5
vol_el   = (SIZE_X / nx) * (SIZE_Y / ny) * (SIZE_Z / nz)
char_len = vol_el ** (1.0 / 5.0)

ROCK_RP = 0.90e5          # Па  (как в вашем МКЭ-тесте — ОЧЕНЬ БОЛЬШОЕ)
ROCK_RC = 1.50e20

A_matrix = np.eye(9) * ROCK_RP**2

cp_mat = CPMaterial(
    mu      = 0.5,
    A_tensor= A_matrix,
    Rpx=ROCK_RP, Rpy=ROCK_RP, Rpz=ROCK_RP,
    Rcx=ROCK_RC, Rcy=ROCK_RC, Rcz=ROCK_RC,
)

joint_params = {
    'phi'         : 0.0,
    'psi'         :  0.0,    # убрали дилатансию, чтобы не мешала
    'phi_r'       :  0.0,
    'cp_material' : cp_mat,
    'l_c'         : char_len,
    'Gf_t'        :  100.0,
    'Gf_c'        : 5000.0,
    'Gf_s'        :  5000.0,
    'a_t'         :    0.0,
    'a_s'         :    0.0,
    'mu'          :    0.1,
    'fcr_over_fc' :    0.0,
}

G_shear = ROCK_E / (2.0 * (1.0 + ROCK_NU))

print("=" * 65)
print("  ПАРАМЕТРЫ ТЕСТА")
print("=" * 65)
print(f"  E  = {ROCK_E:.3e} Па   nu = {ROCK_NU}")
print(f"  G  = {G_shear:.3e} Па")
print(f"  Rp = {ROCK_RP:.3e} Па   ← критерий срабатывания")
print(f"  l_c = {char_len:.5f} м  (char. size)")
print(f"  Gf_s (регуляр.) = {joint_params['Gf_s']/char_len:.2f} Дж/м²")
print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# ВАРИАНТ A: Автоматическая блокировка (как в МКЭ)
# ─────────────────────────────────────────────────────────────────────────────
def run_AUTO(n_steps=200, gamma_max_factor=5.0):
    """Плоскость блокируется автоматически через критерии."""
    mat   = RockMaterial(E=ROCK_E, nu=ROCK_NU, joint_params=joint_params)
    model = UbiquitousJointModel3D(mat)

    tau_el  = G_shear * 1e4           # малый пробный масштаб
    gamma_u = gamma_max_factor * tau_el / G_shear

    strains = np.linspace(0, gamma_u, n_steps)

    locked_at = None
    gamma_hist, tau_hist, D_s_hist = [], [], []

    print("\n  [ВАРИАНТ A] Автоматическая блокировка")
    print(f"  {'шаг':>4}  {'γ_xz':>10}  {'τ_xz [Па]':>14}  {'D_s':>8}  {'locked?':>8}")
    print("  " + "-" * 55)

    for i, gamma in enumerate(strains):
        eps = np.zeros(6)
        eps[5] = gamma               # γ_xz — индекс 5

        stress, _ = model.update_state(eps)
        model.commit()

        if model.is_locked and locked_at is None:
            locked_at = gamma
            print(f"  >>> ПЛОСКОСТЬ ЗАФИКСИРОВАНА на γ = {gamma:.4e}, "
                  f"нормаль = {model.fixed_normal}")
            print(f"      c = {model.c:.3e} Па,  ft = {model.f_t:.3e} Па,  "
                  f"fc = {model.f_c:.3e} Па")

        gamma_hist.append(gamma)
        tau_hist.append(stress[5])
        D_s_hist.append(model.D_s_old)

        if i % (n_steps // 20) == 0:
            print(f"  {i:>4}  {gamma:>10.4e}  {stress[5]:>14.4e}  "
                  f"{model.D_s_old:>8.4f}  {model.is_locked!s:>8}")

    if locked_at is None:
        print("\n  ⚠️  ПЛОСКОСТЬ ТАК И НЕ ЗАФИКСИРОВАЛАСЬ!")
        print(f"      Максимальный τ достигнут = {max(tau_hist):.3e} Па")
        print(f"      Rp = {ROCK_RP:.3e} Па")
        print(f"      → Нагрузка в {ROCK_RP / max(tau_hist):.1f}x меньше порога срабатывания")

    return np.array(gamma_hist), np.array(tau_hist), np.array(D_s_hist), locked_at


# ─────────────────────────────────────────────────────────────────────────────
# ВАРИАНТ B: Явная блокировка (как в рабочем примере)
# ─────────────────────────────────────────────────────────────────────────────
def run_MANUAL(n_steps=200, gamma_max_factor=5.0):
    """Плоскость фиксируется вручную. Поведение материала без МКЭ-цикла."""
    mat   = RockMaterial(E=ROCK_E, nu=ROCK_NU, joint_params=joint_params)
    model = UbiquitousJointModel3D(mat)

    # Фиксируем горизонтальную плоскость (нормаль Z)
    model._lock_plane(np.array([0.0, 0.0, 1.0]), StressTensor(0,0,0,0,0,0))

    print(f"\n  [ВАРИАНТ B] Явная блокировка  n = [0, 0, 1]")
    print(f"      c = {model.c:.3e} Па,  ft = {model.f_t:.3e} Па,  "
          f"fc = {model.f_c:.3e} Па")
    print(f"      Ht = {model.H_t:.3e} Па,  En = {model.E_n:.3e} Па,  "
          f"G_s = {model.G_s:.3e} Па")

    tau_yield = model.c   # phi=30°, phi_r=0°, sigN=0 → τ = c − 0*tan(φ)
    gamma_u   = gamma_max_factor * tau_yield / G_shear

    strains = np.concatenate([
        np.linspace(0, +gamma_u*1.1, n_steps),
        # np.linspace(+gamma_u, -gamma_u, 2 * n_steps)[1:],
        # np.linspace(-gamma_u, +gamma_u / 2, n_steps)[1:],
    ])

    gamma_hist, tau_hist, D_s_hist = [], [], []

    print(f"\n  τ_yield (при σn=0) = {tau_yield:.3e} Па")
    print(f"  γ_max = {gamma_u:.4e}")
    print(f"\n  {'шаг':>4}  {'γ_xz':>10}  {'τ_xz [Па]':>14}  {'σzz [Па]':>14}  "
          f"{'D_s':>8}  {'Wpl_s':>10}")
    print("  " + "-" * 70)

    for i, gamma in enumerate(strains):
        eps    = np.zeros(6)
        eps[5] = gamma

        stress, _ = model.update_state(eps)
        model.commit()

        gamma_hist.append(gamma)
        tau_hist.append(stress[5])
        D_s_hist.append(model.D_s_old)

        if i % (len(strains) // 25) == 0:
            print(f"  {i:>4}  {gamma:>10.4e}  {stress[5]:>14.4e}  "
                  f"{stress[2]:>14.4e}  {model.D_s_old:>8.4f}  "
                  f"{model.W_pl_s_old:>10.4e}")

    print(f"\n  Итоговый D_s = {model.D_s_old:.6f}")
    return np.array(gamma_hist), np.array(tau_hist), np.array(D_s_hist)


# ─────────────────────────────────────────────────────────────────────────────
# ВАРИАНТ C: Диагностика критериев срабатывания
# ─────────────────────────────────────────────────────────────────────────────
def diagnose_locking_criteria(n_steps=50):
    """Показывает динамику f_sh, f_t, v_c до срабатывания."""
    from FEM.Integration_Point_Level.CriticalPlane.criterion import (
        find_critical_plane_shear,
        find_critical_plane_tensile,
        get_compression_limit,
    )

    mat   = RockMaterial(E=ROCK_E, nu=ROCK_NU, joint_params=joint_params)
    model = UbiquitousJointModel3D(mat)

    D_rock = model.D_rock

    # Диапазон — до значения при котором ожидается локализация
    gamma_range = np.linspace(0, ROCK_RP / G_shear * 1.2, n_steps)

    print("\n  [ДИАГНОСТИКА] Критерии срабатывания vs γ_xz")
    print(f"  Rp = {ROCK_RP:.3e}  G = {G_shear:.3e}")
    print(f"  Ожидаемое γ при локализации ≈ {ROCK_RP / G_shear:.4e}")
    print(f"\n  {'γ_xz':>10}  {'τ_xz[Па]':>14}  {'f_sh':>12}  "
          f"{'f_t':>12}  {'v_c':>12}  {'active?':>8}")
    print("  " + "-" * 75)

    for gamma in gamma_range:
        eps      = np.zeros(6);  eps[5] = gamma
        sig_tr   = D_rock @ eps

        st = StressTensor(*sig_tr)

        f_t_sc, n_t, _ = find_critical_plane_tensile(st, cp_mat, mode='3D')
        f_sh,   n_sh,_ = find_critical_plane_shear(st, cp_mat, mode='3D')

        S = np.array([
            [sig_tr[0], sig_tr[3], sig_tr[5]],
            [sig_tr[3], sig_tr[1], sig_tr[4]],
            [sig_tr[5], sig_tr[4], sig_tr[2]],
        ])
        evals, evecs = np.linalg.eigh(S)
        n_c   = evecs[:, 0]
        f_c_l = get_compression_limit(n_c, cp_mat)
        v_c   = -evals[0] - f_c_l

        active = f_sh > 0 or f_t_sc > 0 or v_c > 0
        print(f"  {gamma:>10.4e}  {sig_tr[5]:>14.4e}  {f_sh:>12.4e}  "
              f"{f_t_sc:>12.4e}  {v_c:>12.4e}  {'✓ YES' if active else 'no':>8}")

        if active:
            print(f"\n  >>> СРАБАТЫВАНИЕ! Нормаль: {n_sh}")
            break
    else:
        print(f"\n  ⚠️ Локализация НЕ наступила в диапазоне γ ∈ [0, {gamma_range[-1]:.3e}]")
        print(f"     → Rp = {ROCK_RP:.3e} Па слишком велик")
        print(f"     → Реальный предел по τ при данном Rp: γ ≈ {ROCK_RP / G_shear:.3e}")


# ─────────────────────────────────────────────────────────────────────────────
# ЗАПУСК
# ─────────────────────────────────────────────────────────────────────────────
diagnose_locking_criteria(n_steps=60)

g_A, tau_A, Ds_A, locked_at = run_AUTO(n_steps=150, gamma_max_factor=8.0)
g_B, tau_B, Ds_B             = run_MANUAL(n_steps=150, gamma_max_factor=3.0)


# ─────────────────────────────────────────────────────────────────────────────
# ГРАФИКИ
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Отладка UbiquitousJointModel3D — сдвиговый тест\n"
    f"E={ROCK_E:.2e} Па,  G={G_shear:.2e} Па,  Rp={ROCK_RP:.2e} Па,  "
    f"φ={joint_params['phi']}°",
    fontsize=13,
)

# ── A: τ–γ автоматическая блокировка ──
ax = axes[0, 0]
ax.plot(g_A * 1e3, tau_A, 'b-', lw=2)
if locked_at is not None:
    ax.axvline(locked_at * 1e3, color='r', ls='--', label=f'Lock γ={locked_at:.2e}')
ax.set_xlabel('γ_xz (×10⁻³)')
ax.set_ylabel('τ_xz (Па)')
ax.set_title('A: τ–γ  (авто-блокировка)')
ax.legend(); ax.grid(True, ls=':')

# ── B: τ–γ ручная блокировка ──
ax = axes[0, 1]
ax.plot(g_B * 1e3, tau_B, 'g-', lw=2)
ax.axhline(+cp_mat.Rpz, color='r', ls='--', alpha=0.6, label=f'Rp={ROCK_RP:.2e}')
ax.axhline(-cp_mat.Rpz, color='r', ls='--', alpha=0.6)
ax.set_xlabel('γ_xz (×10⁻³)')
ax.set_ylabel('τ_xz (Па)')
ax.set_title('B: τ–γ  (явная блокировка n=[0,0,1])')
ax.legend(); ax.grid(True, ls=':')

# ── C: D_s авто ──
ax = axes[1, 0]
ax.plot(g_A * 1e3, Ds_A, 'b-', lw=2)
ax.set_xlabel('γ_xz (×10⁻³)')
ax.set_ylabel('D_s')
ax.set_ylim(-0.02, 1.05)
ax.set_title('A: Damage D_s  (авто-блокировка)')
ax.grid(True, ls=':')

# ── D: D_s ручная ──
ax = axes[1, 1]
ax.plot(g_B * 1e3, Ds_B, 'g-', lw=2)
ax.set_xlabel('γ_xz (×10⁻³)')
ax.set_ylabel('D_s')
ax.set_ylim(-0.02, 1.05)
ax.set_title('B: Damage D_s  (явная блокировка)')
ax.grid(True, ls=':')

plt.tight_layout()
plt.show()
