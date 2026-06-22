"""
test_debug_single_point.py

Тесты на уровне одной точки интегрирования для отладки сходимости NR.

Аналог FEM-тестов из main.py, но без сборки матрицы жёсткости и решателя —
деформация задаётся напрямую в конститутивную модель.

Это позволяет:
  • изолировать NR-проблемы от ошибок МКЭ;
  • быстро перебирать параметры нагружения;
  • получать подробный отчёт о сходимости на каждом шаге.

Запуск всех тестов:
    python test_debug_single_point.py

Отдельный тест:
    python test_debug_single_point.py --test shear
    python test_debug_single_point.py --test tension
    python test_debug_single_point.py --test step_size
"""

import sys
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import FEM.Integration_Point_Level.CriticalPlane.material as cp_mt
from FEM.Abstract.Integration_Point_Level import Material
from FEM.Integration_Point_Level.UbiquitousJointModel3D import UbiquitousJointModel3D


# ═══════════════════════════════════════════════════════════════════════════════
#  МАТЕРИАЛ-ОБЁРТКА (как в main.py)
# ═══════════════════════════════════════════════════════════════════════════════

class JointedMaterial(Material):
    """Обёртка для передачи параметров в ConstitutiveModel."""
    def __init__(self, E, nu, joint_params):
        super().__init__(E, nu)
        self.joint_params = joint_params


# ═══════════════════════════════════════════════════════════════════════════════
#  РЕЗУЛЬТАТ ОДНОГО НАГРУЖЕНИЯ
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SinglePointResult:
    """Полная история одного теста на точке интегрирования."""
    label:            str
    strain_history:   List[np.ndarray]   = field(default_factory=list)  # [6]
    stress_history:   List[np.ndarray]   = field(default_factory=list)  # [6]
    iter_history:     List[int]          = field(default_factory=list)  # кол-во итер./шаг
    order_history:    List[float]        = field(default_factory=list)  # порядок сходимости
    linear_rate_hist: List[float]        = field(default_factory=list)  # линейный коэф.
    active_set_hist:  List[str]          = field(default_factory=list)  # метка набора
    warning_counts:   List[int]          = field(default_factory=list)  # пред./шаг
    failed_steps:     List[int]          = field(default_factory=list)  # номера упавших шагов
    total_steps:      int                = 0


# ═══════════════════════════════════════════════════════════════════════════════
#  ФАБРИКА МОДЕЛИ
# ═══════════════════════════════════════════════════════════════════════════════

def _make_isotropic_A(scale: float = 0.001) -> np.ndarray:
    """Изотропный тензор структуры cp_material (9×9)."""
    return np.eye(9) * scale


def make_model(
    E:            float = 4.6e19,
    nu:           float = 0.2,
    f_t:          float = 8.0e6,
    f_c:          float = 1.5e8,
    phi_deg:      float = 0.0,
    psi_deg:      float = 0.0,
    phi_r_deg:    float = 10.0,
    Gf_t:         float = 100.0,
    Gf_c:         float = 5000.0,
    Gf_s:         float = 100.0,
    a_t:          float = 1.0,
    a_s:          float = 0.0,
    mu:           float = 0.1,
    fcr_over_fc:  float = 0.0,
    A_scale:      float = 0.001,
    l_c:          float = 1.0,
    debug_mode:   bool  = True,
) -> UbiquitousJointModel3D:
    """
    Создаёт модель с изотропным cp_material и заданными параметрами трещиноватости.

    Parameters
    ----------
    E, nu         : упругие константы породы
    f_t, f_c      : прочность на растяжение и сжатие на плоскости
    phi_deg       : угол трения (градусы)
    psi_deg       : угол дилатансии (градусы)
    phi_r_deg     : остаточный угол трения (градусы)
    Gf_t/c/s      : энергии разрушения (до деления на l_c)
    a_t, a_s      : перекрёстные коэффициенты damage
    mu            : параметр нормальной деформации Минга
    fcr_over_fc   : доля остаточной сжимающей прочности
    A_scale       : масштаб тензора A для cp_material
    l_c           : характеристическая длина (регуляризация Gf)
    debug_mode    : True → включить сбор истории NR
    """
    A_matrix = _make_isotropic_A(A_scale)

    cp_material = cp_mt.Material(
        mu=0.5,
        A_tensor=A_matrix,
        Rpx=f_t, Rpy=f_t, Rpz=f_t,
        Rcx=f_c, Rcy=f_c, Rcz=f_c,
    )

    joint_params = {
        'cp_material':  cp_material,
        'phi':          phi_deg,
        'psi':          psi_deg,
        'phi_r':        phi_r_deg,
        'l_c':          l_c,
        'Gf_t':         Gf_t,
        'Gf_c':         Gf_c,
        'Gf_s':         Gf_s,
        'a_t':          a_t,
        'a_s':          a_s,
        'mu':           mu,
        'fcr_over_fc':  fcr_over_fc,
    }

    material = JointedMaterial(E=E, nu=nu, joint_params=joint_params)
    return UbiquitousJointModel3D(material, debug_mode=debug_mode)


# ═══════════════════════════════════════════════════════════════════════════════
#  ЯДРО: ПРОГОН МОДЕЛИ ПО ПУТИ ДЕФОРМАЦИЙ
# ═══════════════════════════════════════════════════════════════════════════════

def run_strain_path(
    model:        UbiquitousJointModel3D,
    strain_path:  np.ndarray,       # shape (N, 6) — весь путь
    label:        str = "test",
    verbose:      bool = True,
) -> SinglePointResult:
    """
    Прогоняет модель через заданный путь деформаций шаг за шагом.

    Parameters
    ----------
    model       : экземпляр UbiquitousJointModel3D (с debug_mode=True для диагностики)
    strain_path : массив (N_steps, 6) — инженерные деформации Войгта
                  [εxx, εyy, εzz, γxy, γyz, γxz]
    label       : название теста (для графиков и отчётов)
    verbose     : True → печатать прогресс в консоль

    Returns
    -------
    SinglePointResult с полной историей напряжений и сходимости
    """
    result = SinglePointResult(label=label, total_steps=len(strain_path))
    model.reset_convergence_history()

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  ТЕСТ: {label}")
        print(f"  Шагов: {len(strain_path)}")
        print(f"{'─'*60}")

    for step_idx, strain in enumerate(strain_path):
        try:
            stress, _ = model.update_state(strain)
            model.commit()

            result.strain_history.append(strain.copy())
            result.stress_history.append(stress.copy())

            # Собираем метрики из debug-истории
            if model.debug_mode and model._convergence_history:
                rec = model._convergence_history[-1]
                win_att = next(
                    (a for a in rec.attempts if a.label == rec.winning_label),
                    None
                )
                n_iter  = len(win_att.iterations) if win_att else 0
                orders  = win_att.conv_rates_order if win_att else []
                lin_r   = win_att.conv_rates_linear if win_att else []

                result.iter_history.append(n_iter)
                result.order_history.append(
                    float(np.mean(orders[-3:])) if len(orders) >= 2 else float('nan')
                )
                result.linear_rate_hist.append(
                    float(np.mean(lin_r)) if lin_r else float('nan')
                )
                result.active_set_hist.append(rec.winning_label or 'elastic')
                result.warning_counts.append(len(rec.diag_warnings))
            else:
                result.iter_history.append(0)
                result.order_history.append(float('nan'))
                result.linear_rate_hist.append(float('nan'))
                result.active_set_hist.append('elastic')
                result.warning_counts.append(0)

            if verbose and (step_idx % max(1, len(strain_path) // 10) == 0):
                sig   = result.stress_history[-1]
                iters = result.iter_history[-1]
                aset  = result.active_set_hist[-1]
                print(f"  Шаг {step_idx+1:>4}/{len(strain_path)} | "
                      f"σzz={sig[2]:>12.4g}  τxz={sig[5]:>12.4g} | "
                      f"iter={iters:>3}  набор={aset}")

        except RuntimeError as e:
            result.failed_steps.append(step_idx)
            if verbose:
                print(f"\n  ✗ ШАГ {step_idx+1} РАСХОДИТСЯ: {e}")
            # При расхождении продолжаем со старым напряжением
            if result.stress_history:
                result.strain_history.append(strain.copy())
                result.stress_history.append(result.stress_history[-1].copy())
            result.iter_history.append(-1)
            result.order_history.append(float('nan'))
            result.linear_rate_hist.append(float('nan'))
            result.active_set_hist.append('FAILED')
            result.warning_counts.append(0)

    if verbose:
        n_ok   = result.total_steps - len(result.failed_steps)
        ok_iters = [i for i in result.iter_history if i >= 0]
        print(f"\n  Итог: {n_ok}/{result.total_steps} успешных шагов")
        if ok_iters:
            print(f"  Итерации: min={min(ok_iters)}, max={max(ok_iters)}, "
                  f"mean={np.mean(ok_iters):.1f}")
        if result.failed_steps:
            print(f"  ✗ Расходящиеся шаги: {result.failed_steps}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  ВИЗУАЛИЗАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════════

def plot_stress_strain(
    results: List[SinglePointResult],
    strain_comp: int = 2,   # компонента деформации для оси X
    stress_comp: int = 2,   # компонента напряжения для оси Y
    strain_label: str = "εzz",
    stress_label: str = "σzz [Па]",
    title: str = "Кривая напряжение-деформация",
):
    """Строит кривые σ-ε для списка результатов (сравнение тестов)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors

    for idx, res in enumerate(results):
        if not res.strain_history:
            continue
        eps = [e[strain_comp] for e in res.strain_history]
        sig = [s[stress_comp] for s in res.stress_history]

        # Отмечаем расходящиеся шаги
        ok_mask   = [i for i in range(len(eps)) if i not in res.failed_steps]
        fail_mask = res.failed_steps

        ax.plot([eps[i] for i in ok_mask],
                [sig[i] for i in ok_mask],
                color=colors[idx % 10], lw=2, marker='o', ms=3,
                label=res.label)
        if fail_mask:
            ax.scatter([eps[i] for i in fail_mask if i < len(eps)],
                       [sig[i] for i in fail_mask if i < len(sig)],
                       color='red', zorder=5, s=60, marker='X',
                       label=f'{res.label} — расхождение')

    ax.axhline(0, color='black', lw=0.8)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel(strain_label, fontsize=12)
    ax.set_ylabel(stress_label, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_convergence_dashboard(result: SinglePointResult, model: UbiquitousJointModel3D):
    """
    Дашборд из 4 графиков для одного теста:
      1. Кол-во итераций на шаг + активный набор
      2. Порядок сходимости (должен быть ≈ 2)
      3. Средний линейный коэф. убывания |R|
      4. Количество предупреждений на шаг
    """
    steps = list(range(1, len(result.iter_history) + 1))
    if not steps:
        print("[plot_convergence_dashboard] Нет данных.")
        return

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(f"Дашборд сходимости NR: «{result.label}»",
                 fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax_iter  = fig.add_subplot(gs[0, 0])
    ax_order = fig.add_subplot(gs[0, 1])
    ax_lrate = fig.add_subplot(gs[1, 0])
    ax_warn  = fig.add_subplot(gs[1, 1])

    # Цвет столбцов по активному набору
    set_colors = {
        'elastic':   '#aec7e8',
        'initial':   '#1f77b4',
        'T':         '#ff7f0e',
        'C':         '#d62728',
        'S':         '#2ca02c',
        'T+S':       '#9467bd',
        'C+S':       '#8c564b',
        'FAILED':    '#000000',
    }
    bar_colors = [set_colors.get(s, '#7f7f7f') for s in result.active_set_hist]
    iters_plot = [max(i, 0) for i in result.iter_history]

    # 1. Итерации
    bars = ax_iter.bar(steps, iters_plot, color=bar_colors, edgecolor='white', lw=0.5)
    ax_iter.set_title('Итерации на шаг')
    ax_iter.set_xlabel('Шаг')
    ax_iter.set_ylabel('N итераций')
    ax_iter.axhline(model.nr_max_iter, color='red', ls=':', lw=1,
                    label=f'max={model.nr_max_iter}')
    ax_iter.legend(fontsize=8)
    ax_iter.grid(True, axis='y', alpha=0.3)

    # Легенда активных наборов
    from matplotlib.patches import Patch
    used_sets = list(dict.fromkeys(result.active_set_hist))  # сохраняем порядок
    legend_patches = [Patch(facecolor=set_colors.get(s, '#7f7f7f'), label=s)
                      for s in used_sets]
    ax_iter.legend(handles=legend_patches, fontsize=7,
                   loc='upper right', title='Набор')

    # 2. Порядок сходимости
    valid_ord = [(s, o) for s, o in zip(steps, result.order_history)
                 if not np.isnan(o)]
    if valid_ord:
        sx, oy = zip(*valid_ord)
        ax_order.plot(sx, oy, color='#1f77b4', lw=1.5, marker='s', ms=4)
        ax_order.axhline(2.0, color='green', ls='--', lw=1.5,
                         label='p=2 (квадратичная)')
        ax_order.axhline(1.0, color='orange', ls='--', lw=1,
                         label='p=1 (линейная)')
        ax_order.set_ylim([-0.3, 3.5])
        ax_order.fill_between(sx, 1.8, 2.2, alpha=0.1, color='green',
                              label='зона NR [1.8, 2.2]')
    ax_order.set_title('Порядок сходимости p')
    ax_order.set_xlabel('Шаг')
    ax_order.set_ylabel('p')
    ax_order.legend(fontsize=8)
    ax_order.grid(True, alpha=0.3)

    # 3. Линейный коэф. убывания невязки
    valid_lr = [(s, r) for s, r in zip(steps, result.linear_rate_hist)
                if not np.isnan(r)]
    if valid_lr:
        sx, ry = zip(*valid_lr)
        ax_lrate.semilogy(sx, ry, color='#9467bd', lw=1.5, marker='D', ms=4)
        ax_lrate.axhline(1.0, color='red', ls=':', lw=1,
                         label='rate=1 (граница расходимости)')
        ax_lrate.axhline(0.1, color='green', ls=':', lw=1,
                         label='rate=0.1 (хорошая сходимость)')
    ax_lrate.set_title('Ср. коэф. убывания невязки')
    ax_lrate.set_xlabel('Шаг')
    ax_lrate.set_ylabel('||R_{k+1}|| / ||R_k||')
    ax_lrate.legend(fontsize=8)
    ax_lrate.grid(True, which='both', alpha=0.3)

    # 4. Предупреждения
    ax_warn.bar(steps, result.warning_counts,
                color='#ff7f0e', edgecolor='white', lw=0.5)
    ax_warn.set_title('Предупреждения на шаг')
    ax_warn.set_xlabel('Шаг')
    ax_warn.set_ylabel('Кол-во ⚠')
    ax_warn.grid(True, axis='y', alpha=0.3)

    if result.failed_steps:
        for ax in [ax_iter, ax_order, ax_lrate, ax_warn]:
            for fs in result.failed_steps:
                ax.axvline(fs + 1, color='black', ls='--', lw=1.5, alpha=0.7)

    plt.tight_layout()
    plt.show()


def plot_step_size_comparison(
    step_size_results: Dict[int, SinglePointResult],
    metric: str = 'iterations',  # 'iterations' | 'order' | 'rate'
):
    """
    Сравнивает характеристики сходимости при разных размерах шага.

    Parameters
    ----------
    step_size_results : {n_steps: SinglePointResult}
    metric            : что сравнивать
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Влияние размера шага на сходимость NR",
                 fontsize=13, fontweight='bold')

    n_steps_list = sorted(step_size_results.keys())
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(n_steps_list)))

    # Левый: кол-во итераций по шагу нагружения (кривые)
    ax_l = axes[0]
    for (n_steps, color) in zip(n_steps_list, colors):
        res = step_size_results[n_steps]
        step_load_frac = np.linspace(0, 1, len(res.iter_history))
        iters_ok = [max(i, 0) for i in res.iter_history]
        lbl = f"{n_steps} шагов (шаг={1/n_steps:.4f})"
        if res.failed_steps:
            lbl += f" ✗{len(res.failed_steps)} расх."
        ax_l.plot(step_load_frac, iters_ok, color=color, lw=1.5,
                  marker='o', ms=2, label=lbl)

    ax_l.set_title('Итерации по ходу нагружения')
    ax_l.set_xlabel('Доля пути нагружения')
    ax_l.set_ylabel('N итераций')
    ax_l.legend(fontsize=8)
    ax_l.grid(True, alpha=0.3)

    # Правый: сводная статистика (средн. итерации, макс. итерации, % успеха)
    ax_r = axes[1]
    mean_iters, max_iters, success_rates = [], [], []
    for n_steps in n_steps_list:
        res = step_size_results[n_steps]
        ok = [i for i in res.iter_history if i >= 0]
        mean_iters.append(np.mean(ok) if ok else 0)
        max_iters.append(max(ok) if ok else 0)
        success_rates.append(
            100 * (res.total_steps - len(res.failed_steps)) / max(res.total_steps, 1)
        )

    x = np.arange(len(n_steps_list))
    w = 0.3
    ax_r.bar(x - w, mean_iters, width=w, label='Средн. итерации',
             color='#1f77b4', alpha=0.8)
    ax_r.bar(x,     max_iters,  width=w, label='Макс. итерации',
             color='#ff7f0e', alpha=0.8)
    ax_r2 = ax_r.twinx()
    ax_r2.plot(x + w/2, success_rates, color='green', lw=2,
               marker='D', ms=6, label='% успеха')
    ax_r2.set_ylabel('% успешных шагов', color='green')
    ax_r2.tick_params(axis='y', labelcolor='green')
    ax_r2.set_ylim([0, 105])

    ax_r.set_xticks(x)
    ax_r.set_xticklabels([f"{n}" for n in n_steps_list], fontsize=9)
    ax_r.set_xlabel('Кол-во шагов нагружения')
    ax_r.set_ylabel('Итерации NR')
    ax_r.set_title('Сводная статистика по размерам шага')
    ax_r.legend(loc='upper left', fontsize=8)
    ax_r2.legend(loc='upper right', fontsize=8)
    ax_r.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  ТЕСТ 1: МОНОТОННОЕ РАСТЯЖЕНИЕ ПО Z
# ═══════════════════════════════════════════════════════════════════════════════

def run_tension_test(n_steps: int = 50, eps_max: float = 5e-4,
                     debug: bool = True) -> SinglePointResult:
    """
    Монотонное растяжение по Z.
    Цель: проверить поверхность F1 (растяжение), корректность D_nt.

    Strain path: εzz: 0 → eps_max, остальные компоненты = 0.
    """
    print("\n" + "═"*60)
    print("  ТЕСТ 1: Монотонное растяжение по Z")
    print("═"*60)

    model = make_model(
        f_t=8e6, f_c=1.5e8, phi_deg=0.0, psi_deg=0.0,
        Gf_t=100.0, mu=0.1, debug_mode=debug,
    )

    # Путь деформации: только εzz растёт
    eps_path = np.zeros((n_steps, 6))
    eps_path[:, 2] = np.linspace(0, eps_max, n_steps + 1)[1:]  # εzz

    result = run_strain_path(model, eps_path, label=f'Растяжение Z ({n_steps} шагов)')

    # Кривая σzz vs εzz
    plot_stress_strain(
        [result],
        strain_comp=2, stress_comp=2,
        strain_label='εzz', stress_label='σzz [Па]',
        title='Тест 1: Монотонное растяжение по Z',
    )
    if debug:
        plot_convergence_dashboard(result, model)
        model.print_convergence_report()

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  ТЕСТ 2: МОНОТОННОЕ СЖАТИЕ ПО Z
# ═══════════════════════════════════════════════════════════════════════════════

def run_compression_test(n_steps: int = 50, eps_max: float = 2e-3,
                         debug: bool = True) -> SinglePointResult:
    """
    Монотонное сжатие по Z.
    Цель: проверить поверхность F3 (сжатие), корректность D_nc.

    Strain path: εzz: 0 → -eps_max.
    """
    print("\n" + "═"*60)
    print("  ТЕСТ 2: Монотонное сжатие по Z")
    print("═"*60)

    model = make_model(
        f_t=8e6, f_c=1.5e8, phi_deg=0.0, psi_deg=0.0,
        Gf_c=5000.0, mu=0.1, debug_mode=debug,
    )

    eps_path = np.zeros((n_steps, 6))
    eps_path[:, 2] = np.linspace(0, -eps_max, n_steps + 1)[1:]  # εzz < 0

    result = run_strain_path(model, eps_path, label=f'Сжатие Z ({n_steps} шагов)')

    plot_stress_strain(
        [result],
        strain_comp=2, stress_comp=2,
        strain_label='εzz', stress_label='σzz [Па]',
        title='Тест 2: Монотонное сжатие по Z',
    )
    if debug:
        plot_convergence_dashboard(result, model)
        model.print_convergence_report()

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  ТЕСТ 3: ЧИСТЫЙ СДВИГ XZ  ← КЛЮЧЕВОЙ ТЕСТ СХОДИМОСТИ
# ═══════════════════════════════════════════════════════════════════════════════

def run_shear_xz_test(n_steps: int = 50, gamma_max: float = 2e-3,
                      debug: bool = True) -> SinglePointResult:
    """
    Чистый сдвиг в плоскости XZ.
    Цель: проверить поверхность F2 (сдвиг Мора-Кулона), D_s,
          и устойчивость при τ → 0 (начало нагружения).

    Strain path: γxz: 0 → gamma_max.
    Ключевые риски:
      • τ = 0 при первом шаге → сингулярность направления
      • при phi>0: связность нормали и сдвига → угловая зона
    """
    print("\n" + "═"*60)
    print("  ТЕСТ 3: Чистый сдвиг XZ (КЛЮЧЕВОЙ ТЕСТ NR)")
    print("═"*60)

    model = make_model(
        f_t=8e6, f_c=1.5e8,
        phi_deg=30.0, psi_deg=10.0, phi_r_deg=10.0,
        Gf_s=100.0, a_s=0.0, debug_mode=debug,
    )

    eps_path = np.zeros((n_steps, 6))
    eps_path[:, 5] = np.linspace(0, gamma_max, n_steps + 1)[1:]  # γxz

    result = run_strain_path(model, eps_path, label=f'Сдвиг XZ ({n_steps} шагов)')

    plot_stress_strain(
        [result],
        strain_comp=5, stress_comp=5,
        strain_label='γxz', stress_label='τxz [Па]',
        title='Тест 3: Чистый сдвиг XZ',
    )
    if debug:
        plot_convergence_dashboard(result, model)
        model.print_convergence_report()

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  ТЕСТ 4: ЦИКЛИЧЕСКОЕ НАГРУЖЕНИЕ (растяжение → сжатие → растяжение)
# ═══════════════════════════════════════════════════════════════════════════════

def run_cyclic_test(n_steps_per_leg: int = 30, debug: bool = True) -> SinglePointResult:
    """
    Циклическое нагружение по Z: растяжение → сжатие → растяжение.
    Цель: проверить переключение поверхностей T ↔ C,
          корректность пластической деформации при смене знака.

    Аналог run_cyclic_test() в main.py, но без МКЭ.
    """
    print("\n" + "═"*60)
    print("  ТЕСТ 4: Циклическое нагружение Z (T→C→T)")
    print("═"*60)

    model = make_model(
        f_t=8e6, f_c=1.5e8,
        phi_deg=0.0, psi_deg=0.0,
        Gf_t=100.0, Gf_c=5000.0, mu=0.1,
        debug_mode=debug,
    )

    eps_t1  = np.linspace(0,      5e-4,  n_steps_per_leg + 1)[1:]  # растяжение
    eps_c   = np.linspace(5e-4,  -1e-3,  n_steps_per_leg + 1)[1:]  # сжатие
    eps_t2  = np.linspace(-1e-3,  2e-4,  n_steps_per_leg + 1)[1:]  # обратно

    eps_all  = np.concatenate([eps_t1, eps_c, eps_t2])
    eps_path = np.zeros((len(eps_all), 6))
    eps_path[:, 2] = eps_all

    result = run_strain_path(model, eps_path, label='Цикл Z (T→C→T)')

    plot_stress_strain(
        [result],
        strain_comp=2, stress_comp=2,
        strain_label='εzz', stress_label='σzz [Па]',
        title='Тест 4: Циклическое нагружение (T→C→T)',
    )
    if debug:
        plot_convergence_dashboard(result, model)
        model.print_convergence_report()

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  ТЕСТ 5: СОВМЕСТНОЕ НАГРУЖЕНИЕ НОРМАЛЬ + СДВИГ (угловая зона T+S)
# ═══════════════════════════════════════════════════════════════════════════════

def run_mixed_test(n_steps: int = 60, debug: bool = True) -> SinglePointResult:
    """
    Пропорциональный рост εzz и γxz одновременно → угловая зона T+S.
    Цель: проверить активацию T+S, корректность Якобиана в угловой зоне.
    """
    print("\n" + "═"*60)
    print("  ТЕСТ 5: Совместное нагружение (εzz + γxz)")
    print("═"*60)

    model = make_model(
        f_t=8e6, f_c=1.5e8,
        phi_deg=30.0, psi_deg=10.0,
        Gf_t=100.0, Gf_s=100.0,
        debug_mode=debug,
    )

    eps_path = np.zeros((n_steps, 6))
    eps_path[:, 2] = np.linspace(0, 4e-4,  n_steps + 1)[1:]   # εzz
    eps_path[:, 5] = np.linspace(0, 8e-4,  n_steps + 1)[1:]   # γxz

    result = run_strain_path(model, eps_path, label='Смешанное (εzz+γxz)')

    # Две кривые рядом
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Тест 5: Совместное нагружение (угловая зона T+S)',
                 fontsize=12, fontweight='bold')

    eps_z = [e[2] for e in result.strain_history]
    eps_g = [e[5] for e in result.strain_history]
    sig_z = [s[2] for s in result.stress_history]
    tau_x = [s[5] for s in result.stress_history]

    axes[0].plot(eps_z, sig_z, color='#1f77b4', lw=2, marker='o', ms=3)
    axes[0].set_xlabel('εzz');  axes[0].set_ylabel('σzz [Па]')
    axes[0].set_title('Нормальная компонента');  axes[0].grid(True, alpha=0.3)

    axes[1].plot(eps_g, tau_x, color='#2ca02c', lw=2, marker='o', ms=3)
    axes[1].set_xlabel('γxz');  axes[1].set_ylabel('τxz [Па]')
    axes[1].set_title('Касательная компонента');  axes[1].grid(True, alpha=0.3)

    plt.tight_layout();  plt.show()

    if debug:
        plot_convergence_dashboard(result, model)
        model.print_convergence_report()

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  ТЕСТ 6: ИССЛЕДОВАНИЕ ВЛИЯНИЯ РАЗМЕРА ШАГА НА СХОДИМОСТЬ
# ═══════════════════════════════════════════════════════════════════════════════

def run_step_size_study(
    mode:        str   = 'shear',    # 'shear' | 'tension' | 'compression'
    n_steps_list: List[int] = None,
    debug:       bool  = True,
) -> Dict[int, SinglePointResult]:
    """
    Запускает один и тот же тест с разным числом шагов нагружения
    и сравнивает характеристики сходимости NR.

    Позволяет ответить на вопрос:
      «При каком размере шага NR ещё сходится?»
      «Как размер шага влияет на порядок сходимости?»

    Parameters
    ----------
    mode          : тип нагружения ('shear', 'tension', 'compression')
    n_steps_list  : список количеств шагов (грубые → мелкие)
    """
    if n_steps_list is None:
        n_steps_list = [5, 10, 20, 50, 100]

    print("\n" + "═"*60)
    print(f"  ТЕСТ 6: Исследование размера шага [{mode}]")
    print(f"  Варианты: {n_steps_list}")
    print("═"*60)

    results = {}

    for n_steps in n_steps_list:
        model = make_model(
            f_t=8e6, f_c=1.5e8,
            phi_deg=30.0 if mode == 'shear' else 0.0,
            psi_deg=10.0 if mode == 'shear' else 0.0,
            Gf_t=100.0, Gf_s=100.0, Gf_c=5000.0,
            debug_mode=debug,
        )

        if mode == 'shear':
            eps_path = np.zeros((n_steps, 6))
            eps_path[:, 5] = np.linspace(0, 2e-3, n_steps + 1)[1:]
        elif mode == 'tension':
            eps_path = np.zeros((n_steps, 6))
            eps_path[:, 2] = np.linspace(0, 5e-4, n_steps + 1)[1:]
        else:  # compression
            eps_path = np.zeros((n_steps, 6))
            eps_path[:, 2] = np.linspace(0, -2e-3, n_steps + 1)[1:]

        res = run_strain_path(
            model, eps_path,
            label=f'{mode} {n_steps} шагов',
            verbose=False,
        )
        results[n_steps] = res

        n_ok = res.total_steps - len(res.failed_steps)
        ok_iters = [i for i in res.iter_history if i >= 0]
        ord_vals  = [o for o in res.order_history if not np.isnan(o)]

        print(f"  n={n_steps:>4}: успех {n_ok:>4}/{res.total_steps} | "
              f"iter mean={np.mean(ok_iters):.1f} max={max(ok_iters) if ok_iters else 0} | "
              f"порядок mean={np.mean(ord_vals):.2f}"
              if ord_vals else
              f"  n={n_steps:>4}: успех {n_ok:>4}/{res.total_steps} | "
              f"iter mean={np.mean(ok_iters):.1f} max={max(ok_iters) if ok_iters else 0} | "
              f"порядок —")

    plot_step_size_comparison(results)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  ТЕСТ 7: ПРЯМОЕ ЗОНДИРОВАНИЕ NR В ЗАДАННОЙ ТОЧКЕ
# ═══════════════════════════════════════════════════════════════════════════════

def probe_nr_at_stress_state(
    sig_n_tr:   float = 10e6,    # пробное нормальное напряжение [Па]
    tau_23_tr:  float = 0.0,     # пробное τ23 [Па]
    tau_13_tr:  float = 5e6,     # пробное τ13 [Па]
    lam_t_old:  float = 0.0,     # предыдущий множитель пластичности
    lam_s_old:  float = 0.0,
    phi_deg:    float = 30.0,
    debug:      bool  = True,
) -> None:
    """
    Прямое зондирование NR-решателя в одной точке напряжённого пространства.

    Позволяет:
      • воспроизвести конкретный шаг, на котором модель расходится;
      • изучить поведение NR без прохождения всего пути деформаций.

    Как использовать:
      1. Запустите основной тест.
      2. Из отчёта возьмите sig_n_tr, tau_23_tr, tau_13_tr проблемного шага.
      3. Вставьте сюда и запустите probe.
    """
    print("\n" + "═"*60)
    print("  ТЕСТ 7: Прямое зондирование NR")
    print(f"  sig_n_tr={sig_n_tr:.4g}  tau_23={tau_23_tr:.4g}  tau_13={tau_13_tr:.4g}")
    print(f"  lam_t_old={lam_t_old:.4g}  lam_s_old={lam_s_old:.4g}")
    print("═"*60)

    model = make_model(phi_deg=phi_deg, debug_mode=True)

    # Принудительно блокируем плоскость (нормаль по Z)
    from FEM.Integration_Point_Level.CriticalPlane.tensor import StressTensor
    dummy_st = StressTensor(sig_n_tr, sig_n_tr, sig_n_tr, 0, 0, 0)
    model._lock_plane(np.array([0.0, 0.0, 1.0]), dummy_st)

    # Устанавливаем историю пластической деформации
    model.lam_t_old = lam_t_old
    model.lam_s_old = lam_s_old

    # Запускаем диагностику
    try:
        result = model._return_mapping_nr(sig_n_tr, tau_23_tr, tau_13_tr)
        print(f"\n  Результат NR:")
        print(f"    sig_n  = {result[0]:.6g}")
        print(f"    tau_23 = {result[1]:.6g}")
        print(f"    tau_13 = {result[2]:.6g}")
        print(f"    d_lam_t= {result[3]:.6g}")
        print(f"    d_lam_c= {result[4]:.6g}")
        print(f"    d_lam_s= {result[5]:.6g}")
    except RuntimeError as e:
        print(f"\n  ✗ NR расходится: {e}")

    if debug and model._convergence_history:
        rec = model._convergence_history[-1]
        model._print_step_record(rec)
        model.plot_convergence(step=rec.step_idx, show_all_attempts=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  ЗАПУСК ВСЕХ ТЕСТОВ
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_tests():
    print("\n" + "█"*60)
    print("  ПОЛНЫЙ НАБОР ОТЛАДОЧНЫХ ТЕСТОВ (Single Point)")
    print("█"*60)

    r1 = run_tension_test(n_steps=50)
    r2 = run_compression_test(n_steps=50)
    r3 = run_shear_xz_test(n_steps=50)
    r4 = run_cyclic_test(n_steps_per_leg=30)
    r5 = run_mixed_test(n_steps=60)

    # Совместный график σzz-εzz для T и C
    plot_stress_strain(
        [r1, r2],
        strain_comp=2, stress_comp=2,
        strain_label='εzz', stress_label='σzz [Па]',
        title='Сравнение: Растяжение vs Сжатие',
    )

    # Исследование размера шага для сдвига
    run_step_size_study(mode='shear', n_steps_list=[5, 10, 20, 50, 100])

    print("\n" + "█"*60)
    print("  ТЕСТЫ ЗАВЕРШЕНЫ")
    print("█"*60)


# ═══════════════════════════════════════════════════════════════════════════════
#  ТОЧКА ВХОДА
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Отладочные тесты UbiquitousJointModel3D')
    parser.add_argument(
        '--test',
        choices=['all', 'tension', 'compression', 'shear', 'cyclic', 'mixed',
                 'step_size', 'probe'],
        default='all',
        help='Выбор теста (по умолчанию: all)',
    )
    parser.add_argument('--n_steps', type=int, default=50,
                        help='Количество шагов нагружения')
    parser.add_argument('--no_debug', action='store_true',
                        help='Отключить debug_mode (без диагностики NR)')
    args = parser.parse_args()

    debug = not args.no_debug
    n    = args.n_steps

    dispatch = {
        'all':         run_all_tests,
        'tension':     lambda: run_tension_test(n_steps=n,           debug=debug),
        'compression': lambda: run_compression_test(n_steps=n,       debug=debug),
        'shear':       lambda: run_shear_xz_test(n_steps=n,          debug=debug),
        'cyclic':      lambda: run_cyclic_test(n_steps_per_leg=n//3, debug=debug),
        'mixed':       lambda: run_mixed_test(n_steps=n,             debug=debug),
        'step_size':   lambda: run_step_size_study(mode='shear',     debug=debug),
        'probe':       lambda: probe_nr_at_stress_state(
                           sig_n_tr=10e6, tau_13_tr=5e6, phi_deg=30.0
                       ),
    }

    dispatch[args.test]()
