# -*- coding: utf-8 -*-
"""
Бенчмарк узких мест решателя StagedNRControl2D.

Запускает реальный сценарий (стена с проёмом из 2dtestir_opening.py),
но с инструментированным решателем: замеряется время основных секций
  - _internal_forces (внутренние силы + касательная матрица);
      из них отдельно — вызовы материала cm.update_state(...) и commit/revert;
  - сборка K (_assemble_K, bincount);
  - LU-факторизация (splu);
  - LU-решение (lu.solve);
  - общее время solve.

Исходный код решателя НЕ изменяется: всё делается через наследование
и обёртки над методами моделей материала.

Примеры запуска (из корня проекта):
    python MyTesting/benchmark_staged_nr.py
    python MyTesting/benchmark_staged_nr.py --nx 12 --ny 12 --n-shear 30
    python MyTesting/benchmark_staged_nr.py --profile   # cProfile внутри материала
    python MyTesting/benchmark_staged_nr.py --verbose   # не глушить лог итераций
"""

import argparse
import importlib
import io
import os
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

# --- пути: корень проекта (для FEM) и папка со сценарием ---
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from FEM.Structure_Level.StagedNRControl2D import StagedNRControl2D  # noqa: E402

VERBOSE = False  # глушить ли пошаговый вывод решателя


# =====================================================================
#  Аккумулятор замеров (общий для обеих стадий: предсжатие + сдвиг)
# =====================================================================

class Bench:
    def __init__(self):
        self.t = {}  # имя секции -> суммарное время, с
        self.n = {}  # имя секции -> число вызовов

    def add(self, name, dt):
        self.t[name] = self.t.get(name, 0.0) + dt
        self.n[name] = self.n.get(name, 0) + 1

    def count(self, name):
        self.n[name] = self.n.get(name, 0) + 1


BENCH = Bench()


class _TimedLU:
    """Обёртка над SuperLU: замеряет время lu.solve(...)."""

    def __init__(self, lu):
        self._lu = lu

    def solve(self, b):
        t0 = time.perf_counter()
        x = self._lu.solve(b)
        BENCH.add('lu_solve', time.perf_counter() - t0)
        return x

    def __getattr__(self, item):
        return getattr(self._lu, item)


# =====================================================================
#  Инструментированный решатель
# =====================================================================

class TimedStagedNRControl2D(StagedNRControl2D):
    """StagedNRControl2D + замеры времени по секциям."""

    # --- после предрасчёта топологии оборачиваем все модели материала ---
    def _precompute_topology_and_kinematics(self):
        super()._precompute_topology_and_kinematics()
        self._wrap_material_models()

    def _wrap_material_models(self):
        for grp in self.groups:
            for cm_row in grp['cms']:
                for cm in cm_row:
                    # объекты материала переиспользуются между стадиями —
                    # не оборачиваем дважды
                    if getattr(cm, '_bench_wrapped', False):
                        continue
                    cm._bench_wrapped = True

                    orig_update = cm.update_state
                    # Численный тангенс вызывает self.update_state внутри
                    # (см. UbiquitousJointModel2D, пертурбации eps_pert) —
                    # без защиты от реентерабельности время считается многократно.
                    state = {'depth': 0}

                    def update_state(strain, _orig=orig_update, _state=state):
                        if _state['depth'] > 0:
                            return _orig(strain)
                        _state['depth'] += 1
                        t0 = time.perf_counter()
                        try:
                            return _orig(strain)
                        finally:
                            BENCH.add('material_update_state',
                                      time.perf_counter() - t0)
                            _state['depth'] -= 1

                    cm.update_state = update_state

                    if hasattr(cm, 'commit'):
                        orig_commit = cm.commit

                        def commit(_orig=orig_commit):
                            t0 = time.perf_counter()
                            _orig()
                            BENCH.add('material_commit', time.perf_counter() - t0)

                        cm.commit = commit

                    if hasattr(cm, 'revert'):
                        orig_revert = cm.revert

                        def revert(_orig=orig_revert):
                            t0 = time.perf_counter()
                            _orig()
                            BENCH.add('material_revert', time.perf_counter() - t0)

                        cm.revert = revert

    # --- внутренние силы: общее время + счётчик вызовов с/без K ---
    def _internal_forces(self, U, need_K):
        t0 = time.perf_counter()
        F = super()._internal_forces(U, need_K)
        BENCH.add('internal_forces_total', time.perf_counter() - t0)
        BENCH.count('internal_forces_with_K' if need_K else 'internal_forces_noK')
        return F

    def _assemble_K(self):
        t0 = time.perf_counter()
        K = super()._assemble_K()
        BENCH.add('assemble_K', time.perf_counter() - t0)
        return K

    def _factorize(self, K):
        t0 = time.perf_counter()
        lu = super()._factorize(K)
        BENCH.add('lu_factorize', time.perf_counter() - t0)
        return _TimedLU(lu)

    def solve(self, initial_displacements=None):
        t0 = time.perf_counter()
        if VERBOSE:
            super().solve(initial_displacements)
        else:
            with redirect_stdout(io.StringIO()):
                super().solve(initial_displacements)
        BENCH.add('solve_total', time.perf_counter() - t0)


# =====================================================================
#  Отчёт
# =====================================================================

def _fmt_avg(t, n):
    """Среднее время вызова с автоподбором единиц."""
    avg = t / n
    if avg < 1e-3:
        return f"{avg * 1e6:.2f} мкс"
    if avg < 1.0:
        return f"{avg * 1e3:.2f} мс"
    return f"{avg:.3f} с"


def _row(name, t, total, n=None):
    pct = 100.0 * t / total if total > 0 else 0.0
    line = f"  {name:<42} {t:>10.3f} с  {pct:>6.1f} %"
    if n:
        line += f"   ({n} выз., ср. {_fmt_avg(t, n)})"
    print(line)


def print_report():
    total = BENCH.t.get('solve_total', 0.0)
    t_if = BENCH.t.get('internal_forces_total', 0.0)
    t_mat = BENCH.t.get('material_update_state', 0.0)
    t_asm = BENCH.t.get('assemble_K', 0.0)
    t_lu = BENCH.t.get('lu_factorize', 0.0)
    t_solve = BENCH.t.get('lu_solve', 0.0)
    t_commit = BENCH.t.get('material_commit', 0.0)
    t_revert = BENCH.t.get('material_revert', 0.0)
    t_mech = t_if - t_mat  # einsum-деформации/силы/K_els, add.at, пружины
    accounted = t_if + t_asm + t_lu + t_solve + t_commit + t_revert
    t_other = max(total - accounted, 0.0)

    print("\n" + "=" * 74)
    print("ОТЧЁТ ПО ВРЕМЕНИ (суммарно по всем стадиям нагружения)")
    print("=" * 74)
    print(f"  CPU-ядер в системе: {os.cpu_count()}")
    print(f"  Полное время solve():                          {total:>10.3f} с")
    print("-" * 74)

    n_if = BENCH.n.get('internal_forces_total', 0)
    _row("Внутренние силы _internal_forces (ВСЕГО)", t_if, total, n_if)
    _row("  из них материал update_state", t_mat, total,
         BENCH.n.get('material_update_state', 0))
    _row("  из них механика (einsum, add.at, пружины)", t_mech, total)
    _row("Сборка K (_assemble_K, bincount)", t_asm, total, BENCH.n.get('assemble_K', 0))
    _row("LU-факторизация (splu)", t_lu, total, BENCH.n.get('lu_factorize', 0))
    _row("LU-решение (lu.solve)", t_solve, total, BENCH.n.get('lu_solve', 0))
    _row("commit материала", t_commit, total, BENCH.n.get('material_commit', 0))
    _row("revert материала", t_revert, total, BENCH.n.get('material_revert', 0))
    _row("Прочее (невязки, F_ext, line search обвязка)", t_other, total)

    print("-" * 74)
    n_k = BENCH.n.get('internal_forces_with_K', 0)
    n_nok = BENCH.n.get('internal_forces_noK', 0)
    print(f"  Вызовов _internal_forces: {n_if} (с K: {n_k}, без K: {n_nok})")
    n_upd = BENCH.n.get('material_update_state', 0)
    print(f"  Вызовов update_state (точек интегрирования): {n_upd}")

    print("\n  ИНТЕРПРЕТАЦИЯ:")
    if total <= 0:
        print("  Нет данных.")
        return
    share = {
        'материал update_state': t_mat,
        'механика сборки (einsum)': t_mech,
        'LU-факторизация': t_lu,
        'LU-решение': t_solve,
    }
    bottleneck = max(share, key=share.get)
    print(f"  Узкое место: {bottleneck} ({100.0 * share[bottleneck] / total:.1f} % времени solve).")
    if bottleneck == 'материал update_state':
        print("  -> Распараллеливание/векторизация модели материала ИМЕЕТ смысл.")
        print("     Макс. ускорение по закону Амдала при идеальной параллельности материала:")
        for cores in (2, 4, 8, 16):
            s = 1.0 / ((1 - t_mat / total) + (t_mat / total) / cores)
            print(f"       {cores:>2} ядер: ~{s:.2f}x")
    elif bottleneck in ('LU-факторизация', 'LU-решение'):
        print("  -> Параллелить материал почти бесполезно: смотреть на решатель СЛАУ")
        print("     (PyPardiso/MUMPS, перестановки, реже refactor_every).")
    else:
        print("  -> Основное время в einsum-сборке: проверять формы массивов/BLAS,")
        print("     параллелизм материала даст мало.")
    print("=" * 74)


# =====================================================================
#  Запуск
# =====================================================================

def main():
    global VERBOSE

    parser = argparse.ArgumentParser(description="Бенчмарк узких мест StagedNRControl2D")
    parser.add_argument('--sample', default='J4D', choices=['J4D', 'J6D', 'J7D'],
                        help='образец (вертикальная нагрузка)')
    parser.add_argument('--nx', type=int, default=10, help='элементов по X')
    parser.add_argument('--ny', type=int, default=10, help='элементов по Y')
    parser.add_argument('--n-shear', type=int, default=50, help='шагов сдвига (стадия 2)')
    parser.add_argument('--n-preload', type=int, default=21, help='шагов предсжатия (стадия 1)')
    parser.add_argument('--target-disp', type=float, default=0.004, help='целевой сдвиг, м')
    parser.add_argument('--verbose', action='store_true', help='не глушить лог итераций')
    parser.add_argument('--profile', action='store_true',
                        help='дополнительно cProfile (топ-40 по cumtime; цифры замеров завышены)')
    args = parser.parse_args()
    VERBOSE = args.verbose

    # Импортируем сценарий и подменяем в нём решатель на инструментированный
    scenario = importlib.import_module('2dtestir_opening')
    scenario.StagedNRControl2D = TimedStagedNRControl2D

    # Не блокировать выполнение на plt.show()
    import matplotlib.pyplot as plt
    plt.show = lambda *a, **k: None

    print(f"Сценарий: стена с проёмом, образец {args.sample}, "
          f"сетка {args.nx}x{args.ny}, сдвиг {args.target_disp * 1000:.1f} мм, "
          f"шагов: {args.n_preload} (предсжатие) + {args.n_shear} (сдвиг)")

    def run():
        scenario.run_masonry_wall_opening_test(
            sample_name=args.sample,
            nx=args.nx, ny=args.ny,
            target_disp=args.target_disp,
            n_shear=args.n_shear,
            n_preload=args.n_preload,
        )

    t0 = time.perf_counter()
    if args.profile:
        import cProfile
        import pstats
        prof = cProfile.Profile()
        prof.enable()
        run()
        prof.disable()
        print("\n===== cProfile: топ-40 по суммарному времени (cumtime) =====")
        pstats.Stats(prof).sort_stats('cumtime').print_stats(40)
        print("\n[ВНИМАНИЕ] cProfile включён: абсолютные цифры ниже завышены, "
              "смотреть только ДОЛИ.")
    else:
        run()
    t_wall = time.perf_counter() - t0

    print_report()
    print(f"\n  Полное время сценария (включая построение сетки): {t_wall:.3f} с")


if __name__ == '__main__':
    main()
