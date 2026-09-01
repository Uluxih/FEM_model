"""Исследование влияния числа элементов (nq) на сходимость и результат теста 2dtestir."""
import sys
import importlib.util

import matplotlib
matplotlib.use('Agg')  # без GUI — plt.show не вызывается, кривые сохраняются в PNG

spec = importlib.util.spec_from_file_location("test2d", "MyTesting/2dtestir.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

nqs = [4, 8, 12, 16]
for nq in nqs:
    log_path = f"mesh_log_nq{nq}.txt"
    log = open(log_path, "w", encoding="utf-8", buffering=1)
    old_out = sys.stdout
    sys.stdout = log
    try:
        control, fx_kN, ux_mm = mod.run_masonry_wall_test("J7D", nq=nq, show_plot=False)
        n_steps = len(control.history_U)
        peak = max(fx_kN) if fx_kN else float('nan')
        print(f"RESULT nq={nq}: converged_steps={n_steps}/49, peak_Fx={peak:.2f} kN")
    except Exception as e:
        print(f"RESULT nq={nq}: EXCEPTION: {type(e).__name__}: {e}")
    finally:
        sys.stdout = old_out
        log.close()
    print(f"done nq={nq} -> {log_path}")

print("ALL DONE")
