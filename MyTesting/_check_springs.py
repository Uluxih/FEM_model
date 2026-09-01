import os
os.environ["MPLBACKEND"] = "Agg"

import importlib.util
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

spec = importlib.util.spec_from_file_location(
    "t2d", os.path.join(os.path.dirname(__file__), "2dtestir_opening.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Укороченный прогон: 10 мм сдвига за 25 шагов — достаточно, чтобы увидеть
# расходимость (она начинается сразу после пика/повреждения).
configs = [
    dict(nx=15, ny=15),                                        # базовая
    dict(nx=30, ny=30),                                        # мелкая, как есть
    dict(nx=30, ny=30, scale_kx_by_nodes=True),                # мелкая, kx/n_top
    dict(nx=30, ny=30, k_beam=1e10),                           # мелкая, мягкая балка
]

results = []
for cfg in configs:
    print("\n" + "=" * 70, flush=True)
    t0 = time.time()
    full = dict(target_disp=0.010, n_shear=25, n_preload=11)
    full.update(cfg)
    try:
        ux, fx = mod.run_masonry_wall_opening_test("J4D", **full)
        n_steps = len(ux)
        peak = max(fx) if fx else float("nan")
        u_last = ux[-1] if ux else float("nan")
        msg = (f">>> {cfg}: шагов={n_steps}/24, u_последний={u_last:.3f} мм, "
               f"пик Fx={peak:.1f} кН, время={time.time() - t0:.1f} c")
    except Exception as e:
        msg = f">>> {cfg}: ОШИБКА {type(e).__name__}: {e}"
    print(msg, flush=True)
    results.append(msg)

print("\n\n########## СВОДКА ##########", flush=True)
for m in results:
    print(m, flush=True)
