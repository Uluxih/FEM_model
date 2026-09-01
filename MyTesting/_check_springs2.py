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

# Вариант выбирается аргументом командной строки: scale | beam
mode = sys.argv[1] if len(sys.argv) > 1 else "scale"
if mode == "scale":
    cfg = dict(nx=30, ny=30, scale_kx_by_nodes=True)
elif mode == "beam":
    cfg = dict(nx=30, ny=30, k_beam=1e10)
else:
    raise SystemExit(f"unknown mode {mode}")

t0 = time.time()
try:
    ux, fx = mod.run_masonry_wall_opening_test(
        "J4D", target_disp=0.003, n_shear=10, n_preload=11, **cfg)
    print(f">>> {cfg}: шагов={len(ux)}, u_последний={ux[-1] if ux else float('nan'):.4f} мм, "
          f"пик Fx={max(fx) if fx else float('nan'):.1f} кН, время={time.time() - t0:.1f} c",
          flush=True)
except Exception as e:
    print(f">>> {cfg}: ОШИБКА {type(e).__name__}: {e}", flush=True)
