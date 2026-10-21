"""Снятие эталонных результатов старой реализации criterion.py и замер времени."""
import sys, time, json
import numpy as np

sys.path.insert(0, r"C:\Users\Uluxih\FEM_model")
sys.path.insert(0, r"C:\Users\Uluxih\FEM_model\.tasks\opt-criterion")

import criterion_old as cr
from FEM.Integration_Point_Level.CriticalPlane.tensor import StressTensor
import FEM.Integration_Point_Level.CriticalPlane.material as mt

rng = np.random.default_rng(42)

results = []
t_old = 0.0
for case in range(30):
    a = rng.normal(size=(3, 3)) * 100
    m = a + a.T
    st = StressTensor(m[0, 0], m[1, 1], m[2, 2], m[0, 1], m[1, 2], m[0, 2])
    A = rng.normal(size=(9, 9))
    A = A @ A.T + 9 * np.eye(9)  # положительно определенный тензор анизотропии
    mat = mt.Material(mu=0.4, A_tensor=A,
                      Rpx=10, Rpy=20, Rpz=15, Rcx=50, Rcy=60, Rcz=55)
    for mode, npl in [('2D_XY', 10), ('2D_XY', 37), ('3D', 50), ('3D', 1)]:
        t0 = time.perf_counter()
        f_s, n_s, u_s = cr.find_critical_plane_shear(st, mat, mode=mode, num_planes=npl)
        f_t, n_t, u_t = cr.find_critical_plane_tensile(st, mat, mode=mode, num_planes=npl)
        f_c, n_c, u_c = cr.find_critical_plane_compression(st, mat, mode=mode, num_planes=npl)
        t_old += time.perf_counter() - t0
        results.append(dict(case=case, mode=mode, npl=npl,
                            f_s=f_s, n_s=list(n_s), u_s=u_s,
                            f_t=f_t, n_t=list(n_t), u_t=u_t,
                            f_c=f_c, n_c=list(n_c), u_c=u_c))

def default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, float) and np.isinf(o):
        return "inf"
    return o

with open(r"C:\Users\Uluxih\FEM_model\.tasks\opt-criterion\reference.json", "w") as fh:
    json.dump(results, fh, default=default)

print(f"cases: {len(results)}, old total time: {t_old:.4f} s")
