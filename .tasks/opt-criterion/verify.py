"""Сверка новой векторизованной реализации criterion.py с эталоном + бенчмарк."""
import sys, time, json
import numpy as np

sys.path.insert(0, r"C:\Users\Uluxih\FEM_model")
sys.path.insert(0, r"C:\Users\Uluxih\FEM_model\.tasks\opt-criterion")

from FEM.Integration_Point_Level.CriticalPlane import criterion as cr_new
import criterion_old as cr_old
from FEM.Integration_Point_Level.CriticalPlane.tensor import StressTensor
import FEM.Integration_Point_Level.CriticalPlane.material as mt

with open(r"C:\Users\Uluxih\FEM_model\.tasks\opt-criterion\reference.json") as fh:
    ref = json.load(fh)

rng = np.random.default_rng(42)

def to_val(x):
    return float('inf') if x == "inf" else float(x)

errors = 0
max_diff = 0.0
case_idx = -1
t_new = 0.0
for case in range(30):
    a = rng.normal(size=(3, 3)) * 100
    m = a + a.T
    st = StressTensor(m[0, 0], m[1, 1], m[2, 2], m[0, 1], m[1, 2], m[0, 2])
    A = rng.normal(size=(9, 9))
    A = A @ A.T + 9 * np.eye(9)
    mat = mt.Material(mu=0.4, A_tensor=A,
                      Rpx=10, Rpy=20, Rpz=15, Rcx=50, Rcy=60, Rcz=55)
    for mode, npl in [('2D_XY', 10), ('2D_XY', 37), ('3D', 50), ('3D', 1)]:
        case_idx += 1
        r = ref[case_idx]
        t0 = time.perf_counter()
        f_s, n_s, u_s = cr_new.find_critical_plane_shear(st, mat, mode=mode, num_planes=npl)
        f_t, n_t, u_t = cr_new.find_critical_plane_tensile(st, mat, mode=mode, num_planes=npl)
        f_c, n_c, u_c = cr_new.find_critical_plane_compression(st, mat, mode=mode, num_planes=npl)
        t_new += time.perf_counter() - t0

        for tag, fv, nv, uv, rk_f, rk_n, rk_u in [
            ('shear', f_s, n_s, u_s, 'f_s', 'n_s', 'u_s'),
            ('tensile', f_t, n_t, u_t, 'f_t', 'n_t', 'u_t'),
            ('comp', f_c, n_c, u_c, 'f_c', 'n_c', 'u_c'),
        ]:
            d_f = abs(fv - r[rk_f])
            d_n = np.max(np.abs(np.asarray(nv) - np.asarray(r[rk_n])))
            ru, nu = r[rk_u], uv
            if ru == "inf" or (isinstance(ru, float) and np.isinf(ru)):
                ok_u = np.isinf(nu)
            else:
                ok_u = abs(nu - to_val(ru)) < 1e-8
            max_diff = max(max_diff, d_f, d_n)
            if d_f > 1e-8 or d_n > 1e-8 or not ok_u:
                errors += 1
                print(f"MISMATCH case={case} mode={mode} npl={npl} {tag}: "
                      f"d_f={d_f:.2e} d_n={d_n:.2e} u_new={nu} u_ref={ru}")

# Скорость старой версии (повторный замер в этом же процессе)
rng2 = np.random.default_rng(7)
cases = []
for case in range(30):
    a = rng2.normal(size=(3, 3)) * 100
    m = a + a.T
    A = rng2.normal(size=(9, 9))
    A = A @ A.T + 9 * np.eye(9)
    cases.append((StressTensor(m[0, 0], m[1, 1], m[2, 2], m[0, 1], m[1, 2], m[0, 2]),
                  mt.Material(mu=0.4, A_tensor=A, Rpx=10, Rpy=20, Rpz=15,
                              Rcx=50, Rcy=60, Rcz=55)))

def bench(mod):
    t0 = time.perf_counter()
    for st, mat in cases:
        for mode, npl in [('2D_XY', 10), ('2D_XY', 37), ('3D', 50)]:
            mod.find_critical_plane_shear(st, mat, mode=mode, num_planes=npl)
            mod.find_critical_plane_tensile(st, mat, mode=mode, num_planes=npl)
            mod.find_critical_plane_compression(st, mat, mode=mode, num_planes=npl)
    return time.perf_counter() - t0

t_old = bench(cr_old)
t_new_b = bench(cr_new)

print(f"errors: {errors}, max_diff(f,n): {max_diff:.2e}")
print(f"old: {t_old:.4f} s, new: {t_new_b:.4f} s, speedup: {t_old/t_new_b:.1f}x")
