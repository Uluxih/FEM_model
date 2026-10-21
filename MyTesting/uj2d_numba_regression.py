# -*- coding: utf-8 -*-
"""
Регрессионный бенчмарк для UbiquitousJointModel2D (проверка Numba-оптимизации).

Прогоняет набор детерминированных сценариев нагружения одной точки
интегрирования, покрывающих все ветви конститутивной модели:
растяжение / сдвиг / сжатие шва, углы поверхности текучести,
пластичность матрицы (Друкер-Прагер), свободный поиск критической
плоскости, режим numerical-тангенса.

Использование (из корня проекта):
  python MyTesting/uj2d_numba_regression.py save  MyTesting/uj2d_baseline.npz
  python MyTesting/uj2d_numba_regression.py check MyTesting/uj2d_baseline.npz
  python MyTesting/uj2d_numba_regression.py bench
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from FEM.Abstract.Integration_Point_Level import Material
import FEM.Integration_Point_Level.CriticalPlane.material as cp_mt
from FEM.Integration_Point_Level.UbiquitousJointModel2D import UbiquitousJointModel2D


class JointedMaterial(Material):
    def __init__(self, E, nu, joint_params, matrix_params=None):
        super().__init__(E, nu)
        self.joint_params = joint_params
        self.matrix_params = matrix_params if matrix_params is not None else {}


def make_cp_material(cohesion=2.0e6, Rp_scale=1.0, Rc_scale=1.0):
    return cp_mt.Material(
        mu=0.56,
        A_tensor=np.eye(9) * cohesion ** 2,
        Rpx=0.45e6 * Rp_scale, Rpy=0.35e6 * Rp_scale, Rpz=0.45e6 * Rp_scale,
        Rcx=9.0e6 * Rc_scale, Rcy=12.0e6 * Rc_scale, Rcz=12.0e6 * Rc_scale,
    )


def make_model(joint_overrides=None, matrix_params=None, cp_material=None):
    jp = dict(
        tangent_mode='secant',
        phi=30.0, psi=5.0, phi_r=20.0,
        cp_material=cp_material if cp_material is not None else make_cp_material(),
        l_c=0.1,
        Gf_t=200, Gf_c=12000, Gf_s=2000,
        a_t=0.5, a_s=0.5,
        mu=0.6,
        fcr_over_fc=0.1,
    )
    if joint_overrides:
        jp.update(joint_overrides)
    mp = matrix_params if matrix_params is not None else {
        'c': 1.4e6, 'phi': 30.0, 'psi': 0.0}
    mat = JointedMaterial(E=3500e6, nu=0.2, joint_params=jp, matrix_params=mp)
    return UbiquitousJointModel2D(mat)


def stack_path(ex, ey, gxy):
    return [np.array([a, b, c]) for a, b, c in zip(ex, ey, gxy)]


def scenario_list():
    """Возвращает [(name, model, eps_path), ...] — детерминированно."""
    sc = []

    # 1. Одноосное растяжение вдоль X, трещина с нормалью X: нагрузка-разгрузка-повторная
    ex = np.concatenate([
        np.linspace(0.0, 3e-3, 100),
        np.linspace(3e-3, -5e-4, 80),
        np.linspace(-5e-4, 3e-3, 100),
    ])
    sc.append(("tension_preset",
               make_model({'preset_plane_normal': np.array([1.0, 0.0, 0.0])}),
               stack_path(ex, -0.2 * ex, np.zeros_like(ex))))

    # 2. Сдвиг горизонтального шва при постоянном сжатии (ветвь F2, D_s при sn<0)
    #    Прочная матрица (c=50e6), чтобы сдвиг ушёл именно в шов, а не в DP матрицы
    ey = np.concatenate([np.linspace(0.0, -1e-3, 20), np.full(280, -1e-3)])
    g = np.concatenate([np.linspace(0.0, 8e-3, 150), np.linspace(8e-3, -8e-3, 150)])
    sc.append(("shear_compression_preset",
               make_model({'preset_plane_normal': np.array([0.0, 1.0, 0.0])},
                          matrix_params={'c': 50e6, 'phi': 30.0, 'psi': 0.0}),
               stack_path(np.zeros_like(ey), ey, g)))

    # 3. Одновременное растяжение + сдвиг (угол F1+F2)
    ey3 = np.concatenate([np.linspace(0.0, 2e-3, 200), np.linspace(2e-3, 0.0, 100)])
    g3 = np.concatenate([np.linspace(0.0, 4e-3, 200), np.linspace(4e-3, 0.0, 100)])
    sc.append(("corner_tension_shear",
               make_model({'preset_plane_normal': np.array([0.0, 1.0, 0.0])}),
               stack_path(np.zeros_like(ey3), ey3, g3)))

    # 4. Свободный поиск критической плоскости (без preset), случайное блуждание
    rng = np.random.default_rng(123)
    inc = rng.normal(0.0, 1.5e-4, size=(300, 3))
    inc[:, 0] += 5e-5   # дрейф на растяжение, чтобы трещина образовалась
    path4 = np.cumsum(inc, axis=0)
    sc.append(("free_crack_random",
               make_model(),
               [row.copy() for row in path4]))

    # 5. Пластичность матрицы Друкера-Прагера (трещина подавлена прочным cp)
    strong_cp = make_cp_material(cohesion=1.0e12, Rp_scale=1e6, Rc_scale=1e6)
    rng5 = np.random.default_rng(7)
    inc5 = rng5.normal(0.0, 1.0e-4, size=(300, 3))
    inc5[:, 2] += 3e-5  # дрейф сдвига -> выход на поверхность DP
    path5 = np.cumsum(inc5, axis=0)
    sc.append(("matrix_dp",
               make_model({'cp_material': strong_cp},
                          matrix_params={'c': 0.3e6, 'phi': 30.0, 'psi': 5.0}),
               [row.copy() for row in path5]))

    # 6. Сжатие шва до раздавливания (ветвь F3, D_nc)
    ex6 = np.concatenate([np.linspace(0.0, -1e-2, 150),
                          np.linspace(-1e-2, -2e-3, 100)])
    sc.append(("compression_preset",
               make_model({'preset_plane_normal': np.array([1.0, 0.0, 0.0])}),
               stack_path(ex6, np.zeros_like(ex6), np.zeros_like(ex6))))

    # 7. Режим numerical-тангенса (FD-пробы вызывают update_state рекурсивно)
    ex7 = np.linspace(0.0, 2e-3, 100)
    sc.append(("numerical_tangent",
               make_model({'preset_plane_normal': np.array([1.0, 0.0, 0.0]),
                           'tangent_mode': 'numerical'}),
               stack_path(ex7, np.zeros_like(ex7), np.zeros_like(ex7))))

    # 8. Угол сжатие+сдвиг шва (F2+F3): сильное сжатие по X + большой сдвиг
    ex8 = np.concatenate([np.linspace(0.0, -6e-3, 40), np.full(260, -6e-3)])
    g8 = np.concatenate([np.zeros(40), np.linspace(0.0, 1.2e-2, 260)])
    sc.append(("corner_compression_shear",
               make_model({'preset_plane_normal': np.array([1.0, 0.0, 0.0])},
                          matrix_params={'c': 50e6, 'phi': 30.0, 'psi': 0.0}),
               stack_path(ex8, np.zeros_like(ex8), g8)))

    return sc


def run_all(collect=True):
    results = {}
    times = {}
    for name, model, path in scenario_list():
        out = []
        t0 = time.perf_counter()
        for eps in path:
            sig, D = model.update_state(eps)
            model.commit()
            if collect:
                out.append(np.concatenate([sig, D.ravel()]))
        times[name] = time.perf_counter() - t0
        if collect:
            results[name] = np.array(out)
            results[name + '__state'] = np.array([
                model.q_old, model.W_pl_t_old, model.W_pl_c_old,
                model.W_pl_s_old, model.D_nt_old, model.D_nc_old,
                model.D_s_old,
                *model.sig_eff_glob_old, *model.eps_p_old,
                *model.eps_p_mat_old,
            ])
    return results, times


def warmup():
    """Прогрев (компиляция numba вне замеров)."""
    for name, model, path in scenario_list():
        for eps in path[:30]:
            model.update_state(eps)
            model.commit()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'bench'
    path = sys.argv[2] if len(sys.argv) > 2 else 'MyTesting/uj2d_baseline.npz'

    warmup()
    results, times = run_all(collect=True)

    total = sum(times.values())
    for k, v in times.items():
        print(f"  time[{k}] = {v:.3f} s")
    print(f"  TOTAL = {total:.3f} s")

    if mode == 'save':
        np.savez(path, **results)
        print(f"Эталон сохранён: {path}")
    elif mode == 'check':
        ref = np.load(path)
        worst_rel = 0.0
        worst_key = None
        for key in results:
            a, b = results[key], ref[key]
            abs_diff = np.max(np.abs(a - b))
            scale = max(np.max(np.abs(b)), 1.0)
            rel = abs_diff / scale
            status = "OK " if rel < 1e-8 else "FAIL"
            print(f"  [{status}] {key}: max_abs_diff={abs_diff:.3e}, "
                  f"max_rel_diff={rel:.3e}")
            if rel > worst_rel:
                worst_rel, worst_key = rel, key
        if worst_rel < 1e-8:
            print(f"REGRESSION PASS (worst rel diff {worst_rel:.3e} on "
                  f"'{worst_key}')")
        else:
            print(f"REGRESSION FAIL (worst rel diff {worst_rel:.3e} on "
                  f"'{worst_key}')")
            sys.exit(1)


if __name__ == '__main__':
    main()
