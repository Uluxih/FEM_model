# -*- coding: utf-8 -*-
"""
Numba-JIT ядра для UbiquitousJointModel2D.

Формулы один-в-один повторяют исходные методы класса
(_return_mapping_matrix, _return_mapping_stress, _calculate_damage,
_apply_damage, _evaluate_joint, секущая матрица) — логика расчёта
не изменена, убраны только накладные расходы интерпретатора и NumPy.

Если numba не установлена, ядра остаются обычными функциями Python
(код работоспособен, просто без JIT-ускорения).
"""
import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        """Заглушка декоратора при отсутствии numba."""
        if args and callable(args[0]):
            return args[0]

        def deco(f):
            return f
        return deco


# ----------------------------------------------------------------------
# Решение системы 2x2 с частичным выбором главного элемента
# (в точности как LAPACK dgesv, вызываемый np.linalg.solve).
# При точной сингулярности (нулевой пивот) возвращает (0, 0) —
# эквивалент fallback'а "except LinAlgError: x = zeros(2)".
# ----------------------------------------------------------------------
@njit(cache=True)
def _solve2x2(a11, a12, a21, a22, b1, b2):
    if abs(a21) > abs(a11):
        a11, a21 = a21, a11
        a12, a22 = a22, a12
        b1, b2 = b2, b1
    if a11 == 0.0:
        return 0.0, 0.0
    m = a21 / a11
    u22 = a22 - m * a12
    y2 = b2 - m * b1
    if u22 == 0.0:
        return 0.0, 0.0
    x1 = y2 / u22
    x0 = (b1 - a12 * x1) / a11
    return x0, x1


# ----------------------------------------------------------------------
# Return mapping матрицы (Друкер-Прагер, эффективные напряжения)
# ----------------------------------------------------------------------
@njit(cache=True)
def rm_matrix_kernel(sig_tr, D_rock, D_rock_inv,
                     sin_phi_m, cos_phi_m, sin_psi_m, c_m):
    p_tr = 0.5 * (sig_tr[0] + sig_tr[1])
    sx = 0.5 * (sig_tr[0] - sig_tr[1])
    txy = sig_tr[2]
    q_tr = np.hypot(sx, txy)

    F_tr = q_tr + p_tr * sin_phi_m - c_m * cos_phi_m
    if F_tr <= 1e-10:
        return sig_tr.copy(), np.zeros(3), D_rock.copy()

    K_2D = 0.5 * (D_rock[0, 0] + D_rock[0, 1])
    G = D_rock[2, 2]

    denom = G + K_2D * sin_phi_m * sin_psi_m
    dlam = F_tr / denom

    p_new = p_tr - K_2D * sin_psi_m * dlam
    q_new = q_tr - G * dlam

    p_apex = c_m * cos_phi_m / sin_phi_m if sin_phi_m > 1e-12 else np.inf

    at_apex = False
    if q_new < 0.0 or p_new > p_apex:
        p_new = min(p_apex, p_tr)
        q_new = 0.0
        at_apex = True

    sig_mat = np.zeros(3)
    if q_tr > 1e-12:
        sx_new = sx * (q_new / q_tr)
        txy_new = txy * (q_new / q_tr)
    else:
        sx_new = 0.0
        txy_new = 0.0

    sig_mat[0] = p_new + sx_new
    sig_mat[1] = p_new - sx_new
    sig_mat[2] = txy_new

    # deps_p = D^{-1} (sig_tr - sig_mat)
    deps_p_mat = D_rock_inv @ (sig_tr - sig_mat)

    # Континуальный эласто-пластический тангенс Друкера-Прагера
    if not at_apex and q_new > 1e-12:
        c_dev = 0.5 * sx_new / q_new
        s_dev = txy_new / q_new
        n_vec = np.array([c_dev + 0.5 * sin_phi_m,
                          -c_dev + 0.5 * sin_phi_m,
                          s_dev])
        m_vec = np.array([c_dev + 0.5 * sin_psi_m,
                          -c_dev + 0.5 * sin_psi_m,
                          s_dev])
    else:
        # Апекс: пластичность только по объёмной составляющей
        n_vec = np.array([0.5 * sin_phi_m, 0.5 * sin_phi_m, 0.0])
        m_vec = np.array([0.5 * sin_psi_m, 0.5 * sin_psi_m, 0.0])

    Dm = D_rock @ m_vec
    den_ep = n_vec @ Dm
    if abs(den_ep) > 1e-12:
        D_ep = D_rock - np.outer(Dm, D_rock @ n_vec) / den_ep
    else:
        D_ep = D_rock.copy()

    return sig_mat, deps_p_mat, D_ep


# ----------------------------------------------------------------------
# Return mapping шва (локальные эффективные напряжения)
# ----------------------------------------------------------------------
@njit(cache=True)
def rm_stress_kernel(sig_tr_local, D_local,
                     E_n, G_s, H_t, f_t, f_c, c_val,
                     q_old, q_lim, tan_phi, tan_psi):
    sn_tr = sig_tr_local[0]
    tau_tr = sig_tr_local[2]

    abs_tau_tr = abs(tau_tr)
    sign_tau = 1.0 if tau_tr >= 0.0 else -1.0

    ft_curr = f_t + q_old
    # _c_curr(q_old)
    if q_old <= q_lim:
        c_curr = c_val
    else:
        c_curr = c_val + (q_old - q_lim) * tan_phi

    F1_tr = sn_tr - ft_curr                          # растяжение
    F3_tr = -sn_tr - f_c                             # сжатие
    F2_tr = abs_tau_tr + sn_tr * tan_phi - c_curr    # сдвиг

    dlams = np.zeros(3)
    tol = 1e-8

    if F1_tr <= tol and F2_tr <= tol and F3_tr <= tol:
        return sig_tr_local.copy(), dlams, 0.0, np.zeros(3)

    denom_t = E_n + H_t
    denom_s = G_s + E_n * tan_phi * tan_psi

    # --- Выбор активного набора поверхностей ---
    if F1_tr > tol and (abs_tau_tr + ft_curr * tan_phi - c_curr) <= tol:
        # Чистое растяжение
        dlams[0] = F1_tr / denom_t

    elif F3_tr > tol and (abs_tau_tr - f_c * tan_phi - c_curr) <= tol:
        # Чистое сжатие
        dlams[1] = F3_tr / E_n

    elif (F1_tr > tol and (abs_tau_tr + ft_curr * tan_phi - c_curr > tol)) or \
            (F2_tr > tol and (F1_tr - (F2_tr / denom_s) * E_n * tan_psi > tol)):
        # Угол растяжение + сдвиг
        x0, x1 = _solve2x2(denom_t, E_n * tan_psi,
                           E_n * tan_phi, denom_s,
                           F1_tr, F2_tr)
        if x0 < 0.0:
            # Активна только сдвиговая поверхность
            dlams[2] = max(F2_tr, 0.0) / denom_s
        elif x1 < 0.0:
            # Активна только поверхность растяжения
            dlams[0] = max(F1_tr, 0.0) / denom_t
        else:
            dlams[0], dlams[2] = x0, x1

    elif (F2_tr > tol and (F3_tr + (F2_tr / denom_s) * E_n * tan_psi > tol)) or \
            (F3_tr > tol and (abs_tau_tr - f_c * tan_phi - c_curr > tol)):
        # Угол сжатие + сдвиг
        x0, x1 = _solve2x2(E_n, -E_n * tan_psi,
                           -E_n * tan_phi, denom_s,
                           F3_tr, F2_tr)
        if x0 < 0.0:
            dlams[2] = max(F2_tr, 0.0) / denom_s
        elif x1 < 0.0:
            dlams[1] = max(F3_tr, 0.0) / E_n
        else:
            dlams[1], dlams[2] = x0, x1

    elif F2_tr > tol:
        # Чистый сдвиг
        dlams[2] = F2_tr / denom_s

    # deps_p_local = dlams[0]*m_t + dlams[1]*m_c + dlams[2]*m_s,
    # m_t=[1,0,0], m_c=[-1,0,0], m_s=[tan_psi,0,sign_tau]
    deps_p_local = np.zeros(3)
    deps_p_local[0] = dlams[0] - dlams[1] + dlams[2] * tan_psi
    deps_p_local[2] = dlams[2] * sign_tau

    sig_eff_new = sig_tr_local - D_local @ deps_p_local
    dq = dlams[0] * H_t

    return sig_eff_new, dlams, dq, deps_p_local


# ----------------------------------------------------------------------
# Поврежденность
# ----------------------------------------------------------------------
@njit(cache=True)
def damage_kernel(W_pl_t, W_pl_c, W_pl_s, q_new, sn_eff,
                  Gf_t, Gf_c, Gf_s, a_t, a_s, f_t, c_val,
                  tan_phi, tan_phi_r, fcr_over_fc,
                  d_nt_max, d_nc_max, d_s_max):
    r_t = min(W_pl_t / Gf_t, 1.0)
    r_c = min(W_pl_c / Gf_c, 1.0)
    r_s = min(W_pl_s / Gf_s, 1.0)

    Fp_t = r_t * (2.0 - r_t)
    Fp_c = 0.5 * (np.sin(np.pi * r_c - 0.5 * np.pi) + 1.0)
    Fp_s = r_s * (2.0 - r_s)

    dt = Fp_t + a_t * Fp_s * (1.0 - Fp_t)
    D_nt = 1.0 - (1.0 - dt) * f_t / (f_t + q_new + 1e-12)

    D_nc = (1.0 - fcr_over_fc) * Fp_c

    ds_base = min(max(a_s * Fp_t * (1.0 - Fp_s) * (1.0 - Fp_c)
                      + Fp_s + Fp_c - Fp_s * Fp_c, 0.0), 1.0)
    if sn_eff < 0.0:
        abs_sn = -sn_eff
        D_s = ds_base * (c_val + abs_sn * (tan_phi - tan_phi_r)) / (
                c_val + abs_sn * tan_phi + 1e-12)
    else:
        D_s = ds_base

    return (min(D_nt, d_nt_max),
            min(D_nc, d_nc_max),
            min(D_s, d_s_max))


@njit(cache=True)
def apply_damage_kernel(sig_eff_local, D_n, D_s, mpc_cap):
    """Эффективные напряжения -> номинальные (поврежденность один раз)."""
    sig_nom = sig_eff_local.copy()
    sig_nom[0] *= (1.0 - D_n)
    sig_nom[2] *= (1.0 - D_s)

    if mpc_cap > 0.0:
        D_parallel = min(max(D_n, D_s), mpc_cap)
        sig_nom[1] *= (1.0 - D_parallel)

    return sig_nom


# ----------------------------------------------------------------------
# Return mapping шва + энергии + поврежденность (единый вызов)
# ----------------------------------------------------------------------
@njit(cache=True)
def evaluate_joint_kernel(sig_tr_local, D_local,
                          E_n, G_s, H_t, f_t, f_c, c_val,
                          q_old, q_lim, tan_phi, tan_psi, tan_phi_r,
                          W_pl_t_old, W_pl_c_old, W_pl_s_old,
                          D_nt_old, D_nc_old, D_s_old,
                          Gf_t, Gf_c, Gf_s, a_t, a_s,
                          fcr_over_fc, d_nt_max, d_nc_max, d_s_max,
                          mpc_cap):
    sig_eff_new, dlams, dq, deps_p_local = rm_stress_kernel(
        sig_tr_local, D_local, E_n, G_s, H_t, f_t, f_c, c_val,
        q_old, q_lim, tan_phi, tan_psi)

    sig_n_yield_start = f_t + q_old
    dW_t = max(0.5 * (sig_n_yield_start + sig_eff_new[0]) * dlams[0], 0.0)
    dW_c = max(abs(sig_eff_new[0]) * dlams[1], 0.0)
    dW_s = max((abs(sig_eff_new[2]) + sig_eff_new[0] * tan_psi) * dlams[2],
               0.0)

    W_pl_t = W_pl_t_old + dW_t
    W_pl_c = W_pl_c_old + dW_c
    W_pl_s = W_pl_s_old + dW_s
    q_new = q_old + dq

    D_nt_calc, D_nc_calc, D_s_calc = damage_kernel(
        W_pl_t, W_pl_c, W_pl_s, q_new, sig_eff_new[0],
        Gf_t, Gf_c, Gf_s, a_t, a_s, f_t, c_val,
        tan_phi, tan_phi_r, fcr_over_fc,
        d_nt_max, d_nc_max, d_s_max)

    # Необратимость поврежденности
    D_nt = max(D_nt_calc, D_nt_old)
    D_nc = max(D_nc_calc, D_nc_old)
    D_s = max(D_s_calc, D_s_old)

    D_n = D_nt if sig_eff_new[0] >= 0.0 else D_nc

    sig_nom_local = apply_damage_kernel(sig_eff_new, D_n, D_s, mpc_cap)

    return (sig_nom_local, sig_eff_new, deps_p_local,
            W_pl_t, W_pl_c, W_pl_s, q_new, D_nt, D_nc, D_s, D_n)


# ----------------------------------------------------------------------
# Секущая матрица в глобальных осях (случай зафиксированной трещины)
# ----------------------------------------------------------------------
@njit(cache=True)
def secant_locked_kernel(T_sig, D_matrix_ep, T_eps_inv, T_eps,
                         f_n, f_s, f_p):
    D_sec_local = T_sig @ D_matrix_ep @ T_eps_inv

    D_sec_local[0, :] *= f_n
    D_sec_local[:, 0] *= f_n
    D_sec_local[1, :] *= f_p
    D_sec_local[:, 1] *= f_p
    D_sec_local[2, :] *= f_s
    D_sec_local[:, 2] *= f_s

    return T_eps.T @ D_sec_local @ T_eps
