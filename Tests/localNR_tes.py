import numpy as np


# =====================================================================
# ИЗОЛИРОВАННАЯ МОДЕЛЬ (С УЛУЧШЕННЫМ АЛГОРИТМОМ RETURN MAPPING)
# =====================================================================
class UbiquitousJointModel3D_Test:
    def __init__(self):
        # Строго заданные параметры для возможности аналитического расчета
        self.E_n = 20000.0
        self.G_s = 10000.0

        self.f_t = 3.0
        self.f_c = 30.0
        self.c = 5.0

        self.tan_phi = 0.5773502691896257  # Точно tan(30 deg)
        self.tan_psi = 0.17632698070846498  # Точно tan(10 deg)

        self.H_t = -1000.0
        self.H_c = 0.0
        self.H_s = 0.0

        self.q_lim = self.c / self.tan_phi - self.f_t

        self.lam_t_old = 0.0
        self.lam_c_old = 0.0
        self.lam_s_old = 0.0

        self.nr_tol = 1e-9
        self.nr_max_iter = 50

    def _c_curr(self, q):
        if q <= self.q_lim: return self.c
        return self.c + (q - self.q_lim) * self.tan_phi

    def _ft_curr(self, q):
        if q <= self.q_lim: return self.f_t + q
        return np.inf

    def _return_mapping_nr(self, sig_n_tr, tau_23_tr, tau_13_tr):
        tol = self.nr_tol
        max_iter = self.nr_max_iter

        q_old = self.H_t * self.lam_t_old
        tau_tr = np.sqrt(tau_23_tr ** 2 + tau_13_tr ** 2)

        ft_yld = self._ft_curr(q_old)
        c_yld = self._c_curr(q_old)

        f_t_tr = (sig_n_tr - ft_yld) if np.isfinite(ft_yld) else -1.0
        f_c_tr = -sig_n_tr - self.f_c
        f_s_tr = tau_tr + sig_n_tr * self.tan_phi - c_yld

        tol_f_t = tol * max(self.f_t, 1.0)
        tol_f_c = tol * max(self.f_c, 1.0)
        tol_f_s = tol * max(self.c, 1.0)

        # ── Упругий шаг ──────────────────────────────────────────────────────────
        if f_t_tr <= tol_f_t and f_c_tr <= tol_f_c and f_s_tr <= tol_f_s:
            return sig_n_tr, tau_23_tr, tau_13_tr, 0.0, 0.0, 0.0

        # ── Улучшенная начальная догадка ─────────────────────────────────────────
        # act_s определяется по нарушению поверхности сдвига ПОСЛЕ упругой
        # проекции на T/C, а не по исходному пробному напряжению.
        act_t = f_t_tr > tol_f_t and np.isfinite(ft_yld)
        act_c = f_c_tr > tol_f_c and not act_t
        sn_proj = ft_yld if act_t else (-self.f_c if act_c else sig_n_tr)
        act_s = (tau_tr + sn_proj * self.tan_phi - c_yld) > tol_f_s

        # ── Вспомогательные константы ─────────────────────────────────────────────
        scale_nr = max(self.E_n, self.G_s, 1.0)

        # Физическая верхняя граница для множителей пластичности
        max_dlam = (abs(sig_n_tr) + abs(tau_23_tr) + abs(tau_13_tr) +
                    max(self.f_t, self.f_c, self.c)) / min(self.E_n, self.G_s)

        # Допуск для проверки нарушения поверхностей (в единицах напряжений)
        nr_abs_tol = tol * max(self.f_t, self.f_c, self.c) * 1e4

        # ── Внутренний NR-решатель ───────────────────────────────────────────────
        def _nr_solve(act_t, act_c, act_s):
            col_t = 3 if act_t else None
            col_c = (3 + int(act_t)) if act_c else None
            col_s = (3 + int(act_t) + int(act_c)) if act_s else None
            n = 3 + int(act_t) + int(act_c) + int(act_s)

            x = np.zeros(n)
            x[0], x[1], x[2] = sig_n_tr, tau_23_tr, tau_13_tr

            R_vec = np.zeros(n)  # хранит последнюю невязку для флага сходимости

            for _ in range(max_iter):
                sn_i = x[0];
                t23_i = x[1];
                t13_i = x[2]
                dlt_i = x[col_t] if act_t else 0.0
                dlc_i = x[col_c] if act_c else 0.0
                dls_i = x[col_s] if act_s else 0.0

                tau_i = np.sqrt(t23_i ** 2 + t13_i ** 2)
                tau_si = max(tau_i, 1e-14)
                n23_i = t23_i / tau_si
                n13_i = t13_i / tau_si

                q_i = self.H_t * (self.lam_t_old + dlt_i)
                ft_i = self._ft_curr(q_i)
                c_i = self._c_curr(q_i)

                # Невязка
                R_vec = np.zeros(n)
                R_vec[0] = sn_i - sig_n_tr + self.E_n * (dlt_i - dlc_i + dls_i * self.tan_psi)
                R_vec[1] = t23_i - tau_23_tr + self.G_s * dls_i * n23_i
                R_vec[2] = t13_i - tau_13_tr + self.G_s * dls_i * n13_i
                row = 3
                if act_t:
                    R_vec[row] = sn_i - (ft_i if np.isfinite(ft_i) else 1e30);
                    row += 1
                if act_c:
                    R_vec[row] = -sn_i - self.f_c;
                    row += 1
                if act_s:
                    R_vec[row] = tau_i + sn_i * self.tan_phi - c_i;
                    row += 1

                if np.linalg.norm(R_vec) / scale_nr < tol:
                    break

                # Якобиан
                J = np.zeros((n, n))
                J[0, 0] = 1.0
                if tau_si > 1e-10:
                    J[1, 1] = 1.0 + self.G_s * dls_i * n13_i ** 2 / tau_si
                    J[1, 2] = -self.G_s * dls_i * n23_i * n13_i / tau_si
                    J[2, 1] = J[1, 2]
                    J[2, 2] = 1.0 + self.G_s * dls_i * n23_i ** 2 / tau_si
                else:
                    J[1, 1] = 1.0
                    J[2, 2] = 1.0

                if act_t: J[0, col_t] = self.E_n
                if act_c: J[0, col_c] = -self.E_n
                if act_s:
                    J[0, col_s] = self.E_n * self.tan_psi
                    if tau_si > 1e-10:
                        J[1, col_s] = self.G_s * n23_i
                        J[2, col_s] = self.G_s * n13_i

                row = 3
                if act_t:
                    J[row, 0] = 1.0
                    J[row, col_t] = -self.H_t
                    row += 1
                if act_c:
                    J[row, 0] = -1.0
                    row += 1
                if act_s:
                    J[row, 0] = self.tan_phi
                    if tau_si > 1e-10:
                        J[row, 1] = n23_i
                        J[row, 2] = n13_i
                    if act_t and q_i > self.q_lim:
                        J[row, col_t] = -self.H_t * self.tan_phi
                    row += 1

                try:
                    x -= np.linalg.solve(J, R_vec)
                except np.linalg.LinAlgError:
                    break  # сингулярный Якобиан → выход, converged=False

            # Флаг сходимости: финальная невязка в 1e5 раз мягче основного критерия,
            # но всё равно жёстче, чем у любого расходящегося решения
            converged = np.linalg.norm(R_vec) / scale_nr < tol * 1e5

            return (x[0], x[1], x[2],
                    x[col_t] if act_t else 0.0,
                    x[col_c] if act_c else 0.0,
                    x[col_s] if act_s else 0.0,
                    converged)

        # ── Вспомогательные функции проверки ─────────────────────────────────────
        def _is_valid(sn, t23, t13, dl_t):
            """Все поверхности текучести не нарушены (с допуском)."""
            tau = np.sqrt(t23 ** 2 + t13 ** 2)
            q = self.H_t * (self.lam_t_old + dl_t)
            ft_v = self._ft_curr(q)
            ft_check = (sn - ft_v) if np.isfinite(ft_v) else (sn - 1e30)
            return (ft_check <= nr_abs_tol and
                    -sn - self.f_c <= nr_abs_tol and
                    tau + sn * self.tan_phi - self._c_curr(q) <= nr_abs_tol)

        def _check_multipliers(act_t, act_c, act_s, dlt, dlc, dls):
            """Множители пластичности неотрицательны и физически разумны."""
            tol_lam = 1e-10
            if act_t and (dlt < -tol_lam or dlt > max_dlam): return False
            if act_c and (dlc < -tol_lam or dlc > max_dlam): return False
            if act_s and (dls < -tol_lam or dls > max_dlam): return False
            return True

        # ── Шаг 1: проверяем начальную догадку ───────────────────────────────────
        sig_n, tau_23, tau_13, d_lam_t, d_lam_c, d_lam_s, conv = \
            _nr_solve(act_t, act_c, act_s)

        if (conv
                and _check_multipliers(act_t, act_c, act_s, d_lam_t, d_lam_c, d_lam_s)
                and _is_valid(sig_n, tau_23, tau_13, d_lam_t)):
            return sig_n, tau_23, tau_13, d_lam_t, d_lam_c, d_lam_s

        # ── Шаг 2: систематический перебор активных наборов (Active Set Search) ──
        combinations = [
            (True, False, True),  # Отрыв  + Сдвиг
            (False, True, True),  # Сжатие + Сдвиг
            (True, False, False),  # Только отрыв
            (False, True, False),  # Только сжатие
            (False, False, True),  # Только сдвиг
        ]

        for test_t, test_c, test_s in combinations:
            sn, t23, t13, dlt, dlc, dls, conv = _nr_solve(test_t, test_c, test_s)

            if not conv:                                          continue
            if not _check_multipliers(test_t, test_c, test_s,
                                      dlt, dlc, dls):            continue
            if _is_valid(sn, t23, t13, dlt):
                return sn, t23, t13, dlt, dlc, dls

        # ── Fallback: для корректных выпуклых поверхностей сюда не должны доходить
        raise RuntimeError(
            f"Return mapping did not converge: "
            f"sig_n_tr={sig_n_tr:.6g}, "
            f"tau_23_tr={tau_23_tr:.6g}, "
            f"tau_13_tr={tau_13_tr:.6g}"
        )


# =====================================================================
# ТЕСТОВЫЙ ФРЕЙМВОРК С АНАЛИТИЧЕСКИМИ РЕШЕНИЯМИ
# =====================================================================
model = UbiquitousJointModel3D_Test()


def assert_close(name, numerical, analytical, tol=1e-5):
    diff = abs(numerical - analytical)
    if diff < tol:
        print(f"  [OK] {name:15s} : {numerical:10.5f} == {analytical:10.5f}")
    else:
        print(f"  [FAIL] {name:13s} : {numerical:10.5f} != {analytical:10.5f} (Diff: {diff:.2e})")


print("\n" + "=" * 70)
print("ЗАПУСК АНАЛИТИЧЕСКИХ ТЕСТОВ RETURN MAPPING ALGORITHM")
print("=" * 70)

# ---------------------------------------------------------
# ТЕСТ 1: Упругий шаг
# ---------------------------------------------------------
print("\n--- ТЕСТ 1: Упругость (Без пластики) ---")
sn_tr, t23_tr, t13_tr = 2.0, 1.0, 1.0
sn, t23, t13, dlt, dlc, dls = model._return_mapping_nr(sn_tr, t23_tr, t13_tr)

assert_close("sig_n", sn, sn_tr)
assert_close("tau_23", t23, t23_tr)
assert_close("d_lam_t", dlt, 0.0)

# ---------------------------------------------------------
# ТЕСТ 2: Чистый отрыв (Mode I)
# ---------------------------------------------------------
print("\n--- ТЕСТ 2: Чистый отрыв (Линейное разупрочнение) ---")
sn_tr, t23_tr, t13_tr = 10.0, 0.0, 0.0
sn, t23, t13, dlt, dlc, dls = model._return_mapping_nr(sn_tr, t23_tr, t13_tr)

dlt_ana = (sn_tr - model.f_t) / (model.E_n + model.H_t)
sn_ana = sn_tr - model.E_n * dlt_ana

assert_close("sig_n", sn, sn_ana)
assert_close("d_lam_t", dlt, dlt_ana)
assert_close("tau_23", t23, 0.0)

# ---------------------------------------------------------
# ТЕСТ 3: Чистое сжатие (Cap)
# ---------------------------------------------------------
print("\n--- ТЕСТ 3: Чистое сжатие (Идеальная пластичность) ---")
sn_tr, t23_tr, t13_tr = -40.0, 0.0, 0.0
sn, t23, t13, dlt, dlc, dls = model._return_mapping_nr(sn_tr, t23_tr, t13_tr)

sn_ana = -model.f_c
dlc_ana = (-sn_ana + sn_tr) / -model.E_n

assert_close("sig_n", sn, sn_ana)
assert_close("d_lam_c", dlc, dlc_ana)

# ---------------------------------------------------------
# ТЕСТ 4: Чистый сдвиг (С учетом дилатансии)
# ---------------------------------------------------------
print("\n--- ТЕСТ 4: Чистый сдвиг (Mohr-Coulomb + Dilation) ---")
sn_tr, t23_tr, t13_tr = -10.0, 20.0, 0.0
sn, t23, t13, dlt, dlc, dls = model._return_mapping_nr(sn_tr, t23_tr, t13_tr)

fs_tr = t23_tr + sn_tr * model.tan_phi - model.c
denom = model.G_s + model.E_n * model.tan_psi * model.tan_phi
dls_ana = fs_tr / denom
sn_ana = sn_tr - model.E_n * dls_ana * model.tan_psi
t23_ana = t23_tr - model.G_s * dls_ana

assert_close("sig_n", sn, sn_ana)
assert_close("tau_23", t23, t23_ana)
assert_close("d_lam_s", dls, dls_ana)

# ---------------------------------------------------------
# ТЕСТ 5: Угол "Сжатие + Сдвиг"
# ---------------------------------------------------------
print("\n--- ТЕСТ 5: Угол (Сжатие + Сдвиг) ---")
sn_tr, t23_tr, t13_tr = -50.0, 30.0, 0.0
sn, t23, t13, dlt, dlc, dls = model._return_mapping_nr(sn_tr, t23_tr, t13_tr)

sn_ana = -model.f_c
t23_ana = model.c - sn_ana * model.tan_phi
dls_ana = (t23_tr - t23_ana) / model.G_s
dlc_ana = (sn_ana - sn_tr + model.E_n * dls_ana * model.tan_psi) / model.E_n

assert_close("sig_n", sn, sn_ana)
assert_close("tau_23", t23, t23_ana)
assert_close("d_lam_c", dlc, dlc_ana)
assert_close("d_lam_s", dls, dls_ana)

# ---------------------------------------------------------
# ТЕСТ 6: Угол "Отрыв + Сдвиг" (Coupled Hardening)
# ---------------------------------------------------------
print("\n--- ТЕСТ 6: Угол (Отрыв + Сдвиг со связанным разупрочнением) ---")
# Увеличили sn_tr до 15.0, чтобы гарантированно пробить дилатансию и остаться в отрыве
sn_tr, t23_tr, t13_tr = 15.0, 10.0, 0.0
sn, t23, t13, dlt, dlc, dls = model._return_mapping_nr(sn_tr, t23_tr, t13_tr)

A = np.array([
    [model.E_n + model.H_t, model.E_n * model.tan_psi],
    [model.H_t * model.tan_phi, -model.G_s]
])
B = np.array([
    sn_tr - model.f_t,
    -t23_tr - model.f_t * model.tan_phi + model.c
])
x_ana = np.linalg.solve(A, B)
dlt_ana, dls_ana = x_ana[0], x_ana[1]
sn_ana = model.f_t + model.H_t * dlt_ana
t23_ana = t23_tr - model.G_s * dls_ana

assert_close("sig_n", sn, sn_ana)
assert_close("tau_23", t23, t23_ana)
assert_close("d_lam_t", dlt, dlt_ana)
assert_close("d_lam_s", dls, dls_ana)

print("\n" + "=" * 70)
print("ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ!")
print("=" * 70)
