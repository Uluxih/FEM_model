import numpy as np
import pandas as pd


class MaterialLogger:
    """
    Сбор данных с точек интегрирования по шагам нагружения:
      - развитие трещин (повреждённости D, доли исчерпания энергий R = W_pl/Gf);
      - критерии разрушения (режим образования трещины, пластика матрицы DP).

    Логируются только ПОДТВЕРЖДЁННЫЕ (committed) состояния — вызывать
    log_state() строго после cm.commit().
    """

    FAILURE_R = 0.99  # R >= порога => энергия разрушения исчерпана (полное разрушение)

    def __init__(self):
        self.records = []

    def log_state(self, step, load_factor, el_id, ip_id, cx, cy, model):
        """Сбор данных с конкретной точки интегрирования (committed-значения)."""
        W_t = getattr(model, 'W_pl_t_old', 0.0)
        W_c = getattr(model, 'W_pl_c_old', 0.0)
        W_s = getattr(model, 'W_pl_s_old', 0.0)
        Gf_t = getattr(model, 'Gf_t', np.nan)
        Gf_c = getattr(model, 'Gf_c', np.nan)
        Gf_s = getattr(model, 'Gf_s', np.nan)

        stress = getattr(model, 'stress', np.zeros(3))
        strain = getattr(model, 'strain', np.zeros(3))

        is_locked = getattr(model, 'is_locked', False)
        n = getattr(model, 'fixed_normal', None)
        angle = (float(np.degrees(np.arctan2(n[1], n[0])))
                 if (is_locked and n is not None) else np.nan)

        self.records.append({
            'Step': step,
            'Load_Factor': load_factor,
            'Element_ID': el_id,
            'IP_ID': ip_id,
            'X': cx,
            'Y': cy,
            # Глобальные напряжения и деформации (номинальные)
            'Sig_xx': stress[0],
            'Sig_yy': stress[1],
            'Tau_xy': stress[2],
            'Eps_xx': strain[0],
            'Eps_yy': strain[1],
            'Gamma_xy': strain[2],
            # --- Трещина (ослабленная плоскость / шов) ---
            'Cracked': bool(is_locked),
            'Crack_Mode': getattr(model, 'crack_mode', None) or '',
            'Crack_Normal_Angle_deg': angle,
            'D_nt': getattr(model, 'D_nt_old', 0.0),
            'D_nc': getattr(model, 'D_nc_old', 0.0),
            'D_s': getattr(model, 'D_s_old', 0.0),
            # Доля исчерпанной энергии разрушения (0..1) — индикатор развития трещины
            'R_t': min(W_t / Gf_t, 1.0) if Gf_t else np.nan,
            'R_c': min(W_c / Gf_c, 1.0) if Gf_c else np.nan,
            'R_s': min(W_s / Gf_s, 1.0) if Gf_s else np.nan,
            # Пластические работы и упрочнение
            'W_pl_t': W_t,
            'W_pl_c': W_c,
            'W_pl_s': W_s,
            'q_hardening': getattr(model, 'q_old', 0.0),
            # --- Пластика матрицы (Друкер-Прагер) и шва ---
            'Eps_p_matrix': float(np.linalg.norm(getattr(model, 'eps_p_mat_old', np.zeros(3)))),
            'Eps_p_joint': float(np.linalg.norm(getattr(model, 'eps_p_old', np.zeros(3)))),
        })

    def save_to_excel(self, filename="material_log.xlsx"):
        """Сохранение накопленных данных в Excel"""
        if not self.records:
            print("Нет данных для сохранения в лог.")
            return

        df = pd.DataFrame(self.records)
        df.to_excel(filename, index=False)
        print(f"\n[+] Лог материалов сохранен: {filename} ({len(df)} записей)")

    def print_summary(self):
        """Текстовый отчёт: зарождение/развитие трещин и критерии разрушения."""
        if not self.records:
            print("Лог пуст — нечего анализировать.")
            return

        df = pd.DataFrame(self.records)
        key = ['Element_ID', 'IP_ID']
        n_points = df.groupby(key).ngroups

        print("\n" + "=" * 60)
        print("ОТЧЁТ О РАЗВИТИИ ТРЕЩИН И КРИТЕРИЯХ РАЗРУШЕНИЯ")
        print("=" * 60)

        # 1. Зарождение трещин: первый шаг, где Cracked == True, по каждой точке
        cracked = df[df['Cracked']]
        if cracked.empty:
            print("\n1. ЗАРОЖДЕНИЕ ТРЕЩИН: трещины не образовались.")
        else:
            init = (cracked.sort_values('Step')
                    .groupby(key, as_index=False)
                    .first()
                    .sort_values(['Step', 'Element_ID', 'IP_ID']))
            print(f"\n1. ЗАРОЖДЕНИЕ ТРЕЩИН: {len(init)} из {n_points} точек интегрирования")
            print(f"   {'Эл':>4} {'IP':>3} {'Шаг':>5} {'Фактор':>8} {'Режим':>12} "
                  f"{'Нормаль,°':>10} {'X':>7} {'Y':>7}")
            for _, r in init.iterrows():
                print(f"   {int(r['Element_ID']):>4} {int(r['IP_ID']):>3} "
                      f"{int(r['Step']):>5} {r['Load_Factor']:>8.3f} "
                      f"{str(r['Crack_Mode']):>12} "
                      f"{r['Crack_Normal_Angle_deg']:>10.1f} "
                      f"{r['X']:>7.3f} {r['Y']:>7.3f}")

        # 2. Полное разрушение по механизмам: R >= порога (энергия Gf исчерпана)
        print(f"\n2. ИСЧЕРПАНИЕ ЭНЕРГИИ РАЗРУШЕНИЯ (R = W_pl/Gf >= {self.FAILURE_R}):")
        any_fail = False
        for col, name in [('R_t', 'растяжение'), ('R_s', 'сдвиг'), ('R_c', 'сжатие')]:
            sub = df[df[col] >= self.FAILURE_R]
            if sub.empty:
                print(f"   {name:<11}: не достигнуто")
                continue
            any_fail = True
            first = sub.sort_values('Step').groupby(key, as_index=False).first()
            r0 = first.sort_values('Step').iloc[0]
            print(f"   {name:<11}: {len(first)} точек; первая — эл {int(r0['Element_ID'])}, "
                  f"IP {int(r0['IP_ID'])}, шаг {int(r0['Step'])} "
                  f"(фактор {r0['Load_Factor']:.3f})")

        # 3. Разрушение матрицы (пластические деформации Друкера-Прагера)
        last = df.sort_values('Step').groupby(key, as_index=False).last()
        mat = last[last['Eps_p_matrix'] > 1e-8]
        msg = f"\n3. ПЛАСТИКА МАТРИЦЫ (Друкер-Прагер): {len(mat)} точек с остаточными деформациями"
        if len(mat):
            msg += f", max |eps_p| = {mat['Eps_p_matrix'].max():.3e}"
        print(msg)

        # 4. Итоговая повреждённость на последнем шаге
        print("\n4. ИТОГОВАЯ ПОВРЕЖДЁННОСТЬ (последний шаг):")
        print(f"   D_nt: max = {last['D_nt'].max():.3f} | "
              f"D_nc: max = {last['D_nc'].max():.3f} | "
              f"D_s:  max = {last['D_s'].max():.3f}")
        print("=" * 60)
