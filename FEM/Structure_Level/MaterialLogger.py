import pandas as pd


class MaterialLogger:
    """
    Класс для сбора данных с точек интегрирования и выгрузки их в Excel.
    """

    def __init__(self):
        self.records = []

    def log_state(self, step, load_factor, el_id, ip_id, model):
        """Сбор данных с конкретной точки интегрирования"""
        record = {
            'Step': step,
            'Load_Factor': load_factor,
            'Element_ID': el_id,
            'IP_ID': ip_id,
            # Глобальные напряжения и деформации
            'Sig_xx': model.stress[0],
            'Sig_yy': model.stress[1],
            'Tau_xy': model.stress[2],
            'Eps_xx': model.strain[0],
            'Eps_yy': model.strain[1],
            'Gamma_xy': model.strain[2],
            # Параметры поврежденности
            'Damage_N': getattr(model, 'D_n', 0.0),
            'Damage_S': getattr(model, 'D_s', 0.0),
            # Пластическая работа и упрочнение
            'Work_T': model.W_pl_t_old,
            'Work_C': model.W_pl_c_old,
            'Work_S': model.W_pl_s_old,
            'q_hardening': model.q_old,
            # Статус
            'Is_Locked': model.is_locked
        }
        self.records.append(record)

    def save_to_excel(self, filename="material_log.xlsx"):
        """Сохранение накопленных данных в Excel"""
        if not self.records:
            print("Нет данных для сохранения в лог.")
            return

        df = pd.DataFrame(self.records)
        df.to_excel(filename, index=False)
        print(f"\n[+] Лог материалов успешно сохранен в файл: {filename}")