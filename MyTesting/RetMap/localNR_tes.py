import numpy as np
# Импортируем ваш новый класс
from FEM.Integration_Point_Level.UbiquitousJointModel2D import UbiquitousJointModel2D


# =====================================================================
# ЗАГЛУШКА ДЛЯ МАТЕРИАЛА (MOCK MATERIAL)
# =====================================================================
class DummyMaterial:
    """Заглушка класса материала для инициализации модели в тестах"""
    def __init__(self):
        # При E = 20000 и nu = 0.0 компоненты D_rock[0,0] и D_rock[2,2]
        # будут в точности равны E_n = 20000.0 и G_s = 10000.0
        self.E = 20000.0
        self.nu = 0.0
        self.joint_params = {
            'phi': 30.0,
            'psi': 10.0,
            'phi_r': 30.0,
            'cp_material': "dummy_cp_material",
            'preset_plane_normal': np.array([0.0, 1.0, 0.0]),
            'force_horizontal': True,
            'l_c': 1.0,
            'Gf_t': 1.0,
            'Gf_c': 1.0,
            'Gf_s': 1.0,
        }


def setup_test_model():
    """Создает и настраивает экземпляр реальной модели с нужными параметрами"""
    material = DummyMaterial()
    model = UbiquitousJointModel2D(material)

    # Строго заданные параметры для возможности аналитического расчета
    model.E_n = 20000.0
    model.G_s = 10000.0

    model.f_t = 3.0
    model.f_c = 30.0
    model.c = 5.0

    model.tan_phi = 0.5773502691896257  # Точно tan(30 deg)
    model.tan_psi = 0.17632698070846498  # Точно tan(10 deg)

    model.H_t = -1000.0
    model.H_c = 0.0
    model.H_s = 0.0

    model.q_lim = model.c / model.tan_phi - model.f_t

    model.q_old = 0.0
    model.W_pl_t_old = 0.0
    model.W_pl_c_old = 0.0
    model.W_pl_s_old = 0.0

    # Перезаписываем локальную матрицу жесткости, чтобы она соответствовала тестам
    model.D_local = np.zeros((3, 3))
    model.D_local[0, 0] = model.E_n
    model.D_local[2, 2] = model.G_s

    return model


def run_test_case(model, sn_tr, t23_tr, t13_tr):
    """
    Адаптер для вызова нового метода _return_mapping_stress.
    Передает пробные напряжения через sig_eff_old и запускает шаг с нулевым deps_l.
    """
    # Задаем пробные напряжения напрямую в историю как старые эффективные напряжения
    model.sig_eff_old = np.array([sn_tr, t13_tr, t23_tr])
    model.q_old = 0.0  # Сбрасываем упрочнение для чистоты теста

    # Вызываем новый метод с нулевым приращением деформаций deps_l
    sig_eff_new, dlams, dq = model._return_mapping_stress(np.zeros(3))

    # Возвращаем: sn, t23, t13, dlt, dlc, dls
    return sig_eff_new[0], sig_eff_new[2], sig_eff_new[1], dlams[0], dlams[1], dlams[2]


def assert_close(name, numerical, analytical, tol=1e-5):
    diff = abs(numerical - analytical)
    if diff < tol:
        print(f"  [OK] {name:15s} : {numerical:10.5f} == {analytical:10.5f}")
    else:
        print(f"  [FAIL] {name:13s} : {numerical:10.5f} != {analytical:10.5f} (Diff: {diff:.2e})")


if __name__ == "__main__":
    model = setup_test_model()

    print("\n" + "=" * 70)
    print("ЗАПУСК АНАЛИТИЧЕСКИХ ТЕСТОВ RETURN MAPPING ALGORITHM (РЕАЛЬНЫЙ МОДУЛЬ)")
    print("=" * 70)

    # ---------------------------------------------------------
    # ТЕСТ 1: Упругий шаг
    # ---------------------------------------------------------
    print("\n--- ТЕСТ 1: Упругость (Без пластики) ---")
    sn_tr, t23_tr, t13_tr = 2.0, 1.0, 1.0
    sn, t23, t13, dlt, dlc, dls = run_test_case(model, sn_tr, t23_tr, t13_tr)

    assert_close("sig_n", sn, sn_tr)
    assert_close("tau_23", t23, t23_tr)
    assert_close("d_lam_t", dlt, 0.0)

    # ---------------------------------------------------------
    # ТЕСТ 2: Чистый отрыв (Mode I)
    # ---------------------------------------------------------
    print("\n--- ТЕСТ 2: Чистый отрыв (Линейное разупрочнение) ---")
    sn_tr, t23_tr, t13_tr = 10.0, 0.0, 0.0
    sn, t23, t13, dlt, dlc, dls = run_test_case(model, sn_tr, t23_tr, t13_tr)

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
    sn, t23, t13, dlt, dlc, dls = run_test_case(model, sn_tr, t23_tr, t13_tr)

    sn_ana = -model.f_c
    dlc_ana = (-sn_ana + sn_tr) / -model.E_n

    assert_close("sig_n", sn, sn_ana)
    assert_close("d_lam_c", dlc, dlc_ana)

    # ---------------------------------------------------------
    # ТЕСТ 4: Чистый сдвиг (С учетом дилатансии)
    # ---------------------------------------------------------
    print("\n--- ТЕСТ 4: Чистый сдвиг (Mohr-Coulomb + Dilation) ---")
    sn_tr, t23_tr, t13_tr = -10.0, 20.0, 0.0
    sn, t23, t13, dlt, dlc, dls = run_test_case(model, sn_tr, t23_tr, t13_tr)

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
    sn, t23, t13, dlt, dlc, dls = run_test_case(model, sn_tr, t23_tr, t13_tr)

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
    sn_tr, t23_tr, t13_tr = 15.0, 10.0, 0.0
    sn, t23, t13, dlt, dlc, dls = run_test_case(model, sn_tr, t23_tr, t13_tr)

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