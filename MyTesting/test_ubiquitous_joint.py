"""
Тесты надёжности для алгоритма Return Mapping (UbiquitousJointModel3D_Test).

Запуск:
    pytest test_ubiquitous_joint.py -v
    pytest test_ubiquitous_joint.py -v --cov   # с покрытием
"""

import numpy as np
import pytest

# Предполагается, что класс импортируется из вашего модуля:
# from ubiquitous_joint import UbiquitousJointModel3D_Test
# Для самодостаточности замените на ваш реальный импорт.
from localNR_tes import UbiquitousJointModel3D_Test


# =====================================================================
#  ФИКСТУРЫ И ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =====================================================================
@pytest.fixture
def model():
    """Свежая модель на КАЖДЫЙ тест — изоляция от накопленного состояния."""
    return UbiquitousJointModel3D_Test()


def yield_values(m, sn, t23, t13, dlt):
    """Возвращает (f_t, f_c, f_s) в конечном состоянии с учётом упрочнения по q."""
    tau = np.hypot(t23, t13)
    q = m.H_t * (m.lam_t_old + dlt)
    ft = m._ft_curr(q)
    f_t = (sn - ft) if np.isfinite(ft) else -np.inf
    f_c = -sn - m.f_c
    f_s = tau + sn * m.tan_phi - m._c_curr(q)
    return f_t, f_c, f_s


def mapping_residual(m, tr, res):
    """
    Остаток уравнений return-mapping (упругий предиктор - пластическая коррекция).
    tr  = (sn_tr, t23_tr, t13_tr)
    res = (sn, t23, t13, dlt, dlc, dls)
    """
    sn_tr, t23_tr, t13_tr = tr
    sn, t23, t13, dlt, dlc, dls = res
    tau = max(np.hypot(t23, t13), 1e-14)
    r0 = sn  - sn_tr  + m.E_n * (dlt - dlc + dls * m.tan_psi)
    r1 = t23 - t23_tr + m.G_s * dls * t23 / tau
    r2 = t13 - t13_tr + m.G_s * dls * t13 / tau
    return np.array([r0, r1, r2])


# Допуски
ATOL_F   = 1e-6     # допуск нарушения поверхности текучести
ATOL_LAM = 1e-10    # допуск знака множителя
ATOL_RES = 1e-6     # допуск остатка уравнений
ATOL_KKT = 1e-5     # допуск комплементарности


# =====================================================================
#  (а) РОТАЦИОННАЯ ИНВАРИАНТНОСТЬ СДВИГА
#  Сдвиговая поверхность изотропна => результат не зависит от
#  направления вектора (t23, t13) при фиксированном модуле.
#  Ловит ошибки в нормализации n23/n13 и в недиагональных членах якобиана.
# =====================================================================
@pytest.mark.parametrize("theta", np.linspace(0.0, 2 * np.pi, 12, endpoint=False))
@pytest.mark.parametrize("sn_tr,tau", [(-10.0, 20.0), (-30.0, 25.0), (5.0, 8.0)])
def test_shear_rotational_invariance(theta, sn_tr, tau):
    base = UbiquitousJointModel3D_Test()._return_mapping_nr(sn_tr, tau, 0.0)
    rot  = UbiquitousJointModel3D_Test()._return_mapping_nr(
        sn_tr, tau * np.cos(theta), tau * np.sin(theta)
    )

    # Инварианты: sig_n, |tau|, множители — не зависят от поворота
    assert rot[0] == pytest.approx(base[0], rel=1e-7, abs=1e-9)            # sig_n
    assert np.hypot(rot[1], rot[2]) == pytest.approx(
        np.hypot(base[1], base[2]), rel=1e-7, abs=1e-9)                    # |tau|
    assert rot[3] == pytest.approx(base[3], rel=1e-7, abs=1e-9)           # d_lam_t
    assert rot[4] == pytest.approx(base[4], rel=1e-7, abs=1e-9)           # d_lam_c
    assert rot[5] == pytest.approx(base[5], rel=1e-7, abs=1e-9)           # d_lam_s


def test_shear_direction_preserved(model):
    """Вектор касательных напряжений сонаправлен пробному (радиальный возврат)."""
    sn_tr, t23_tr, t13_tr = -10.0, 18.0, 24.0   # произвольное направление
    sn, t23, t13, *_ = model._return_mapping_nr(sn_tr, t23_tr, t13_tr)
    # Кросс-произведение направлений ~ 0 (коллинеарны)
    cross = t23_tr * t13 - t13_tr * t23
    assert abs(cross) < 1e-6
    # Знаки сохранены
    assert np.sign(t23) == np.sign(t23_tr)
    assert np.sign(t13) == np.sign(t13_tr)


# =====================================================================
#  (б) KKT-УСЛОВИЯ НА РАНДОМИЗИРОВАННЫХ ВХОДАХ
#  Самый сильный тест: покрывает всё пространство напряжений,
#  включая углы между поверхностями. Не требует знания точного ответа.
# =====================================================================
@pytest.mark.parametrize("seed", range(300))
def test_kkt_conditions_random(seed):
    m = UbiquitousJointModel3D_Test()
    rng = np.random.default_rng(seed)
    sn_tr  = rng.uniform(-80.0, 30.0)
    t23_tr = rng.uniform(-50.0, 50.0)
    t13_tr = rng.uniform(-50.0, 50.0)

    sn, t23, t13, dlt, dlc, dls = m._return_mapping_nr(sn_tr, t23_tr, t13_tr)
    f_t, f_c, f_s = yield_values(m, sn, t23, t13, dlt)

    # 1) Допустимость: все поверхности не нарушены
    assert f_t <= ATOL_F, f"f_t={f_t} нарушена (seed={seed})"
    assert f_c <= ATOL_F, f"f_c={f_c} нарушена (seed={seed})"
    assert f_s <= ATOL_F, f"f_s={f_s} нарушена (seed={seed})"

    # 2) Неотрицательность множителей
    assert dlt >= -ATOL_LAM
    assert dlc >= -ATOL_LAM
    assert dls >= -ATOL_LAM

    # 3) Комплементарность: d_lam_i * f_i = 0
    assert dlt * abs(f_t) <= ATOL_KKT
    assert dlc * abs(f_c) <= ATOL_KKT
    assert dls * abs(f_s) <= ATOL_KKT

    # 4) Взаимоисключение tension/compression
    assert not (dlt > ATOL_LAM and dlc > ATOL_LAM), \
        "Отрыв и сжатие не могут быть активны одновременно"


# =====================================================================
#  (в) ОСТАТОК УРАВНЕНИЙ RETURN-MAPPING
#  Независимо от активного набора уравнения связи должны выполняться.
# =====================================================================
@pytest.mark.parametrize("seed", range(300))
def test_mapping_residual_random(seed):
    m = UbiquitousJointModel3D_Test()
    rng = np.random.default_rng(seed)
    tr = (rng.uniform(-80.0, 30.0),
          rng.uniform(-50.0, 50.0),
          rng.uniform(-50.0, 50.0))
    res = m._return_mapping_nr(*tr)
    r = mapping_residual(m, tr, res)
    assert np.max(np.abs(r)) < ATOL_RES, f"остаток={r} (seed={seed})"


# =====================================================================
#  (г) ГРАНИЧНЫЕ / ВЫРОЖДЕННЫЕ СЛУЧАИ
# =====================================================================
def test_exactly_on_tension_surface(model):
    """sn_tr ровно = f_t -> упругая ветка, нулевые множители."""
    sn, t23, t13, dlt, dlc, dls = model._return_mapping_nr(model.f_t, 0.0, 0.0)
    assert sn == pytest.approx(model.f_t, abs=1e-9)
    assert (dlt, dlc, dls) == (0.0, 0.0, 0.0)


def test_just_above_tension_surface(model):
    """sn_tr чуть выше f_t -> малая, но корректная пластичность, без NaN."""
    eps = 1e-6
    sn, t23, t13, dlt, dlc, dls = model._return_mapping_nr(model.f_t + eps, 0.0, 0.0)
    assert np.isfinite(sn) and np.isfinite(dlt)
    assert dlt >= -ATOL_LAM
    f_t, _, _ = yield_values(model, sn, 0.0, 0.0, dlt)
    assert f_t <= ATOL_F


def test_zero_shear_with_active_shear_surface(model):
    """
    Чистое отрицательное sig_n при tau_tr -> 0:
    проверка ветки tau_si < 1e-10 в якобиане (нет деления на ноль).
    """
    sn, t23, t13, dlt, dlc, dls = model._return_mapping_nr(-50.0, 0.0, 0.0)
    assert np.isfinite(sn)
    assert np.isfinite(t23) and np.isfinite(t13)
    # tau остаётся нулевым
    assert np.hypot(t23, t13) < 1e-9


def test_mohr_coulomb_apex(model):
    """
    Вершина (apex) Мора-Кулона: sn_tr выше точки пересечения f_t и f_s.
    Классическое место отказа return-mapping. Должно сойтись и быть допустимым.
    """
    sn_tr = model.c / model.tan_phi + 5.0   # за апексом
    sn, t23, t13, dlt, dlc, dls = model._return_mapping_nr(sn_tr, 30.0, 0.0)
    f_t, f_c, f_s = yield_values(model, sn, t23, t13, dlt)
    assert f_t <= ATOL_F and f_c <= ATOL_F and f_s <= ATOL_F
    assert dlt >= -ATOL_LAM and dls >= -ATOL_LAM


def test_deep_compression_idempotent(model):
    """Очень глубокое сжатие: возврат строго на cap, sig_n = -f_c."""
    sn, *_ = model._return_mapping_nr(-1e4, 0.0, 0.0)
    assert sn == pytest.approx(-model.f_c, abs=1e-6)


# =====================================================================
#  (д) ИДЕМПОТЕНТНОСТЬ ПРОЕКЦИИ
#  Повторное применение return-mapping к уже допустимому состоянию
#  ничего не меняет (множители = 0).
# =====================================================================
@pytest.mark.parametrize("tr", [
    (10.0, 0.0, 0.0),
    (-40.0, 0.0, 0.0),
    (-10.0, 20.0, 0.0),
    (-50.0, 30.0, 0.0),
    (15.0, 10.0, 0.0),
])
def test_projection_idempotent(tr):
    m1 = UbiquitousJointModel3D_Test()
    sn, t23, t13, *_ = m1._return_mapping_nr(*tr)

    # Повторно проектируем уже спроецированное состояние
    m2 = UbiquitousJointModel3D_Test()
    sn2, t232, t132, dlt2, dlc2, dls2 = m2._return_mapping_nr(sn, t23, t13)

    assert sn2  == pytest.approx(sn,  abs=1e-7)
    assert t232 == pytest.approx(t23, abs=1e-7)
    assert t132 == pytest.approx(t13, abs=1e-7)
    # Допустимое состояние не должно порождать новой пластики
    assert abs(dlt2) < 1e-7 and abs(dlc2) < 1e-7 and abs(dls2) < 1e-7


# =====================================================================
#  (е) МОНОТОННОСТЬ / ФИЗИЧЕСКАЯ СОГЛАСОВАННОСТЬ
# =====================================================================
@pytest.mark.parametrize("sn_tr", [3.0001, 4.0, 6.0, 10.0, 20.0])
def test_tension_softening_monotonic(sn_tr):
    """С ростом пробного растяжения d_lam_t монотонно растёт (разупрочнение)."""
    m = UbiquitousJointModel3D_Test()
    *_, dlt, _, _ = m._return_mapping_nr(sn_tr, 0.0, 0.0)
    assert dlt >= 0.0


def test_tension_softening_increases_with_load():
    vals = []
    for sn_tr in [4.0, 6.0, 8.0, 10.0]:
        m = UbiquitousJointModel3D_Test()
        *_, dlt, _, _ = m._return_mapping_nr(sn_tr, 0.0, 0.0)
        vals.append(dlt)
    assert all(b > a for a, b in zip(vals, vals[1:])), \
        f"d_lam_t не монотонно растёт: {vals}"
