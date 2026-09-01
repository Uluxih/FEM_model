"""
Проверка алгоритмически согласованной касательной UbiquitousJointModel2D.

Запуск из корня проекта::

    python -m pytest MyTesting/test_algorithmic_tangent_ubiquitous_joint_2d.py -v

Тесты сравнивают матрицу, возвращаемую ``update_state(...)[1]``, с центральной
конечно-разностной производной напряжений. Часть тестов ожидаемо падает, пока
в production-коде не реализована формула из
``Algorithmically_Consistent_Tangent_UbiquitousJointModel2D.md``: текущая
``_compute_secant_matrix`` не учитывает return mapping матрицы и шва, active set
и производную повреждённости. Эти расхождения намеренно не помечены xfail.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest

from FEM.Abstract.Integration_Point_Level import Material as BaseMaterial
from FEM.Integration_Point_Level.CriticalPlane.material import Material as CriticalPlaneMaterial
from FEM.Integration_Point_Level.UbiquitousJointModel2D import UbiquitousJointModel2D


@dataclass(frozen=True)
class Case:
    strain: np.ndarray
    preset_normal: Optional[np.ndarray] = None
    matrix_cohesion: float = 1.0e12
    disable_normal_damage: bool = False
    disable_shear_damage: bool = False
    fracture_energy_t: float = 1.0e6
    fracture_energy_s: float = 1.0e6


class JointedMaterial(BaseMaterial):
    """Минимальный реальный material верхнего уровня для UJ2D."""

    def __init__(self, joint_params, matrix_params):
        super().__init__(E=20_000.0, nu=0.2)
        self.joint_params = joint_params
        self.matrix_params = matrix_params


@pytest.fixture
def material_factory():
    """Создаёт независимый материал с настоящим CriticalPlane Material."""

    def make(case: Case):
        # sqrt(v @ A @ v) = 5 для единичных взаимно ортогональных n и s.
        cp_material = CriticalPlaneMaterial(
            mu=np.tan(np.radians(20.0)),
            A_tensor=np.eye(9) * 5.0**2,
            Rpx=3.0,
            Rpy=3.0,
            Rpz=3.0,
            Rcx=30.0,
            Rcy=30.0,
            Rcz=30.0,
        )
        joint_params = {
            "phi": 20.0,
            "psi": 8.0,
            "phi_r": 5.0,
            "cp_material": cp_material,
            "lock_on_yield": True,
            "l_c": 1.0,
            "Gf_t": case.fracture_energy_t,
            "Gf_c": 1.0e6,
            "Gf_s": case.fracture_energy_s,
            "a_t": 0.0,
            "a_s": 0.0,
            "d_nt_max": 0.0 if case.disable_normal_damage else 0.99,
            "d_s_max": 0.0 if case.disable_shear_damage else 0.99,
            "min_stiff_factor": 0.01,
        }
        if case.preset_normal is not None:
            # Замораживает ориентацию для base и для каждого +/- вычисления.
            joint_params["preset_plane_normal"] = case.preset_normal.copy()
        matrix_params = {
            "phi": 30.0,
            "psi": 10.0,
            "c": case.matrix_cohesion,
        }
        return JointedMaterial(joint_params, matrix_params)

    return make


def _fresh_model(case, material_factory):
    """Новая модель исключает перенос trial/committed state между вызовами."""
    return UbiquitousJointModel2D(material_factory(case))


def _stress(case, strain, material_factory):
    model = _fresh_model(case, material_factory)
    return model.update_state(np.asarray(strain, dtype=float))[0]


def _central_difference(case, material_factory, h=1.0e-8):
    numerical = np.empty((3, 3))
    for column in range(3):
        perturbation = np.zeros(3)
        perturbation[column] = h
        # Намеренно две разные свежие модели: ни trial-, ни committed-state не общие.
        sigma_plus = _stress(case, case.strain + perturbation, material_factory)
        sigma_minus = _stress(case, case.strain - perturbation, material_factory)
        numerical[:, column] = (sigma_plus - sigma_minus) / (2.0 * h)
    return numerical


def _assert_consistent(case, material_factory, *, expected_damage=None):
    model = _fresh_model(case, material_factory)
    _, tangent = model.update_state(case.strain)
    numerical = _central_difference(case, material_factory)

    assert tangent.shape == (3, 3)
    assert numerical.shape == (3, 3)
    assert np.isfinite(tangent).all()
    assert np.isfinite(numerical).all()
    if expected_damage == "normal":
        assert model.D_nt_trial > 0.0
    elif expected_damage == "shear":
        assert model.D_s_trial > 0.0

    np.testing.assert_allclose(tangent, numerical, rtol=2.0e-5, atol=2.0e-3)


def test_elastic_matrix_without_active_joint(material_factory):
    case = Case(strain=np.array([2.0e-5, -1.0e-5, 1.0e-5]))
    _assert_consistent(case, material_factory)


def test_smooth_plastic_matrix_return_without_joint(material_factory):
    case = Case(
        strain=np.array([1.0e-3, -2.0e-4, 4.0e-4]),
        matrix_cohesion=5.0,
    )
    _assert_consistent(case, material_factory)


def test_elastic_joint_with_frozen_plane(material_factory):
    case = Case(
        strain=np.array([5.0e-5, 2.0e-5, 3.0e-5]),
        preset_normal=np.array([1.0, 0.0, 0.0]),
        disable_normal_damage=True,
        disable_shear_damage=True,
    )
    _assert_consistent(case, material_factory)


def test_pure_tensile_joint_return_with_frozen_plane(material_factory):
    case = Case(
        strain=np.array([4.0e-4, 0.0, 0.0]),
        preset_normal=np.array([1.0, 0.0, 0.0]),
        disable_normal_damage=True,
        disable_shear_damage=True,
    )
    _assert_consistent(case, material_factory)


def test_pure_shear_joint_return_with_frozen_plane(material_factory):
    case = Case(
        strain=np.array([-1.0e-4, 0.0, 1.2e-3]),
        preset_normal=np.array([1.0, 0.0, 0.0]),
        disable_normal_damage=True,
        disable_shear_damage=True,
    )
    _assert_consistent(case, material_factory)


def test_active_tensile_damage_derivative_with_frozen_plane(material_factory):
    case = Case(
        strain=np.array([5.0e-4, 0.0, 0.0]),
        preset_normal=np.array([1.0, 0.0, 0.0]),
        fracture_energy_t=1.0e-2,
        disable_shear_damage=True,
    )
    _assert_consistent(case, material_factory, expected_damage="normal")
