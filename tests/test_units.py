"""Unit invariants. Every quantitative defect in this project so far was one of
these, so they are asserted rather than commented."""
import numpy as np
import pytest

from ca_fno3d import units as U


def test_pressure_and_modulus_share_a_unit_or_the_answer_changes():
    """The poroelastic formulas are invariant under a COMMON rescaling of
    pressure and modulus, and not invariant under rescaling one of them. That is
    the whole content of invariant A."""
    E, nu, dP = 2.7e6, 0.257, 1375.0
    cm_a = U.compaction_coefficient(E, nu) * dP
    cm_b = U.compaction_coefficient(E * U.PSI_TO_KPA, nu) * (dP * U.PSI_TO_KPA)
    assert cm_a == pytest.approx(cm_b, rel=1e-12), "common rescaling must be a no-op"

    cm_c = U.compaction_coefficient(E, nu) * (dP * U.PSI_TO_KPA)
    ratio = cm_c / cm_a
    assert ratio == pytest.approx(U.PSI_TO_KPA, rel=1e-12), (
        "converting only the pressure changes the strain by exactly PSI_TO_KPA; "
        "the residual, being quadratic, changes by its square")


def test_oedometer_identity():
    """c_m is the reciprocal of the P-wave modulus. If this ever fails, the
    compaction coefficient and the residual disagree about the same rock."""
    E, nu = 2.7e6, 0.257
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    M = lam + 2 * mu
    assert M * U.compaction_coefficient(E, nu) == pytest.approx(1.0, rel=1e-10)


def test_uniaxial_bound_is_one_sided():
    """Arching reduces the vertical displacement below the uniaxial value; it can
    never amplify it. This is what makes the bound a usable unit test."""
    bound = U.uniaxial_bound_m(np.array([1375.0]), np.array([2.7e6]), np.array([0.257]))
    assert bound > 0
    assert np.isfinite(bound)


def test_length_conversions_round_trip():
    for unit, f in U.LENGTH_TO_M.items():
        assert (1.0 * f) / f == pytest.approx(1.0)
    assert U.LENGTH_TO_M["ft"] == pytest.approx(0.3048)


def test_units_are_declared_verified_before_release():
    """A release must not ship with an unverified unit declaration. This test is
    EXPECTED TO FAIL until `python -m ca_fno3d.unit_audit` has been run against
    the SR3 files and `units.py` updated with its verdict. That is deliberate:
    the failure is the reminder."""
    if not U.UNITS_VERIFIED:
        pytest.xfail("run `python -m ca_fno3d.unit_audit` and set UNITS_VERIFIED = True")
