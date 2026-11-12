#!/usr/bin/env python3

"""
Example tests of unit-ball volumes and the scaled intersection volume alpha_d.

Oracles: closed forms, the regularized incomplete beta function and a direct
quadrature of the lens volume (two spherical caps).

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import math

# ----- numerics -----
import numpy as np
import pytest

# ----- project -----
from theory_oracle import lens_fraction_quad
from tolerances import ATOL_ALPHA_QUAD, RTOL_ALPHA_BETA
from xla_cco.theory.geometry import (
    _intersection_fraction_beta,
    intersection_fraction,
    unit_ball_volume,
)

# ============================================================
# TESTS
# ============================================================


@pytest.mark.critical
def test_unit_ball_volume_known_values() -> None:
    """v1(1) = 2, v1(2) = pi, v1(3) = 4 pi / 3, v1(4) = pi^2 / 2."""
    expected = {1: 2.0, 2: math.pi, 3: 4.0 * math.pi / 3.0, 4: math.pi**2 / 2.0}
    for d, v in expected.items():
        assert unit_ball_volume(d) == pytest.approx(v, rel=1e-15)


@pytest.mark.critical
@pytest.mark.parametrize("d", [1, 2, 3])
def test_intersection_fraction_closed_forms_match_incomplete_beta(d: int) -> None:
    x = np.linspace(0.0, 2.5, 251)
    np.testing.assert_allclose(
        intersection_fraction(x, d),
        _intersection_fraction_beta(x, d),
        rtol=RTOL_ALPHA_BETA,
        atol=RTOL_ALPHA_BETA,
    )


@pytest.mark.critical
@pytest.mark.parametrize("d", [1, 2, 3, 4, 5])
def test_intersection_fraction_matches_lens_quadrature(d: int) -> None:
    x = np.linspace(0.0, 2.0, 41)
    expected = np.array([lens_fraction_quad(float(xi), d) for xi in x])
    np.testing.assert_allclose(
        intersection_fraction(x, d), expected, rtol=0.0, atol=ATOL_ALPHA_QUAD
    )


@pytest.mark.high
def test_intersection_fraction_special_points() -> None:
    """alpha(0) = 1, alpha(x >= 2) = 0, alpha_1(1) = 1/2, alpha_3(1) = 5/16."""
    for d in (1, 2, 3, 4):
        out = intersection_fraction(np.array([0.0, 2.0, 2.5, 10.0]), d)
        np.testing.assert_array_equal(out, [1.0, 0.0, 0.0, 0.0])
    assert float(intersection_fraction(1.0, 1)) == pytest.approx(0.5, rel=1e-15)
    assert float(intersection_fraction(1.0, 3)) == pytest.approx(5 / 16, rel=1e-15)
    two_d = 2.0 / math.pi * (math.pi / 3.0 - 0.5 * math.sqrt(0.75))
    assert float(intersection_fraction(1.0, 2)) == pytest.approx(two_d, rel=1e-15)


@pytest.mark.medium
def test_intersection_fraction_preserves_shape() -> None:
    x = np.full((3, 4), 0.5)
    assert intersection_fraction(x, 2).shape == (3, 4)
    assert intersection_fraction(0.5, 2).shape == ()


@pytest.mark.medium
@pytest.mark.parametrize(
    "x, d",
    [(-0.1, 2), (float("nan"), 2), (float("inf"), 2), (0.5, 0), (0.5, 1.5)],
)
def test_intersection_fraction_rejects_bad_input(x: float, d: int) -> None:
    with pytest.raises(ValueError):
        intersection_fraction(x, d)


@pytest.mark.medium
@pytest.mark.parametrize("d", [0, -1, 2.0, True])
def test_unit_ball_volume_rejects_bad_dimension(d: int) -> None:
    with pytest.raises(ValueError):
        unit_ball_volume(d)


# ============================================================
# REGRESSION TESTS
# ============================================================


@pytest.mark.high
@pytest.mark.parametrize("x", [1e-5, 1e-7, 1e-9, 3e-3, 1.999999])
def test_regression_D092_beta_form_accurate_at_small_separation(x: float) -> None:
    """D-092: I_{1 - x^2/4} must not lose ~eps/x; compare with exact d = 1, 3."""
    exact = {1: 1.0 - x / 2.0, 3: 1.0 - 0.75 * x + x**3 / 16.0}
    for d, value in exact.items():
        got = float(_intersection_fraction_beta(x, d))
        assert got == pytest.approx(value, rel=RTOL_ALPHA_BETA, abs=RTOL_ALPHA_BETA)
