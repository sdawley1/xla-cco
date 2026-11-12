#!/usr/bin/env python3

"""
Example tests of the fixed-K and fixed-chi equations of state.

Oracles: the closed forms of the ideal and Gaussian levels, central
differences of the pressure in n (every level and closure), the PHS entropy
derivatives, and the RPA / harmonic Z formulas.

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
from tolerances import (
    ATOL_FD_PRESSURE,
    ATOL_PY_TCOEF_EXTERNAL,
    ATOL_T0_PRESSURE,
    RTOL_EOS_FORMULA,
)
from xla_cco.theory.eos import (
    compressibility_fixed_k,
    dp_dn_fixed_k,
    manifold_entropy_fixed_chi,
    pressure_fixed_k,
    z_fixed_chi,
)
from xla_cco.theory.phs import c_k, pair_entropy_derivs

# ============================================================
# HELPERS
# ============================================================

LEVELS = ["ideal", "gaussian", "phs"]
CLOSURES = ["py", "low_chi"]
FD_REL_STEP = 1e-4  # central-difference step in n, relative (see ATOL_FD_PRESSURE)


def _n_of_chi(chi: float, d: int, K: float = 1.0) -> float:
    """Density with chi = c_K / (n d)."""
    return c_k(K, d) / (d * chi)


# ============================================================
# TESTS: FIXED K
# ============================================================


@pytest.mark.critical
@pytest.mark.parametrize("d", [1, 2, 3])
def test_ideal_and_gaussian_closed_forms(d: int) -> None:
    K, T = 1.7, np.array([0.01, 0.3, 2.0])
    n = _n_of_chi(0.31, d, K)
    ck, chi = c_k(K, d), 0.31
    p_ideal = n * n / 2 + ck * T * np.log(2 * math.pi * T) + n * T
    p_gauss = n * n / 2 + n * T + ck * T * (np.log(T / n) + 1.0)
    kw = {"d": d, "K": K}
    np.testing.assert_allclose(
        pressure_fixed_k(n, T, level="ideal", **kw), p_ideal, rtol=RTOL_EOS_FORMULA
    )
    np.testing.assert_allclose(
        pressure_fixed_k(n, T, level="gaussian", **kw), p_gauss, rtol=RTOL_EOS_FORMULA
    )
    np.testing.assert_allclose(
        dp_dn_fixed_k(n, T, level="ideal", **kw), n + T, rtol=RTOL_EOS_FORMULA
    )
    np.testing.assert_allclose(
        dp_dn_fixed_k(n, T, level="gaussian", **kw),
        n + T * (1.0 - d * chi),
        rtol=RTOL_EOS_FORMULA,
    )


@pytest.mark.critical
@pytest.mark.parametrize("d", [1, 2, 3])
def test_ideal_and_gaussian_compressibility(d: int) -> None:
    """n T kappa = T/(n + T) (ideal) and T/(n + T (1 - d chi)) (gaussian)."""
    T, chi = np.array([0.05, 0.5, 1.5]), 0.2
    n = _n_of_chi(chi, d)
    np.testing.assert_allclose(
        compressibility_fixed_k(n, T, d=d, level="ideal"),
        T / (n + T),
        rtol=RTOL_EOS_FORMULA,
    )
    np.testing.assert_allclose(
        compressibility_fixed_k(n, T, d=d, level="gaussian"),
        T / (n + T * (1.0 - d * chi)),
        rtol=RTOL_EOS_FORMULA,
    )


@pytest.mark.critical
@pytest.mark.parametrize("closure", CLOSURES)
@pytest.mark.parametrize("d", [1, 2, 3])
def test_phs_level_adds_entropy_terms(closure: str, d: int) -> None:
    """p_phs - p_gauss = T c_K h and dp/dn_phs - dp/dn_gauss = -T d chi^2 h'."""
    chi, T = 0.37, 0.8
    n = _n_of_chi(chi, d)
    _, h, hp = pair_entropy_derivs(chi, d, closure)
    kw = {"d": d, "closure": closure}
    dp = pressure_fixed_k(n, T, level="phs", **kw) - pressure_fixed_k(
        n, T, level="gaussian", **kw
    )
    assert float(dp) == pytest.approx(T * c_k(1.0, d) * h, rel=RTOL_EOS_FORMULA)
    ds = dp_dn_fixed_k(n, T, level="phs", **kw) - dp_dn_fixed_k(
        n, T, level="gaussian", **kw
    )
    assert float(ds) == pytest.approx(-T * d * chi * chi * hp, rel=RTOL_EOS_FORMULA)


@pytest.mark.critical
@pytest.mark.parametrize("closure", CLOSURES)
@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("d", [1, 2, 3])
def test_dp_dn_matches_finite_differences(level: str, closure: str, d: int) -> None:
    for chi, T in ((0.05, 0.3), (0.25, 1.0), (0.45, 3.0)):
        n = _n_of_chi(chi, d)
        step = FD_REL_STEP * n
        kw = {"d": d, "level": level, "closure": closure}
        fd = (
            pressure_fixed_k(n + step, T, **kw) - pressure_fixed_k(n - step, T, **kw)
        ) / (2.0 * step)
        exact = dp_dn_fixed_k(n, T, **kw)
        assert float(exact) == pytest.approx(float(fd), abs=ATOL_FD_PRESSURE * (n + T))


@pytest.mark.critical
@pytest.mark.parametrize("level", LEVELS)
def test_pressure_zero_temperature_limit(level: str) -> None:
    """p -> n^2/2 as T -> 0 (the ground-state energy term only)."""
    n = np.array([0.5, 1.0, 3.0])
    p = pressure_fixed_k(n, 1e-12, d=2, level=level)
    np.testing.assert_allclose(p, n * n / 2.0, rtol=0.0, atol=ATOL_T0_PRESSURE)


@pytest.mark.high
@pytest.mark.parametrize(
    "chi, expected", [(0.094, 0.0032556), (0.291, 0.1152153), (0.488, 0.7910968)]
)
def test_phs_py_dpdn_t_coefficient_d2_external(chi: float, expected: float) -> None:
    """(dp/dn_phs - dp/dn_gauss)/T = -d chi^2 h', d = 2, py closure, against the
    auditor's independent real-space PY solver (AUDIT-phs-theory.md, PR #20)."""
    n, T = _n_of_chi(chi, 2), 0.7
    kw = {"d": 2, "closure": "py"}
    extra = (
        dp_dn_fixed_k(n, T, level="phs", **kw)
        - dp_dn_fixed_k(n, T, level="gaussian", **kw)
    ) / T
    assert float(extra) == pytest.approx(expected, abs=ATOL_PY_TCOEF_EXTERNAL)


@pytest.mark.high
def test_fixed_k_broadcasting() -> None:
    n = np.linspace(0.2, 2.0, 4)[:, None]
    T = np.linspace(0.1, 1.0, 5)[None, :]
    for fn in (pressure_fixed_k, dp_dn_fixed_k, compressibility_fixed_k):
        out = fn(n, T, d=2)
        assert out.shape == (4, 5)
        assert float(out[1, 2]) == pytest.approx(
            float(fn(n[1, 0], T[0, 2], d=2)), rel=1e-14
        )


@pytest.mark.high
def test_k_scaling_of_fixed_k_eos() -> None:
    """chi depends on n / K^d only; the gaussian dp/dn depends on (n, T, chi)."""
    a = dp_dn_fixed_k(0.3, 0.5, d=2, K=1.0, level="phs")
    b = dp_dn_fixed_k(0.3 * 4.0, 0.5, d=2, K=2.0, level="phs")
    chi = c_k(1.0, 2) / (2 * 0.3)
    _, _, hp = pair_entropy_derivs(chi, 2)
    assert float(b) - 1.2 == pytest.approx(float(a) - 0.3, rel=RTOL_EOS_FORMULA)
    expected = 0.3 + 0.5 * (1 - 2 * chi - 2 * chi * chi * hp)
    assert float(a) == pytest.approx(expected, rel=RTOL_EOS_FORMULA)


# ============================================================
# TESTS: FIXED CHI
# ============================================================


@pytest.mark.critical
@pytest.mark.parametrize("d", [1, 2, 3])
def test_z_fixed_chi_formulas(d: int) -> None:
    chi, T, n = np.array([0.0, 0.1, 0.4]), 0.7, 2.0
    np.testing.assert_allclose(z_fixed_chi(chi, T, d=d), 1.0 + d * chi, rtol=1e-15)
    np.testing.assert_allclose(
        z_fixed_chi(chi, T, d=d, N=10), 1.0 + d * chi * 0.9, rtol=1e-15
    )
    np.testing.assert_allclose(
        z_fixed_chi(chi, T, d=d, n=n, level="rpa"),
        1.0 + d * chi * n / (n + T),
        rtol=1e-15,
    )
    np.testing.assert_allclose(
        z_fixed_chi(chi, T, d=d, n=n, N=5, level="rpa"),
        1.0 + d * chi * 0.8 * n / (n + T),
        rtol=1e-15,
    )


@pytest.mark.critical
@pytest.mark.parametrize("closure", CLOSURES)
@pytest.mark.parametrize("d", [1, 2, 3])
def test_manifold_entropy_fixed_chi_levels(closure: str, d: int) -> None:
    chi, n = np.array([0.0, 0.2, 0.45]), 1.3
    kw = {"d": d, "n": n, "closure": closure}
    np.testing.assert_array_equal(
        manifold_entropy_fixed_chi(chi, level="ideal", **kw), 0.0
    )
    gauss = -d * chi * math.log(2 * math.pi * n)
    np.testing.assert_allclose(
        manifold_entropy_fixed_chi(chi, level="gaussian", **kw), gauss, rtol=1e-15
    )
    g = np.array([pair_entropy_derivs(float(c), d, closure)[0] for c in chi])
    np.testing.assert_allclose(
        manifold_entropy_fixed_chi(chi, level="phs", **kw),
        gauss + d * chi * g,
        rtol=1e-14,
    )


# ============================================================
# TESTS: CONTRACTS
# ============================================================


@pytest.mark.medium
@pytest.mark.parametrize(
    "n, T, kwargs",
    [
        (0.01, 1.0, {"d": 2}),  # chi = c_K/(2 n) = 1.99 >= 1/2
        (1.0, 0.0, {"d": 2}),
        (1.0, -1.0, {"d": 2}),
        (0.0, 1.0, {"d": 2}),
        (-1.0, 1.0, {"d": 2}),
        (float("nan"), 1.0, {"d": 2}),
        (1.0, 1.0, {"d": 4}),
        (1.0, 1.0, {"d": 2, "level": "rpa"}),
        (1.0, 1.0, {"d": 2, "closure": "hnc"}),
        (1.0, 1.0, {"d": 2, "K": 0.0}),
    ],
)
def test_fixed_k_rejects_bad_input(n: float, T: float, kwargs: dict) -> None:  # type: ignore[type-arg]
    for fn in (pressure_fixed_k, dp_dn_fixed_k, compressibility_fixed_k):
        with pytest.raises(ValueError):
            fn(n, T, **kwargs)


@pytest.mark.medium
@pytest.mark.parametrize(
    "chi, T, kwargs",
    [
        (0.5, 1.0, {"d": 2}),
        (-0.1, 1.0, {"d": 2}),
        (0.2, 0.0, {"d": 2}),
        (0.2, 1.0, {"d": 0}),
        (0.2, 1.0, {"d": 2, "N": 1}),
        (0.2, 1.0, {"d": 2, "N": 2.5}),
        (0.2, 1.0, {"d": 2, "n": -1.0}),
        (0.2, 1.0, {"d": 2, "level": "phs"}),
    ],
)
def test_z_fixed_chi_rejects_bad_input(chi: float, T: float, kwargs: dict) -> None:  # type: ignore[type-arg]
    with pytest.raises(ValueError):
        z_fixed_chi(chi, T, **kwargs)


@pytest.mark.medium
def test_manifold_entropy_rejects_bad_input() -> None:
    for kwargs in (
        {"d": 2, "level": "rpa"},
        {"d": 2, "closure": "x"},
        {"d": 5},
        {"d": 2, "n": 0.0},
    ):
        with pytest.raises(ValueError):
            manifold_entropy_fixed_chi(0.2, **kwargs)
    with pytest.raises(ValueError):
        manifold_entropy_fixed_chi(0.5, d=2)


# ============================================================
# REGRESSION TESTS
# ============================================================


@pytest.mark.high
def test_regression_D093_compressibility_nan_where_unstable() -> None:
    """D-093: gaussian, d = 3, chi = 0.45: dp/dn = n + T (1 - 3 chi) <= 0 for
    T >= T_c = n / (3 chi - 1); there n T kappa is NaN, below T_c it is finite."""
    chi = 0.45
    n = _n_of_chi(chi, 3)
    t_c = n / (3.0 * chi - 1.0)
    T = np.array([0.5 * t_c, 0.999 * t_c, t_c, 2.0 * t_c])
    kappa = compressibility_fixed_k(n, T, d=3, level="gaussian")
    assert np.all(np.isnan(kappa[2:])), kappa
    expected = T[:2] / (n + T[:2] * (1.0 - 3.0 * chi))
    np.testing.assert_allclose(kappa[:2], expected, rtol=RTOL_EOS_FORMULA)
    assert np.all(kappa[:2] > 0.0)


@pytest.mark.medium
def test_regression_D094_out_of_range_inputs_raise_value_error() -> None:
    """D-094: chi or c_K under/overflow must raise ValueError, not AssertionError."""
    with pytest.raises(ValueError):
        pressure_fixed_k(1e300, 1.0, d=3, K=1e-100)  # chi underflows to 0
    with pytest.raises(ValueError):
        dp_dn_fixed_k(1e300, 1.0, d=3, K=1e-100)
    with pytest.raises(ValueError):
        c_k(1e-200, 3)  # K^3 underflows
    with pytest.raises(ValueError):
        c_k(1e200, 3)  # K^3 overflows
    with pytest.raises(ValueError):
        pressure_fixed_k(1.8961503816218355e154, 1.0, d=1, level="ideal")  # n^2
