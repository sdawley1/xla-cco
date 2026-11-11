#!/usr/bin/env python3

"""
Property tests of the theory layer: ball geometry, hard-sphere oracles, the
PHS structure factor and pair entropy, and the equations of state.

PY-closure structure factors need one PY solve per chi, so their strategies
draw chi from a fixed set (cached solves); the py pair entropy uses the
cached Chebyshev interpolant and takes any chi.

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
from hypothesis import example, given
from hypothesis import strategies as st
from scipy import integrate

# ----- project -----
from theory_oracle import lens_fraction_quad
from tolerances import (
    ATOL_FD_PRESSURE,
    RTOL_ALPHA_BETA,
    RTOL_B_QUAD,
    RTOL_EOS_FORMULA,
    RTOL_QUAD_ORACLE,
    RTOL_Z_COLD,
)
from xla_cco.theory.eos import (
    DPDN_FLOOR_ULPS,
    compressibility_fixed_k,
    dp_dn_fixed_k,
    manifold_entropy_fixed_chi,
    pressure_fixed_k,
    z_fixed_chi,
)
from xla_cco.theory.geometry import (
    _intersection_fraction_beta,
    intersection_fraction,
    unit_ball_volume,
)
from xla_cco.theory.hardsphere import (
    _eta_cache_key,
    c_py_3d,
    contact_value,
    g2_exact_1d,
    g2_low_density,
)
from xla_cco.theory.phs import (
    b_coefficient,
    c_k,
    pair_entropy_derivs,
    structure_factor,
)

# ============================================================
# STRATEGIES
# ============================================================

DIMS = st.sampled_from((1, 2, 3))
CLOSURES = st.sampled_from(("py", "low_chi"))
LEVELS = st.sampled_from(("ideal", "gaussian", "phs"))
CHI = st.floats(0.0, 0.499, allow_nan=False, allow_subnormal=False)
CHI_POS = st.floats(1e-3, 0.499, allow_nan=False)
CHI_PY_GRID = st.sampled_from((0.02, 0.15, 0.3, 0.42, 0.49))  # cached PY solves
TEMP = st.floats(1e-3, 10.0, allow_nan=False)
SEP = st.floats(0.0, 3.0, allow_nan=False, allow_subnormal=False)
Q = st.floats(0.0, 30.0, allow_nan=False, allow_subnormal=False)
ETA = st.floats(0.0, 0.95, allow_nan=False, allow_subnormal=False)
ETA_PY = st.floats(0.0, 0.6, allow_nan=False, allow_subnormal=False)
ETA_PY_GRID = st.sampled_from((0.0, 0.05, 0.1, 0.2, 0.3))  # cached d = 2 solves


def _density(chi: float, d: int) -> float:
    """n with c_K(K = 1) / (n d) = chi."""
    return c_k(1.0, d) / (d * chi)


# ============================================================
# GEOMETRY AND HARD-SPHERE ORACLES
# ============================================================


@pytest.mark.high
@given(SEP, SEP, st.integers(1, 6))
def test_alpha_bounded_and_nonincreasing(x1: float, x2: float, d: int) -> None:
    lo, hi = sorted((x1, x2))
    a_lo, a_hi = (
        float(intersection_fraction(lo, d)),
        float(intersection_fraction(hi, d)),
    )
    assert 0.0 <= a_hi <= a_lo <= 1.0


@pytest.mark.critical
@given(SEP, DIMS)
@example(1e-5, 1)  # D-092 repro (shrunk Hypothesis input)
def test_alpha_closed_form_matches_incomplete_beta(x: float, d: int) -> None:
    np.testing.assert_allclose(
        intersection_fraction(x, d),
        _intersection_fraction_beta(x, d),
        rtol=RTOL_ALPHA_BETA,
        atol=RTOL_ALPHA_BETA,
    )


@pytest.mark.high
@given(st.integers(3, 12))
def test_unit_ball_volume_recurrence(d: int) -> None:
    """v1(d) = (2 pi / d) v1(d - 2)."""
    expected = 2.0 * math.pi / d * unit_ball_volume(d - 2)
    assert unit_ball_volume(d) == pytest.approx(expected, rel=1e-14)


@pytest.mark.high
@given(Q, st.floats(0.0, 1.0, allow_nan=False, exclude_max=True), DIMS)
def test_g2_low_density_nonnegative_and_zero_in_core(
    r: float, eta: float, d: int
) -> None:
    g = float(g2_low_density(r, eta, d))
    assert g >= 0.0
    assert (g == 0.0) == (r < 1.0)


@pytest.mark.high
@given(st.floats(0.0, 40.0, allow_nan=False), ETA)
def test_g2_exact_1d_bounds(r: float, eta: float) -> None:
    """g >= 0, zero in the core, contact maximum 1/(1 - eta) for r < 2."""
    g = float(g2_exact_1d(r, eta))
    assert g >= 0.0
    assert (g == 0.0) == (r < 1.0) or (eta > 0.9 and r > 1.0)  # underflow far out
    if 1.0 <= r < 2.0:
        assert g <= 1.0 / (1.0 - eta) * (1.0 + 1e-12)


@pytest.mark.high
@given(st.floats(0.0, 0.6, allow_nan=False))
@example(8.0004e-4 + 3e-13)  # D-091 repro
def test_eta_cache_key_is_relative(eta: float) -> None:
    """D-091: the cache key perturbs eta by <= 5e-15 relative."""
    key = _eta_cache_key(eta)
    assert abs(key - eta) <= 5e-15 * eta
    assert _eta_cache_key(key) == key  # idempotent


# ============================================================
# PHS STRUCTURE FACTOR AND PAIR ENTROPY
# ============================================================


@pytest.mark.critical
@given(Q, CHI_PY_GRID, DIMS, CLOSURES)
def test_structure_factor_nonnegative_zero_inside(
    q: float, chi: float, d: int, closure: str
) -> None:
    s = float(structure_factor(q, chi, d, closure))
    assert s >= 0.0
    assert (s == 0.0) == (q <= 1.0)


@pytest.mark.critical
@given(st.floats(1.0, 30.0, allow_nan=False, exclude_min=True), CHI, DIMS)
def test_low_chi_structure_factor_bounds(q: float, chi: float, d: int) -> None:
    """1 <= S <= 1 + chi outside the core (peak at q = 1+, the b(d) choice)."""
    s = float(structure_factor(q, chi, d, "low_chi"))
    assert 1.0 <= s <= 1.0 + chi * (1.0 + 1e-12)


@pytest.mark.critical
@given(st.one_of(st.just(0.0), st.floats(1e-100, 0.499)), DIMS, CLOSURES)
def test_pair_entropy_nonpositive(chi: float, d: int, closure: str) -> None:
    """g(chi) is a relative entropy: <= 0, = 0 only at chi = 0.

    chi >= 1e-100 keeps g ~ -a chi^2 above the float underflow (~1e-308).
    """
    g, h, hp = pair_entropy_derivs(chi, d, closure)
    assert g <= 0.0 and math.isfinite(h) and math.isfinite(hp)
    assert (g == 0.0) == (chi == 0.0)


@pytest.mark.high
@given(st.floats(0.01, 10.0, allow_nan=False), DIMS)
def test_c_k_homogeneous_of_degree_d(scale: float, d: int) -> None:
    assert c_k(scale * 1.3, d) == pytest.approx(scale**d * c_k(1.3, d), rel=1e-13)


# ============================================================
# EQUATIONS OF STATE
# ============================================================


@pytest.mark.critical
@given(CHI_POS, TEMP, DIMS, LEVELS, CLOSURES)
def test_dp_dn_matches_pressure_differences(
    chi: float, T: float, d: int, level: str, closure: str
) -> None:
    n = _density(min(chi, 0.4985), d)  # keep n - step on the branch
    step = 1e-4 * n
    kw = {"d": d, "level": level, "closure": closure}
    fd = (pressure_fixed_k(n + step, T, **kw) - pressure_fixed_k(n - step, T, **kw)) / (
        2.0 * step
    )
    assert float(dp_dn_fixed_k(n, T, **kw)) == pytest.approx(
        float(fd), abs=ATOL_FD_PRESSURE * (n + T)
    )


@pytest.mark.critical
@given(CHI_POS, TEMP, TEMP, st.sampled_from((1, 2)), LEVELS, CLOSURES)
def test_compressibility_bounded_and_increasing_in_t(
    chi: float, t1: float, t2: float, d: int, level: str, closure: str
) -> None:
    """d <= 2: dp/dn >= n + T (1 - d chi) > n, so 0 < n T kappa < T/n; and
    n T kappa = T/(n + a T) increases with T."""
    n = _density(chi, d)
    lo, hi = sorted((t1, t2))
    kw = {"d": d, "level": level, "closure": closure}
    k_lo = float(compressibility_fixed_k(n, lo, **kw))
    k_hi = float(compressibility_fixed_k(n, hi, **kw))
    assert 0.0 < k_lo < lo / n and 0.0 < k_hi < hi / n
    assert k_lo <= k_hi * (1.0 + RTOL_EOS_FORMULA)


@pytest.mark.medium
@given(
    st.floats(1e-300, 1e300, allow_nan=False),
    st.floats(1e-300, 1e300, allow_nan=False),
    DIMS,
    st.sampled_from(("ideal", "gaussian")),
)
@example(1e300, 1e-100, 3, "gaussian")  # D-094 repro: chi underflows to 0
@example(1.0, 1e200, 3, "ideal")  # D-094: K^3 overflows
@example(1.8961503816218355e154, 1.0, 1, "ideal")  # D-094: n^2 overflows
def test_extreme_inputs_finite_or_value_error(
    n: float, K: float, d: int, level: str
) -> None:
    """D-094: any (n, K) gives a finite pressure or a ValueError, nothing else."""
    try:
        p = pressure_fixed_k(n, 1.0, d=d, K=K, level=level)
    except ValueError:
        return
    assert np.all(np.isfinite(p))


@pytest.mark.high
@given(CHI_POS, st.floats(1e-3, 1e3, allow_nan=False), DIMS, LEVELS)
@example(0.45, 1.0, 3, "gaussian")  # D-093: T >= T_c = n/(3 chi - 1) there
def test_compressibility_positive_or_nan(
    chi: float, t_rel: float, d: int, level: str
) -> None:
    """D-093: n T kappa is either NaN (dp/dn not positive) or T/(dp/dn) > 0."""
    n = _density(chi, d)
    T = t_rel * n
    kappa = float(compressibility_fixed_k(n, T, d=d, level=level, closure="low_chi"))
    slope = float(dp_dn_fixed_k(n, T, d=d, level=level, closure="low_chi"))
    if math.isnan(kappa):
        assert slope <= DPDN_FLOOR_ULPS * np.finfo(float).eps * (n + T)
    else:
        assert kappa > 0.0 and kappa == pytest.approx(T / slope, rel=RTOL_EOS_FORMULA)


@pytest.mark.high
@given(CHI, TEMP, DIMS, st.one_of(st.none(), st.integers(2, 10_000)))
def test_z_rpa_between_one_and_harmonic(
    chi: float, T: float, d: int, N: object
) -> None:
    harm = float(z_fixed_chi(chi, T, d=d, N=N))  # type: ignore[arg-type]
    rpa = float(z_fixed_chi(chi, T, d=d, N=N, level="rpa"))  # type: ignore[arg-type]
    cold = float(z_fixed_chi(chi, 1e-12, d=d, N=N, level="rpa"))  # type: ignore[arg-type]
    assert 1.0 <= rpa <= harm <= 1.0 + d * chi * (1.0 + 1e-15)
    assert cold == pytest.approx(harm, rel=RTOL_Z_COLD)


@pytest.mark.high
@given(CHI, st.floats(0.01, 100.0, allow_nan=False), DIMS, CLOSURES)
def test_manifold_entropy_phs_below_gaussian(
    chi: float, n: float, d: int, closure: str
) -> None:
    kw = {"d": d, "n": n, "closure": closure}
    gauss = float(manifold_entropy_fixed_chi(chi, level="gaussian", **kw))
    phs = float(manifold_entropy_fixed_chi(chi, level="phs", **kw))
    assert phs <= gauss + 1e-15 * abs(gauss)


@pytest.mark.high
@given(st.integers(1, 8))
def test_b_coefficient_matches_lens_quadrature(d: int) -> None:
    """b(d) = 1/(2^d alpha_d(1)) with alpha_d(1) from the lens quadrature
    (independent of geometry.py): the overlap of two unit balls 1 apart."""
    expected = 1.0 / (2**d * lens_fraction_quad(1.0, d))
    assert b_coefficient(d) == pytest.approx(expected, rel=RTOL_B_QUAD)


@pytest.mark.critical
@given(st.floats(0.0, 0.95, allow_nan=False, allow_subnormal=False))
def test_c_py_3d_identities(eta: float) -> None:
    """Wertheim-Thiele c: c(1-) = -g(1+) = -(1 + eta/2)/(1 - eta)^2 (PY:
    c = -y inside, y continuous), c increasing on [0, 1], 0 outside, and the
    PY compressibility 1 - rho c^(0) = (1 + 2 eta)^2/(1 - eta)^4 by quadrature."""
    r = np.linspace(0.0, 1.0, 101)
    c = c_py_3d(r, eta)
    contact = (1.0 + eta / 2.0) / (1.0 - eta) ** 2
    assert float(c[-1]) == pytest.approx(-contact, rel=RTOL_QUAD_ORACLE)
    ulps = 4.0 * np.finfo(float).eps * float(np.max(np.abs(c)))  # rounding of c
    assert np.all(np.diff(c) >= -ulps) and float(c_py_3d(1.5, eta)) == 0.0
    c0, _ = integrate.quad(
        lambda x: 4.0 * math.pi * x * x * float(c_py_3d(x, eta)),
        0.0,
        1.0,
        epsabs=1e-13,
        epsrel=1e-13,
    )
    rho = 6.0 * eta / math.pi
    expected = (1.0 + 2.0 * eta) ** 2 / (1.0 - eta) ** 4
    assert 1.0 - rho * c0 == pytest.approx(expected, rel=RTOL_QUAD_ORACLE)


@pytest.mark.critical
@given(ETA_PY, ETA_PY, st.sampled_from((1, 3)))
def test_contact_value_analytic_dims(e1: float, e2: float, d: int) -> None:
    """d = 1: equals the Tonks g2(1+); d = 3: equals -c(1-) (PY); both >= 1
    and increasing in eta."""
    lo, hi = sorted((e1, e2))
    other = float(g2_exact_1d(1.0, lo)) if d == 1 else -float(c_py_3d(1.0, lo))
    assert contact_value(lo, d) == pytest.approx(other, rel=RTOL_QUAD_ORACLE)
    assert 1.0 <= contact_value(lo, d) <= contact_value(hi, d)


@pytest.mark.high
@given(ETA_PY_GRID, ETA_PY_GRID)
def test_contact_value_2d_increasing(e1: float, e2: float) -> None:
    """d = 2 (numerical): contact value >= 1 + 2^2 alpha_2(1) eta (positive
    O(eta^2) terms) and increasing in eta; eta from a fixed grid (cached)."""
    lo, hi = sorted((e1, e2))
    first = 1.0 + 4.0 * lens_fraction_quad(1.0, 2) * lo
    assert first <= contact_value(lo, 2) <= contact_value(hi, 2)
