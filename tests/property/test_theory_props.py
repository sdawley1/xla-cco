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

# ----- project -----
from tolerances import ATOL_FD_PRESSURE, RTOL_ALPHA_BETA, RTOL_EOS_FORMULA
from xla_cco.theory.eos import (
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
from xla_cco.theory.hardsphere import _eta_cache_key, g2_exact_1d, g2_low_density
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


@pytest.mark.high
@given(CHI, TEMP, DIMS, st.one_of(st.none(), st.integers(2, 10_000)))
def test_z_rpa_between_one_and_harmonic(
    chi: float, T: float, d: int, N: object
) -> None:
    harm = float(z_fixed_chi(chi, T, d=d, N=N))  # type: ignore[arg-type]
    rpa = float(z_fixed_chi(chi, T, d=d, N=N, level="rpa"))  # type: ignore[arg-type]
    cold = float(z_fixed_chi(chi, 1e-12, d=d, N=N, level="rpa"))  # type: ignore[arg-type]
    assert 1.0 <= rpa <= harm <= 1.0 + d * chi * (1.0 + 1e-15)
    assert cold == pytest.approx(harm, rel=1e-11)


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
@given(CHI_POS, DIMS)
def test_b_coefficient_sets_peak(chi: float, d: int) -> None:
    """S_low(1+) = 1 + 2^d b chi alpha_d(1) = 1 + chi for every d."""
    peak = 1.0 + 2**d * b_coefficient(d) * chi * float(intersection_fraction(1.0, d))
    assert peak == pytest.approx(1.0 + chi, rel=1e-14)
