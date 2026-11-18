#!/usr/bin/env python3

"""
Example tests of the PHS ansatz: b(d), c_K, S(q; chi) for both closures, the
pair entropy g(chi) and its derivatives h = (chi g)', h'.

Oracles: closed forms, a lattice count of wavevectors, the exact hard-rod g2
(PHS with b(1) = 1 in d = 1), the analytic small-chi limit g = -a chi^2,
finite differences, and direct PY solves for the Chebyshev interpolant.

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
from scipy import integrate, special

# ----- project -----
from tolerances import (
    ATOL_FD_CHI,
    ATOL_PY_TCOEF_EXTERNAL,
    ATOL_QUOTED_4DP,
    ATOL_QUOTED_A2,
    ATOL_QUOTED_B2,
    ATOL_S_PY_1D,
    C_G_CUBIC,
    C_S_SECOND_ORDER,
    RTOL_LATTICE_COUNT,
    RTOL_PROXY_DIRECT,
    RTOL_PY_G_EXTERNAL,
    RTOL_PY_G_LIMIT,
    RTOL_QUAD_ORACLE,
    SECOND_ORDER_RATIO_TOL,
)
from xla_cco.theory.hardsphere import contact_value, g2_exact_1d
from xla_cco.theory.phs import (
    _pair_entropy_direct,
    b_coefficient,
    c_k,
    pair_entropy_derivs,
    pair_entropy_ratio,
    structure_factor,
)

# ============================================================
# HELPERS
# ============================================================

FD_STEP = 1e-4  # central-difference step in chi (see ATOL_FD_CHI)
# The task's estimates of -d chi^2 h'(chi), d = 2, low_chi (4 decimals)
PHS_T_COEFF_D2 = {
    0.094: 0.0029,
    0.197: 0.0242,
    0.291: 0.0726,
    0.386: 0.1584,
    0.457: 0.2507,
    0.488: 0.2993,
}
# py closure, d = 2: g(chi) and -d chi^2 h'(chi) from the auditor's independent
# real-space PY solver (no Hankel transforms) with 5-point differences
# (scratchpad/phs_audit/oracle_g.py, AUDIT-phs-theory.md, PR #20)
PY_G_AUDIT_D2 = {0.094: -0.0028155981, 0.291: -0.0288228266, 0.488: -0.0908345288}
PY_TCOEF_AUDIT_D2 = {0.094: 0.0032556, 0.291: 0.1152153, 0.488: 0.7910968}


def _alpha_beta(x: float, d: int) -> float:
    """alpha_d(x) from the incomplete beta function (independent of geometry.py)."""
    return (
        float(special.betainc((d + 1) / 2.0, 0.5, 1.0 - x * x / 4.0)) if x < 2 else 0.0
    )


def _a_coefficient(d: int) -> float:
    """a_d = (d/2) int_1^2 q^(d-1) (alpha_d(q)/alpha_d(1))^2 dq, so g ~ -a chi^2."""
    a1 = _alpha_beta(1.0, d)
    val, _ = integrate.quad(
        lambda q: q ** (d - 1) * (_alpha_beta(q, d) / a1) ** 2,
        1.0,
        2.0,
        epsabs=1e-14,
        epsrel=1e-13,
    )
    return d / 2.0 * val


def _count_half_space(d: int, k_cut: float, length: float) -> int:
    """Number of k = 2 pi n / L (n integer, n != 0) with |k| <= k_cut, halved."""
    n_max = int(k_cut * length / (2.0 * math.pi)) + 1
    axis = np.arange(-n_max, n_max + 1, dtype=float)
    mesh = np.meshgrid(*([axis] * d), indexing="ij")
    k2 = sum(m * m for m in mesh) * (2.0 * math.pi / length) ** 2
    inside = int(np.count_nonzero(k2 <= k_cut * k_cut)) - 1  # drop k = 0
    assert inside % 2 == 0, "the ball is symmetric under k -> -k"
    return inside // 2


# ============================================================
# TESTS: COEFFICIENTS
# ============================================================


@pytest.mark.critical
def test_b_coefficient_values() -> None:
    """b(1) = 1, b(3) = 2/5, b(2) = 1/(4 (2/3 - sqrt(3)/(2 pi))) = 0.63938..."""
    assert b_coefficient(1) == pytest.approx(1.0, rel=1e-15)
    assert b_coefficient(3) == pytest.approx(0.4, rel=1e-15)
    b2 = 1.0 / (4.0 * (2.0 / 3.0 - math.sqrt(3.0) / (2.0 * math.pi)))
    assert b_coefficient(2) == pytest.approx(b2, rel=1e-14)
    assert b_coefficient(2) == pytest.approx(0.63938, abs=ATOL_QUOTED_B2)


@pytest.mark.critical
def test_c_k_closed_forms() -> None:
    """c_K = K/(2 pi), K^2/(8 pi), K^3/(12 pi^2) in d = 1, 2, 3."""
    assert c_k(1.0, 1) == pytest.approx(1.0 / (2.0 * math.pi), rel=1e-15)
    assert c_k(2.0, 2) == pytest.approx(4.0 / (8.0 * math.pi), rel=1e-15)
    assert c_k(1.5, 3) == pytest.approx(1.5**3 / (12.0 * math.pi**2), rel=1e-15)


@pytest.mark.high
@pytest.mark.parametrize("d, length", [(1, 2.0e5), (2, 2000.0), (3, 200.0)])
def test_c_k_counts_half_space_wavevectors(d: int, length: float) -> None:
    """M / V of a periodic box -> c_K (lattice-point count in a half ball)."""
    count = _count_half_space(d, 1.0, length)
    assert count / length**d == pytest.approx(c_k(1.0, d), rel=RTOL_LATTICE_COUNT)


# ============================================================
# TESTS: STRUCTURE FACTOR
# ============================================================


@pytest.mark.critical
@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("chi", [0.05, 0.3, 0.49])
def test_low_chi_structure_factor(d: int, chi: float) -> None:
    """S = 0 for q <= 1, S(1+) = 1 + chi (b(d) choice), S = 1 for q >= 2."""
    s = structure_factor(
        np.array([0.0, 0.5, 1.0, 1.0 + 1e-13, 2.0, 2.7]), chi, d, "low_chi"
    )
    np.testing.assert_array_equal(s[[0, 1, 2]], 0.0)
    assert s[3] == pytest.approx(1.0 + chi, rel=1e-12)
    np.testing.assert_array_equal(s[[4, 5]], 1.0)


@pytest.mark.critical
def test_low_chi_structure_factor_interior_3d() -> None:
    """d = 3, q = 3/2: alpha_3 = 11/128, alpha_3(1) = 5/16, so S = 1 + 0.275 chi."""
    s = structure_factor(1.5, 0.4, 3, "low_chi")
    assert float(s) == pytest.approx(1.0 + 0.275 * 0.4, rel=1e-14)


@pytest.mark.critical
@pytest.mark.parametrize("chi", [0.1, 0.3, 0.45])
def test_py_structure_factor_1d_is_exact_hard_rods(chi: float) -> None:
    """d = 1: eta = chi and PY is exact, so S(q) = g2_Tonks(q; chi) off the nodes."""
    q = np.array([1.0004, 1.3337, 1.5, 2.5, 3.7, 7.77, 25.0])
    np.testing.assert_allclose(
        structure_factor(q, chi, 1, "py"),
        g2_exact_1d(q, chi),
        rtol=0.0,
        atol=ATOL_S_PY_1D,
    )


@pytest.mark.critical
@pytest.mark.parametrize("d", [1, 2, 3])
def test_py_structure_factor_peak_is_contact_value(d: int) -> None:
    chi = 0.3
    peak = float(structure_factor(1.0 + 1e-12, chi, d, "py"))
    assert peak == pytest.approx(
        contact_value(b_coefficient(d) * chi, d), abs=ATOL_S_PY_1D
    )
    assert float(structure_factor(1.0, chi, d, "py")) == 0.0


@pytest.mark.critical
@pytest.mark.parametrize("d", [1, 2, 3])
def test_closures_agree_to_second_order(d: int) -> None:
    """max_q |S_py - S_low| <= C chi^2, and the ratio at chi, 2 chi is ~4."""
    q = np.linspace(0.0, 6.0, 6001)
    gaps = []
    for chi in (0.01, 0.02):
        diff = structure_factor(q, chi, d, "py") - structure_factor(
            q, chi, d, "low_chi"
        )
        gaps.append(float(np.max(np.abs(diff))))
        assert gaps[-1] <= C_S_SECOND_ORDER * chi**2, gaps[-1] / chi**2
    assert abs(gaps[1] / gaps[0] - 4.0) <= SECOND_ORDER_RATIO_TOL, gaps


# ============================================================
# TESTS: PAIR ENTROPY
# ============================================================


@pytest.mark.critical
@pytest.mark.parametrize("d", [1, 2, 3])
def test_low_chi_entropy_small_chi_limit(d: int) -> None:
    """g/chi^2 = -a + O(chi), |O(chi)| <= C chi (C = (d/3) int q^(d-1) u^3 < 0.25)."""
    a = _a_coefficient(d)
    for chi in (1e-4, 1e-3):
        ratio = pair_entropy_ratio(chi, d, "low_chi") / chi**2
        assert abs(ratio + a) <= C_G_CUBIC * chi, (ratio + a) / chi


@pytest.mark.high
def test_a_coefficient_two_dimensions() -> None:
    """a_2 = int_1^2 q (alpha_2(q)/alpha_2(1))^2 dq = 0.311950 (task: ~0.3118)."""
    assert _a_coefficient(2) == pytest.approx(0.3119501, abs=ATOL_QUOTED_A2)
    assert _a_coefficient(1) == pytest.approx(1.0 / 6.0, rel=RTOL_QUAD_ORACLE)


@pytest.mark.critical
@pytest.mark.parametrize("d", [1, 2, 3])
def test_py_entropy_small_chi_limit(d: int) -> None:
    """PY is exact to O(eta), so the py closure also has g/chi^2 -> -a."""
    a = _a_coefficient(d)
    ratio = pair_entropy_ratio(1e-4, d, "py") / 1e-8
    assert abs(ratio + a) <= RTOL_PY_G_LIMIT * a, (ratio + a) / a


@pytest.mark.critical
@pytest.mark.parametrize("d", [1, 2, 3])
def test_py_interpolant_matches_direct_solves(d: int) -> None:
    for chi in (0.0013, 0.137, 0.271, 0.333, 0.46, 0.4999):
        direct = _pair_entropy_direct(chi, d)
        assert pair_entropy_ratio(chi, d, "py") == pytest.approx(
            direct, rel=RTOL_PROXY_DIRECT
        )


@pytest.mark.critical
@pytest.mark.parametrize("closure", ["py", "low_chi"])
@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("chi", [0.05, 0.3, 0.45])
def test_entropy_derivatives_match_finite_differences(
    closure: str, d: int, chi: float
) -> None:
    """h = g + chi g' and h' = dh/dchi against central differences of g and h."""
    g0, h, hp = pair_entropy_derivs(chi, d, closure)
    gp_fd = (
        pair_entropy_ratio(chi + FD_STEP, d, closure)
        - pair_entropy_ratio(chi - FD_STEP, d, closure)
    ) / (2.0 * FD_STEP)
    h_hi = pair_entropy_derivs(chi + FD_STEP, d, closure)[1]
    h_lo = pair_entropy_derivs(chi - FD_STEP, d, closure)[1]
    assert g0 == pytest.approx(pair_entropy_ratio(chi, d, closure), rel=1e-15)
    assert h == pytest.approx(g0 + chi * gp_fd, abs=ATOL_FD_CHI)
    assert hp == pytest.approx((h_hi - h_lo) / (2.0 * FD_STEP), abs=ATOL_FD_CHI)


@pytest.mark.high
@pytest.mark.parametrize("chi, expected", sorted(PHS_T_COEFF_D2.items()))
def test_phs_dpdn_t_coefficient_d2_low_chi(chi: float, expected: float) -> None:
    """-d chi^2 h'(chi), d = 2, low_chi: the task's 4-decimal estimates."""
    _, _, hp = pair_entropy_derivs(chi, 2, "low_chi")
    assert -2.0 * chi * chi * hp == pytest.approx(expected, abs=ATOL_QUOTED_4DP)


@pytest.mark.critical
@pytest.mark.parametrize("chi", sorted(PY_G_AUDIT_D2))
def test_py_pair_entropy_external_pins(chi: float) -> None:
    """py closure, d = 2: g and -d chi^2 h' vs an independent PY solver."""
    g, _, hp = pair_entropy_derivs(chi, 2, "py")
    assert g == pytest.approx(PY_G_AUDIT_D2[chi], rel=RTOL_PY_G_EXTERNAL)
    assert -2.0 * chi * chi * hp == pytest.approx(
        PY_TCOEF_AUDIT_D2[chi], abs=ATOL_PY_TCOEF_EXTERNAL
    )


@pytest.mark.high
@pytest.mark.parametrize("closure", ["py", "low_chi"])
def test_entropy_vanishes_at_zero_chi(closure: str) -> None:
    for d in (1, 2, 3):
        assert pair_entropy_derivs(0.0, d, closure) == (0.0, 0.0, 0.0)


# ============================================================
# TESTS: CONTRACTS
# ============================================================


@pytest.mark.medium
@pytest.mark.parametrize("chi", [-0.01, 0.5, 0.7, float("nan"), float("inf")])
def test_rejects_chi_outside_disordered_branch(chi: float) -> None:
    with pytest.raises(ValueError):
        structure_factor(1.5, chi, 2)
    with pytest.raises(ValueError):
        pair_entropy_ratio(chi, 2)
    with pytest.raises(ValueError):
        pair_entropy_derivs(chi, 2, "low_chi")


@pytest.mark.medium
def test_rejects_bad_dimension_closure_and_k() -> None:
    with pytest.raises(ValueError):
        structure_factor(1.5, 0.2, 4)
    with pytest.raises(ValueError):
        pair_entropy_ratio(0.2, 2, closure="hnc")
    with pytest.raises(ValueError):
        structure_factor(-1.0, 0.2, 2)
    with pytest.raises(ValueError):
        b_coefficient(0)
    for bad in (0.0, -1.0, float("inf")):
        with pytest.raises(ValueError):
            c_k(bad, 2)


@pytest.mark.medium
def test_structure_factor_preserves_shape() -> None:
    q = np.full((2, 3), 1.5)
    for closure in ("py", "low_chi"):
        assert structure_factor(q, 0.2, 2, closure).shape == (2, 3)
        assert structure_factor(1.5, 0.2, 2, closure).shape == ()
