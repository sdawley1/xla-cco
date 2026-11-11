#!/usr/bin/env python3

"""
Example tests of the d-dimensional radial Fourier transform and the
Percus-Yevick hard-sphere solver.

Oracles: analytic transform pairs; the exact hard-rod (Tonks) g2 and c in
d = 1, where PY is exact; the Wertheim-Thiele PY solution in d = 3; the exact
O(eta) density expansion in every d; grid refinement in d = 2.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import dataclasses
import math

# ----- numerics -----
import numpy as np
import pytest
from scipy import integrate, special

# ----- project -----
from tolerances import (
    ATOL_HANKEL_EM2,
    ATOL_HANKEL_EXACT,
    ATOL_PY_CONTACT_RICHARDSON,
    ATOL_PY_LOW_DENSITY,
    ATOL_PY_R_INDEPENDENCE,
    ATOL_PY_TAIL,
    ATOL_TONKS_SUM_RULE,
    C_PY_COINCIDENT_NODE,
    C_PY_GRID_1D,
    C_PY_GRID_3D,
    C_STEP_TRANSFORM,
    PY_CONVERGENCE_RATIO,
    RTOL_BESSEL_ZEROS,
    RTOL_QUAD_ORACLE,
)
from xla_cco.theory.geometry import intersection_fraction
from xla_cco.theory.hardsphere import (
    PYSolution,
    _bessel_zeros,
    _contact_richardson,
    _forward,
    _inverse,
    _omega,
    _omega_generic,
    _radial_grid,
    c_py_3d,
    contact_value,
    g2_exact_1d,
    g2_low_density,
    solve_py,
)

# ============================================================
# HELPERS
# ============================================================


def _node(sol: PYSolution, x: float) -> int:
    """Index of the grid node at r = x (x a multiple of the spacing)."""
    dr = float(sol.r[1] - sol.r[0])
    i = int(round(x / dr))
    assert abs(sol.r[i] - x) < 1e-9, f"r = {x} is not a node"
    return i


def _step_transform(k: np.ndarray, d: int) -> np.ndarray:
    """Analytic d-dimensional Fourier transform of Theta(1 - r)."""
    kk = np.maximum(k, 1e-300)
    if d == 1:
        out = 2.0 * np.sin(kk) / kk
    elif d == 2:
        out = 2.0 * math.pi * special.j1(kk) / kk
    else:
        out = 4.0 * math.pi * (np.sin(kk) - kk * np.cos(kk)) / kk**3
    vol = {1: 2.0, 2: math.pi, 3: 4.0 * math.pi / 3.0}[d]
    return np.where(k < 1e-8, vol, out)


# ============================================================
# TESTS: TRANSFORM
# ============================================================


@pytest.mark.high
def test_bessel_zeros_match_closed_forms_and_scipy() -> None:
    m = np.arange(1, 201)
    np.testing.assert_allclose(
        _bessel_zeros(-0.5, 200), (m - 0.5) * np.pi, rtol=RTOL_BESSEL_ZEROS
    )
    np.testing.assert_allclose(
        _bessel_zeros(0.5, 200), m * np.pi, rtol=RTOL_BESSEL_ZEROS
    )
    np.testing.assert_allclose(
        _bessel_zeros(0.0, 200), special.jn_zeros(0, 200), rtol=RTOL_BESSEL_ZEROS
    )


@pytest.mark.high
@pytest.mark.parametrize("d", [1, 2, 3])
def test_omega_closed_forms_match_generic_bessel(d: int) -> None:
    x = np.linspace(0.0, 3000.0, 300_001)
    np.testing.assert_allclose(
        _omega(d, x), _omega_generic(d, x), rtol=0.0, atol=ATOL_HANKEL_EXACT
    )


@pytest.mark.critical
@pytest.mark.parametrize("d", [1, 2, 3])
def test_radial_transform_gaussian_pair(d: int) -> None:
    """exp(-r^2) <-> pi^(d/2) exp(-k^2/4) in d dimensions, both directions."""
    grid = _radial_grid(d, 20.0, 4096)
    f = np.exp(-(grid.r**2))
    fk = math.pi ** (d / 2) * np.exp(-(grid.k**2) / 4.0)
    atol_fwd = ATOL_HANKEL_EM2 if d == 2 else ATOL_HANKEL_EXACT
    np.testing.assert_allclose(_forward(grid, f), fk, rtol=0.0, atol=atol_fwd)
    np.testing.assert_allclose(_inverse(grid, fk), f, rtol=0.0, atol=ATOL_HANKEL_EXACT)


@pytest.mark.high
@pytest.mark.parametrize("d", [1, 2, 3])
def test_radial_transform_step_second_order(d: int) -> None:
    """Theta(1 - r) with the half value at the node r = 1: O((dr k)^2) error."""
    grid = _radial_grid(d, 20.0, 4096)
    f = np.where(np.arange(grid.r.size) < grid.m, 1.0, 0.0)
    f[grid.m] = 0.5
    keep = grid.k <= 20.0
    err = np.abs(_forward(grid, f) - _step_transform(grid.k, d))[keep]
    bound = C_STEP_TRANSFORM * grid.dr**2 * (1.0 + grid.k[keep] ** 2)
    assert np.all(err <= bound), f"max err/bound {np.max(err / bound):.3g}"


# ============================================================
# TESTS: PY SOLVER VS EXACT RESULTS
# ============================================================


@pytest.mark.critical
@pytest.mark.parametrize("eta", [0.1, 0.3, 0.5])
def test_py_1d_matches_exact_hard_rods(eta: float) -> None:
    """d = 1 PY is exact: g2 (Tonks) and c(x) = -(1 - eta x)/(1 - eta)^2."""
    sol = solve_py(eta, 1)
    assert sol.converged
    i0, i1, i2 = 0, _node(sol, 1.0), _node(sol, 2.0)
    dr = float(sol.r[1])
    c_exact = -(1.0 - eta * sol.r[: i1 + 1]) / (1.0 - eta) ** 2
    c_err = np.abs(sol.c[: i1 + 1] - c_exact)
    g_exact = g2_exact_1d(sol.r, eta)
    g_err = np.abs(sol.g - g_exact)
    # smooth nodes: second order; r = 0 and r = 2: first order (see solve_py)
    smooth_c = np.delete(c_err, [i0])
    smooth_g = np.delete(g_err, [i2])
    assert smooth_c.max() <= C_PY_GRID_1D * dr**2, smooth_c.max()
    assert smooth_g.max() <= C_PY_GRID_1D * dr**2, smooth_g.max()
    assert max(c_err[i0], g_err[i2]) <= C_PY_COINCIDENT_NODE * dr


@pytest.mark.critical
@pytest.mark.parametrize("eta", [0.1, 0.3, 0.45])
def test_py_3d_matches_wertheim_thiele(eta: float) -> None:
    sol = solve_py(eta, 3)
    assert sol.converged
    i1 = _node(sol, 1.0)
    dr = float(sol.r[1])
    a = (1.0 + 2.0 * eta) ** 2 / (1.0 - eta) ** 4  # -c(0), the scale of c
    bound = C_PY_GRID_3D * a * a * dr**2
    inner = (sol.r >= 0.2) & (np.arange(sol.r.size) <= i1)
    err = np.abs(sol.c - c_py_3d(sol.r, eta))[inner]
    assert err.max() <= bound, err.max() / bound
    wt_contact = (1.0 + eta / 2.0) / (1.0 - eta) ** 2
    assert abs(sol.g[i1] - wt_contact) <= bound


@pytest.mark.critical
@pytest.mark.parametrize("d", [1, 2, 3])
def test_py_first_order_in_eta(d: int) -> None:
    """g = Theta(r - 1)[1 + 2^d eta alpha_d(r)] + O(eta^2); contact slope 2^d a(1)."""
    eta = 1e-4
    sol = solve_py(eta, d)
    out = sol.r >= 1.0 - 1e-12
    if d == 1:  # drop the first-order node r = 2 (see test_py_1d_matches_exact)
        out[_node(sol, 2.0)] = False
    slope = (sol.g[out] - 1.0) / eta
    exact = 2**d * intersection_fraction(sol.r[out], d)
    np.testing.assert_allclose(slope, exact, rtol=0.0, atol=ATOL_PY_LOW_DENSITY)
    first = (contact_value(eta, d) - 1.0) / eta
    assert first == pytest.approx(
        2**d * float(intersection_fraction(1.0, d)), abs=ATOL_PY_LOW_DENSITY
    )


@pytest.mark.high
@pytest.mark.parametrize("d", [1, 2, 3])
def test_py_grid_convergence_is_second_order(d: int) -> None:
    """Contact values at dr = 1/100, 1/200, 1/400: error ratio ~ 4."""
    vals = [
        solve_py(0.3, d, r_max=10.0, n_grid=n).g[100 * f]
        for n, f in ((1001, 1), (2001, 2), (4001, 4))
    ]
    ratio = (vals[1] - vals[0]) / (vals[2] - vals[1])
    assert abs(ratio - 4.0) <= PY_CONVERGENCE_RATIO, ratio


@pytest.mark.high
@pytest.mark.parametrize("d", [1, 2, 3])
def test_py_independent_of_outer_radius(d: int) -> None:
    """Same spacing, r_max 10 vs 20: no periodic-image or k-grid error."""
    a = solve_py(0.3, d, r_max=10.0, n_grid=2051)
    b = solve_py(0.3, d, r_max=20.0, n_grid=4101)
    n = a.r.size - 600  # compare away from the outer edge of the small grid
    np.testing.assert_allclose(a.r[:n], b.r[:n], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(a.g[:n], b.g[:n], rtol=0.0, atol=ATOL_PY_R_INDEPENDENCE)


@pytest.mark.high
@pytest.mark.parametrize("d", [1, 2, 3])
def test_py_solution_structure(d: int) -> None:
    """g = 0 in the core, c = 0 outside, g > 0 outside, g -> 1 at large r."""
    sol = solve_py(0.3, d)
    i1 = _node(sol, 1.0)
    assert np.all(sol.g[:i1] == 0.0)
    assert np.all(sol.c[i1 + 1 :] == 0.0)
    assert np.all(sol.g[i1:] > 0.0)
    assert np.all(sol.c[: i1 + 1] < 0.0)
    assert np.max(np.abs(sol.g[sol.r > 15.0] - 1.0)) <= ATOL_PY_TAIL
    assert sol.converged and sol.residual < 1e-12 and sol.iterations >= 1


@pytest.mark.high
@pytest.mark.parametrize("d", [1, 2, 3])
def test_py_zero_density_is_step(d: int) -> None:
    sol = solve_py(0.0, d)
    i1 = _node(sol, 1.0)
    expected = np.where(np.arange(sol.r.size) >= i1, 1.0, 0.0)
    np.testing.assert_array_equal(sol.g, expected)
    assert sol.converged and sol.iterations == 1


@pytest.mark.high
@pytest.mark.parametrize("d", [1, 2, 3])
def test_py_converges_at_eta_max(d: int) -> None:
    sol = solve_py(0.6, d)
    assert sol.converged and np.all(np.isfinite(sol.g))


# ============================================================
# TESTS: CONTACT VALUES
# ============================================================


@pytest.mark.critical
@pytest.mark.parametrize("d", [1, 3])
@pytest.mark.parametrize("eta", [0.1, 0.2, 0.32])
def test_contact_richardson_matches_analytic(d: int, eta: float) -> None:
    """The d = 2 extrapolation, checked where the answer is known (eta of PHS)."""
    assert _contact_richardson(eta, d) == pytest.approx(
        contact_value(eta, d), abs=ATOL_PY_CONTACT_RICHARDSON
    )


@pytest.mark.critical
@pytest.mark.parametrize("eta", [0.1, 0.2, 0.3])
def test_contact_value_2d_matches_finer_extrapolation(eta: float) -> None:
    """Default (dr = 1/102, 1/205) vs finer (dr = 1/200, 1/400) Richardson."""
    fine = [solve_py(eta, 2, r_max=10.0, n_grid=n) for n in (2001, 4001)]
    g_c, g_f = fine[0].g[200], fine[1].g[400]
    fine_value = g_f + (g_f - g_c) / 3.0
    assert contact_value(eta, 2) == pytest.approx(
        fine_value, abs=ATOL_PY_CONTACT_RICHARDSON
    )


@pytest.mark.critical
def test_contact_value_analytic_forms() -> None:
    """d = 1: 1/(1 - eta) (Tonks); d = 3: (1 + eta/2)/(1 - eta)^2 (PY, WT)."""
    assert contact_value(0.5, 1) == pytest.approx(2.0, rel=1e-15)
    assert contact_value(0.5, 3) == pytest.approx(5.0, rel=1e-15)
    assert contact_value(0.0, 2) == 1.0


# ============================================================
# TESTS: ANALYTIC ORACLES THEMSELVES
# ============================================================


@pytest.mark.high
@pytest.mark.parametrize("eta", [0.1, 0.4, 0.7])
def test_g2_exact_1d_sum_rules(eta: float) -> None:
    """Contact 1/(1 - eta); S(0) = 1 + 2 eta int_0^inf (g - 1) dx = (1 - eta)^2.

    The core contributes -1 exactly; the trapezoid runs over [1, 80].
    """
    x = np.linspace(1.0, 80.0, 790_001)
    g = g2_exact_1d(x, eta)
    assert float(g2_exact_1d(1.0, eta)) == pytest.approx(1.0 / (1.0 - eta), rel=1e-14)
    assert float(g2_exact_1d(1.0 - 1e-12, eta)) == 0.0
    s0 = 1.0 + 2.0 * eta * (-1.0 + integrate.trapezoid(g - 1.0, x))
    assert s0 == pytest.approx((1.0 - eta) ** 2, abs=ATOL_TONKS_SUM_RULE)
    assert abs(g[-1] - 1.0) < ATOL_TONKS_SUM_RULE


@pytest.mark.high
@pytest.mark.parametrize("eta", [0.1, 0.3, 0.5])
def test_wertheim_thiele_compressibility_and_contact(eta: float) -> None:
    """1/S(0) = 1 - rho c^(0) = (1 + 2 eta)^2/(1 - eta)^4; c(1-) = -g(1+) (PY)."""
    rho = 6.0 * eta / math.pi
    c0, _ = integrate.quad(
        lambda r: 4.0 * math.pi * r * r * float(c_py_3d(r, eta)),
        0.0,
        1.0,
        epsabs=1e-13,
        epsrel=1e-13,
    )
    expected = (1.0 + 2.0 * eta) ** 2 / (1.0 - eta) ** 4
    assert 1.0 - rho * c0 == pytest.approx(expected, rel=RTOL_QUAD_ORACLE)
    assert float(c_py_3d(1.0, eta)) == pytest.approx(-contact_value(eta, 3), rel=1e-14)
    assert float(c_py_3d(1.0 + 1e-9, eta)) == 0.0


@pytest.mark.high
def test_g2_low_density_formula() -> None:
    r = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    for d in (1, 2, 3):
        expected = np.where(
            r >= 1.0, 1.0 + 2**d * 0.1 * intersection_fraction(r, d), 0.0
        )
        np.testing.assert_allclose(g2_low_density(r, 0.1, d), expected, rtol=1e-15)


# ============================================================
# TESTS: CONTRACTS
# ============================================================


@pytest.mark.medium
def test_py_solution_is_frozen_and_read_only() -> None:
    sol = solve_py(0.2, 1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        sol.eta = 0.3  # type: ignore[misc]
    with pytest.raises(ValueError):
        sol.g[0] = 1.0
    assert sol.d == 1 and sol.eta == 0.2
    assert sol.r.shape == sol.g.shape == sol.c.shape == (4096,)


@pytest.mark.medium
def test_py_reports_nonconvergence() -> None:
    sol = solve_py(0.3, 3, max_iter=2)
    assert not sol.converged and sol.iterations == 2 and sol.residual > 1e-12


@pytest.mark.low
def test_py_solutions_are_cached() -> None:
    assert solve_py(0.25, 2) is solve_py(0.25, 2)


@pytest.mark.medium
@pytest.mark.parametrize(
    "kwargs",
    [
        {"eta": 0.3, "d": 4},
        {"eta": -0.1, "d": 2},
        {"eta": 0.61, "d": 2},
        {"eta": float("nan"), "d": 2},
        {"eta": 0.3, "d": 2, "r_max": 3.0},
        {"eta": 0.3, "d": 2, "n_grid": 50},
        {"eta": 0.3, "d": 2, "tol": 0.0},
        {"eta": 0.3, "d": 2, "max_iter": 0},
    ],
)
def test_solve_py_rejects_bad_input(kwargs: dict) -> None:  # type: ignore[type-arg]
    with pytest.raises(ValueError):
        solve_py(**kwargs)


@pytest.mark.medium
def test_oracles_reject_bad_input() -> None:
    with pytest.raises(ValueError):
        g2_exact_1d(1.5, 1.0)
    with pytest.raises(ValueError):
        g2_exact_1d(-1.0, 0.3)
    with pytest.raises(ValueError):
        c_py_3d(0.5, -0.1)
    with pytest.raises(ValueError):
        g2_low_density(1.5, 0.1, 0)
    with pytest.raises(ValueError):
        contact_value(0.7, 2)


# ============================================================
# REGRESSION TESTS
# ============================================================


@pytest.mark.high
@pytest.mark.parametrize("eta", [8.0004e-4 + 3e-13, 1.234567890123e-6, 0.3 + 4e-13])
def test_regression_D091_eta_cache_key_keeps_relative_precision(eta: float) -> None:
    """D-091: the cache key must keep eta to ~1e-14 relative, not 12 decimals."""
    sol = solve_py(eta, 1)
    assert abs(sol.eta / eta - 1.0) <= 1e-14
    assert sol.converged
