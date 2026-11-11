#!/usr/bin/env python3

"""
Pseudo-hard-sphere (PHS) ansatz for the T -> 0 structure factor of stealthy
ground states and the pair-level entropy of the unconstrained waves.

PHS (Torquato, Zhang & Stillinger, PRX 5, 021020 (2015), Sec. VII A, Eqs.
69-72, 77): for q = k/K > 1 the structure factor is the hard-sphere g2 at
r = q (diameter 1 <-> K) and packing fraction eta = b(d) chi; S = 0 for
q <= 1. b(d) = 1 / (2^d alpha_d(1)) makes the peak S(K+) = 1 + chi + O(chi^2)
in every d (the d = 1 value; b(1) = 1, b(3) = 2/5).

Closures for g2:
- "low_chi": first order in eta, S = 1 + chi alpha_d(q)/alpha_d(1) on
  1 < q < 2 and S = 1 for q >= 2.
- "py": Percus-Yevick at eta = b chi (``hardsphere.solve_py``, default grid),
  linearly interpolated between grid nodes.

Pair entropy: with c_K = v1(d) K^d / (2 (2 pi)^d) constrained half-space
wavevectors per volume, the Gaussian-field relative entropy of the waves with
|k| > K is

    sigma = (1/2) int_{|k|>K} d^dk/(2 pi)^d [ln S - S + 1] = c_K g(chi),
    g(chi) = d int_1^inf q^(d-1) [ln S(q) - S(q) + 1] dq <= 0

(d^dk = d v1 K^d q^(d-1) dq). ``pair_entropy_derivs`` also returns
h = d(chi g)/d chi and h' = dh/d chi, which enter the equations of state.

Derivatives, py closure: G(chi) = g(chi) / chi^2 is analytic on [0, 1/2]
(g = -a chi^2 + O(chi^3)); it is interpolated by a Chebyshev series through
direct PY solves at N_CHEB first-kind Chebyshev nodes (built once per d,
cached) and g, g', g'' follow from the series analytically. This is smooth,
consistent (h is exactly (chi g)' of the returned g) and makes grid sweeps
cheap; the interpolant agrees with direct solves to ~1e-13 relative
(tests/unit/test_theory_phs.py), so its accuracy is that of the PY grid
(O(dr^2) ~ 3e-5 relative in g at the default dr = 1/205). Derivatives,
low_chi closure: exact integrals by adaptive quadrature.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import functools
import math
from typing import Callable, Tuple

# ----- numerics -----
import numpy as np
from numpy.polynomial import chebyshev
from numpy.typing import ArrayLike
from scipy import integrate  # type: ignore[import-untyped]

# ----- project -----
from xla_cco.theory.geometry import (
    SUPPORTED_DIMS,
    check_dim,
    check_nonnegative,
    intersection_fraction,
    unit_ball_volume,
)
from xla_cco.theory.hardsphere import PYSolution, solve_py

# ============================================================
# CONSTANTS
# ============================================================

CHI_MAX: float = 0.5  # disordered stealthy branch: 0 <= chi < 1/2
CLOSURES: Tuple[str, ...] = ("py", "low_chi")
N_CHEB: int = 20  # Chebyshev nodes of the py interpolant of g(chi)/chi^2
QUAD_EPS: float = 1e-13  # absolute/relative accuracy requested from quad
PY_TOL: float = 1e-13  # PY iteration tolerance (roundoff floor ~1e-14)
LNM_SERIES_X: float = 1e-3  # below |x| use the series of log1p(x) - x

# ============================================================
# VALIDATION
# ============================================================


def check_chi(chi: float) -> float:
    """Return chi as a float in [0, 1/2); raise ValueError otherwise."""
    val = float(chi)
    if not (math.isfinite(val) and 0.0 <= val < CHI_MAX):
        raise ValueError(
            f"chi must lie in [0, {CHI_MAX}) (disordered branch), got {chi}"
        )
    return val


def check_closure(closure: str) -> str:
    """Return ``closure`` if it is a known closure; raise ValueError otherwise."""
    if closure not in CLOSURES:
        raise ValueError(f"closure must be one of {CLOSURES}, got {closure!r}")
    return closure


# ============================================================
# COEFFICIENTS
# ============================================================


def b_coefficient(d: int) -> float:
    """Return b(d) = 1 / (2^d alpha_d(1)), so that eta = b(d) chi."""
    dim = check_dim(d)
    out = 1.0 / (2.0**dim * float(intersection_fraction(1.0, dim)))
    assert 0.0 < out <= 1.0, "b(d) decreases from b(1) = 1"
    return out


def c_k(K: float, d: int) -> float:
    """Return c_K = v1(d) K^d / (2 (2 pi)^d), constrained wavevectors per volume."""
    dim = check_dim(d)
    kval = float(K)
    if not (math.isfinite(kval) and kval > 0.0):
        raise ValueError(f"K must be finite and > 0, got {K}")
    out = unit_ball_volume(dim) * kval**dim / (2.0 * (2.0 * math.pi) ** dim)
    assert out > 0.0, "c_K must be positive"
    return out


# ============================================================
# STRUCTURE FACTOR
# ============================================================


def _py_solution(chi: float, d: int) -> PYSolution:
    """Return the converged PY solution at eta = b(d) chi (raise if not)."""
    sol = solve_py(b_coefficient(d) * chi, d, tol=PY_TOL)
    if not sol.converged:
        raise RuntimeError(f"PY did not converge at chi = {chi}, d = {d}")
    assert sol.d == d, "solution dimension mismatch"
    return sol


def structure_factor(
    q: ArrayLike, chi: float, d: int, closure: str = "py"
) -> np.ndarray:
    """Return the PHS structure factor S(q = k/K; chi) (0 for q <= 1).

    Parameters
    ----------
    q : ArrayLike
        Scaled wavenumbers k/K (>= 0, finite).
    chi : float
        Stealthiness in [0, 1/2).
    d : int
        Dimension, 1, 2 or 3.
    closure : str
        "py" or "low_chi".

    Returns
    -------
    np.ndarray
        S(q), same shape as ``q``; S(1+) is the contact value.
    """
    dim = check_dim(d, SUPPORTED_DIMS)
    val = check_chi(chi)
    check_closure(closure)
    qa = check_nonnegative(q, "q")
    if closure == "low_chi":
        bump = (
            val
            * intersection_fraction(qa, dim)
            / float(intersection_fraction(1.0, dim))
        )
        out = np.where(qa <= 1.0, 0.0, 1.0 + bump)
    else:
        sol = _py_solution(val, dim)
        m = int(round(1.0 / float(sol.r[1])))
        inside = np.interp(qa, sol.r[m:], sol.g[m:], right=1.0)
        out = np.where(qa <= 1.0, 0.0, inside)
    assert out.shape == qa.shape, "shape must be preserved"
    assert np.all(out >= 0.0), "S must be non-negative"
    return np.asarray(out, dtype=float)


# ============================================================
# PAIR ENTROPY
# ============================================================


def _log1p_minus(x: np.ndarray) -> np.ndarray:
    """Return ln(1 + x) - x without cancellation for small |x| (x > -1)."""
    assert np.all(x > -1.0), "need S = 1 + x > 0"
    series = x * x * (-0.5 + x * (1.0 / 3.0 + x * (-0.25 + x * 0.2)))
    return np.asarray(np.where(np.abs(x) < LNM_SERIES_X, series, np.log1p(x) - x))


def _pair_entropy_direct(chi: float, d: int) -> float:
    """Return g(chi) for the py closure by trapezoid over one PY solution."""
    sol = _py_solution(chi, d)
    dr = float(sol.r[1])
    m = int(round(1.0 / dr))
    q = sol.r[m:]
    assert abs(q[0] - 1.0) < 1e-12 and sol.g[m] > 0.0, "node m is the contact"
    f = q ** (d - 1) * _log1p_minus(sol.g[m:] - 1.0)
    out = d * dr * (float(np.sum(f)) - 0.5 * float(f[0] + f[-1]))
    assert out <= 0.0 or chi == 0.0, "relative entropy must be <= 0"
    return out


@functools.lru_cache(maxsize=len(SUPPORTED_DIMS))
def _py_ratio_series(d: int) -> chebyshev.Chebyshev:
    """Chebyshev interpolant of G(chi) = g(chi)/chi^2 on [0, 1/2] (py)."""
    assert d in SUPPORTED_DIMS, "py closure needs d in {1, 2, 3}"
    nodes = 0.5 * CHI_MAX * (1.0 - np.cos(np.pi * (np.arange(N_CHEB) + 0.5) / N_CHEB))
    vals = np.array([_pair_entropy_direct(float(x), d) / x**2 for x in nodes])
    series = chebyshev.Chebyshev.fit(nodes, vals, N_CHEB - 1, domain=[0.0, CHI_MAX])
    assert np.all(vals < 0.0), "G must be negative on (0, 1/2)"
    assert np.allclose(series(nodes), vals, rtol=1e-10, atol=0.0), (
        "fit must interpolate"
    )
    return series


def _low_chi_integral(chi: float, d: int, power: int) -> float:
    """Return d int_1^2 q^(d-1) F(chi u) dq, u = alpha(q)/alpha(1), for F by power.

    power 0: ln(1 + x) - x; power 1: -u x / (1 + x); power 2: -u^2 / (1 + x)^2,
    i.e. the integrands of g, g' and g'' of the low_chi closure.
    """
    assert power in (0, 1, 2), "power selects g, g' or g''"
    a1 = float(intersection_fraction(1.0, d))
    forms: Tuple[Callable[[float, float], float], ...] = (
        lambda u, x: float(_log1p_minus(np.array(x))),
        lambda u, x: -u * x / (1.0 + x),
        lambda u, x: -u * u / (1.0 + x) ** 2,
    )

    def integrand(q: float) -> float:
        u = float(intersection_fraction(q, d)) / a1
        return d * q ** (d - 1) * forms[power](u, chi * u)

    val, _ = integrate.quad(integrand, 1.0, 2.0, epsabs=QUAD_EPS, epsrel=QUAD_EPS)
    assert math.isfinite(val), "quadrature failed"
    return float(val)


def pair_entropy_derivs(
    chi: float, d: int, closure: str = "py"
) -> Tuple[float, float, float]:
    """Return (g, h, h') with h = d(chi g)/d chi = g + chi g', h' = 2 g' + chi g''.

    Parameters
    ----------
    chi : float
        Stealthiness in [0, 1/2).
    d : int
        Dimension, 1, 2 or 3.
    closure : str
        "py" (Chebyshev interpolant of direct PY solves) or "low_chi" (quad).

    Returns
    -------
    Tuple[float, float, float]
        g(chi) <= 0, h(chi) and h'(chi).
    """
    dim = check_dim(d, SUPPORTED_DIMS)
    val = check_chi(chi)
    check_closure(closure)
    if closure == "low_chi":
        g0, g1, g2 = (_low_chi_integral(val, dim, p) for p in (0, 1, 2))
    else:
        series = _py_ratio_series(dim)
        s0, s1, s2 = series(val), series.deriv(1)(val), series.deriv(2)(val)
        g0 = val * val * s0
        g1 = 2.0 * val * s0 + val * val * s1
        g2 = 2.0 * s0 + 4.0 * val * s1 + val * val * s2
    out = (float(g0), float(g0 + val * g1), float(2.0 * g1 + val * g2))
    assert all(math.isfinite(v) for v in out), "non-finite entropy derivative"
    assert out[0] <= 0.0, "relative entropy must be <= 0"
    return out


def pair_entropy_ratio(chi: float, d: int, closure: str = "py") -> float:
    """Return g(chi) = sigma / c_K = d int_1^inf q^(d-1) [ln S - S + 1] dq <= 0."""
    g0, _, _ = pair_entropy_derivs(chi, d, closure)
    assert math.isfinite(g0), "g must be finite"
    return g0
