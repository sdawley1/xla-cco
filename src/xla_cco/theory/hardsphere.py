#!/usr/bin/env python3

"""
Percus-Yevick (PY) hard spheres in d = 1, 2, 3 and analytic oracles.

Units: sphere diameter 1, packing fraction eta = rho v1(d) / 2^d.

Ornstein-Zernike (OZ) with the PY closure is solved by iterating on the
continuous indirect function gamma = h - c:

    c(r) = -(1 + gamma(r))  for r < 1,   c(r) = 0  for r > 1   (g = 0 inside),
    gamma^(k) = rho c^(k)^2 / (1 - rho c^(k)),

with Ng (1974) acceleration (J. Chem. Phys. 61, 2680) on top of damped Picard
steps. One generic d-dimensional radial Fourier (Hankel) transform serves
every d:

    f^(k) = int_0^inf r^(d-1) Omega_d(k r) f(r) dr,
    f(r)  = (2 pi)^(-d) int_0^inf k^(d-1) Omega_d(k r) f^(k) dk,
    Omega_d(x) = (2 pi)^(d/2) x^(-nu) J_nu(x),   nu = d/2 - 1

(Omega_1 = 2 cos x, Omega_2 = 2 pi J_0(x), Omega_3 = 4 pi sin(x)/x).

Discretization (dense matrices; why this choice):

- r-nodes r_i = i dr, i = 0..N (N = n_grid - 1), with 1/dr an integer so the
  core boundary r = 1 is a node. The forward transform is the trapezoid rule,
  with the PY jump of c at r = 1 carried by the half value (second order).
  In d = 2 the integrand r J_0(k r) f(r) is odd in r, so the first
  Euler-Maclaurin term (dr^2 / 12) Omega_2(0) f(0) is added; in d = 1, 3 the
  integrand is even and the correction vanishes.
- k-nodes k_m = j_{nu,m} / R (zeros of J_nu, R = N dr) and the inverse is the
  Fourier-Bessel series of order nu on [0, R], which is exact for functions
  supported in [0, R] (truncated at m = N, i.e. k_max ~ pi / dr). For d = 1, 3
  the pair reduces to the DCT-III / DST-I, i.e. exact discrete convolution of
  the periodically extended functions. A trapezoid rule in k instead would
  leave a non-convergent O((pi/R)^2) error in d = 2 (k J_0(k r) is odd in k);
  the Fourier-Bessel inverse removes it (results independent of R to 1e-11).

Accuracy: O(dr^2) everywhere except at the nodes r = 0 and r = 2 (d = 1),
where both factors of the OZ convolution integrand jump at the same point and
the trapezoid rule is first order; the error there is a single-node
O(rho dr) spike in d = 1 (spread as O(dr / r) near r = 0 in d = 3). Contact
values (node r = 1) converge at second order in every d.

References: Percus & Yevick, Phys. Rev. 110, 1 (1958); Wertheim, PRL 10, 321
(1963); Thiele, J. Chem. Phys. 39, 474 (1963); Tonks, Phys. Rev. 50, 955
(1936); Salsburg, Zwanzig & Kirkwood, J. Chem. Phys. 21, 1098 (1953);
Lado, J. Comput. Phys. 8, 417 (1971); Santos, A Concise Course on the Theory
of Classical Liquids (Springer, 2016), Ch. 4-5.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import functools
import math
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple

# ----- numerics -----
import numpy as np
from numpy.typing import ArrayLike
from scipy import special  # type: ignore[import-untyped]

# ----- project -----
from xla_cco.theory.geometry import (
    SUPPORTED_DIMS,
    check_dim,
    check_nonnegative,
    intersection_fraction,
    unit_ball_volume,
)

# ============================================================
# CONSTANTS
# ============================================================

ETA_MAX: float = 0.6  # largest packing fraction accepted (iteration tested there)
ETA_CACHE_SIG: int = 15  # cache key: eta to 15 significant digits (D-091)
MIN_R_MAX: float = 5.0  # g - 1 must have decayed inside the box
MIN_NODES_PER_UNIT: int = 8  # coarsest accepted spacing dr = 1/8
PICARD_MIX: float = 0.5  # damping of plain Picard steps
NG_DEPTH: int = 3  # Ng acceleration uses the last three iterates
NEWTON_MAX: int = 50  # bound on the Bessel-zero Newton refinement
RICHARDSON_COARSE: int = 2048  # coarse n_grid of the Richardson pair (dr = 1/102)
RICHARDSON_FINE: int = 4096  # fine n_grid of the Richardson pair (dr = 1/205)
# PY tolerance of the Richardson pair; contact_value and phs share these cached
# solves (the roundoff floor of the residual is ~1e-14).
RICHARDSON_TOL: float = 1e-13
TONKS_MAX_R: float = 1.0e4  # bounds the neighbour sum of g2_exact_1d

# ============================================================
# DATA
# ============================================================


@dataclass(frozen=True)
class PYSolution:
    """Converged (or last) PY iterate on the r-grid.

    ``r`` holds the nodes r_i = i dr (r = 1 is a node); ``g`` is g2(r) with
    the contact value g(1+) at r = 1; ``c`` is the direct correlation function
    with c(1-) at r = 1. Arrays are read-only (solutions are cached).
    """

    r: np.ndarray
    g: np.ndarray
    c: np.ndarray
    eta: float
    d: int
    converged: bool
    iterations: int
    residual: float


class _RadialGrid(NamedTuple):
    """Nodes and transform matrices of the d-dimensional radial transform."""

    d: int
    r: np.ndarray  # (n_r,) r-nodes i dr
    k: np.ndarray  # (n_k,) k-nodes j_{nu,m} / R
    m: int  # index of the node r = 1
    dr: float
    omega: np.ndarray  # (n_k, n_r) Omega_d(k_m r_i)
    w_fwd: np.ndarray  # (n_r,) trapezoid weights times r^(d-1)
    w_inv: np.ndarray  # (n_k,) Fourier-Bessel weights
    em0: float  # Euler-Maclaurin weight of f(0) in every f^(k) (d = 2 only)


# ============================================================
# RADIAL FOURIER TRANSFORM
# ============================================================


def _omega(d: int, x: np.ndarray) -> np.ndarray:
    """Return Omega_d(x) for x >= 0: elementary forms of the Bessel kernel.

    Omega_1 = 2 cos x, Omega_2 = 2 pi J_0(x), Omega_3 = 4 pi sin(x) / x are
    the generic (2 pi)^(d/2) x^(-nu) J_nu(x) (``_omega_generic``, tested
    equal) evaluated ~10x faster than scipy's fractional-order ``jv``.
    """
    assert np.all(x >= 0.0) and d in SUPPORTED_DIMS, "Omega_d needs x >= 0"
    if d == 1:
        out = 2.0 * np.cos(x)
    elif d == 2:
        out = 2.0 * math.pi * special.j0(x)
    else:
        safe = np.where(x == 0.0, 1.0, x)
        out = 4.0 * math.pi * np.where(x == 0.0, 1.0, np.sin(safe) / safe)
    assert np.all(np.isfinite(out)), "Omega_d not finite"
    return np.asarray(out, dtype=float)


def _omega_generic(d: int, x: np.ndarray) -> np.ndarray:
    """Return Omega_d(x) = (2 pi)^(d/2) x^(-nu) J_nu(x), nu = d/2 - 1, x >= 0."""
    assert np.all(x >= 0.0), "Omega_d needs x >= 0"
    nu = d / 2.0 - 1.0
    small = x == 0.0
    safe = np.where(small, 1.0, x)
    val = safe ** (-nu) * special.jv(nu, safe)
    val = np.where(small, 1.0 / (2.0**nu * math.gamma(nu + 1.0)), val)
    assert np.all(np.isfinite(val)), "Omega_d not finite"
    return np.asarray((2.0 * math.pi) ** (d / 2.0) * val, dtype=float)


def _bessel_zeros(nu: float, count: int) -> np.ndarray:
    """Return the first ``count`` positive zeros of J_nu (McMahon + Newton)."""
    assert count >= 1 and nu > -1.0, "need count >= 1 and nu > -1"
    beta = (np.arange(1, count + 1) + nu / 2.0 - 0.25) * math.pi
    mu = 4.0 * nu * nu
    x = beta - (mu - 1.0) / (8.0 * beta)
    x = x - 4.0 * (mu - 1.0) * (7.0 * mu - 31.0) / (3.0 * (8.0 * beta) ** 3)
    for _ in range(NEWTON_MAX):
        deriv = 0.5 * (special.jv(nu - 1.0, x) - special.jv(nu + 1.0, x))
        step = special.jv(nu, x) / deriv
        x = x - step
        if np.max(np.abs(step) / x) < 1e-15:
            break
    assert np.all(np.diff(x) > 2.0), "zeros must be increasing, spacing ~ pi"
    assert np.max(np.abs(special.jv(nu, x))) < 1e-12, "Newton did not converge"
    return np.asarray(x, dtype=float)


@functools.lru_cache(maxsize=4)  # <= 4 x 134 MB at n_grid = 4096
def _radial_grid(d: int, r_max: float, n_grid: int) -> _RadialGrid:
    """Build (and cache) the nodes and matrices for one (d, r_max, n_grid)."""
    assert d in SUPPORTED_DIMS and n_grid >= 2, "bad grid request"
    n_int = n_grid - 1
    per_unit = max(1, round(n_int / r_max))
    dr = 1.0 / per_unit
    big_r = n_int * dr
    r = np.arange(n_grid) * dr
    nu = d / 2.0 - 1.0
    zeros = _bessel_zeros(nu, n_int)
    k = zeros / big_r
    omega = _omega(d, np.outer(k, r))
    w = np.ones(n_grid)
    w[0] = w[-1] = 0.5
    w_fwd = w * r ** (d - 1) * dr
    fb = 2.0 / (big_r**2 * special.jv(nu + 1.0, zeros) ** 2)
    w_inv = k ** (d - 2) * fb / (2.0 * math.pi) ** d
    em0 = dr * dr / 12.0 * float(_omega(d, np.zeros(1))[0]) if d == 2 else 0.0
    for arr in (r, k, omega, w_fwd, w_inv):
        arr.setflags(write=False)
    assert abs(per_unit * dr - 1.0) < 1e-12, "r = 1 must be a node"
    return _RadialGrid(d, r, k, per_unit, dr, omega, w_fwd, w_inv, em0)


def _forward(grid: _RadialGrid, f: np.ndarray) -> np.ndarray:
    """Return f^(k_m) from samples f(r_i) (trapezoid + Euler-Maclaurin)."""
    assert f.shape == grid.r.shape, "f must live on the r-grid"
    out = grid.omega @ (grid.w_fwd * f) + grid.em0 * f[0]
    assert out.shape == grid.k.shape, "output must live on the k-grid"
    return np.asarray(out, dtype=float)


def _inverse(grid: _RadialGrid, fk: np.ndarray) -> np.ndarray:
    """Return f(r_i) from samples f^(k_m) (Fourier-Bessel series)."""
    assert fk.shape == grid.k.shape, "fk must live on the k-grid"
    out = grid.omega.T @ (grid.w_inv * fk)
    assert out.shape == grid.r.shape, "output must live on the r-grid"
    return np.asarray(out, dtype=float)


# ============================================================
# PY ITERATION
# ============================================================


def _number_density(eta: float, d: int) -> float:
    """Return rho = 2^d eta / v1(d) for spheres of diameter 1."""
    assert eta >= 0.0, "eta must be >= 0"
    rho = 2.0**d * eta / unit_ball_volume(d)
    assert rho >= 0.0 and math.isfinite(rho), "bad density"
    return rho


def _core_weights(grid: _RadialGrid) -> np.ndarray:
    """Return 1 inside the core, 1/2 at r = 1 (jump midpoint), 0 outside."""
    w = np.where(np.arange(grid.r.size) < grid.m, 1.0, 0.0)
    w[grid.m] = 0.5
    assert w.sum() == grid.m + 0.5, "core weights miscounted"
    assert abs(grid.r[grid.m] - 1.0) < 1e-12, "node m must be r = 1"
    return w


def _oz_map(
    grid: _RadialGrid, rho: float, core: np.ndarray, gam: np.ndarray
) -> Optional[np.ndarray]:
    """One OZ + PY step gamma -> gamma'.

    Parameters
    ----------
    grid : _RadialGrid
        Nodes and transform matrices.
    rho : float
        Number density (>= 0).
    core : np.ndarray
        Core weights (``_core_weights``): c = -core (1 + gam).
    gam : np.ndarray
        Current indirect function gamma = h - c on the r-grid.

    Returns
    -------
    Optional[np.ndarray]
        gamma' = F^-1[rho c^2 / (1 - rho c^)], or None if 1 - rho c^ <= 0
        somewhere or the result is not finite.
    """
    assert gam.shape == core.shape, "gamma must live on the r-grid"
    assert rho >= 0.0, "density must be >= 0"
    ck = _forward(grid, -core * (1.0 + gam))
    denom = 1.0 - rho * ck
    if not np.all(denom > 0.0):
        return None
    out = _inverse(grid, rho * ck * ck / denom)
    return out if np.all(np.isfinite(out)) else None


def _ng_step(ins: List[np.ndarray], outs: List[np.ndarray]) -> Optional[np.ndarray]:
    """Ng's three-point extrapolation; None if its 2 x 2 system is singular."""
    assert len(ins) == len(outs) == NG_DEPTH, "Ng needs three iterates"
    res = [o - i for i, o in zip(ins, outs, strict=True)]
    d01, d02 = res[2] - res[1], res[2] - res[0]
    mat = np.array([[d01 @ d01, d01 @ d02], [d01 @ d02, d02 @ d02]])
    assert np.all(np.isfinite(mat)), "Ng iterates must be finite"
    rhs = np.array([res[2] @ d01, res[2] @ d02])
    det = mat[0, 0] * mat[1, 1] - mat[0, 1] ** 2
    if not det > 1e-14 * mat[0, 0] * mat[1, 1]:
        return None
    a1, a2 = np.linalg.solve(mat, rhs)
    nxt = (1.0 - a1 - a2) * outs[2] + a1 * outs[1] + a2 * outs[0]
    return nxt if np.all(np.isfinite(nxt)) else None


def _iterate(
    grid: _RadialGrid, rho: float, tol: float, max_iter: int
) -> Tuple[np.ndarray, bool, int, float]:
    """Iterate gamma to the OZ + PY fixed point (Ng-accelerated Picard).

    A trial gamma whose c^ makes 1 - rho c^ <= 0 (possible for an Ng
    extrapolation at high eta) is pulled halfway back to the last valid
    iterate and the Ng history is cleared.

    Parameters
    ----------
    grid : _RadialGrid
        Nodes and transform matrices.
    rho : float
        Number density (>= 0, finite).
    tol : float
        Convergence threshold on max |gamma_out - gamma_in|.
    max_iter : int
        Iteration bound (every OZ step, accepted or not, counts).

    Returns
    -------
    Tuple[np.ndarray, bool, int, float]
        gamma (converged output, else last input), converged flag, number
        of iterations and last residual (inf if no valid step).
    """
    assert tol > 0.0 and max_iter >= 1, "need tol > 0 and max_iter >= 1"
    assert rho >= 0.0 and math.isfinite(rho), "density must be finite and >= 0"
    core = _core_weights(grid)
    gam = np.zeros(grid.r.size)
    good = gam
    ins: List[np.ndarray] = []
    outs: List[np.ndarray] = []
    residual = math.inf
    for it in range(1, max_iter + 1):
        out = _oz_map(grid, rho, core, gam)
        if out is None:
            if gam is good:
                return gam, False, it, math.inf
            gam, ins, outs = 0.5 * (gam + good), [], []
            continue
        good = gam
        residual = float(np.max(np.abs(out - gam)))
        if residual < tol:
            return out, True, it, residual
        ins, outs = (ins + [gam])[-NG_DEPTH:], (outs + [out])[-NG_DEPTH:]
        nxt = _ng_step(ins, outs) if len(ins) == NG_DEPTH else None
        gam = nxt if nxt is not None else gam + PICARD_MIX * (out - gam)
    return gam, False, max_iter, residual


# ============================================================
# PUBLIC SOLVER
# ============================================================


def _check_eta(eta: float, upper: float) -> float:
    """Return eta as a float in [0, upper]; raise ValueError otherwise."""
    val = float(eta)
    if not (math.isfinite(val) and 0.0 <= val <= upper):
        raise ValueError(f"eta must lie in [0, {upper}], got {eta}")
    return val


def _eta_cache_key(eta: float) -> float:
    """Return eta rounded to ETA_CACHE_SIG significant digits (relative, D-091)."""
    assert eta >= 0.0 and math.isfinite(eta), "eta must be finite and >= 0"
    key = float(f"{eta:.{ETA_CACHE_SIG - 1}e}")
    assert abs(key - eta) <= 1e-14 * eta, "rounding must be relative"
    return key


def solve_py(
    eta: float,
    d: int,
    *,
    r_max: float = 20.0,
    n_grid: int = 4096,
    tol: float = 1e-12,
    max_iter: int = 10_000,
) -> PYSolution:
    """Solve OZ + PY for hard spheres of diameter 1 at packing fraction eta.

    Parameters
    ----------
    eta : float
        Packing fraction in [0, 0.6]; rounded to 15 significant digits
        (the cache key; relative change <= 5e-15).
    d : int
        Dimension, 1, 2 or 3.
    r_max : float
        Approximate outer radius (>= 5). The spacing is
        dr = 1 / round((n_grid - 1) / r_max) so that r = 1 is a node; the
        last node is (n_grid - 1) dr.
    n_grid : int
        Number of r-nodes (at least 8 per unit length).
    tol : float
        Convergence threshold on max |gamma_out - gamma_in|.
    max_iter : int
        Iteration bound.

    Returns
    -------
    PYSolution
        Cached, read-only solution; check ``converged``.

    Raises
    ------
    ValueError
        On out-of-range arguments.
    """
    dim = check_dim(d, SUPPORTED_DIMS)
    val = _eta_cache_key(_check_eta(eta, ETA_MAX))
    if not (math.isfinite(r_max) and r_max >= MIN_R_MAX):
        raise ValueError(f"r_max must be >= {MIN_R_MAX}, got {r_max}")
    if int(n_grid) != n_grid or (n_grid - 1) / r_max < MIN_NODES_PER_UNIT - 0.5:
        raise ValueError(f"n_grid must give >= {MIN_NODES_PER_UNIT} nodes per unit")
    if not (tol > 0.0 and int(max_iter) == max_iter and max_iter >= 1):
        raise ValueError("need tol > 0 and an integer max_iter >= 1")
    return _solve_py_cached(
        val, dim, float(r_max), int(n_grid), float(tol), int(max_iter)
    )


@functools.lru_cache(maxsize=128)
def _solve_py_cached(
    eta: float, d: int, r_max: float, n_grid: int, tol: float, max_iter: int
) -> PYSolution:
    """Solve and package one PY problem.

    Parameters
    ----------
    eta : float
        Packing fraction (already validated and rounded by ``solve_py``).
    d : int
        Dimension, 1, 2 or 3.
    r_max, n_grid, tol, max_iter
        As in ``solve_py`` (validated).

    Returns
    -------
    PYSolution
        Read-only arrays g (g(1+) at r = 1) and c (c(1-) at r = 1).
    """
    grid = _radial_grid(d, r_max, n_grid)
    gam, converged, iters, residual = _iterate(
        grid, _number_density(eta, d), tol, max_iter
    )
    idx = np.arange(grid.r.size)
    g = np.where(idx >= grid.m, 1.0 + gam, 0.0)
    c = np.where(idx <= grid.m, -(1.0 + gam), 0.0)
    r = np.array(grid.r)
    for arr in (r, g, c):
        arr.setflags(write=False)
    assert g.shape == c.shape == r.shape, "solution arrays must share the grid"
    assert iters >= 1, "at least one OZ step is taken"
    return PYSolution(r, g, c, eta, d, converged, iters, residual)


# ============================================================
# ANALYTIC RESULTS
# ============================================================


def g2_low_density(r: ArrayLike, eta: float, d: int) -> np.ndarray:
    """Return the hard-sphere g2 to first order in eta (diameter 1).

    g2 = Theta(r - 1) [1 + 2^d eta alpha_d(r)], with g(1) = g(1+).

    Parameters
    ----------
    r : ArrayLike
        Distances (>= 0, finite).
    eta : float
        Packing fraction in [0, 1].
    d : int
        Dimension (>= 1).

    Returns
    -------
    np.ndarray
        g2, same shape as ``r``.
    """
    dim = check_dim(d)
    ra = check_nonnegative(r, "r")
    val = _check_eta(eta, 1.0)
    out = np.where(
        ra >= 1.0, 1.0 + 2.0**dim * val * intersection_fraction(ra, dim), 0.0
    )
    assert out.shape == ra.shape, "shape must be preserved"
    assert np.all(out >= 0.0), "g2 must be non-negative"
    return np.asarray(out, dtype=float)


def g2_exact_1d(r: ArrayLike, eta: float) -> np.ndarray:
    """Return the exact hard-rod g2 (rod length 1, density eta).

    Gaps between neighbouring rods minus 1 are i.i.d. exponential with rate
    lam = eta / (1 - eta), so the j-th neighbour sits at j + Gamma(j, lam)
    and eta g2(x) = sum_j lam^j (x - j)^(j-1) e^(-lam (x - j)) / (j - 1)!
    for x > j (Salsburg, Zwanzig & Kirkwood 1953). g2(1) = g2(1+) = 1/(1 - eta).

    Raises
    ------
    ValueError
        If eta is outside [0, 1), or r is negative, not finite or > 1e4.
    """
    ra = check_nonnegative(r, "r")
    val = _check_eta(eta, 1.0)
    if val >= 1.0 or (ra.size and ra.max() > TONKS_MAX_R):
        raise ValueError("need eta < 1 and r <= 1e4")
    if val == 0.0:
        return np.where(ra >= 1.0, 1.0, 0.0)
    lam = val / (1.0 - val)
    total = np.zeros(ra.shape)
    n_terms = int(math.floor(ra.max())) if ra.size else 0
    for j in range(1, n_terms + 1):
        y = ra - j
        pos = y > 0.0
        logy = np.log(np.where(pos, y, 1.0))
        term = np.exp(j * math.log(lam) + (j - 1) * logy - lam * y - math.lgamma(j))
        total += np.where(pos, term, 0.0)
    total = np.where(ra == 1.0, lam, total)  # contact: the j = 1 term at y = 0+
    assert np.all(total >= 0.0), "g2 must be non-negative"
    assert np.all(total[ra < 1.0] == 0.0), "g2 must vanish inside the core"
    return np.asarray(total / val, dtype=float)


def c_py_3d(r: ArrayLike, eta: float) -> np.ndarray:
    """Return the Wertheim-Thiele PY c(r) for d = 3 (diameter 1).

    c = -(a + b r + (eta a / 2) r^3) for r <= 1 (c(1) = c(1-)), 0 for r > 1,
    a = (1 + 2 eta)^2 / (1 - eta)^4, b = -6 eta (1 + eta/2)^2 / (1 - eta)^4.
    """
    ra = check_nonnegative(r, "r")
    val = _check_eta(eta, 1.0)
    if val >= 1.0:
        raise ValueError("need eta < 1")
    a = (1.0 + 2.0 * val) ** 2 / (1.0 - val) ** 4
    b = -6.0 * val * (1.0 + val / 2.0) ** 2 / (1.0 - val) ** 4
    out = np.where(ra <= 1.0, -(a + b * ra + 0.5 * val * a * ra**3), 0.0)
    assert out.shape == ra.shape, "shape must be preserved"
    assert np.all(out <= 0.0), "PY hard-sphere c is non-positive"
    return np.asarray(out, dtype=float)


def _richardson_pair(eta: float, d: int) -> Tuple[PYSolution, PYSolution]:
    """Return the converged (coarse, fine) PY solutions used for extrapolation.

    Grids: default r_max, n_grid = RICHARDSON_COARSE and RICHARDSON_FINE
    (dr = 1/102 and 1/205), tol = RICHARDSON_TOL; cached through ``solve_py``.

    Raises
    ------
    RuntimeError
        If either iteration does not converge.
    """
    coarse = solve_py(eta, d, n_grid=RICHARDSON_COARSE, tol=RICHARDSON_TOL)
    fine = solve_py(eta, d, n_grid=RICHARDSON_FINE, tol=RICHARDSON_TOL)
    if not (fine.converged and coarse.converged):
        raise RuntimeError(f"PY iteration did not converge at eta = {eta}, d = {d}")
    assert float(coarse.r[1]) > float(fine.r[1]) > 0.0, "coarse must be coarser"
    return coarse, fine


def _richardson(coarse: float, fine: float, h_coarse: float, h_fine: float) -> float:
    """Return the dr^2 Richardson extrapolation of two grid values."""
    assert h_coarse > h_fine > 0.0, "need h_coarse > h_fine > 0"
    out = fine + (fine - coarse) * h_fine**2 / (h_coarse**2 - h_fine**2)
    assert math.isfinite(out), "extrapolation must be finite"
    return out


def _contact_richardson(eta: float, d: int) -> float:
    """Return g(1+) from the solver, Richardson-extrapolated in dr^2."""
    coarse, fine = _richardson_pair(eta, d)
    h_f, h_c = float(fine.r[1]), float(coarse.r[1])
    g_f = float(fine.g[int(round(1.0 / h_f))])
    g_c = float(coarse.g[int(round(1.0 / h_c))])
    assert g_f > 0.0 and g_c > 0.0, "contact values must be positive"
    assert abs(fine.r[int(round(1.0 / h_f))] - 1.0) < 1e-12, "contact node at r = 1"
    return _richardson(g_c, g_f, h_c, h_f)


def contact_value(eta: float, d: int) -> float:
    """Return the PY contact value g(1+) at packing fraction eta in [0, 0.6].

    d = 1: 1 / (1 - eta) (exact); d = 3: (1 + eta/2) / (1 - eta)^2 (PY);
    d = 2: numerical, Richardson extrapolation of n_grid = 2048 and 4096
    (``_richardson_pair``); error <= 3e-8 at eta <= 0.3 against an independent
    real-space PY solver (audit of PR #20).
    """
    dim = check_dim(d, SUPPORTED_DIMS)
    val = _check_eta(eta, ETA_MAX)
    if dim == 1:
        return 1.0 / (1.0 - val)
    if dim == 3:
        return (1.0 + val / 2.0) / (1.0 - val) ** 2
    out = 1.0 if val == 0.0 else _contact_richardson(val, dim)
    assert out >= 1.0, "PY contact value grows with eta"
    assert math.isfinite(out), "contact value must be finite"
    return out
