#!/usr/bin/env python3

"""
Low-temperature equations of state of the stealthy disordered ground states.

Units: k_B = 1, mass 1, v~(k) = 1 for |k| <= K and v~(0) = 1. Number density
n = N/V; c_K = v1(d) K^d / (2 (2 pi)^d); stealthiness chi = c_K / (n d)
(thermodynamic limit); disordered branch 0 <= chi < 1/2.

Manifold entropy (Prop. 12, Eq. 79 of the notes): near the ground-state
manifold Z = exp(-Phi_0/T) (2 pi T)^M W(V) with W = int dH / sqrt(det K),
K = (2/V) J J^T, and by the coarea formula W = (V/2)^M V^N p0 (p0: density
at u = 0 of the 2M collective coordinates of N independent uniform particles).
Levels of ln W:

- "ideal":    N ln V                                (To15 Eq. 92 level)
- "gaussian": N ln V - M ln(2 pi n)                 (u ~ N(0, N/2) i.i.d.)
- "phs":      gaussian + V sigma = gaussian + M g(chi)   (``phs.py``)

Fixed K (family F2): p = n^2/2 + c_K T ln(2 pi T) + T d(ln W)/dV |_{N,K}:

- ideal:    p = n^2/2 + c_K T ln(2 pi T) + n T;           dp/dn = n + T
- gaussian: p = n^2/2 + n T + c_K T [ln(T/n) + 1];        dp/dn = n + T (1 - d chi)
- phs:      gaussian + T c_K h(chi);        dp/dn adds -T d chi^2 h'(chi)

(d(V g(chi))/dV = g + chi g' = h since chi is proportional to V at fixed N, K;
dchi/dn = -chi/n.) Compressibility n T kappa_T = T / (dp/dn) (= S(0+) at fixed
K); negative values signal dp/dn < 0, i.e. the low-T expansion breaking down.

Fixed chi (family F1, Eqs. 76-78): Z = 1 + <Phi>/(N T) with
M/N = d chi (1 - 1/N) (d chi for N = None, the thermodynamic limit):
"harmonic" Z = 1 + M/N, "rpa" Z = 1 + (M/N) n / (n + T); per-particle manifold
entropy s_W = (ln W - N ln V)/N = 0, -d chi ln(2 pi n), d chi [g(chi) - ln(2 pi n)].

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import math
from typing import Optional, Tuple

# ----- numerics -----
import numpy as np
from numpy.typing import ArrayLike

# ----- project -----
from xla_cco.theory.geometry import SUPPORTED_DIMS, check_dim
from xla_cco.theory.phs import CHI_MAX, c_k, check_closure, pair_entropy_derivs

# ============================================================
# CONSTANTS
# ============================================================

ENTROPY_LEVELS: Tuple[str, ...] = ("ideal", "gaussian", "phs")
Z_LEVELS: Tuple[str, ...] = ("harmonic", "rpa")

# ============================================================
# VALIDATION
# ============================================================


def _positive(x: ArrayLike, name: str) -> np.ndarray:
    """Return ``x`` as a float array; raise ValueError unless finite and > 0."""
    arr = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"{name} must be finite and > 0")
    return arr


def _check_level(level: str, allowed: Tuple[str, ...]) -> str:
    """Return ``level`` if allowed; raise ValueError otherwise."""
    if level not in allowed:
        raise ValueError(f"level must be one of {allowed}, got {level!r}")
    return level


def _check_chi_array(chi: np.ndarray) -> np.ndarray:
    """Raise ValueError unless every chi lies in [0, 1/2); return chi."""
    if not np.all(np.isfinite(chi)) or np.any(chi < 0.0) or np.any(chi >= CHI_MAX):
        raise ValueError(f"chi must lie in [0, {CHI_MAX}) (disordered branch)")
    return chi


def _fixed_k_inputs(
    n: ArrayLike, T: ArrayLike, d: int, K: float, level: str, closure: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Validate and broadcast; return (n, T, chi, c_K, d)."""
    dim = check_dim(d, SUPPORTED_DIMS)
    _check_level(level, ENTROPY_LEVELS)
    check_closure(closure)
    n_b, t_b = np.broadcast_arrays(_positive(n, "n"), _positive(T, "T"))
    ck = c_k(K, dim)
    chi = _check_chi_array(ck / (n_b * dim))
    assert n_b.shape == t_b.shape == chi.shape, "broadcast failed"
    assert ck > 0.0 and np.all(chi > 0.0), "fixed K gives chi > 0"
    return n_b, t_b, chi, ck, dim


def _entropy_terms(
    chi: np.ndarray, d: int, closure: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return arrays (g, h, h') at every chi (one evaluation per unique chi)."""
    uniq, inverse = np.unique(chi, return_inverse=True)
    assert inverse.size == chi.size, "every chi needs a table row"
    table = np.empty((uniq.size, 3))
    for i in range(uniq.size):
        table[i] = pair_entropy_derivs(float(uniq[i]), d, closure)
    out = tuple(table[inverse.reshape(chi.shape), j] for j in range(3))
    assert all(o.shape == chi.shape for o in out), "shape must follow chi"
    return out[0], out[1], out[2]


# ============================================================
# FIXED K (FAMILY F2)
# ============================================================


def pressure_fixed_k(
    n: ArrayLike,
    T: ArrayLike,
    *,
    d: int,
    K: float = 1.0,
    level: str = "phs",
    closure: str = "py",
) -> np.ndarray:
    """Return p = n^2/2 + c_K T ln(2 pi T) + T d(ln W)/dV at fixed N and K.

    Parameters
    ----------
    n, T : ArrayLike
        Number density and temperature (> 0), broadcast together.
    d : int
        Dimension, 1, 2 or 3.
    K : float
        Constraint radius (> 0); chi = c_K / (n d) must lie in [0, 1/2).
    level : str
        "ideal", "gaussian" or "phs".
    closure : str
        PHS closure, "py" or "low_chi" (used by level "phs" only).

    Returns
    -------
    np.ndarray
        Pressure, broadcast shape of (n, T).
    """
    n_b, t_b, chi, ck, dim = _fixed_k_inputs(n, T, d, K, level, closure)
    dlnw = n_b.copy()  # ideal: d(N ln V)/dV = n
    if level != "ideal":
        dlnw = dlnw - ck * np.log(2.0 * math.pi * n_b) + ck
    if level == "phs":
        dlnw = dlnw + ck * _entropy_terms(chi, dim, closure)[1]
    out = 0.5 * n_b * n_b + ck * t_b * np.log(2.0 * math.pi * t_b) + t_b * dlnw
    assert out.shape == n_b.shape, "pressure must follow the broadcast shape"
    assert np.all(np.isfinite(out)), "pressure must be finite"
    return np.asarray(out, dtype=float)


def dp_dn_fixed_k(
    n: ArrayLike,
    T: ArrayLike,
    *,
    d: int,
    K: float = 1.0,
    level: str = "phs",
    closure: str = "py",
) -> np.ndarray:
    """Return (dp/dn)_{T,K}: n + T, n + T (1 - d chi), and - T d chi^2 h' for phs.

    Arguments as in ``pressure_fixed_k``.
    """
    n_b, t_b, chi, _, dim = _fixed_k_inputs(n, T, d, K, level, closure)
    coeff = np.ones_like(chi)  # T-coefficient of dp/dn
    if level != "ideal":
        coeff = coeff - dim * chi
    if level == "phs":
        coeff = coeff - dim * chi * chi * _entropy_terms(chi, dim, closure)[2]
    out = n_b + t_b * coeff
    assert out.shape == n_b.shape, "dp/dn must follow the broadcast shape"
    assert np.all(np.isfinite(out)), "dp/dn must be finite"
    return np.asarray(out, dtype=float)


def compressibility_fixed_k(
    n: ArrayLike,
    T: ArrayLike,
    *,
    d: int,
    K: float = 1.0,
    level: str = "phs",
    closure: str = "py",
) -> np.ndarray:
    """Return n T kappa_T = T / (dp/dn)_{T,K} (inf where dp/dn = 0).

    Arguments as in ``pressure_fixed_k``. Negative where dp/dn < 0.
    """
    _, t_b, _, _, _ = _fixed_k_inputs(n, T, d, K, level, closure)
    slope = dp_dn_fixed_k(n, T, d=d, K=K, level=level, closure=closure)
    with np.errstate(divide="ignore"):
        out = t_b / slope
    assert out.shape == slope.shape, "shape must follow dp/dn"
    assert not np.any(np.isnan(out)), "compressibility must not be NaN"
    return np.asarray(out, dtype=float)


# ============================================================
# FIXED CHI (FAMILY F1)
# ============================================================


def _modes_per_particle(chi: np.ndarray, d: int, N: Optional[int]) -> np.ndarray:
    """Return M/N = d chi (1 - 1/N), or d chi for N = None (thermodynamic limit)."""
    if N is None:
        return d * chi
    if isinstance(N, bool) or not isinstance(N, (int, np.integer)) or N < 2:
        raise ValueError(f"N must be an integer >= 2 or None, got {N!r}")
    out = d * chi * (1.0 - 1.0 / int(N))
    assert np.all(out >= 0.0), "mode count must be non-negative"
    return out


def z_fixed_chi(
    chi: ArrayLike,
    T: ArrayLike,
    *,
    d: int,
    n: float = 1.0,
    N: Optional[int] = None,
    level: str = "harmonic",
) -> np.ndarray:
    """Return Z = 1 + <Phi>/(N T) at fixed chi.

    Parameters
    ----------
    chi, T : ArrayLike
        Stealthiness in [0, 1/2) and temperature (> 0), broadcast together.
    d : int
        Dimension, 1, 2 or 3.
    n : float
        Number density (> 0); enters the "rpa" level only.
    N : Optional[int]
        Particle number (>= 2), or None for the thermodynamic limit.
    level : str
        "harmonic" (1 + M/N) or "rpa" (1 + (M/N) n/(n + T)).

    Returns
    -------
    np.ndarray
        Z, broadcast shape of (chi, T, n).
    """
    dim = check_dim(d, SUPPORTED_DIMS)
    _check_level(level, Z_LEVELS)
    chi_b, t_b, n_b = np.broadcast_arrays(
        _check_chi_array(np.asarray(chi, dtype=float)),
        _positive(T, "T"),
        _positive(n, "n"),
    )
    ratio = _modes_per_particle(chi_b, dim, N)
    if level == "rpa":
        ratio = ratio * n_b / (n_b + t_b)
    out = 1.0 + ratio
    assert out.shape == chi_b.shape, "Z must follow the broadcast shape"
    assert np.all(out >= 1.0), "Z must be >= 1"
    return np.asarray(out, dtype=float)


def manifold_entropy_fixed_chi(
    chi: ArrayLike,
    *,
    d: int,
    n: ArrayLike = 1.0,
    level: str = "phs",
    closure: str = "py",
) -> np.ndarray:
    """Return s_W = (ln W - N ln V)/N in the thermodynamic limit.

    ideal: 0; gaussian: -d chi ln(2 pi n); phs: d chi [g(chi) - ln(2 pi n)].
    chi in [0, 1/2) and n > 0 are broadcast together.
    """
    dim = check_dim(d, SUPPORTED_DIMS)
    _check_level(level, ENTROPY_LEVELS)
    check_closure(closure)
    chi_b, n_b = np.broadcast_arrays(
        _check_chi_array(np.asarray(chi, dtype=float)), _positive(n, "n")
    )
    out = np.zeros(chi_b.shape)
    if level != "ideal":
        out = -dim * chi_b * np.log(2.0 * math.pi * n_b)
    if level == "phs":
        out = out + dim * chi_b * _entropy_terms(chi_b, dim, closure)[0]
    assert out.shape == chi_b.shape, "s_W must follow the broadcast shape"
    assert np.all(np.isfinite(out)), "s_W must be finite"
    return np.asarray(out, dtype=float)
