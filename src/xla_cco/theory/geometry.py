#!/usr/bin/env python3

"""
Ball geometry: unit-ball volumes and the scaled intersection volume alpha_d.

alpha_d(x) is the volume of the intersection of two balls of radius 1 whose
centres are x apart, divided by the volume of one ball: alpha_d(0) = 1 and
alpha_d(x) = 0 for x >= 2. It is the first-order density coefficient of the
hard-sphere pair correlation, g2(r) = Theta(r - 1) [1 + 2^d eta alpha_d(r)]
for diameter 1 at packing fraction eta (Torquato & Stillinger, PRE 2006).

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
from scipy import special  # type: ignore[import-untyped]

# ============================================================
# CONSTANTS
# ============================================================

SUPPORTED_DIMS: Tuple[int, ...] = (1, 2, 3)  # dimensions of the PY/PHS layer

# ============================================================
# VALIDATION
# ============================================================


def check_dim(d: object, allowed: Optional[Tuple[int, ...]] = None) -> int:
    """Return ``d`` as an int, or raise ValueError (bools and floats rejected)."""
    if isinstance(d, bool) or not isinstance(d, (int, np.integer)) or int(d) < 1:
        raise ValueError(f"dimension must be an integer >= 1, got {d!r}")
    if allowed is not None and int(d) not in allowed:
        raise ValueError(f"dimension must be one of {allowed}, got {d}")
    return int(d)


def check_nonnegative(x: ArrayLike, name: str) -> np.ndarray:
    """Return ``x`` as a float array; raise ValueError if not finite and >= 0."""
    arr = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError(f"{name} must be finite and >= 0")
    return arr


# ============================================================
# GEOMETRY
# ============================================================


def unit_ball_volume(d: int) -> float:
    """Return v1(d) = pi^(d/2) / Gamma(1 + d/2), the volume of the unit d-ball."""
    dim = check_dim(d)
    out = math.pi ** (dim / 2.0) / math.gamma(1.0 + dim / 2.0)
    assert out > 0.0 and math.isfinite(out), "unit-ball volume must be positive"
    return float(out)


def intersection_fraction(x: ArrayLike, d: int) -> np.ndarray:
    """Return alpha_d(x), the scaled intersection volume of two unit balls.

    Closed forms for d <= 3; the regularized incomplete beta function
    I_{1 - x^2/4}((d + 1)/2, 1/2) otherwise.

    Parameters
    ----------
    x : ArrayLike
        Centre separations (>= 0, finite), in units of the ball radius.
    d : int
        Spatial dimension (>= 1).

    Returns
    -------
    np.ndarray
        alpha_d(x), same shape as ``x``, in [0, 1].

    Raises
    ------
    ValueError
        If ``x`` is negative or not finite, or ``d`` is not an integer >= 1.
    """
    dim = check_dim(d)
    xa = check_nonnegative(x, "x")
    t = np.minimum(xa, 2.0) / 2.0  # half separation, clipped to the support
    if dim == 1:
        out = 1.0 - t
    elif dim == 2:
        out = (2.0 / math.pi) * (np.arccos(t) - t * np.sqrt(1.0 - t * t))
    elif dim == 3:
        out = 1.0 - 1.5 * t + 0.5 * t**3
    else:
        out = _intersection_fraction_beta(xa, dim)
    out = np.where(xa >= 2.0, 0.0, out)
    assert out.shape == xa.shape, "shape must be preserved"
    assert np.all((out >= -1e-15) & (out <= 1.0 + 1e-15)), "alpha out of [0, 1]"
    return np.clip(out, 0.0, 1.0)


def _intersection_fraction_beta(x: ArrayLike, d: int) -> np.ndarray:
    """Return alpha_d(x) = I_{1 - x^2/4}((d + 1)/2, 1/2), any d (0 for x >= 2).

    For x^2/4 < 1/2 the complement 1 - I_{x^2/4}(1/2, (d + 1)/2) is used, so
    1 - x^2/4 is never rounded where I is steep (D-092).
    """
    dim = check_dim(d)
    xa = check_nonnegative(x, "x")
    w = np.minimum(xa * xa / 4.0, 1.0)  # 1 - z, exact up to one rounding
    a = (dim + 1) / 2.0
    direct = special.betainc(a, 0.5, 1.0 - w)
    complement = 1.0 - special.betainc(0.5, a, w)
    out = np.where(xa >= 2.0, 0.0, np.where(w < 0.5, complement, direct))
    assert out.shape == xa.shape, "shape must be preserved"
    assert np.all((out >= 0.0) & (out <= 1.0)), "alpha out of [0, 1]"
    return np.asarray(out, dtype=float)
