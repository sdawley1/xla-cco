#!/usr/bin/env python3

"""
Independent quadrature oracles for the theory tests (xla_cco.theory).

Nothing here calls the code under test: ball volumes come from math.gamma and
the overlap of two unit balls from slicing the lens into (d-1)-balls.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import math

# ----- numerics -----
from scipy import integrate

# ============================================================
# ORACLES
# ============================================================


def ball_volume(d: int) -> float:
    """Volume of the unit d-ball, pi^(d/2) / Gamma(d/2 + 1) (d >= 0)."""
    assert d >= 0, "d must be >= 0"
    out = math.pi ** (d / 2) / math.gamma(d / 2 + 1)
    assert out > 0.0, "volume must be positive"
    return out


def lens_volume_quad(x: float, d: int) -> float:
    """Overlap volume of two unit d-balls with centres x apart (0 <= x <= 2).

    Two caps of height 1 - x/2, each sliced into (d-1)-balls of radius
    sqrt(1 - t^2) for t in [x/2, 1]; scipy quad to ~1e-14.
    """
    assert 0.0 <= x <= 2.0 and d >= 1, "need 0 <= x <= 2 and d >= 1"
    cap, _ = integrate.quad(
        lambda t: ball_volume(d - 1) * (1.0 - t * t) ** ((d - 1) / 2),
        x / 2.0,
        1.0,
        epsabs=1e-14,
        epsrel=1e-13,
    )
    assert cap >= 0.0, "cap volume must be non-negative"
    return 2.0 * cap


def lens_fraction_quad(x: float, d: int) -> float:
    """alpha_d(x): the lens volume divided by the volume of one unit ball."""
    out = lens_volume_quad(x, d) / ball_volume(d)
    assert -1e-14 <= out <= 1.0 + 1e-14, "fraction must lie in [0, 1]"
    return out
