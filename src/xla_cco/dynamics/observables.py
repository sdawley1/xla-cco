#!/usr/bin/env python3

"""
Density waves along MD trajectories: rho_k, centre-of-mass correction, winding.

rho_k = sum_j exp(i k . r_j) for Cartesian r (unwrapped) and k; stored as
(..., 2) = (Re, Im). The constrained waves' rho equals the energy's
collective density (phase 2 pi n . s = k . r with s = r inv(B)).

``com_correct`` multiplies rho_k by exp(-i k . R_cm), removing the phase a
rigid translation adds (needed when the total momentum is not conserved,
Andersen and BAOAB). ``winding_update`` accumulates Delta theta =
atan2(Im(rho_new conj rho_old), Re(rho_new conj rho_old)) per wave:
wind_abs += |Delta theta|, wind_net += Delta theta, n_valid += 1, only where
``active`` and S = |rho|^2 / n_part exceeds ``phase_floor`` at both ends;
rho_prev is replaced by rho_new only where ``active``.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from typing import Optional, Tuple

# ----- numerics -----
from jax import Array
from jaxtyping import Bool, Float, Int

# ============================================================
# FUNCTIONS
# ============================================================


def wave_densities(
    r: Float[Array, "n d"], kvec: Float[Array, "w d"], chunk_size: Optional[int] = None
) -> Float[Array, "w 2"]:
    """rho_k (Re, Im) of every tracked wave at Cartesian positions r."""
    raise NotImplementedError


def center_of_mass(r: Float[Array, "n d"]) -> Float[Array, " d"]:
    """Mean position (mass 1, unwrapped coordinates)."""
    raise NotImplementedError


def com_correct(
    rho: Float[Array, "w 2"], kvec: Float[Array, "w d"], r_cm: Float[Array, " d"]
) -> Float[Array, "w 2"]:
    """rho_k exp(-i k . R_cm): translation-invariant densities."""
    raise NotImplementedError


def winding_update(
    rho_prev: Float[Array, "w 2"], rho_new: Float[Array, "w 2"],
    wind_abs: Float[Array, " w"], wind_net: Float[Array, " w"],
    n_valid: Int[Array, " w"], active: Bool[Array, " w"], n_part: int,
    phase_floor: float,
) -> Tuple[Float[Array, "w 2"], Float[Array, " w"], Float[Array, " w"],
           Int[Array, " w"]]:  # fmt: skip
    """Accumulate winding; returns (rho_prev', wind_abs', wind_net', n_valid')."""
    raise NotImplementedError
