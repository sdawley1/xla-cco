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
from typing import Optional, Tuple, cast

# ----- numerics -----
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Int

# ----- project -----
from xla_cco._types import require_float64

# ============================================================
# HELPERS
# ============================================================


def _density_rows(r: Float[Array, "n d"], k_block: Float[Array, "c d"]) -> Array:
    """(Re, Im) rho of the wavevectors in ``k_block``: shape (c, 2)."""
    assert k_block.ndim == 2 and r.ndim == 2, "k_block and r must be 2-D"
    assert k_block.shape[1] == r.shape[1], "dimension mismatch"
    theta = k_block @ r.T  # (c, n)
    return jnp.stack([jnp.sum(jnp.cos(theta), 1), jnp.sum(jnp.sin(theta), 1)], -1)


# ============================================================
# FUNCTIONS
# ============================================================


def wave_densities(
    r: Float[Array, "n d"], kvec: Float[Array, "w d"], chunk_size: Optional[int] = None
) -> Float[Array, "w 2"]:
    """rho_k (Re, Im) of every tracked wave at Cartesian positions r.

    ``chunk_size`` waves per block (``jax.lax.map``), so the (c, n) phase
    matrix never holds all w rows; None computes every wave at once.
    """
    require_float64(r, "r")
    require_float64(kvec, "kvec")
    assert r.ndim == 2 and kvec.ndim == 2, "r and kvec must be 2-D"
    assert r.shape[1] == kvec.shape[1], "dimension mismatch"
    assert chunk_size is None or chunk_size >= 1, "chunk_size must be >= 1"
    if chunk_size is None or chunk_size >= kvec.shape[0]:
        return _density_rows(r, kvec)

    def one_row(k_row: Array) -> Array:
        return _density_rows(r, k_row[None, :])[0]

    return cast(Array, jax.lax.map(one_row, kvec, batch_size=chunk_size))


def center_of_mass(r: Float[Array, "n d"]) -> Float[Array, " d"]:
    """Mean position (mass 1, unwrapped coordinates)."""
    require_float64(r, "r")
    assert r.ndim == 2, "r must have shape (n, d)"
    assert r.shape[0] >= 1, "need at least one particle"
    return jnp.mean(r, axis=0)


def com_correct(
    rho: Float[Array, "w 2"], kvec: Float[Array, "w d"], r_cm: Float[Array, " d"]
) -> Float[Array, "w 2"]:
    """rho_k exp(-i k . R_cm): translation-invariant densities."""
    assert rho.ndim == 2 and rho.shape[1] == 2, "rho must have shape (w, 2)"
    assert kvec.shape == (rho.shape[0], r_cm.shape[0]), "kvec must be (w, d)"
    phase = kvec @ r_cm
    c, s = jnp.cos(phase), jnp.sin(phase)
    re, im = rho[:, 0], rho[:, 1]
    return jnp.stack([re * c + im * s, im * c - re * s], axis=-1)


def winding_update(
    rho_prev: Float[Array, "w 2"], rho_new: Float[Array, "w 2"],
    wind_abs: Float[Array, " w"], wind_net: Float[Array, " w"],
    n_valid: Int[Array, " w"], active: Bool[Array, " w"], n_part: int,
    phase_floor: float,
) -> Tuple[Float[Array, "w 2"], Float[Array, " w"], Float[Array, " w"],
           Int[Array, " w"]]:  # fmt: skip
    """Accumulate winding; returns (rho_prev', wind_abs', wind_net', n_valid').

    Delta = atan2(Im p, Re p) with p = rho_new conj(rho_old) (no unwrapping
    of separate phases); a wave's step counts where it is ``active`` and
    S = |rho|^2 / n_part > ``phase_floor`` at both ends.
    """
    assert rho_prev.shape == rho_new.shape == (wind_abs.shape[0], 2), "shapes"
    assert wind_net.shape == n_valid.shape == active.shape == wind_abs.shape, "w"
    assert n_part >= 1, "n_part must be positive"
    o_re, o_im = rho_prev[:, 0], rho_prev[:, 1]
    n_re, n_im = rho_new[:, 0], rho_new[:, 1]
    delta = jnp.arctan2(n_im * o_re - n_re * o_im, n_re * o_re + n_im * o_im)
    s_old = (o_re * o_re + o_im * o_im) / n_part
    s_new = (n_re * n_re + n_im * n_im) / n_part
    ok = active & (s_old > phase_floor) & (s_new > phase_floor)
    return (
        jnp.where(active[:, None], rho_new, rho_prev),
        wind_abs + jnp.where(ok, jnp.abs(delta), 0.0),
        wind_net + jnp.where(ok, delta, 0.0),
        n_valid + ok.astype(n_valid.dtype),
    )
