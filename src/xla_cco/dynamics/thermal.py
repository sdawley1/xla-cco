#!/usr/bin/env python3

"""
Thermal sampling: Maxwell-Boltzmann velocities, kinetic energy, harmonic positions.

Mass 1, k_B = 1. ``harmonic_sample`` draws a canonical Gaussian displacement
of a ground state along the nonzero modes of the Cartesian Hessian (host
NumPy, one eigh), so that low-temperature runs start equilibrated in the
2M stiff directions (NVE is nearly integrable there).

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import math

# ----- numerics -----
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float, UInt32

# ----- project -----
from xla_cco._types import require_float64
from xla_cco.energy import AbstractEnergy

# ============================================================
# HELPERS
# ============================================================


def _require_key(key: Array) -> None:
    """Raise ``TypeError`` unless ``key`` is a raw uint32 (2,) PRNG key (static)."""
    if getattr(key, "dtype", None) != jnp.uint32 or getattr(key, "shape", ()) != (2,):
        raise TypeError("key must be a raw uint32 PRNG key of shape (2,)")


def _cartesian_hessian(
    energy: AbstractEnergy, s: np.ndarray, basis: np.ndarray
) -> np.ndarray:
    """Cartesian Hessian d^2 Phi / dr dr at r = s B, row-major (n d, n d).

    With s = r inv(B), ds_jc / dr_ja = inv(B)_ac, so
    H_r[(j, a), (l, b)] = sum_ce inv(B)_ac H_s[(j, c), (l, e)] inv(B)_be,
    i.e. (I (x) inv(B)) H_s (I (x) inv(B))^T; symmetrized against roundoff.
    """
    n, d = s.shape
    if basis.shape != (d, d) or not np.isfinite(basis).all():
        raise ValueError(f"basis must be a finite ({d}, {d}) array")
    inv = np.linalg.inv(basis)
    h_s = np.asarray(energy.hessian(jnp.asarray(s, dtype=jnp.float64)))
    assert h_s.shape == (n * d, n * d), f"hessian shape {h_s.shape}"
    h4 = np.einsum("ac,jcle,be->jalb", inv, h_s.reshape(n, d, n, d), inv)
    h_r = h4.reshape(n * d, n * d)
    assert np.isfinite(h_r).all(), "non-finite Cartesian Hessian"
    return np.asarray(0.5 * (h_r + h_r.T), dtype=np.float64)


# ============================================================
# FUNCTIONS
# ============================================================


def maxwell_boltzmann(
    key: UInt32[Array, " 2"], n: int, d: int, temperature: Float[Array, ""],
    remove_com: bool,
) -> Float[Array, "n d"]:  # fmt: skip
    """Velocities (n, d) ~ N(0, T) per component; zero total momentum if asked.

    With ``remove_com`` the mean velocity is subtracted (the result then has
    d(n - 1) degrees of freedom).

    Raises
    ------
    ValueError
        If ``n`` or ``d`` is not positive.
    TypeError
        If ``key`` is not a raw uint32 PRNG key of shape (2,).
    """
    if n < 1 or d < 1:
        raise ValueError(f"n and d must be positive, got n={n}, d={d}")
    _require_key(key)
    t = jnp.asarray(temperature, dtype=jnp.float64)
    v = jnp.sqrt(t) * jax.random.normal(key, (n, d), dtype=jnp.float64)
    if remove_com:
        v = v - jnp.mean(v, axis=0, keepdims=True)
    assert v.shape == (n, d) and v.dtype == jnp.float64, "bad velocity draw"
    return v


def kinetic_energy(v: Float[Array, "n d"]) -> Float[Array, ""]:
    """Kinetic energy sum_j |v_j|^2 / 2 (mass 1)."""
    require_float64(v, "v")
    assert v.ndim == 2, "v must have shape (n, d)"
    out: Array = 0.5 * jnp.sum(v * v)
    return out


def kinetic_temperature(v: Float[Array, "n d"], dof: int) -> Float[Array, ""]:
    """Instantaneous temperature 2 KE / dof (dof = d(n-1) without COM motion)."""
    if dof < 1:
        raise ValueError(f"dof must be positive, got {dof}")
    out: Array = 2.0 * kinetic_energy(v) / dof
    assert out.shape == (), "temperature must be a scalar"
    return out


def harmonic_sample(
    rng: np.random.Generator, energy: AbstractEnergy, s_gs: np.ndarray,
    basis: np.ndarray, temperature: float, zero_rel: float = 1e-9,
) -> np.ndarray:  # fmt: skip
    """Cartesian positions (n, d): ground state plus a canonical harmonic draw.

    The Cartesian Hessian H_r = (I (x) inv(B)) H_s (I (x) inv(B))^T is built
    from ``energy.hessian`` at ``s_gs``; modes with lambda > zero_rel *
    lambda_max get amplitude N(0, T / lambda), the rest none. Returns
    s_gs @ B + displacement (unwrapped).

    The draw uses ``rng.standard_normal(k)`` once, k = number of kept modes
    (ascending eigenvalue order).

    Raises
    ------
    ValueError
        On a negative or non-finite temperature, a basis that is not (d, d),
        or a Hessian without positive eigenvalues.
    """
    if not (math.isfinite(temperature) and temperature >= 0.0):
        raise ValueError(f"temperature must be finite and >= 0, got {temperature}")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    s = np.asarray(s_gs, dtype=np.float64)
    b = np.asarray(basis, dtype=np.float64)
    if s.ndim != 2:
        raise ValueError(f"s_gs must have shape (n, d), got {s.shape}")
    lam, vec = np.linalg.eigh(_cartesian_hessian(energy, s, b))
    if not lam[-1] > 0.0:
        raise ValueError("Hessian has no positive eigenvalue")
    keep = lam > zero_rel * lam[-1]
    amp = np.sqrt(temperature / lam[keep]) * rng.standard_normal(int(keep.sum()))
    out = s @ b + (vec[:, keep] @ amp).reshape(s.shape)
    assert out.shape == s.shape and np.isfinite(out).all(), "bad harmonic draw"
    return np.asarray(out, dtype=np.float64)
