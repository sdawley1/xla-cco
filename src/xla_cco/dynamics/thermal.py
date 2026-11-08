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

# ----- numerics -----
import numpy as np
from jax import Array
from jaxtyping import Float, UInt32

# ----- project -----
from xla_cco.energy import AbstractEnergy

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
    """
    raise NotImplementedError


def kinetic_energy(v: Float[Array, "n d"]) -> Float[Array, ""]:
    """Kinetic energy sum_j |v_j|^2 / 2 (mass 1)."""
    raise NotImplementedError


def kinetic_temperature(v: Float[Array, "n d"], dof: int) -> Float[Array, ""]:
    """Instantaneous temperature 2 KE / dof (dof = d(n-1) without COM motion)."""
    raise NotImplementedError


def harmonic_sample(
    rng: np.random.Generator, energy: AbstractEnergy, s_gs: np.ndarray,
    basis: np.ndarray, temperature: float, zero_rel: float = 1e-9,
) -> np.ndarray:  # fmt: skip
    """Cartesian positions (n, d): ground state plus a canonical harmonic draw.

    The Cartesian Hessian H_r = (I (x) inv(B)) H_s (I (x) inv(B))^T is built
    from ``energy.hessian`` at ``s_gs``; modes with lambda > zero_rel *
    lambda_max get amplitude N(0, T / lambda), the rest none. Returns
    s_gs @ B + displacement (unwrapped).
    """
    raise NotImplementedError
