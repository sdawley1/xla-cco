#!/usr/bin/env python3

"""
Periodic simulation box: fractional/Cartesian maps and reciprocal basis.

Convention: ``basis`` rows are lattice vectors a_i (density 1, so the box is
the unit basis times N^(1/d) and V = N). Reciprocal rows satisfy
b_i . a_j = 2 pi delta_ij.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- numerics -----
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float

# ----- project -----
from xla_cco._types import Positions

# ============================================================
# BOX
# ============================================================


class Box(eqx.Module):
    """Periodic box defined by a (d, d) lattice basis (rows = a_i).

    Attributes
    ----------
    basis : Float[Array, "d d"]
        Lattice vectors as rows, in length units with density 1.
    """

    basis: Float[Array, "d d"]

    @property
    def dim(self) -> int:
        """Spatial dimension d."""
        return int(self.basis.shape[0])

    @property
    def volume(self) -> Float[Array, ""]:
        """Box volume |det(basis)|."""
        return jnp.abs(jnp.linalg.det(self.basis))

    def reciprocal(self) -> Float[Array, "d d"]:
        """Return reciprocal rows b_i with b_i . a_j = 2 pi delta_ij."""
        out: Array = 2.0 * jnp.pi * jnp.linalg.inv(self.basis).T
        return out

    def to_cartesian(self, s: Positions) -> Float[Array, "n d"]:
        """Map fractional positions to Cartesian, r = s @ basis."""
        out: Array = s @ self.basis
        return out

    def to_fractional(self, r: Float[Array, "n d"]) -> Positions:
        """Map Cartesian positions to fractional, s = r @ inv(basis)."""
        out: Array = r @ jnp.linalg.inv(self.basis)
        return out

    def cartesian_forces(self, grad_s: Positions) -> Float[Array, "n d"]:
        """Convert dPhi/ds to Cartesian forces, F = -grad_s @ inv(B).T."""
        out: Array = -grad_s @ jnp.linalg.inv(self.basis).T
        return out


def cubic_box(n_particles: int, dim: int) -> Box:
    """Return the cubic box of side N^(1/d) (density 1, V = N).

    Parameters
    ----------
    n_particles : int
        Number of particles N >= 1.
    dim : int
        Spatial dimension d in {1, 2, 3}.
    """
    assert n_particles >= 1, "n_particles must be >= 1"
    assert dim in (1, 2, 3), "dim must be 1, 2 or 3"
    side = float(n_particles) ** (1.0 / dim)
    return Box(basis=jnp.asarray(side * np.eye(dim), dtype=jnp.float64))


def wrap_fractional(s: Positions) -> Positions:
    """Wrap fractional coordinates into [0, 1)."""
    wrapped = s - jnp.floor(s)
    # s - floor(s) rounds to exactly 1.0 for tiny negative s; keep strictly < 1
    below_one = jnp.nextafter(jnp.asarray(1.0, dtype=wrapped.dtype), 0.0)
    return jnp.minimum(wrapped, below_one)
