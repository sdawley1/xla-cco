#!/usr/bin/env python3

"""
Stealthy energy as an Equinox PyTree in fractional coordinates.

Phi = (1/V) sum_k w_k |rho_k|^2, rho_k = sum_j exp(2 pi i n_k . s_j); v = 1,
no factor 1/2, Shift = 0. ``__call__`` is plain jnp (autodiff to any order);
``value_and_grad`` is the analytic chunked fast path with dPhi/ds equal to
the C++ gradient -(F_i . a_j).

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import abc
from typing import Optional, Tuple

# ----- numerics -----
import equinox as eqx
from jax import Array
from jaxtyping import Float

# ----- project -----
from xla_cco._types import Positions
from xla_cco.box import Box
from xla_cco.kvectors import KSet

# ============================================================
# ENERGIES
# ============================================================


class AbstractEnergy(eqx.Module):
    """Extension point for energies on fractional positions."""

    @abc.abstractmethod
    def __call__(self, s: Positions) -> Float[Array, ""]:
        """Return the scalar energy for fractional positions ``s``."""
        raise NotImplementedError

    def value_and_grad(self, s: Positions) -> Tuple[Float[Array, ""], Positions]:
        """Return (energy, dE/ds); default is ``jax.value_and_grad``."""
        raise NotImplementedError


class StealthyEnergy(AbstractEnergy):
    """Stealthy energy over a padded k-set.

    Attributes
    ----------
    n_vectors : Float[Array, "m d"]
        Integer-valued float64 reciprocal coefficients; zero rows = padding.
    weights : Float[Array, "m"]
        Per-vector weights, 0.0 on padding rows.
    volume : Float[Array, ""]
        Box volume V.
    chunk_size : Optional[int]
        Static chunk length over k for ``value_and_grad`` (None = unchunked).
    """

    n_vectors: Float[Array, "m d"]
    weights: Float[Array, "m"]
    volume: Float[Array, ""]
    chunk_size: Optional[int] = eqx.field(static=True, default=None)

    def __call__(self, s: Positions) -> Float[Array, ""]:
        """Return Phi(s) (dimensionless, density 1)."""
        raise NotImplementedError

    def value_and_grad(self, s: Positions) -> Tuple[Float[Array, ""], Positions]:
        """Return (Phi, dPhi/ds) with one chunked pass for rho and one for grad."""
        raise NotImplementedError


def collective_density(
    s: Positions,
    n_vectors: Float[Array, "m d"],
    chunk_size: Optional[int] = None,
) -> Tuple[Float[Array, "m"], Float[Array, "m"]]:
    """Return (Re rho, Im rho) with rho_k = sum_j exp(2 pi i n_k . s_j)."""
    raise NotImplementedError


def make_stealthy_energy(
    box: Box,
    kset: KSet,
    chunk_size: Optional[int] = None,
    pad_to: int = 64,
) -> StealthyEnergy:
    """Build a ``StealthyEnergy`` from a box and k-set.

    Parameters
    ----------
    box : Box
        Periodic box (supplies the volume).
    kset : KSet
        Half-space k-set.
    chunk_size : Optional[int]
        Static chunk length over k, or None.
    pad_to : int
        Pad M up to a multiple of this value with zero-weight rows.
    """
    raise NotImplementedError
