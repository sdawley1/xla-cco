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
from typing import Optional, Tuple, cast

# ----- numerics -----
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float

# ----- project -----
from xla_cco._types import Positions, require_float64
from xla_cco.box import Box
from xla_cco.kvectors import KSet

# ============================================================
# HELPERS
# ============================================================

_TWO_PI = 2.0 * np.pi


def _density_block(
    s: Positions, n_block: Float[Array, "c d"]
) -> Tuple[Float[Array, "c"], Float[Array, "c"]]:
    """Return (Re rho, Im rho) for the k rows in ``n_block``."""
    theta = _TWO_PI * (n_block @ s.T)  # (c, N)
    return jnp.sum(jnp.cos(theta), axis=1), jnp.sum(jnp.sin(theta), axis=1)


def _pad_rows(x: Array, total: int) -> Array:
    """Append zero rows along axis 0 up to ``total`` rows."""
    pad = [(0, total - x.shape[0])] + [(0, 0)] * (x.ndim - 1)
    return jnp.pad(x, pad)


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
        out = jax.value_and_grad(self.__call__)(s)
        return cast(Tuple[Float[Array, ""], Positions], out)


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
        require_float64(s, "s")
        assert s.ndim == 2, "s must have shape (n, d)"
        assert s.shape[1] == self.n_vectors.shape[1], "dimension mismatch"
        re, im = collective_density(s, self.n_vectors, self.chunk_size)
        out: Array = jnp.sum(self.weights * (re * re + im * im)) / self.volume
        return out

    def value_and_grad(self, s: Positions) -> Tuple[Float[Array, ""], Positions]:
        """Return (Phi, dPhi/ds) with one chunked pass for rho and one for grad.

        dPhi/ds_j = (2/V) sum_k w_k (Im rho_k cos th_kj - Re rho_k sin th_kj)
        2 pi n_k, th_kj = 2 pi n_k . s_j; this equals -B F_j for the C++ force.
        """
        require_float64(s, "s")
        assert s.ndim == 2, "s must have shape (n, d)"
        assert s.shape[1] == self.n_vectors.shape[1], "dimension mismatch"
        re, im = collective_density(s, self.n_vectors, self.chunk_size)
        value = jnp.sum(self.weights * (re * re + im * im)) / self.volume
        return value, self._gradient(s, re, im)

    def _gradient(
        self, s: Positions, re: Float[Array, "m"], im: Float[Array, "m"]
    ) -> Positions:
        """Accumulate dPhi/ds over k-chunks with ``lax.scan``."""
        m = self.n_vectors.shape[0]
        size = m if self.chunk_size is None else min(self.chunk_size, m)
        n_chunks = -(-m // size)
        total = n_chunks * size
        n_blk = _pad_rows(self.n_vectors, total).reshape(n_chunks, size, -1)
        # zero-weight padding rows contribute exactly 0 whatever their rho
        w_blk = _pad_rows(self.weights, total).reshape(n_chunks, size)
        re_blk = _pad_rows(re, total).reshape(n_chunks, size)
        im_blk = _pad_rows(im, total).reshape(n_chunks, size)

        def body(acc: Positions, blk: Tuple[Array, ...]) -> Tuple[Positions, None]:
            n_c, w_c, re_c, im_c = blk
            theta = _TWO_PI * (s @ n_c.T)  # (N, c)
            coef = w_c * (im_c * jnp.cos(theta) - re_c * jnp.sin(theta))
            return acc + coef @ n_c, None

        init = jnp.zeros_like(s)
        acc, _ = jax.lax.scan(body, init, (n_blk, w_blk, re_blk, im_blk))
        out: Array = (2.0 * _TWO_PI / self.volume) * acc
        return out


def collective_density(
    s: Positions,
    n_vectors: Float[Array, "m d"],
    chunk_size: Optional[int] = None,
) -> Tuple[Float[Array, "m"], Float[Array, "m"]]:
    """Return (Re rho, Im rho) with rho_k = sum_j exp(2 pi i n_k . s_j).

    Parameters
    ----------
    s : Positions
        Fractional positions (N, d), float64.
    n_vectors : Float[Array, "m d"]
        Integer-valued float64 coefficients (M, d).
    chunk_size : Optional[int]
        Number of k rows per block (None = all at once); remainders are
        handled by ``jax.lax.map``.
    """
    require_float64(s, "s")
    require_float64(n_vectors, "n_vectors")
    assert s.shape[1] == n_vectors.shape[1], "dimension mismatch"
    assert chunk_size is None or chunk_size >= 1, "chunk_size must be >= 1"
    if chunk_size is None or chunk_size >= n_vectors.shape[0]:
        return _density_block(s, n_vectors)

    def one_row(n_row: Array) -> Tuple[Array, Array]:
        re, im = _density_block(s, n_row[None, :])
        return re[0], im[0]

    return cast(
        Tuple[Array, Array],
        jax.lax.map(one_row, n_vectors, batch_size=chunk_size),
    )


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

    Raises
    ------
    ValueError
        If the k-set is empty.
    """
    require_float64(box.basis, "box.basis")
    assert pad_to >= 1, "pad_to must be >= 1"
    assert chunk_size is None or chunk_size >= 1, "chunk_size must be >= 1"
    m = int(kset.n.shape[0])
    if m == 0:
        raise ValueError("empty k-set: no k-vectors in the shell")
    total = -(-m // pad_to) * pad_to
    n_vec = np.zeros((total, kset.n.shape[1]), dtype=np.float64)
    n_vec[:m] = kset.n
    wts = np.zeros(total, dtype=np.float64)
    wts[:m] = kset.weights
    return StealthyEnergy(
        n_vectors=jnp.asarray(n_vec),
        weights=jnp.asarray(wts),
        volume=jnp.asarray(box.volume),
        chunk_size=chunk_size,
    )
