#!/usr/bin/env python3

"""
Stealthy energy as an Equinox PyTree in fractional coordinates.

Phi = (1/V) sum_k w_k |rho_k|^2, rho_k = sum_j exp(2 pi i n_k . s_j); v = 1,
no factor 1/2, Shift = 0. ``__call__`` is plain jnp (autodiff to any order);
``value_and_grad`` is the analytic chunked fast path with dPhi/ds equal to
the C++ gradient -(F_i . a_j). When the integer bound ``n_max`` on |n_a| is
known (``make_stealthy_energy`` sets it), the fast path replaces the per-(k, j)
trig by per-axis tables of exp(2 pi i s_a)^p built with cumulative products.

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


def _power_table(s: Positions, n_max: int) -> Array:
    """Return z^p, p = 0..n_max, with z = exp(2 pi i s): shape (n_max + 1, n, d).

    Built by cumulative products, so the (k, j) phases cost no trig.
    """
    assert n_max >= 0, "n_max must be non-negative"
    z = jnp.cos(_TWO_PI * s) + 1j * jnp.sin(_TWO_PI * s)
    ones = jnp.ones((1,) + s.shape, dtype=z.dtype)
    rest = jnp.broadcast_to(z, (n_max,) + s.shape)
    out: Array = jnp.cumprod(jnp.concatenate([ones, rest], axis=0), axis=0)
    return out


def _phase_block(table: Array, n_block: Float[Array, "c d"]) -> Array:
    """Return exp(2 pi i n_k . s_j) for the rows of ``n_block``: shape (c, n).

    Per axis a the factor is table[|n_ka|, :, a], conjugated for n_ka < 0;
    an index beyond the table yields NaN rather than a wrong finite value.
    """
    assert table.ndim == 3, "table must have shape (p, n, d)"
    assert n_block.shape[1] == table.shape[2], "dimension mismatch"
    idx = jnp.abs(n_block).astype(jnp.int32)
    phase = None
    for a in range(n_block.shape[1]):
        fac = jnp.take(
            table[:, :, a], idx[:, a], axis=0, mode="fill", fill_value=jnp.nan
        )
        fac = jnp.where((n_block[:, a] < 0)[:, None], jnp.conj(fac), fac)
        phase = fac if phase is None else phase * fac
    assert phase is not None, "need at least one axis"
    return phase


def auto_chunk_size(
    n_particles: int,
    n_vectors: int,
    budget_bytes: int = 1 << 30,
    bytes_per_pair: int = 96,
) -> Optional[int]:
    """Return a ``chunk_size`` whose (c, N) work arrays fit ``budget_bytes``.

    Internal helper (the factory default stays unchunked): ``bytes_per_pair``
    is the live bytes per (k, j) pair of one block (a few complex128 arrays).
    Returns None when the whole (M, N) matrix already fits.
    """
    assert n_particles >= 1 and n_vectors >= 1, "sizes must be positive"
    assert budget_bytes >= 1 and bytes_per_pair >= 1, "budget must be positive"
    size = max(1, budget_bytes // (n_particles * bytes_per_pair))
    return None if size >= n_vectors else size


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

    def hessian(self, s: Positions) -> Float[Array, "nd nd"]:
        """Return d^2E/ds^2 as an (n d, n d) matrix (row-major flattening of s).

        The default is ``jax.hessian`` of ``__call__``; subclasses may override
        it with a closed form (its memory grows with the size of every
        intermediate times n d).
        """
        n, d = s.shape
        hess = jax.hessian(lambda y: self(y.reshape(n, d)))(s.reshape(-1))
        return cast(Float[Array, "nd nd"], hess)


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
    n_max : Optional[int]
        Static upper bound on max |n_ka|; enables the phase-table fast path
        (None = per-(k, j) trig). Set by ``make_stealthy_energy``.
    """

    n_vectors: Float[Array, "m d"]
    weights: Float[Array, "m"]
    volume: Float[Array, ""]
    chunk_size: Optional[int] = eqx.field(static=True, default=None)
    n_max: Optional[int] = eqx.field(static=True, default=None)

    def __check_init__(self) -> None:
        """Validate ``n_vectors`` and ``n_max`` (skipped for traced arrays).

        Raises
        ------
        ValueError
            If ``n_max`` is negative or smaller than max |n|, or if
            ``n_vectors`` is not integer-valued (the table path would
            truncate it silently).
        """
        if isinstance(self.n_vectors, jax.core.Tracer):
            return
        a = np.asarray(self.n_vectors)
        if self.n_max is not None and self.n_max < 0:
            raise ValueError(f"n_max must be non-negative, got {self.n_max}")
        if self.n_max is not None and a.size and np.abs(a).max() > self.n_max:
            raise ValueError(
                f"n_max={self.n_max} is smaller than max|n|={np.abs(a).max():g}"
            )
        if np.any(a != np.round(a)):
            raise ValueError("n_vectors must be integer-valued")

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
        if self.n_max is not None:
            return self._value_and_grad_tables(s, self.n_max)
        re, im = collective_density(s, self.n_vectors, self.chunk_size)
        value = jnp.sum(self.weights * (re * re + im * im)) / self.volume
        return value, self._gradient(s, re, im)

    def _value_and_grad_tables(
        self, s: Positions, n_max: int
    ) -> Tuple[Float[Array, ""], Positions]:
        """Phase-table fast path: one block (unchunked) or two scans (chunked)."""
        m = self.n_vectors.shape[0]
        size = m if self.chunk_size is None else min(self.chunk_size, m)
        n_chunks = -(-m // size)
        total = n_chunks * size
        table = _power_table(s, n_max)
        n_blk = _pad_rows(self.n_vectors, total).reshape(n_chunks, size, -1)
        w_blk = _pad_rows(self.weights, total).reshape(n_chunks, size)
        pref = 2.0 * _TWO_PI / self.volume

        def coef_of(ph: Array, w_c: Array, rho: Array) -> Array:
            return w_c[:, None] * (
                rho.imag[:, None] * ph.real - rho.real[:, None] * ph.imag
            )

        if n_chunks == 1:  # the (m, n) phase matrix is already resident: reuse it
            ph = _phase_block(table, n_blk[0])
            rho = jnp.sum(ph, axis=1)  # complex sum: fuses far better than re/im
            value = jnp.sum(w_blk[0] * (rho.real**2 + rho.imag**2)) / self.volume
            return value, pref * (coef_of(ph, w_blk[0], rho).T @ n_blk[0])

        def rho_body(_: None, n_c: Array) -> Tuple[None, Array]:
            return None, jnp.sum(_phase_block(table, n_c), axis=1)

        _, rho_blk = jax.lax.scan(rho_body, None, n_blk)
        re_blk, im_blk = rho_blk.real, rho_blk.imag
        value = jnp.sum(w_blk * (re_blk * re_blk + im_blk * im_blk)) / self.volume

        def grad_body(acc: Positions, blk: Tuple[Array, ...]) -> Tuple[Positions, None]:
            n_c, w_c, rho_c = blk
            ph = _phase_block(table, n_c)
            return acc + coef_of(ph, w_c, rho_c).T @ n_c, None

        init = jnp.zeros_like(s)
        acc, _ = jax.lax.scan(grad_body, init, (n_blk, w_blk, rho_blk))
        return value, pref * acc

    def hessian(self, s: Positions) -> Float[Array, "nd nd"]:
        """Closed-form Hessian, flattened row-major to (n d, n d).

        d^2 Phi / ds_ja ds_lb = (2 (2 pi)^2 / V) sum_k w_k n_ka n_kb
        [cos th_kj cos th_kl + sin th_kj sin th_kl
         - delta_jl (Re rho_k cos th_kj + Im rho_k sin th_kj)],
        i.e. two (n, m) x (m, n) products per (a, b) pair: O(n^2 m d^2) work
        and O(n m + n^2 d^2) memory, unlike autodiff (O(n^2 m d) memory).
        """
        require_float64(s, "s")
        assert s.ndim == 2, "s must have shape (n, d)"
        assert s.shape[1] == self.n_vectors.shape[1], "dimension mismatch"
        n, d = s.shape
        theta = _TWO_PI * (s @ self.n_vectors.T)  # (n, m)
        cos, sin = jnp.cos(theta), jnp.sin(theta)
        re, im = jnp.sum(cos, axis=0), jnp.sum(sin, axis=0)
        pref = 2.0 * _TWO_PI**2 / self.volume
        diag_core = re * cos + im * sin  # (n, m)
        blocks = []
        for a in range(d):
            row = []
            for b in range(d):
                wab = self.weights * self.n_vectors[:, a] * self.n_vectors[:, b]
                pair = (cos * wab) @ cos.T + (sin * wab) @ sin.T  # (n, n)
                row.append(pref * (pair - jnp.diag(diag_core @ wab)))
            blocks.append(jnp.stack(row, axis=-1))  # (n, n, d_b)
        hess4 = jnp.stack(blocks, axis=1)  # (n, d_a, n, d_b)
        return hess4.reshape(n * d, n * d)

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

    Notes
    -----
    The factory always selects the per-(k, j) trig kernel (``n_max=None``).
    The per-axis phase-table kernel (``n_max = max|n|``) is reachable by
    constructing ``StealthyEnergy(..., n_max=...)`` directly. It is faster in
    isolation (1.4x to 1.9x per lane at M = 101, 128) but not end to end in
    ``relax``: job 16992894 measured 1200 configurations at d=3, N=100 with
    the CLI settings on an A100 and found tables 1.03x faster at chi = 0.35
    (M = 101) and 1.08x slower at chi = 0.45 (M = 128), an ambiguous result
    that defaults to trig (benchmarks/RESULTS.md, ruling R32). Both kernels
    agree to 1e-9 in ensemble statistics (checked during development).

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
        n_max=None,
    )
