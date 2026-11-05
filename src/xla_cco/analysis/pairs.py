#!/usr/bin/env python3

"""
Pair statistics: nearest-neighbour distances, histograms and g2.

Distances use the minimum-image convention in length units with density 1.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import itertools

# ----- numerics -----
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float

# ----- project -----
from xla_cco._types import BatchPositions, Positions, require_float64
from xla_cco.box import Box
from xla_cco.kvectors import unit_ball_volume

# ============================================================
# CONSTANTS
# ============================================================

ROW_CHUNK: int = 32  # particles per row block; memory is O(ROW_CHUNK * N)

# ============================================================
# PAIR STATISTICS
# ============================================================


def nearest_neighbor_distances(s: Positions, box: Box) -> Float[Array, "n"]:
    """Return minimum-image nearest-neighbour distance of every particle.

    Uses ``ds - round(ds)`` on fractional differences, then scans all 3^d
    neighbouring image shifts. A lone particle (N = 1) sees its own nearest
    periodic image.

    Preconditions (not checked): positions are wrapped to [0, 1), and the
    cell is reduced (Minkowski/Gauss-reduced). In a strongly non-reduced
    cell the nearest image can lie beyond the scanned shifts, so distances
    are overestimated (errors up to ~1.9 were measured for skew of
    2-30 x |a1|).

    Raises
    ------
    TypeError
        If ``s`` is not float64.
    """
    require_float64(s, "s")
    assert s.ndim == 2 and s.shape[1] == box.dim, "s must have shape (n, d)"
    out: Array = _nn_jit(s, box.basis)
    return out


def pair_distance_histogram(
    s: BatchPositions, box: Box, edges: np.ndarray
) -> np.ndarray:
    """Return summed pair-distance counts over the batch for bin ``edges``.

    Every unordered pair i < j is counted once per configuration, using the
    minimum-image distance. Binning follows ``np.histogram``: bins are
    left-closed, [e_k, e_{k+1}), except the last, which is closed. Distances
    outside [edges[0], edges[-1]] are not counted. As for
    ``nearest_neighbor_distances``, positions must be wrapped to [0, 1) and
    the cell must be reduced (Minkowski/Gauss-reduced); otherwise distances
    are overestimated.

    Parameters
    ----------
    s : BatchPositions
        Fractional positions (b, n, d), float64.
    box : Box
        Periodic box.
    edges : np.ndarray
        Strictly increasing, non-negative bin edges (len >= 2).

    Returns
    -------
    np.ndarray
        int64 counts, shape (len(edges) - 1,).

    Raises
    ------
    TypeError
        If ``s`` is not float64.
    ValueError
        If ``edges`` is not 1-D, strictly increasing and non-negative.
    """
    require_float64(s, "s")
    _check_edges(edges)
    assert s.ndim == 3 and s.shape[2] == box.dim, "s must have shape (b, n, d)"
    counts = _hist_jit(jnp.asarray(s), box.basis, jnp.asarray(edges, dtype=jnp.float64))
    return np.rint(np.asarray(counts)).astype(np.int64)


def g2_from_histogram(
    counts: np.ndarray,
    edges: np.ndarray,
    n_particles: int,
    n_configs: int,
    volume: float,
    dim: int,
) -> np.ndarray:
    """Normalize pair counts to g2(r) (shell volume, density N / V).

    g2 = counts / [(V_d r_hi^d - V_d r_lo^d) * 0.5 * rho * N * n_configs]
    with rho = N / V and pairs counted once (C++ ``PairCorrelation``).
    """
    _check_edges(edges)
    if counts.shape != (edges.shape[0] - 1,):
        raise ValueError("counts must have length len(edges) - 1")
    assert n_particles >= 1 and n_configs >= 1 and dim >= 1, "bad sizes"
    assert volume > 0.0, "volume must be positive"
    v_d = unit_ball_volume(dim)
    shell = v_d * edges[1:] ** dim - v_d * edges[:-1] ** dim
    rho = n_particles / volume
    return np.asarray(counts, dtype=np.float64) / (
        shell * 0.5 * rho * n_particles * n_configs
    )


# ============================================================
# HELPERS
# ============================================================


@jax.jit
def _nn_jit(s: Array, basis: Array) -> Array:
    """Nearest-neighbour distances, scanning row blocks with ``lax.map``."""
    n = s.shape[0]
    out = jax.lax.map(
        lambda rows: jnp.min(_block_distances(s, basis, rows), axis=1),
        _row_blocks(n),
    )
    return jnp.asarray(out).reshape(-1)[:n]


@jax.jit
def _hist_jit(s: Array, basis: Array, edges: Array) -> Array:
    """Pair-distance counts summed over configs and row blocks."""
    n = s.shape[1]
    n_bins = edges.shape[0] - 1
    blocks = _row_blocks(n)

    def one_config(cfg: Array) -> Array:
        def one_block(rows: Array) -> Array:
            dist = _block_distances(cfg, basis, rows)
            upper = jnp.arange(n)[None, :] > rows[:, None]
            return _bin_counts(dist, upper, edges, n_bins)

        return jnp.asarray(jax.lax.map(one_block, blocks)).sum(axis=0)

    return jnp.asarray(jax.lax.map(one_config, s)).sum(axis=0)


def _check_edges(edges: np.ndarray) -> None:
    """Raise ``ValueError`` unless edges are 1-D, increasing and >= 0."""
    edges = np.asarray(edges)
    if edges.ndim != 1 or edges.shape[0] < 2:
        raise ValueError("edges must be 1-D with at least 2 entries")
    if not np.all(np.isfinite(edges)) or edges[0] < 0.0:
        raise ValueError("edges must be finite and non-negative")
    if not np.all(np.diff(edges) > 0.0):
        raise ValueError("edges must be strictly increasing")


def _row_blocks(n: int) -> np.ndarray:
    """Return row indices (n_blocks, ROW_CHUNK), padded with the last row."""
    assert n >= 1, "need at least one particle"
    n_blocks = -(-n // ROW_CHUNK)
    idx = np.minimum(np.arange(n_blocks * ROW_CHUNK), n - 1)
    return idx.reshape(n_blocks, ROW_CHUNK)


def _block_distances(s: Array, basis: Array, rows: Array) -> Array:
    """Return min-image distances (len(rows), N); self at zero shift is inf."""
    n, dim = s.shape
    ds = s[None, :, :] - s[rows][:, None, :]
    ds = ds - jnp.round(ds)
    is_self = rows[:, None] == jnp.arange(n)[None, :]
    best = jnp.full(is_self.shape, jnp.inf)
    for shift in itertools.product((-1.0, 0.0, 1.0), repeat=dim):
        vec = (ds + jnp.asarray(shift)) @ basis
        r2 = jnp.sum(vec * vec, axis=-1)
        if not any(shift):
            r2 = jnp.where(is_self, jnp.inf, r2)
        best = jnp.minimum(best, r2)
    return jnp.sqrt(best)


def _bin_counts(dist: Array, mask: Array, edges: Array, n_bins: int) -> Array:
    """Count masked distances per bin (np.histogram semantics)."""
    idx = jnp.searchsorted(edges, dist, side="right") - 1
    idx = jnp.where(dist == edges[-1], n_bins - 1, idx)  # last bin is closed
    valid = mask & (idx >= 0) & (idx < n_bins) & jnp.isfinite(dist)
    weights = valid.astype(jnp.float64).reshape(-1)
    safe = jnp.where(valid, idx, 0).reshape(-1)
    return jnp.bincount(safe, weights=weights, length=n_bins)
