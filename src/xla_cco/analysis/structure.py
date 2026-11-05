#!/usr/bin/env python3

"""
Structure factor S(k) = |rho(k)|^2 / N and its isotropic average.

Wavevectors are given as integer coefficients in the reciprocal basis.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import functools
import math
from typing import Tuple

# ----- numerics -----
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float

# ----- project -----
from xla_cco._types import BatchPositions, Positions, require_float64
from xla_cco.box import Box
from xla_cco.energy import collective_density
from xla_cco.kvectors import enumerate_kvectors

# ============================================================
# CONSTANTS
# ============================================================

K_CHUNK: int = 4096  # k rows per block in the isotropic sum (bounds memory)

# ============================================================
# STRUCTURE FACTOR
# ============================================================


def structure_factor(s: Positions, n_vectors: Float[Array, "m d"]) -> Float[Array, "m"]:
    """Return S(n) = |rho(n)|^2 / N for integer vectors ``n_vectors``."""
    require_float64(s, "s")
    require_float64(n_vectors, "n_vectors")
    assert s.ndim == 2 and n_vectors.ndim == 2, "expected 2-D arrays"
    re, im = collective_density(s, n_vectors)
    out: Array = (re * re + im * im) / s.shape[0]
    return out


def isotropic_structure_factor(
    s: BatchPositions,
    box: Box,
    k_dir_max: float = 5.0,
    k_max: float = 400.0,
    bin_width: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the binned isotropic S(|k|) averaged over a batch.

    Mirrors C++ ``IsotropicStructureFactor``: one k per primitive direction
    with |k| < ``k_dir_max`` (half space), extended by integer multiples to
    |k|^2 <= ``k_max``^2, binned by ``floor(|k| / bin_width)``.

    Parameters
    ----------
    s : BatchPositions
        Fractional positions (b, n, d).
    box : Box
        Periodic box.
    k_dir_max : float
        Cutoff up to which individual directions are kept.
    k_max : float
        Maximum |k| (inverse length); must exceed ``k_dir_max``.
    bin_width : float
        Radial bin width (inverse length).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        For every non-empty bin: mean |k|, mean S, standard error of the
        mean |k| (0 if its variance is not positive) and standard error of S.
        The last column is NOT clamped: a negative variance gives NaN, as in
        C++.

    Raises
    ------
    TypeError
        If ``s`` is not float64.
    ValueError
        On bad shapes, cutoffs or bin width.
    """
    require_float64(s, "s")
    if s.ndim != 3 or s.shape[2] != box.dim:
        raise ValueError(f"s must have shape (b, n, {box.dim}), got {s.shape}")
    if not (0.0 < k_dir_max < k_max and bin_width > 0.0):
        raise ValueError("need 0 < k_dir_max < k_max and bin_width > 0")
    n_vec, k_mod2 = _isotropic_kset(np.asarray(box.basis), k_dir_max, k_max)
    s_mean, s2_mean = _sk_moments(s, n_vec)
    return _bin_moments(np.sqrt(k_mod2), s_mean, s2_mean, int(s.shape[0]), bin_width)


# ============================================================
# HELPERS
# ============================================================


def _isotropic_kset(
    basis: np.ndarray, k_dir_max: float, k_max: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Return integer vectors (M, d) and |k|^2 (M,) in C++ ``GetKs`` order.

    Primitive directions (gcd of coefficients 1) of the half-space shell
    |k| < ``k_dir_max``, sorted by |k|^2, each followed by its multiples
    i k with |i k|^2 <= ``k_max``^2 (equality kept). This equals the C++
    direction-dedup rule: the shortest lattice vector along a direction is
    the primitive one.
    """
    assert basis.ndim == 2 and basis.shape[0] == basis.shape[1], "square basis"
    if basis.shape[0] == 1:  # C++ d = 1 branch: k = i b, independent of k_dir_max
        b = 2.0 * math.pi / basis[0, 0]
        n_prim, k_prim = np.ones((1, 1), dtype=np.int64), np.array([[b]])
        ksq_prim = k_prim[:, 0] ** 2
    else:
        kset = enumerate_kvectors(basis, 0.0, k_dir_max)
        prim = np.gcd.reduce(np.abs(kset.n), axis=1) == 1
        n_prim, k_prim, ksq_prim = kset.n[prim], kset.k[prim], kset.k_sq[prim]
    limit = k_max * k_max
    n_blocks, ksq_blocks = [], []
    for n_row, k_row, q in zip(n_prim, k_prim, ksq_prim, strict=True):
        mult = np.arange(1, int(math.sqrt(limit / q)) + 3, dtype=np.float64)
        kk = mult[:, None] * k_row[None, :]
        modulus2 = np.zeros(mult.shape[0])
        for c in range(kk.shape[1]):
            modulus2 = modulus2 + kk[:, c] * kk[:, c]
        ok = modulus2 <= limit
        n_blocks.append(mult[ok][:, None] * n_row[None, :].astype(np.float64))
        ksq_blocks.append(modulus2[ok])
    if not n_blocks:
        raise ValueError("no reciprocal-lattice vector below k_dir_max")
    return np.concatenate(n_blocks), np.concatenate(ksq_blocks)


@functools.partial(jax.jit, static_argnames=("chunk",))
def _sk_sums(batch: Array, n_all: Array, chunk: int) -> Tuple[Array, Array]:
    """Sum S and S^2 per k over configs (scan keeps memory O(N * chunk))."""
    n_part = batch.shape[1]

    def step(
        carry: Tuple[Array, Array], cfg: Array
    ) -> Tuple[Tuple[Array, Array], None]:
        re, im = collective_density(cfg, n_all, chunk)
        sk = (re * re + im * im) / n_part
        return (carry[0] + sk, carry[1] + sk * sk), None

    zero = jnp.zeros(n_all.shape[0])
    (tot, tot2), _ = jax.lax.scan(step, (zero, zero), batch)
    return tot, tot2


def _sk_moments(s: BatchPositions, n_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return the config-averaged S and S^2 per k, both (M,), float64."""
    n_cfg = int(s.shape[0])
    m = n_vec.shape[0]
    assert n_cfg >= 1 and s.shape[1] >= 1 and m >= 1, "empty input"
    chunk = min(K_CHUNK, m)
    n_pad = np.zeros((-(-m // chunk) * chunk, n_vec.shape[1]))
    n_pad[:m] = n_vec  # zero rows give S = N; sliced off below
    tot, tot2 = _sk_sums(jnp.asarray(s), jnp.asarray(n_pad), chunk)
    return np.asarray(tot)[:m] / n_cfg, np.asarray(tot2)[:m] / n_cfg


def _bin_moments(
    k_mod: np.ndarray,
    s_mean: np.ndarray,
    s2_mean: np.ndarray,
    n_cfg: int,
    bin_width: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bin per-k moments as C++ (sums over configs and k, empty bins dropped)."""
    assert k_mod.shape == s_mean.shape == s2_mean.shape, "shape mismatch"
    idx = np.floor(k_mod / bin_width).astype(np.int64)
    size = int(idx.max()) + 1
    cnt = np.bincount(idx, minlength=size) * float(n_cfg)
    keep = cnt != 0
    cnt = cnt[keep]

    def tot(values: np.ndarray) -> np.ndarray:
        sums: np.ndarray = np.bincount(idx, weights=values, minlength=size)
        return np.asarray(sums[keep] * n_cfg, dtype=np.float64)

    mean_k = tot(k_mod) / cnt
    mean_s = tot(s_mean) / cnt
    var_k = (tot(k_mod * k_mod) / cnt - mean_k**2) / cnt
    err_k = np.sqrt(np.where(var_k > 0.0, var_k, 0.0))
    with np.errstate(invalid="ignore"):
        err_s = np.sqrt((tot(s2_mean) / cnt - mean_s**2) / cnt)
    return mean_k, mean_s, err_k, err_s
