#!/usr/bin/env python3

"""
Wave-vector lattices and density-built couplings of 2D rectangular cells.

The coupling and loop analyses (coupling_mp_*, coupling_phonon_*, loop_*)
work on integer wave vectors n with k = 2 pi (n_x / L_x, n_y / L_y) in a
rectangular cell (diagonal basis, V = L_x L_y), the constrained set
0 < |k| < K, and the density rho(k) = sum_j exp(i k.r_j) on the integer
lattice |n_i| <= 2 nmax_i, nmax_i = ceil(K L_i / 2 pi), which holds every
pair q -+ p of constrained waves. A scalar `side` means a square cell; the
square-cell results are bitwise those of the earlier square-only helpers.

NumPy only (no plotting), so the helpers can be tested in the project
environment.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from pathlib import Path
from typing import List, Sequence, Union

# ----- numerics -----
import numpy as np

# ============================================================
# CONSTANTS
# ============================================================

Side = Union[float, np.ndarray]  # L (square cell) or (L_x, L_y)

# ============================================================
# CELL AND LATTICE
# ============================================================


def cell_lengths(basis: Sequence) -> np.ndarray:
    """(L_x, L_y) of a 2D rectangular cell; sheared cells are rejected."""
    b = np.asarray(basis, dtype=np.float64)
    assert b.shape == (2, 2), "basis must be (2, 2)"
    assert b[0, 1] == 0.0 and b[1, 0] == 0.0, "rectangular (diagonal) cells only"
    out: np.ndarray = np.diag(b).copy()
    assert np.all(out > 0.0), "cell lengths must be positive"
    return out


def axis_lengths(side: Side) -> np.ndarray:
    """(L_x, L_y) from a scalar side (square cell) or a length pair."""
    out = np.broadcast_to(np.asarray(side, dtype=np.float64), (2,)).copy()
    assert out.shape == (2,) and np.all(out > 0.0), "two positive lengths"
    return out


def lattice_limits(side: Side, k_cut: float) -> np.ndarray:
    """Integer reach per axis, nmax_i = ceil(K L_i / 2 pi)."""
    out = np.ceil(k_cut * axis_lengths(side) / (2.0 * np.pi)).astype(int)
    assert out.shape == (2,) and np.all(out >= 1) and k_cut > 0.0, "limits"
    return out


def _limits(nmax: Union[int, np.ndarray]) -> np.ndarray:
    """nmax as an integer pair (a scalar means the same reach on both axes)."""
    out = np.broadcast_to(np.asarray(nmax, dtype=int), (2,)).copy()
    assert np.all(out >= 1), "positive reach"
    return out


def lattice_rho(x: np.ndarray, side: Side, nmax: Union[int, np.ndarray]) -> np.ndarray:
    """rho(k) = sum_j exp(i k.r_j) on the lattice |n_i| <= 2 nmax_i.

    The lattice is (4 nx + 1, 4 ny + 1); index [i, j] holds n = (i - 2 nx,
    j - 2 ny); x is (N, 2) Cartesian.
    """
    lengths, lim = axis_lengths(side), _limits(nmax)
    kx = 2.0 * np.pi * np.arange(-2 * lim[0], 2 * lim[0] + 1) / lengths[0]
    ky = 2.0 * np.pi * np.arange(-2 * lim[1], 2 * lim[1] + 1) / lengths[1]
    out: np.ndarray = np.exp(1j * np.outer(x[:, 0], kx)).T @ np.exp(
        1j * np.outer(x[:, 1], ky)
    )
    assert out.shape == (kx.size, ky.size) and x.shape[1] == 2, "2D lattice"
    return out


def lattice_kmag(nmax: Union[int, np.ndarray], side: Side) -> np.ndarray:
    """|k| on the rho lattice of lattice_rho."""
    lengths, lim = axis_lengths(side), _limits(nmax)
    nx = np.arange(-2 * lim[0], 2 * lim[0] + 1)
    ny = np.arange(-2 * lim[1], 2 * lim[1] + 1) * (lengths[0] / lengths[1])
    out = 2.0 * np.pi / lengths[0] * np.hypot(*np.meshgrid(nx, ny, indexing="ij"))
    assert out.shape == (nx.size, ny.size), "lattice"
    return out


def wave_vectors(n: np.ndarray, side: Side) -> np.ndarray:
    """Cartesian k = 2 pi n / L (per axis) of integer vectors n (..., 2)."""
    out: np.ndarray = n * 2.0 * np.pi / axis_lengths(side)
    assert out.shape == n.shape and n.shape[-1] == 2, "2D vectors"
    return out


def scaled(n: np.ndarray, side: Side) -> np.ndarray:
    """n with n_y rescaled by L_x / L_y: |scaled(n)| = |k| L_x / 2 pi.

    Exactly n (as floats) in a square cell, so norms and dot products of
    the scaled vectors keep the square-cell rounding.
    """
    lengths = axis_lengths(side)
    out: np.ndarray = n * (lengths[0] / lengths)
    assert out.shape == n.shape, "same shape"
    return out


def waves(side: Side, k_cut: float) -> np.ndarray:
    """Integer vectors n of the half-space waves 0 < |k| < K, sorted by |k|."""
    lim = lattice_limits(side, k_cut)
    g = np.stack(np.meshgrid(*[np.arange(-v, v + 1) for v in lim], indexing="ij"), -1)
    n = g.reshape(-1, 2)
    n = n[(n[:, 0] > 0) | ((n[:, 0] == 0) & (n[:, 1] > 0))]
    n = n[np.linalg.norm(wave_vectors(n, side), axis=1) < k_cut]
    out = n[np.argsort(np.linalg.norm(scaled(n, side), axis=1), kind="stable")]
    assert out.ndim == 2 and out.shape[0] > 0, "waves"
    return out


def full_waves(side: Side, k_cut: float) -> np.ndarray:
    """(2M, 2) integer wave vectors {+q, -q}, q the half-space set with |q| < K."""
    nq = waves(side, k_cut)
    out = np.concatenate([nq, -nq])
    assert out.shape[0] == 2 * nq.shape[0], "both signs"
    return out


# ============================================================
# COUPLINGS FROM RHO
# ============================================================


def _pair_rho(rho: np.ndarray, nq: np.ndarray, sign: int) -> np.ndarray:
    """rho(q + sign p) for every pair of integer waves (q, p)."""
    off = (np.asarray(rho.shape) - 1) // 2
    d = nq[:, None, :] + sign * nq[None] + off
    assert np.all(d >= 0) and np.all(d < np.asarray(rho.shape)), "inside the lattice"
    out: np.ndarray = rho[d[..., 0], d[..., 1]]
    return out


def k_matrix(rho: np.ndarray, nq: np.ndarray, side: Side) -> np.ndarray:
    """K = (2/V) J J^T from rho at q -+ p (Re, Im coordinates)."""
    lengths = axis_lengths(side)
    kv = wave_vectors(nq, lengths)
    qp, vol = kv @ kv.T, lengths[0] * lengths[1]
    rm, rp = _pair_rho(rho, nq, -1), _pair_rho(rho, nq, 1)
    ri = -qp / vol * (rp.imag + rm.imag)
    out = np.block([[qp / vol * (rm.real - rp.real), ri],
                    [ri.T, qp / vol * (rm.real + rp.real)]])  # fmt: skip
    assert np.allclose(out, out.T), "symmetric"
    return out


def coupling_from_rho(rho: np.ndarray, nq: np.ndarray, side: Side) -> np.ndarray:
    """C = D^-1/2 K D^-1/2 from rho at q -+ p (D = diag K)."""
    k = k_matrix(rho, nq, side)
    d = np.sqrt(np.diag(k))
    assert np.all(d > 0.0), "positive diagonal"
    out: np.ndarray = k / np.outer(d, d)
    return out


def h_matrix(rho: np.ndarray, nfull: np.ndarray, side: Side) -> np.ndarray:
    """Hermitian H_ab = (Q_a . Q_b) rho(Q_a - Q_b) / V in the complex basis {+q, -q}."""
    lengths = axis_lengths(side)
    kv = wave_vectors(nfull, lengths)
    vol = lengths[0] * lengths[1]
    out: np.ndarray = (kv @ kv.T) * _pair_rho(rho, nfull, -1) / vol
    assert np.allclose(out, out.conj().T), "Hermitian"
    return out


# ============================================================
# RANDOM-PHASE MODELS
# ============================================================


def _centre(shape: Sequence[int]) -> tuple:
    """Index of k = 0 on an odd-sized lattice."""
    assert len(shape) == 2 and all(s % 2 == 1 for s in shape), "odd lattice"
    return (shape[0] // 2, shape[1] // 2)


def phase_randomized(rho: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Same |rho(k)|, independent uniform phases (Hermitian), rho(0) kept."""
    c = _centre(rho.shape)
    z = np.abs(rho) * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, rho.shape))
    # k -> -k reverses the flattened (odd-sized) lattice about its centre
    half = (np.arange(rho.size) < rho.size // 2).reshape(rho.shape)
    out = np.where(half, z, 0.0)
    out = out + np.conj(out[::-1, ::-1])
    out[c] = rho[c]
    assert np.allclose(np.abs(out[half]), np.abs(rho[half])), "moduli kept"
    return out


def surrogate(s: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Hermitian Gaussian rho with E|rho|^2 = N S(k), rho(0) = N."""
    c = _centre(s.shape)
    z = (rng.normal(size=s.shape) + 1j * rng.normal(size=s.shape)) * np.sqrt(
        n * s / 2.0
    )
    out = (z + np.conj(z[::-1, ::-1])) / np.sqrt(2.0)
    out[c] = n
    assert np.allclose(out, np.conj(out[::-1, ::-1])), "Hermitian"
    return out


# ============================================================
# FILES
# ============================================================


def present_tags(root: Path, order: Sequence[str]) -> List[str]:
    """Chi tags of `order` with both flows_chi<tag>.npz and .json in root."""
    out = [t for t in order
           if (root / f"flows_chi{t}.npz").exists()
           and (root / f"flows_chi{t}.json").exists()]  # fmt: skip
    assert len(set(out)) == len(out), "unique tags"
    return out
