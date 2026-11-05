#!/usr/bin/env python3

"""
Host-side (NumPy) construction of the stealthy half-space k-set.

Conventions follow the C++ reference: strict float64 shell comparison
K1^2 < |k|^2 < K2^2, n != 0, first nonzero component of n positive,
|k|^2 summed component by component in index order, unit weights.
K2 = Ka / a with a = (phi / V_d(1))^(1/d); Ka(chi, phi, d) =
(2^(d+1) d chi Gamma(d/2+1)^2 phi)^(1/d); chi = M / (d (N - 1)).

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import dataclasses
import math

# ----- numerics -----
import numpy as np

# ============================================================
# K-SET
# ============================================================


@dataclasses.dataclass(frozen=True)
class KSet:
    """Half-space k-vectors with K1^2 < |k|^2 < K2^2.

    Attributes
    ----------
    n : np.ndarray
        (M, d) int64 coefficients in the reciprocal basis.
    k : np.ndarray
        (M, d) float64 Cartesian vectors, ``k = n @ reciprocal``.
    k_sq : np.ndarray
        (M,) float64 squared moduli (C++ Modulus2 summation order).
    weights : np.ndarray
        (M,) float64, all ones ("step" potential).
    k_inner : float
        Inner cutoff K1.
    k_outer : float
        Outer cutoff K2.
    """

    n: np.ndarray
    k: np.ndarray
    k_sq: np.ndarray
    weights: np.ndarray
    k_inner: float
    k_outer: float


def enumerate_kvectors(basis: np.ndarray, k_inner: float, k_outer: float) -> KSet:
    """Enumerate the half-space k-set of a periodic box.

    Parameters
    ----------
    basis : np.ndarray
        (d, d) float64 lattice basis, rows = lattice vectors.
    k_inner, k_outer : float
        Strict inner and outer cutoffs (K1, K2), inverse length.
    """
    basis = np.asarray(basis, dtype=np.float64)
    _validate_inputs(basis, k_inner, k_outer)
    dim = basis.shape[0]
    recip = 2.0 * math.pi * np.linalg.inv(basis).T
    # n_i = k . a_i / 2 pi, so |n_i| <= k_outer |a_i| / 2 pi for any basis
    bounds = [
        int(math.ceil(k_outer * float(np.linalg.norm(basis[i])) / (2.0 * math.pi))) + 1
        for i in range(dim)
    ]
    axes = [np.arange(-b, b + 1, dtype=np.int64) for b in bounds]
    n = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, dim)
    n = n[_first_nonzero_positive(n)]
    k = np.zeros((n.shape[0], dim))
    for j in range(dim):
        for i in range(dim):  # fixed summation order, no BLAS reassociation
            k[:, j] = k[:, j] + n[:, i] * recip[i, j]
    k_sq = np.zeros(n.shape[0])
    for i in range(dim):
        k_sq = k_sq + k[:, i] * k[:, i]
    keep = (k_inner * k_inner < k_sq) & (k_sq < k_outer * k_outer)
    n, k, k_sq = n[keep], k[keep], k_sq[keep]
    # primary key k_sq, ties by n lexicographic (np.lexsort: last key primary)
    order = np.lexsort(tuple(n[:, i] for i in range(dim - 1, -1, -1)) + (k_sq,))
    n, k, k_sq = n[order], k[order], k_sq[order]
    return KSet(
        n=n,
        k=k,
        k_sq=k_sq,
        weights=np.ones(n.shape[0]),
        k_inner=float(k_inner),
        k_outer=float(k_outer),
    )


def _validate_inputs(basis: np.ndarray, k_inner: float, k_outer: float) -> None:
    """Raise ``ValueError`` for a malformed basis or cutoff pair."""
    if basis.ndim != 2 or basis.shape[0] != basis.shape[1] or basis.shape[0] < 1:
        raise ValueError(f"basis must be square (d, d), got {basis.shape}")
    if not (
        np.all(np.isfinite(basis)) and math.isfinite(k_inner) and math.isfinite(k_outer)
    ):
        raise ValueError("basis and cutoffs must be finite")
    if not 0.0 <= k_inner < k_outer:
        raise ValueError(f"need 0 <= k_inner < k_outer, got {k_inner}, {k_outer}")
    det = float(np.linalg.det(basis))
    if not abs(det) > 0.0 or not math.isfinite(det):
        raise ValueError("basis must have positive |det|")


def _first_nonzero_positive(n: np.ndarray) -> np.ndarray:
    """Return the half-space mask: first nonzero coefficient of each row > 0."""
    undecided = np.ones(n.shape[0], dtype=bool)
    keep = np.zeros(n.shape[0], dtype=bool)
    for i in range(n.shape[1]):
        keep |= undecided & (n[:, i] > 0)
        undecided &= n[:, i] == 0
    return keep


def unit_ball_volume(dim: int) -> float:
    """Return the volume V_d(1) of the unit d-ball."""
    assert dim >= 1, "dim must be >= 1"
    return float(math.pi ** (dim / 2.0)) / math.gamma(dim / 2.0 + 1.0)


def k_outer_from_ka(ka: float, phi: float, dim: int) -> float:
    """Return K2 = Ka / a with a = (phi / V_d(1))^(1/d)."""
    assert phi > 0.0 and dim >= 1, "need phi > 0 and dim >= 1"
    a = (phi / unit_ball_volume(dim)) ** (1.0 / dim)
    return float(ka / a)


def ka_from_chi(chi: float, phi: float, dim: int) -> float:
    """Return Ka = (2^(d+1) d chi Gamma(d/2+1)^2 phi)^(1/d)."""
    assert chi >= 0.0 and phi > 0.0 and dim >= 1, "need chi >= 0, phi > 0, dim >= 1"
    arg = 2.0 ** (dim + 1) * dim * chi * math.gamma(dim / 2.0 + 1.0) ** 2 * phi
    return float(arg ** (1.0 / dim))


def chi_from_count(m: int, n_particles: int, dim: int) -> float:
    """Return chi = M / (d (N - 1)) for M half-space constraints."""
    assert n_particles >= 2 and dim >= 1, "need N >= 2 and dim >= 1"
    return float(m / (dim * (n_particles - 1)))
