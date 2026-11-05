#!/usr/bin/env python3

"""
Brute-force NumPy oracles for the xla_cco test suite.

Every function is a deliberately naive double loop that mirrors the C++
reference conventions (density 1, V = N, Phi = (1/V) sum_k w_k |rho_k|^2,
rho_k = sum_j exp(i k.r_j), no 1/2, Shift = 0). Expected values in tests come
from here, never from the code under test.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import itertools
import math
from typing import List, Tuple

# ----- numerics -----
import numpy as np

# ============================================================
# ENERGY AND FORCES
# ============================================================


def reciprocal_basis(basis: np.ndarray) -> np.ndarray:
    """Return rows b_i with b_i . a_j = 2 pi delta_ij (inverse-transpose)."""
    assert basis.ndim == 2 and basis.shape[0] == basis.shape[1], "square basis"
    return 2.0 * math.pi * np.linalg.inv(basis).T


def energy(
    s: np.ndarray, n_vectors: np.ndarray, weights: np.ndarray, volume: float
) -> float:
    """Return Phi = (1/V) sum_k w_k |sum_j exp(2 pi i n_k . s_j)|^2.

    Parameters
    ----------
    s : np.ndarray
        Fractional positions, shape (N, d).
    n_vectors : np.ndarray
        Integer reciprocal-basis coefficients, shape (M, d).
    weights : np.ndarray
        Per-vector weights, shape (M,).
    volume : float
        Box volume V.

    Returns
    -------
    float
        The stealthy energy (dimensionless, density 1).
    """
    assert s.ndim == 2 and n_vectors.ndim == 2, "expected 2-D arrays"
    assert s.shape[1] == n_vectors.shape[1], "dimension mismatch"
    total = 0.0
    for kk in range(n_vectors.shape[0]):
        rho = 0.0 + 0.0j
        for j in range(s.shape[0]):
            phase = 2.0 * math.pi * float(np.dot(n_vectors[kk], s[j]))
            rho += complex(math.cos(phase), math.sin(phase))
        total += float(weights[kk]) * (rho.real**2 + rho.imag**2)
    return total / volume


def cartesian_forces(
    s: np.ndarray,
    n_vectors: np.ndarray,
    weights: np.ndarray,
    basis: np.ndarray,
    volume: float,
) -> np.ndarray:
    """Return the C++ Cartesian forces F_i = -dPhi/dr_i, shape (N, d).

    F_i = -(1/V) sum_k 2 w_k (Im rho_k cos(k.r_i) - Re rho_k sin(k.r_i)) k,
    with k = n @ reciprocal and r = s @ basis.
    """
    assert s.shape[1] == basis.shape[0], "dimension mismatch"
    assert n_vectors.shape[1] == basis.shape[0], "dimension mismatch"
    recip = reciprocal_basis(basis)
    k = n_vectors @ recip
    r = s @ basis
    forces = np.zeros_like(r)
    for kk in range(k.shape[0]):
        rho = 0.0 + 0.0j
        for j in range(r.shape[0]):
            ph = float(np.dot(k[kk], r[j]))
            rho += complex(math.cos(ph), math.sin(ph))
        for i in range(r.shape[0]):
            ph = float(np.dot(k[kk], r[i]))
            coef = rho.imag * math.cos(ph) - rho.real * math.sin(ph)
            forces[i] += -(2.0 * float(weights[kk]) * coef / volume) * k[kk]
    return forces


# ============================================================
# K-SPACE SCAN
# ============================================================


def modulus2(k: np.ndarray) -> float:
    """Return |k|^2 summed component by component in index order (C++)."""
    acc = 0.0
    for c in range(k.shape[0]):
        acc += float(k[c]) * float(k[c])
    return acc


def halfspace_kvectors(
    basis: np.ndarray, k_inner: float, k_outer: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Scan an integer box for half-space k-vectors with k_inner^2<|k|^2<k_outer^2.

    Strict float64 comparisons; n != 0; the first nonzero component of n is
    positive (one of each +-n pair). The per-axis bound |n_i| <= k_outer |a_i|
    / (2 pi) is exact for any basis, so the box is complete.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (n, k_sq): int64 array (M, d) in lexicographic scan order, and the
        float64 |k|^2 values (M,).
    """
    assert k_outer >= k_inner >= 0.0, "need 0 <= k_inner <= k_outer"
    dim = basis.shape[0]
    recip = reciprocal_basis(basis)
    bounds = [
        int(math.floor(k_outer * float(np.linalg.norm(basis[i])) / (2.0 * math.pi))) + 1
        for i in range(dim)
    ]
    found: List[np.ndarray] = []
    ksq: List[float] = []
    ranges = [range(-b, b + 1) for b in bounds]
    for n in itertools.product(*ranges):
        nz = [c for c in n if c != 0]
        if not nz or nz[0] < 0:
            continue
        nv = np.array(n, dtype=np.int64)
        k = nv.astype(np.float64) @ recip
        q = modulus2(k)
        if k_inner * k_inner < q < k_outer * k_outer:
            found.append(nv)
            ksq.append(q)
    n_out = np.array(found, dtype=np.int64).reshape(-1, dim)
    return n_out, np.array(ksq, dtype=np.float64)


# ============================================================
# STRUCTURE AND PAIRS
# ============================================================


def structure_factor(s: np.ndarray, n_vectors: np.ndarray) -> np.ndarray:
    """Return S(n) = |rho(n)|^2 / N for each integer vector, shape (M,)."""
    assert s.ndim == 2 and n_vectors.shape[1] == s.shape[1], "dimension mismatch"
    out = np.zeros(n_vectors.shape[0])
    for kk in range(n_vectors.shape[0]):
        rho = 0.0 + 0.0j
        for j in range(s.shape[0]):
            phase = 2.0 * math.pi * float(np.dot(n_vectors[kk], s[j]))
            rho += complex(math.cos(phase), math.sin(phase))
        out[kk] = (rho.real**2 + rho.imag**2) / s.shape[0]
    return out


def nearest_neighbor_distances(s: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Return minimum-image nearest-neighbour distances, shape (N,).

    Searches all 3^d periodic images of every other particle; exact for
    reasonably reduced cells (the nearest image lies within one cell).
    Particles alone in the box see only their own images.
    """
    assert s.ndim == 2 and s.shape[1] == basis.shape[0], "dimension mismatch"
    n, dim = s.shape
    shifts = list(itertools.product((-1, 0, 1), repeat=dim))
    out = np.full(n, np.inf)
    for i in range(n):
        for j in range(n):
            for sh in shifts:
                if i == j and not any(sh):
                    continue
                ds = s[j] + np.array(sh, dtype=np.float64) - s[i]
                r = ds @ basis
                out[i] = min(out[i], math.sqrt(modulus2(r)))
    return out
