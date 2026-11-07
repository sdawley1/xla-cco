#!/usr/bin/env python3

"""
Example tests of S(k), the isotropic average, nearest neighbours, histograms, g2.

Expected values come from ``oracle`` or from brute-force sums.

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
import jax.numpy as jnp
import numpy as np
import pytest

# ----- project -----
import oracle
from xla_cco.analysis.pairs import (
    g2_from_histogram,
    nearest_neighbor_distances,
    pair_distance_histogram,
)
from xla_cco.analysis.structure import (
    _bin_moments,
    isotropic_structure_factor,
    structure_factor,
)
from xla_cco.box import Box, cubic_box

# ============================================================
# HELPERS
# ============================================================

# col2 of a bin holding one |k| is sqrt(rounding) ~ 1e-8, so it gets a looser atol
ATOL_WIDE = (1e-10, 1e-10, 1e-7, 1e-10)
SHEARED_2D = np.array([[1.3, 0.0], [0.4, 0.9]])


def _brute_isotropic(
    s: np.ndarray,
    basis: np.ndarray,
    k_dir: float,
    k_max: float,
    width: float,
) -> Tuple[np.ndarray, ...]:
    """Independent C++-rule construction: direction dedup, multiples, bins."""
    dim = basis.shape[0]
    recip = oracle.reciprocal_basis(basis)
    if dim == 1:
        prim = [np.array([1])]
    else:
        reach = int(k_dir * np.linalg.norm(basis, axis=1).max() / (2 * math.pi)) + 2
        found = []
        for n in itertools.product(range(-reach, reach + 1), repeat=dim):
            nz = [c for c in n if c != 0]
            if not nz or nz[0] < 0:
                continue
            q = oracle.modulus2(np.array(n, dtype=float) @ recip)
            if q < k_dir * k_dir:
                found.append((q, n))
        found.sort()
        prim, dirs = [], []
        for _, n in found:  # keep the first (shortest) vector of each direction
            k = np.array(n, dtype=float) @ recip
            u = k / np.linalg.norm(k)
            if not any(np.linalg.norm(u - v) < 1e-10 for v in dirs):
                dirs.append(u)
                prim.append(np.array(n))
    ks, ns = [], []
    for n in prim:
        for i in range(1, 100000):
            k = i * (n.astype(float) @ recip)
            if oracle.modulus2(k) > k_max * k_max:
                break
            ks.append(math.sqrt(oracle.modulus2(k)))
            ns.append(i * n)
    sk = np.array([oracle.structure_factor(c, np.array(ns)) for c in s])
    return _bin_brute(np.array(ks), sk, width)


def _sqrt_nan(x: float) -> float:
    """sqrt without clamp: negative input gives NaN (C++ semantics)."""
    return math.sqrt(x) if x >= 0.0 else math.nan


def _bin_brute(k: np.ndarray, sk: np.ndarray, width: float) -> Tuple[np.ndarray, ...]:
    """Per-bin plain-Python accumulation of the four C++ columns."""
    bins: dict[int, List[Tuple[float, float]]] = {}
    for c in range(sk.shape[0]):
        for kk, ss in zip(k, sk[c], strict=True):
            bins.setdefault(int(math.floor(kk / width)), []).append((kk, ss))
    rows = []
    for b in sorted(bins):
        kv = np.array([p[0] for p in bins[b]])
        sv = np.array([p[1] for p in bins[b]])
        cnt = len(kv)
        mk, ms = kv.mean(), sv.mean()
        vk = ((kv**2).mean() - mk * mk) / cnt
        vs = ((sv**2).mean() - ms * ms) / cnt
        rows.append((mk, ms, math.sqrt(vk) if vk > 0 else 0.0, _sqrt_nan(vs)))
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


def _box(basis: np.ndarray) -> Box:
    return Box(jnp.asarray(basis))


# ============================================================
# S(k)
# ============================================================


def test_structure_factor_oracle_example() -> None:
    rng = np.random.default_rng(1)
    s = rng.uniform(size=(9, 3))
    n = rng.integers(-3, 4, size=(11, 3)).astype(np.float64)
    got = np.asarray(structure_factor(jnp.asarray(s), jnp.asarray(n)))
    np.testing.assert_allclose(
        got, oracle.structure_factor(s, n), rtol=1e-11, atol=1e-13
    )


def test_structure_factor_coincident_particles_equal_n() -> None:
    s = np.tile(np.array([[0.2, 0.7]]), (6, 1))
    n = np.array([[1.0, 0.0], [2.0, 3.0], [-1.0, 4.0]])
    got = np.asarray(structure_factor(jnp.asarray(s), jnp.asarray(n)))
    np.testing.assert_allclose(got, 6.0, rtol=1e-12)


def test_structure_factor_float32_raises() -> None:
    s32 = jnp.zeros((3, 2), dtype=jnp.float32)
    with pytest.raises(TypeError):
        structure_factor(s32, jnp.ones((2, 2)))


# ============================================================
# ISOTROPIC S(k)
# ============================================================


@pytest.mark.parametrize(
    "basis, k_dir, k_max",
    [
        (np.array([[2.7]]), 5.0, 30.0),
        (SHEARED_2D, 6.0, 14.0),
        (np.eye(3) * 1.9, 4.0, 9.0),
    ],
)
def test_isotropic_sk_kset_rule(basis: np.ndarray, k_dir: float, k_max: float) -> None:
    dim = basis.shape[0]
    s = np.random.default_rng(3 + dim).uniform(size=(3, 7, dim))
    want = _brute_isotropic(s, basis, k_dir, k_max, 0.01)
    got = isotropic_structure_factor(s, _box(basis), k_dir, k_max, 0.01)
    assert got[0].shape == want[0].shape, "bin count"
    for col in range(4):
        np.testing.assert_allclose(
            got[col], want[col], rtol=1e-8, atol=1e-10, equal_nan=True
        )


@pytest.mark.parametrize("width", [0.5, 1.0])
@pytest.mark.parametrize(
    "basis, k_dir, k_max",
    [
        (np.array([[20.0]]), 5.0, 8.0),
        (SHEARED_2D * 3.0, 3.0, 10.0),
        (np.eye(3) * 6.0, 2.5, 6.0),
    ],
)
def test_isotropic_sk_wide_bins_all_columns(
    basis: np.ndarray, k_dir: float, k_max: float, width: float
) -> None:
    """Wide bins mix many |k| per bin, so col2 (SEM of |k|) is exercised."""
    dim = basis.shape[0]
    s = np.random.default_rng(11 + dim).uniform(size=(3, 7, dim))
    want = _brute_isotropic(s, basis, k_dir, k_max, width)
    got = isotropic_structure_factor(s, _box(basis), k_dir, k_max, width)
    assert got[0].shape == want[0].shape, "bin count"
    assert np.count_nonzero(want[2] > 0.0) >= 2, "col2 must carry signal"
    for col in range(4):
        np.testing.assert_allclose(
            got[col], want[col], rtol=0.0, atol=ATOL_WIDE[col], equal_nan=True
        )


@pytest.mark.parametrize("dim, k_dir", [(1, 3.0), (2, 3.0), (3, 3.0)])
def test_isotropic_sk_cutoff_equality_included(dim: int, k_dir: float) -> None:
    """Side 2 pi gives reciprocal spacing 1: |k| = k_max = 5 exactly is kept."""
    basis = np.eye(dim) * (2.0 * math.pi)
    s = np.random.default_rng(20 + dim).uniform(size=(2, 4, dim))
    got = isotropic_structure_factor(s, _box(basis), k_dir, 5.0, 0.01)
    want = _brute_isotropic(s, basis, k_dir, 5.0, 0.01)
    assert got[0].shape == want[0].shape, "row count"
    assert got[0].max() == 5.0 and want[0].max() == 5.0
    np.testing.assert_allclose(got[1], want[1], rtol=1e-8, atol=1e-10)


def test_bin_moments_col3_unclamped_col2_clamped() -> None:
    """Negative variances: col3 is NaN (C++), col2 is clamped to 0."""
    k = np.full(3, 0.1)  # equal |k|: float rounding makes the variance < 0
    var_k = ((k * k).sum() / 3.0 - (k.sum() / 3.0) ** 2) / 3.0
    assert var_k < 0.0, "premise: rounding drives the |k| variance negative"
    out = _bin_moments(k, np.ones(3), np.full(3, 0.5), 1, 0.01)  # <S^2> < <S>^2
    assert out[0].shape == (1,)
    assert np.isnan(out[3][0])
    assert out[2][0] == 0.0


def test_isotropic_sk_error_contracts() -> None:
    box = cubic_box(7, 2)
    with pytest.raises(TypeError):
        isotropic_structure_factor(np.zeros((2, 7, 2), dtype=np.float32), box)
    with pytest.raises(ValueError):
        isotropic_structure_factor(np.zeros((7, 2)), box)
    with pytest.raises(ValueError):
        isotropic_structure_factor(np.zeros((2, 7, 2)), box, k_dir_max=9.0, k_max=3.0)


# ============================================================
# NEAREST NEIGHBOURS
# ============================================================


def test_nn_distance_oracle_sheared_example() -> None:
    s = np.random.default_rng(5).uniform(size=(16, 2))
    got = np.asarray(nearest_neighbor_distances(jnp.asarray(s), _box(SHEARED_2D)))
    np.testing.assert_allclose(
        got, oracle.nearest_neighbor_distances(s, SHEARED_2D), rtol=1e-12
    )


def test_nn_single_particle_sees_own_image() -> None:
    got = nearest_neighbor_distances(jnp.array([[0.3, 0.3]]), _box(np.eye(2) * 2.0))
    assert math.isclose(float(got[0]), 2.0, rel_tol=1e-14)


def test_nn_translation_invariant() -> None:
    s = np.random.default_rng(6).uniform(size=(40, 3))
    box = _box(np.eye(3) * 1.5)
    a = nearest_neighbor_distances(jnp.asarray(s), box)
    b = nearest_neighbor_distances(jnp.asarray((s + [0.37, -0.21, 0.5]) % 1.0), box)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12)


def test_nn_float32_raises() -> None:
    with pytest.raises(TypeError):
        nearest_neighbor_distances(
            jnp.zeros((3, 2), dtype=jnp.float32), cubic_box(3, 2)
        )


# ============================================================
# HISTOGRAM AND g2
# ============================================================


def test_pair_histogram_matches_oracle_example() -> None:
    s = np.random.default_rng(7).uniform(size=(2, 16, 2))
    edges = np.linspace(0.0, 1.2, 9)
    want = sum(np.histogram(oracle.pair_distances(c, SHEARED_2D), edges)[0] for c in s)
    got = pair_distance_histogram(jnp.asarray(s), _box(SHEARED_2D), edges)
    np.testing.assert_array_equal(got, want)


def test_pair_histogram_total_count() -> None:
    b, n = 3, 16
    s = np.random.default_rng(8).uniform(size=(b, n, 3))
    box = cubic_box(n, 3)
    r_max = float(np.linalg.norm(np.asarray(box.basis).sum(axis=0))) / 2.0
    got = pair_distance_histogram(jnp.asarray(s), box, np.linspace(0.0, r_max, 7))
    assert int(got.sum()) == b * n * (n - 1) // 2


def test_pair_histogram_last_bin_closed_left_closed() -> None:
    box = _box(np.eye(1) * 8.0)
    s = jnp.asarray([[[0.0], [0.125], [0.375]]])  # exact distances 1, 3, 2
    got = pair_distance_histogram(s, box, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(got, [1, 2])  # 1 -> bin 0; 2 -> bin 1; 3 on last edge
    outside = pair_distance_histogram(s, box, np.array([1.5, 2.5]))
    np.testing.assert_array_equal(outside, [1])


@pytest.mark.parametrize(
    "edges",
    [
        np.array([0.0, 1.0, 1.0]),
        np.array([-1.0, 1.0]),
        np.array([2.0, 1.0]),
        np.ones(1),
    ],
)
def test_bad_edges_raise(edges: np.ndarray) -> None:
    s = jnp.zeros((1, 3, 2))
    with pytest.raises(ValueError):
        pair_distance_histogram(s, cubic_box(3, 2), edges)
    with pytest.raises(ValueError):
        g2_from_histogram(np.zeros(max(edges.size - 1, 1)), edges, 3, 1, 3.0, 2)


def test_histogram_float32_raises() -> None:
    with pytest.raises(TypeError):
        pair_distance_histogram(
            jnp.zeros((1, 3, 2), dtype=jnp.float32),
            cubic_box(3, 2),
            np.array([0.0, 1.0]),
        )


def test_g2_from_histogram_hand_value() -> None:
    edges = np.array([1.0, 2.0])
    got = g2_from_histogram(np.array([30.0]), edges, 10, 2, 20.0, 2)
    shell = math.pi * (4.0 - 1.0)
    assert math.isclose(got[0], 30.0 / (shell * 0.5 * 0.5 * 10 * 2), rel_tol=1e-14)


def test_g2_counts_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        g2_from_histogram(np.zeros(3), np.array([0.0, 1.0, 2.0]), 5, 1, 5.0, 2)


def test_g2_ideal_gas_unity() -> None:
    """Poisson gas: E[g2] = (N-1)/N exactly (fixed-N pairs); |g2 - E| < 5 SEM."""
    b, n, d = 200, 50, 2
    s = np.random.default_rng(2024).uniform(size=(b, n, d))
    box = cubic_box(n, d)
    edges = np.linspace(0.0, float(np.asarray(box.basis)[0, 0]) / 2.0, 13)
    per = np.array(
        [
            g2_from_histogram(
                pair_distance_histogram(jnp.asarray(c[None]), box, edges),
                edges,
                n,
                1,
                float(box.volume),
                d,
            )
            for c in s
        ]
    )
    mean, sem = per.mean(axis=0), per.std(axis=0, ddof=1) / math.sqrt(b)
    assert np.all(np.abs(mean - (n - 1) / n) < 5.0 * sem), (mean, sem)
