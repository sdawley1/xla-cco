#!/usr/bin/env python3

"""
Property tests of S(k), nearest neighbours, pair histograms and g2.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from typing import Tuple

# ----- numerics -----
import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

# ----- project -----
import oracle
from strategies import boxes, dims, fractional_configs, integer_kvectors
from xla_cco.analysis.pairs import (
    g2_from_histogram,
    nearest_neighbor_distances,
    pair_distance_histogram,
)
from xla_cco.analysis.structure import isotropic_structure_factor, structure_factor
from xla_cco.box import Box, cubic_box

# ============================================================
# HELPERS
# ============================================================

FAST = settings(deadline=None)
M_PAD = 32  # fixed k count so jit shapes come from one grid


@st.composite
def sk_inputs(draw: st.DrawFn) -> Tuple[np.ndarray, np.ndarray, int]:
    """Draw (s, padded integer k-vectors, true count) in a common dimension."""
    d = draw(dims())
    s = draw(fractional_configs(dim=d))
    n = draw(integer_kvectors(dim=d, max_m=M_PAD))
    m = n.shape[0]
    pad = np.zeros((M_PAD, d))
    pad[:m] = n
    return s, pad, m


@st.composite
def cells(draw: st.DrawFn) -> Tuple[np.ndarray, np.ndarray]:
    """Draw (positions, basis) for a (possibly sheared) reduced cell."""
    d = draw(dims())
    return draw(fractional_configs(dim=d)), draw(boxes(dim=d))


# ============================================================
# PROPERTIES
# ============================================================


@FAST
@given(sk_inputs())
def test_structure_factor_oracle(case: Tuple[np.ndarray, np.ndarray, int]) -> None:
    s, n, m = case
    got = np.asarray(structure_factor(jnp.asarray(s), jnp.asarray(n)))[:m]
    want = oracle.structure_factor(s, n[:m])
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


@FAST
@given(sk_inputs())
def test_structure_factor_bounds(case: Tuple[np.ndarray, np.ndarray, int]) -> None:
    s, n, _ = case
    n_part = s.shape[0]
    got = np.asarray(structure_factor(jnp.asarray(s), jnp.asarray(n)))
    assert np.all(got >= 0.0) and np.all(got <= n_part * (1.0 + 1e-12))
    same = np.tile(s[:1], (n_part, 1))
    full = np.asarray(structure_factor(jnp.asarray(same), jnp.asarray(n)))
    np.testing.assert_allclose(full, n_part, rtol=1e-12)


@FAST
@given(cells())
def test_nn_distance_oracle(case: Tuple[np.ndarray, np.ndarray]) -> None:
    s, basis = case
    got = np.asarray(
        nearest_neighbor_distances(jnp.asarray(s), Box(jnp.asarray(basis)))
    )
    np.testing.assert_allclose(
        got, oracle.nearest_neighbor_distances(s, basis), rtol=1e-11, atol=1e-12
    )


@FAST
@given(cells(), st.integers(0, 2**31))
def test_nn_translation_invariant(
    case: Tuple[np.ndarray, np.ndarray], seed: int
) -> None:
    s, basis = case
    shift = np.random.default_rng(seed).uniform(size=s.shape[1])
    box = Box(jnp.asarray(basis))
    a = nearest_neighbor_distances(jnp.asarray(s), box)
    b = nearest_neighbor_distances(jnp.asarray((s + shift) % 1.0), box)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-9, atol=1e-12)


@FAST
@given(cells())
def test_pair_histogram_matches_oracle(case: Tuple[np.ndarray, np.ndarray]) -> None:
    s, basis = case
    r_max = float(np.linalg.norm(basis, axis=1).sum())
    edges = np.linspace(0.0, r_max, 6) * 0.9137  # clips some pairs, hits no edge
    want = np.histogram(oracle.pair_distances(s, basis), edges)[0]
    got = pair_distance_histogram(jnp.asarray(s[None]), Box(jnp.asarray(basis)), edges)
    np.testing.assert_array_equal(got, want)


@FAST
@given(cells())
def test_pair_histogram_total_count(case: Tuple[np.ndarray, np.ndarray]) -> None:
    s, basis = case
    n = s.shape[0]
    r_max = float(np.linalg.norm(basis, axis=1).sum())  # >= half diagonal
    edges = np.linspace(0.0, r_max, 5)
    got = pair_distance_histogram(
        jnp.asarray(np.stack([s, s[::-1]])), Box(jnp.asarray(basis)), edges
    )
    assert int(got.sum()) == 2 * n * (n - 1) // 2


@FAST
@given(
    st.lists(st.integers(0, 500), min_size=3, max_size=3),
    st.integers(2, 200),
    st.integers(1, 20),
    st.floats(1.0, 500.0),
    dims(),
)
def test_g2_inverts_shell_normalization(
    counts: list[int], n_part: int, n_cfg: int, volume: float, dim: int
) -> None:
    edges = np.array([0.0, 0.5, 1.5, 3.0])
    c = np.array(counts, dtype=np.float64)
    g = g2_from_histogram(c, edges, n_part, n_cfg, volume, dim)
    # doubling configs and counts leaves g2 unchanged; doubling V halves rho
    np.testing.assert_allclose(
        g2_from_histogram(2 * c, edges, n_part, 2 * n_cfg, volume, dim), g, rtol=1e-12
    )
    np.testing.assert_allclose(
        g2_from_histogram(c, edges, n_part, n_cfg, 2 * volume, dim), 2 * g, rtol=1e-12
    )
    assert np.all(g >= 0.0)


@FAST
@given(st.sampled_from((1, 2, 3)), st.integers(0, 2**31))
def test_isotropic_sk_translation_invariant(dim: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    s = rng.uniform(size=(2, 7, dim))
    box = cubic_box(7, dim)
    a = isotropic_structure_factor(s, box, 4.0, 8.0)
    b = isotropic_structure_factor((s + rng.uniform(size=dim)) % 1.0, box, 4.0, 8.0)
    np.testing.assert_allclose(a[1], b[1], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(a[0], b[0], rtol=1e-13)
    assert np.all(a[1] >= 0.0) and np.all(a[1] <= 7.0 * (1 + 1e-12))
