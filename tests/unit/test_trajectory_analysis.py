#!/usr/bin/env python3

"""
Tests for xla_cco.analysis.trajectory (unwrap, proximity metric).

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- numerics -----
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

# ----- project -----
from xla_cco.analysis.trajectory import proximity_metric, unwrap_fractional

# ============================================================
# STRATEGIES
# ============================================================


@st.composite
def _paths(draw: st.DrawFn) -> np.ndarray:
    """Continuous fractional paths with per-frame steps < 0.4 (unwrappable)."""
    t = draw(st.sampled_from((1, 2, 5, 17)))
    n = draw(st.sampled_from((1, 3, 8)))
    d = draw(st.sampled_from((1, 2, 3)))
    start = draw(hnp.arrays(np.float64, (n, d), elements=st.floats(-3.0, 3.0)))
    steps = draw(hnp.arrays(np.float64, (t - 1, n, d), elements=st.floats(-0.4, 0.4)))
    return np.concatenate([start[None], start + np.cumsum(steps, axis=0)])


# ============================================================
# UNWRAP
# ============================================================


@pytest.mark.high
@settings(deadline=None)
@given(_paths())
def test_unwrap_recovers_continuous_path(path: np.ndarray) -> None:
    wrapped = path - np.floor(path)
    out = unwrap_fractional(wrapped)
    # Same shape/start as the wrapped input; equals the true path up to a
    # constant integer shift fixed by the first frame.
    assert out.shape == path.shape
    np.testing.assert_array_equal(out[0], wrapped[0])
    shift = path[0] - wrapped[0]
    np.testing.assert_allclose(out + shift, path, atol=1e-12)


@pytest.mark.high
def test_unwrap_crossing_boundary_example() -> None:
    frames = np.array([[[0.95]], [[0.05]], [[0.15]]])
    np.testing.assert_allclose(unwrap_fractional(frames)[:, 0, 0], [0.95, 1.05, 1.15])


@pytest.mark.medium
def test_unwrap_rejects_bad_input() -> None:
    with pytest.raises(TypeError):
        unwrap_fractional(np.zeros((2, 3, 2), np.float32))
    with pytest.raises(ValueError):
        unwrap_fractional(np.zeros((0, 3, 2)))
    with pytest.raises(ValueError):
        unwrap_fractional(np.zeros((3, 2)))


# ============================================================
# PROXIMITY METRIC
# ============================================================


@pytest.mark.high
@settings(deadline=None)
@given(_paths(), st.floats(0.5, 4.0), st.floats(0.1, 3.0))
def test_proximity_matches_double_loop(
    path: np.ndarray, scale: float, density: float
) -> None:
    d = path.shape[2]
    basis = np.eye(d) * scale
    got = proximity_metric(path, path[0], basis, density)
    want = np.empty(path.shape[0])
    for t in range(path.shape[0]):  # independent oracle
        acc = 0.0
        for j in range(path.shape[1]):
            dr = (path[t, j] - path[0, j]) @ basis
            acc += float(dr @ dr)
        want[t] = np.sqrt(density * acc)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
    assert got[0] == 0.0 and np.all(got >= 0.0)


@pytest.mark.high
def test_proximity_uniform_translation_example() -> None:
    # All N particles shifted by c: p = sqrt(rho) * sqrt(N) * |c|.
    path = np.zeros((2, 4, 2))
    path[1] += np.array([0.1, 0.0])
    basis = np.eye(2) * 10.0
    got = proximity_metric(path, path[0], basis, density=4.0)
    np.testing.assert_allclose(got, [0.0, 2.0 * 2.0 * 1.0])


@pytest.mark.medium
def test_proximity_rejects_bad_input() -> None:
    path = np.zeros((2, 4, 2))
    with pytest.raises(ValueError):
        proximity_metric(path, np.zeros((3, 2)), np.eye(2), 1.0)
    with pytest.raises(ValueError):
        proximity_metric(path, path[0], np.eye(3), 1.0)
    with pytest.raises(ValueError):
        proximity_metric(path, path[0], np.eye(2), 0.0)
