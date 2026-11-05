#!/usr/bin/env python3

"""
Property tests of the minimizers on random small stealthy systems.

Shapes come from fixed grids so jit compiles a handful of times only.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- numerics -----
import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import example, given, settings
from hypothesis import strategies as st

# ----- project -----
from systems import small_system
from tolerances import ATOL_VMAP_S, RTOL_VMAP_E
from xla_cco.minimize import relax
from xla_cco.minimize.descent import DescentSettings, adaptive_descent
from xla_cco.minimize.lbfgs import LBFGSSettings
from xla_cco.minimize.result import STATUS_RUNNING

# ============================================================
# STRATEGIES
# ============================================================

SHAPES = st.sampled_from([(8, 1), (8, 2), (8, 3), (12, 2), (12, 3)])
SEEDS = st.integers(0, 2**31 - 1)
STEPS = st.sampled_from([1e-3, 3e-2])
EVALS = st.sampled_from([40, 120])

# ============================================================
# PROPERTIES
# ============================================================


@settings(deadline=None)
@given(shape=SHAPES, seed=SEEDS, step=STEPS)
@example(shape=(8, 2), seed=0, step=3e-2)
def test_descent_accepted_energy_monotone(
    shape: tuple[int, int], seed: int, step: float
) -> None:
    """Recorded energies never increase: accepted steps decrease Phi, rejects hold."""
    n, d = shape
    _, _, energy = small_system(n, d, 4.0)
    s0 = jnp.asarray(np.random.default_rng(seed).uniform(size=(n, d)))
    cfg = DescentSettings(step_init=step, max_evals=120, record_every=1, max_records=61)
    res = adaptive_descent(energy, s0, cfg)
    traj = res.trajectory
    assert traj is not None
    e = np.asarray(traj.energy)[: int(traj.count)]
    assert np.all(np.diff(e) <= 0.0)
    assert float(res.energy) == e[-1] and float(res.energy) <= float(energy(s0))
    assert np.all(np.isfinite(np.asarray(res.s)))


@settings(deadline=None)
@given(shape=SHAPES, seed=SEEDS, evals=EVALS, step=STEPS)
@example(shape=(8, 2), seed=1, evals=40, step=3e-2)
def test_relax_vmap_equals_loop_descent(
    shape: tuple[int, int], seed: int, evals: int, step: float
) -> None:
    """Batched relax equals a Python loop of adaptive_descent (floats ~1 ulp/step)."""
    n, d = shape
    _, _, energy = small_system(n, d, 4.0)
    s0 = jnp.asarray(np.random.default_rng(seed).uniform(size=(3, n, d)))
    cfg = DescentSettings(step_init=step, max_evals=evals)
    out = relax(energy, s0, cfg, chunk_iters=9)
    for i in range(3):
        one = adaptive_descent(energy, s0[i], cfg)
        assert int(out.n_iters[i]) == int(one.n_iters)
        assert int(out.status[i]) == int(one.status)
        np.testing.assert_allclose(out.s[i], one.s, atol=ATOL_VMAP_S)
        np.testing.assert_allclose(out.energy[i], one.energy, rtol=RTOL_VMAP_E)


@settings(deadline=None)
@given(shape=st.sampled_from([(8, 2), (12, 2)]), seed=SEEDS)
@example(shape=(8, 2), seed=3)
def test_relax_lbfgs_terminates_and_never_raises_energy(
    shape: tuple[int, int], seed: int
) -> None:
    """L-BFGS lanes all finish, stay finite and do not increase Phi."""
    n, d = shape
    _, _, energy = small_system(n, d, 4.0)
    s0 = jnp.asarray(np.random.default_rng(seed).uniform(size=(2, n, d)))
    out = relax(energy, s0, LBFGSSettings(max_evals_per_pass=20), chunk_iters=8)
    assert np.all(np.asarray(out.status) != STATUS_RUNNING)
    for i in range(2):
        assert float(out.energy[i]) <= float(energy(s0[i])) * (1 + 1e-12)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(out))
