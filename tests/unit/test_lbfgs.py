#!/usr/bin/env python3

"""
Tests of the three-pass L-BFGS minimizer.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from typing import List, Tuple

# ----- numerics -----
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from jax import Array

# ----- project -----
from systems import small_system
from tolerances import ATOL_PASS_RESTART_S
from xla_cco.energy import AbstractEnergy
from xla_cco.kvectors import k_outer_from_ka, ka_from_chi
from xla_cco.minimize.lbfgs import LBFGSSettings, lbfgs, update_stagnation
from xla_cco.minimize.result import (
    STATUS_BUDGET,
    STATUS_ENERGY_TOL,
    STATUS_GRAD_TOL,
    STATUS_LINESEARCH_FAIL,
    STATUS_STAGNATED,
)

# ============================================================
# TEST ENERGIES
# ============================================================


class _Counter:
    """Identity-hashed evaluation counter (usable as a static jit field)."""

    def __init__(self) -> None:
        self.n = 0

    def bump(self) -> None:
        self.n += 1


class _Quadratic(AbstractEnergy):
    """0.5 (s - c)^T A (s - c) with SPD A (optional evaluation counter)."""

    a: Array
    c: Array
    counter: _Counter = eqx.field(static=True, default=None)

    def __call__(self, s: Array) -> Array:
        d = (s - self.c).reshape(-1)
        return 0.5 * d @ self.a @ d

    def value_and_grad(self, s: Array) -> Tuple[Array, Array]:
        if self.counter is not None:
            jax.debug.callback(self.counter.bump)
        d = (s - self.c).reshape(-1)
        grad = (self.a @ d).reshape(s.shape)
        return 0.5 * d @ self.a @ d, grad


class _Cliff(_Quadratic):
    """Quadratic whose value is NaN anywhere except at ``origin`` (no valid step)."""

    origin: Array = eqx.field(default=None)

    def value_and_grad(self, s: Array) -> Tuple[Array, Array]:
        value, grad = super().value_and_grad(s)
        here = jnp.all(s == self.origin)
        return jnp.where(here, value, jnp.nan), grad


def _quadratic(n: int = 6, seed: int = 0, **kw: object) -> _Quadratic:
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(2 * n, 2 * n))
    a = m @ m.T / (2 * n) + np.eye(2 * n)
    c = rng.uniform(0.3, 0.7, size=(n, 2))
    return _Quadratic(jnp.asarray(a), jnp.asarray(c), **kw)  # type: ignore[arg-type]


def _poisson(n: int, d: int, chi: float, seed: int = 0) -> Tuple[AbstractEnergy, Array]:
    ka = ka_from_chi(chi, 0.05, d)
    _, _, energy = small_system(n, d, k_outer_from_ka(ka, 0.05, d))
    return energy, jnp.asarray(np.random.default_rng(seed).uniform(size=(n, d)))


# ============================================================
# TESTS
# ============================================================


@pytest.mark.high
def test_lbfgs_spd_quadratic() -> None:
    """L-BFGS converges on a strictly convex quadratic (default 3 passes)."""
    energy = _quadratic()
    s0 = jnp.asarray(np.random.default_rng(5).uniform(0.3, 0.7, size=(6, 2)))
    res = lbfgs(energy, s0, LBFGSSettings())
    assert float(res.energy) < 1e-18
    assert np.max(np.abs(np.asarray(res.s) - np.asarray(energy.c))) < 1e-8


@pytest.mark.high
def test_lbfgs_rescale_passes() -> None:
    """Three passes run with history reset: equal to three chained single passes."""
    energy, s0 = _poisson(16, 2, 0.25)
    cfg = LBFGSSettings(max_evals_per_pass=12)
    full = lbfgs(energy, s0, cfg)
    cur = s0
    n_evals, n_iters = 1, 0
    for r in cfg.rescales:
        one = lbfgs(energy, cur, LBFGSSettings(rescales=(r,), max_evals_per_pass=12))
        cur = one.s
        n_evals += int(one.n_evals) - 1
        n_iters += int(one.n_iters)
    assert int(full.n_iters) == n_iters and int(full.n_evals) == n_evals
    assert np.max(np.abs(np.asarray(full.s) - np.asarray(cur))) < ATOL_PASS_RESTART_S
    assert float(full.energy) == pytest.approx(float(one.energy), rel=1e-6)
    # each pass is capped by its own budget (+ at most one line search overshoot)
    assert int(full.n_evals) <= 1 + 3 * (12 + 30)
    assert int(full.n_evals) > 1 + 3 * 12  # all three passes consumed budget
    assert float(full.energy) > 1e-6  # short budget: far from roundoff chaos


@pytest.mark.medium
def test_lbfgs_eval_count_is_exact() -> None:
    """n_evals equals the number of energy.value_and_grad calls actually made."""
    counter = _Counter()
    energy = _quadratic(counter=counter)
    s0 = jnp.asarray(np.random.default_rng(5).uniform(0.3, 0.7, size=(6, 2)))
    res = lbfgs(energy, s0, LBFGSSettings())
    jax.block_until_ready(res.s)
    jax.effects_barrier()
    assert counter.n == int(res.n_evals) > 3


@pytest.mark.critical
def test_lbfgs_reaches_ground_state() -> None:
    """Poisson d=2, N=64, chi=0.25 reaches Phi < 1e-20."""
    energy, s0 = _poisson(64, 2, 0.25)
    res = lbfgs(energy, s0, LBFGSSettings())
    print(f"final Phi={float(res.energy):.3e} evals={int(res.n_evals)}", flush=True)
    assert float(res.energy) < 1e-20
    assert float(energy(res.s)) < 1e-20  # re-evaluated at the wrapped output


@pytest.mark.high
def test_lbfgs_energy_and_grad_tol_end_the_run() -> None:
    """energy_tol / grad_tol stop all passes with their own status codes."""
    energy, s0 = _poisson(16, 2, 0.25)
    res = lbfgs(energy, s0, LBFGSSettings(energy_tol=1e-6))
    assert int(res.status) == STATUS_ENERGY_TOL and float(res.energy) < 1e-6
    res = lbfgs(energy, s0, LBFGSSettings(grad_tol=1e-3))
    assert int(res.status) == STATUS_GRAD_TOL and float(res.grad_norm) < 1e-3


@pytest.mark.high
def test_lbfgs_stagnation_and_linesearch_status() -> None:
    """Stagnation and line-search failure are reported, not shown as converged."""
    # at the exact minimum the line search succeeds without decrease -> stagnation
    energy = _quadratic()
    flat = lbfgs(
        energy, energy.c, LBFGSSettings(rescales=(1.0,), stagnation_patience=1)
    )
    assert int(flat.status) == STATUS_STAGNATED
    # no trial step has a finite value -> line-search failure, point not moved
    s0 = jnp.asarray(np.random.default_rng(5).uniform(0.3, 0.7, size=(6, 2)))
    base = _quadratic()
    bad = _Cliff(base.a, base.c, origin=s0)
    res = lbfgs(bad, s0, LBFGSSettings(rescales=(1.0,)))
    assert int(res.status) == STATUS_LINESEARCH_FAIL
    assert float(res.energy) == pytest.approx(float(base(s0)), rel=1e-12)
    assert np.array_equal(np.asarray(res.s), np.asarray(s0))
    # a tiny budget ends the pass with STATUS_BUDGET
    short = lbfgs(energy, s0, LBFGSSettings(rescales=(1.0,), max_evals_per_pass=3))
    assert int(short.status) == STATUS_BUDGET


@pytest.mark.medium
def test_result_dtypes_lbfgs() -> None:
    """Every RelaxResult leaf (with trajectory) is float64 or integer."""
    energy, s0 = _poisson(8, 2, 0.25)
    cfg = LBFGSSettings(max_evals_per_pass=30, record_every=2, max_records=4)
    res = lbfgs(energy, s0, cfg)
    leaves: List[Array] = jax.tree_util.tree_leaves(res)
    assert len(leaves) == 11  # 6 result fields + 5 trajectory fields
    for leaf in leaves:
        assert leaf.dtype == np.float64 or np.issubdtype(leaf.dtype, np.integer)


@pytest.mark.medium
def test_lbfgs_trajectory_records_step_and_iteration() -> None:
    """Records iteration 0, r, 2r...; the step is the accepted line-search length."""
    energy, s0 = _poisson(16, 2, 0.25)
    cfg = LBFGSSettings(max_evals_per_pass=60, record_every=5, max_records=6)
    res = lbfgs(energy, s0, cfg)
    traj = res.trajectory
    assert traj is not None and int(traj.count) == 6
    assert list(np.asarray(traj.iteration)) == [0, 5, 10, 15, 20, 25]
    assert float(traj.step[0]) == 0.0 and bool(jnp.all(traj.step[1:] > 0.0))
    assert bool(jnp.all(jnp.diff(traj.energy) <= 0.0))


@pytest.mark.medium
def test_lbfgs_trajectory_records_wrapped_positions() -> None:
    """Records are wrapped into [0, 1) even when the iterate drifts outside."""
    energy, s0 = _poisson(16, 2, 0.25)
    cfg = LBFGSSettings(max_evals_per_pass=60, record_every=5, max_records=6)
    traj = lbfgs(energy, s0 + 3.0, cfg).trajectory
    assert traj is not None
    rec = np.asarray(traj.s)[: int(traj.count)]
    assert rec.min() >= 0.0 and rec.max() < 1.0


@pytest.mark.medium
def test_lbfgs_rejects_non_float64() -> None:
    """float32 input raises TypeError before any compute."""
    energy, _ = _poisson(8, 2, 0.25)
    with pytest.raises(TypeError):
        lbfgs(energy, jnp.zeros((8, 2), jnp.float32), LBFGSSettings())


@pytest.mark.high
def test_stagnation_counter_resets_on_progress() -> None:
    """Steady strict progress for > patience iterations is not STAGNATED."""
    n = 6
    diag = np.logspace(0, 3, 2 * n)  # ill-conditioned: gradual progress
    energy = _Quadratic(jnp.asarray(np.diag(diag)), jnp.full((n, 2), 0.5))
    s0 = jnp.asarray(np.random.default_rng(5).uniform(0.3, 0.7, size=(n, 2)))
    cfg = LBFGSSettings(
        rescales=(1.0,),
        memory=1,
        max_linesearch_steps=3,
        stagnation_patience=2,
        record_every=1,
        max_records=40,
    )
    res = lbfgs(energy, s0, cfg)
    traj = res.trajectory
    assert traj is not None
    e = np.asarray(traj.energy)[: int(traj.count)]
    run = int(np.argmax(np.diff(e) >= 0.0)) if np.any(np.diff(e) >= 0.0) else len(e)
    assert run > 2 * cfg.stagnation_patience, "need steady progress beyond patience"
    # the run did not stop while it was still strictly decreasing
    assert int(res.n_iters) > run - 1
    assert int(res.status) in (STATUS_STAGNATED, STATUS_LINESEARCH_FAIL, STATUS_BUDGET)


@pytest.mark.medium
@settings(deadline=None)
@given(st.lists(st.sampled_from([0.0, 1.0, 2.0, 3.0]), min_size=2, max_size=30))
@example([3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0])
def test_update_stagnation_counts_consecutive_non_decreases(fs: List[float]) -> None:
    """The counter equals the length of the current run of non-decreasing steps."""
    stag, expected = jnp.asarray(0), 0
    for old, new in zip(fs[:-1], fs[1:], strict=True):
        stag = update_stagnation(jnp.asarray(new), jnp.asarray(old), stag)
        expected = 0 if new < old else expected + 1
        assert int(stag) == expected
