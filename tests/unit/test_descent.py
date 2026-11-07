#!/usr/bin/env python3

"""
Tests of adaptive descent: C++ transcription parity, budgets, statuses, trajectory.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from typing import Any, List, Tuple

# ----- numerics -----
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

# ----- project -----
import oracle
from golden import load_golden
from systems import small_system
from tolerances import ATOL_DESCENT_GOLDEN_S, ATOL_DESCENT_ITERATE
from xla_cco.box import cubic_box
from xla_cco.energy import AbstractEnergy, make_stealthy_energy
from xla_cco.kvectors import enumerate_kvectors
from xla_cco.minimize import relax
from xla_cco.minimize.descent import DescentSettings, adaptive_descent
from xla_cco.minimize.result import (
    STATUS_BUDGET,
    STATUS_ENERGY_TOL,
    STATUS_GRAD_TOL,
    STATUS_STEP_FLOOR,
)

# ============================================================
# C++ TRANSCRIPTION (StructureOptimization.cpp:699-724)
# ============================================================


def _min_image(a: np.ndarray, b: np.ndarray) -> float:
    """Return max |a - b| after minimum-image wrapping of the difference."""
    diff = a - b
    return float(np.max(np.abs(diff - np.round(diff))))


def _cpp_descent(
    s0: np.ndarray,
    kset: Any,
    basis: np.ndarray,
    volume: float,
    grow: float,
    iters: int,
    step0: float,
) -> Tuple[List[np.ndarray], List[bool], List[float]]:
    """Transcribe the C++ loop verbatim; recompute f1 and g every iteration."""
    weights = np.ones(kset.n.shape[0])

    def obj(x: np.ndarray) -> Tuple[float, np.ndarray]:
        f = oracle.energy(x, kset.n, weights, volume)
        forces = oracle.cartesian_forces(x, kset.n, weights, basis, volume)
        return f, -forces @ basis.T  # dPhi/ds = -F B^T

    x = s0.copy()
    step = step0
    xs, accepts, steps = [], [], []
    for _ in range(iters):
        f1, g = obj(x)
        norm = float(np.sqrt(np.sum(g * g)))
        g = g / norm
        xn = x - g * step
        f2, _ = obj(xn)
        df = f1 - f2
        df2 = step * norm
        if abs(df - df2) / df2 > 0.01:
            step *= 0.5
            accepts.append(False)
        else:
            x = xn
            step *= grow  # grow = 1.0 is the C++ rule
            accepts.append(True)
        xs.append(x.copy())
        steps.append(step)
    return xs, accepts, steps


# ============================================================
# TESTS
# ============================================================


@pytest.mark.parametrize("grow", [1.0, 2.0])
def test_descent_matches_transcribed_cpp(grow: float) -> None:
    """100 iterations match the verbatim C++ loop: iterates, accepts, steps."""
    n, d, iters = 16, 2, 100
    box, kset, energy = small_system(n, d, 4.0)
    s0 = np.random.default_rng(3).uniform(size=(n, d))
    basis = np.asarray(box.basis)
    step0 = 3e-2  # large enough that both branches are exercised
    xs, accepts, steps = _cpp_descent(
        s0, kset, basis, float(box.volume), grow, iters, step0
    )
    cfg = DescentSettings(
        step_init=step0,
        grow=grow,
        max_evals=2 * iters,
        record_every=1,
        max_records=iters + 1,
    )
    res = adaptive_descent(energy, jnp.asarray(s0), cfg)
    traj = res.trajectory
    assert traj is not None and int(traj.count) == iters + 1
    got_steps = np.asarray(traj.step)[1:]
    assert np.array_equal(got_steps, np.asarray(steps))  # exact
    prev = np.concatenate([[step0], got_steps[:-1]])
    got_acc = got_steps > prev if grow > 1.0 else got_steps == prev
    got_acc = np.where(got_steps < prev, False, got_acc)
    assert list(got_acc) == accepts
    dev = max(_min_image(np.asarray(traj.s)[m + 1], xs[m]) for m in range(iters))
    print(f"grow={grow}: max |ds| vs transcribed C++ = {dev:.3e}", flush=True)
    assert dev <= ATOL_DESCENT_ITERATE
    assert any(accepts) and not all(accepts), "test must exercise both branches"


def test_descent_first_steps_match_cpp_golden() -> None:
    """First 10 accept decisions and iterates match the C++ golden (d=2, N=64)."""
    g = load_golden("traj_d2_n64_chi0.25")
    p = g["params"]
    box = cubic_box(p["N"], p["d"])
    kset = enumerate_kvectors(
        np.asarray(box.basis),
        float.fromhex(p["k_inner_hex"]),
        float.fromhex(p["k_outer_hex"]),
    )
    assert kset.n.shape[0] == p["M"], "k-set size must match the golden"
    energy = make_stealthy_energy(box, kset)
    m_max = 10
    cfg = DescentSettings(max_evals=2 * m_max, record_every=1, max_records=m_max + 1)
    res = relax(energy, jnp.asarray(g["s0"]), cfg)
    assert res.trajectory is not None
    s_traj = np.asarray(res.trajectory.s)  # (B, m_max + 1, N, d)
    iters = list(g["iters"])
    worst = 0.0
    for b in range(g["s0"].shape[0]):
        for m in range(1, m_max + 1):
            t = iters.index(m)
            mine = bool(np.any(s_traj[b, m] != s_traj[b, m - 1]))
            assert mine == bool(g["accepted"][t, b]), f"accept lane {b} m {m}"
            worst = max(worst, _min_image(s_traj[b, m], g["s"][t, b]))
    print(f"golden first 10 steps: accepts equal, max |ds| = {worst:.3e}", flush=True)
    assert worst <= ATOL_DESCENT_GOLDEN_S


def test_descent_eval_budget() -> None:
    """max_evals = 2m runs exactly m iterations and 2m counted evaluations."""
    _, _, energy = small_system(12, 2, 4.0)
    s0 = jnp.asarray(np.random.default_rng(1).uniform(size=(12, 2)))
    res = adaptive_descent(energy, s0, DescentSettings(max_evals=14))
    assert int(res.n_iters) == 7 and int(res.n_evals) == 14
    assert int(res.status) == STATUS_BUDGET


def test_regression_D003_odd_budget_no_underflow() -> None:
    """D-003: an odd budget 2m+1 runs m iterations and terminates (no underflow)."""
    _, _, energy = small_system(12, 2, 4.0)
    s0 = jnp.asarray(np.random.default_rng(1).uniform(size=(12, 2)))
    res = adaptive_descent(energy, s0, DescentSettings(max_evals=2 * 5 + 1))
    assert int(res.n_iters) == 5 and int(res.n_evals) == 10
    assert int(res.status) == STATUS_BUDGET
    one = adaptive_descent(energy, s0, DescentSettings(max_evals=1))
    assert int(one.n_iters) == 0 and int(one.status) == STATUS_BUDGET


def test_descent_step_floor_status() -> None:
    """Always-reject (rel_tol ~ 0) halves the step to the floor; s is unchanged."""
    _, _, energy = small_system(12, 2, 4.0)
    s0 = jnp.asarray(np.random.default_rng(1).uniform(size=(12, 2)))
    res = adaptive_descent(energy, s0, DescentSettings(rel_tol=1e-30))
    assert int(res.status) == STATUS_STEP_FLOOR
    assert int(res.n_iters) == 40  # 1e-3 * 0.5^40 < 1e-15 <= 1e-3 * 0.5^39
    assert np.array_equal(np.asarray(res.s), np.asarray(s0))


def test_descent_energy_tol_status() -> None:
    """Stops with STATUS_ENERGY_TOL once Phi < energy_tol; zero iterations if met."""
    _, _, energy = small_system(12, 2, 4.0)
    s0 = jnp.asarray(np.random.default_rng(1).uniform(size=(12, 2)))
    f0 = float(energy(s0))
    res = adaptive_descent(energy, s0, DescentSettings(energy_tol=0.5 * f0))
    assert int(res.status) == STATUS_ENERGY_TOL
    assert 0 < int(res.n_iters) < 5000 and float(res.energy) < 0.5 * f0
    met = adaptive_descent(energy, s0, DescentSettings(energy_tol=2.0 * f0))
    assert int(met.status) == STATUS_ENERGY_TOL and int(met.n_iters) == 0


def test_trajectory_recording() -> None:
    """Iterations 0, r, 2r, ... are stored; count is capped; off gives None."""
    _, _, energy = small_system(12, 2, 4.0)
    s0 = jnp.asarray(np.random.default_rng(1).uniform(size=(12, 2)))
    cfg = DescentSettings(max_evals=40, record_every=3, max_records=4)
    res = adaptive_descent(energy, s0, cfg)
    traj = res.trajectory
    assert traj is not None and traj.s.shape == (4, 12, 2)
    assert int(traj.count) == 4
    assert list(np.asarray(traj.iteration)) == [0, 3, 6, 9]
    for t in range(4):
        e_rec = float(energy(traj.s[t]))
        assert e_rec == pytest.approx(float(traj.energy[t]), rel=1e-9)
    roomy = adaptive_descent(
        energy, s0, DescentSettings(max_evals=40, record_every=3, max_records=50)
    )
    assert roomy.trajectory is not None and int(roomy.trajectory.count) == 7
    for off in (
        DescentSettings(max_evals=20, record_every=0, max_records=5),
        DescentSettings(max_evals=20, record_every=1, max_records=0),
    ):
        assert adaptive_descent(energy, s0, off).trajectory is None


def test_descent_rejects_non_float64() -> None:
    """float32 input raises TypeError before any compute."""
    _, _, energy = small_system(8, 2, 4.0)
    with pytest.raises(TypeError):
        adaptive_descent(energy, jnp.zeros((8, 2), jnp.float32), DescentSettings())


def test_result_dtypes_descent() -> None:
    """Every RelaxResult leaf (with trajectory) is float64 or integer."""
    _, _, energy = small_system(8, 2, 4.0)
    s0 = jnp.asarray(np.random.default_rng(0).uniform(size=(8, 2)))
    res = adaptive_descent(
        energy, s0, DescentSettings(max_evals=20, record_every=2, max_records=3)
    )
    for leaf in jax.tree_util.tree_leaves(res):
        assert leaf.dtype == np.float64 or np.issubdtype(leaf.dtype, np.integer)


# ============================================================
# EVALUATION PATH
# ============================================================


class _Calls:
    """Identity-hashed call counter (usable as a static field)."""

    def __init__(self) -> None:
        self.n = 0

    def bump(self) -> None:
        self.n += 1


class _SpyEnergy(AbstractEnergy):
    """Wraps an energy; ``__call__`` is forbidden, value_and_grad is counted."""

    inner: AbstractEnergy
    calls: _Calls = eqx.field(static=True)

    def __call__(self, s: jax.Array) -> jax.Array:
        raise AssertionError("descent must evaluate f only via value_and_grad")

    def value_and_grad(self, s: jax.Array) -> Tuple[jax.Array, jax.Array]:
        jax.debug.callback(self.calls.bump)
        return self.inner.value_and_grad(s)


def test_descent_uses_value_and_grad_once_per_iteration() -> None:
    """f1 and f2 share one code path: exactly 1 value_and_grad per iteration."""
    _, _, energy = small_system(12, 2, 4.0)
    spy = _SpyEnergy(energy, _Calls())
    s0 = jnp.asarray(np.random.default_rng(1).uniform(size=(12, 2)))
    res = adaptive_descent(spy, s0, DescentSettings(max_evals=40, step_init=3e-2))
    jax.block_until_ready(res.s)
    jax.effects_barrier()
    assert int(res.n_iters) == 20
    assert spy.calls.n == 1 + int(res.n_iters)  # initial + one per iteration


def _single_particle_energy() -> AbstractEnergy:
    """N = 1: Phi is constant and the exact gradient vanishes everywhere."""
    return make_stealthy_energy(
        cubic_box(1, 2), enumerate_kvectors(np.eye(2), 0.0, 7.0)
    )


@pytest.mark.parametrize("grow", [1.0, 2.0])
def test_regression_D012_zero_gradient_stops_finite(grow: float) -> None:
    """At s = 0 the computed gradient is exactly zero.

    The 0/0 accept test used to accept forever; with grow = 2 the step
    overflowed and the positions became NaN. Descent must stop at once with
    STATUS_GRAD_TOL and leave the configuration unchanged.
    """
    s0 = jnp.zeros((1, 2))
    settings_ = DescentSettings(grow=grow, max_evals=2200)
    res = adaptive_descent(_single_particle_energy(), s0, settings_)
    assert int(res.status) == STATUS_GRAD_TOL
    assert int(res.n_iters) == 0
    np.testing.assert_array_equal(np.asarray(res.s), np.asarray(s0))
    assert np.isfinite(float(res.energy))


@settings(deadline=None, max_examples=20)
@given(
    grow=st.sampled_from((1.0, 2.0)),
    start=st.lists(st.floats(0.0, 1.0, exclude_max=True), min_size=2, max_size=2),
)
@example(grow=2.0, start=[0.0, 0.0])
@example(grow=2.0, start=[0.0, 0.625])
def test_descent_flat_energy_stays_finite(grow: float, start: List[float]) -> None:
    """On a flat energy (gradient zero or round-off) descent ends finite."""
    s0 = jnp.asarray([start])
    settings_ = DescentSettings(grow=grow, max_evals=2200)
    res = adaptive_descent(_single_particle_energy(), s0, settings_)
    assert np.all(np.isfinite(np.asarray(res.s)))
    assert np.isfinite(float(res.energy))
    assert int(res.status) in (STATUS_GRAD_TOL, STATUS_STEP_FLOOR, STATUS_BUDGET)
