#!/usr/bin/env python3

"""
Tests of the batched ``relax`` driver: equality with per-config runs, per-lane
stopping, compaction, and compile counts.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import logging
from typing import Any, Callable, Union

# ----- numerics -----
import jax
import jax.numpy as jnp
import numpy as np
import pytest

# ----- project -----
from systems import small_system, square_lattice
from tolerances import ATOL_VMAP_S, RTOL_VMAP_E
from xla_cco.minimize import _loop, relax
from xla_cco.minimize import descent as descent_mod
from xla_cco.minimize.descent import DescentSettings, adaptive_descent
from xla_cco.minimize.lbfgs import LBFGSSettings, lbfgs
from xla_cco.minimize.result import (
    STATUS_BUDGET,
    STATUS_ENERGY_TOL,
    STATUS_RUNNING,
    RelaxResult,
)

Settings = Union[DescentSettings, LBFGSSettings]

# ============================================================
# HELPERS
# ============================================================


def _same(a: Any, b: Any) -> bool:
    """True when two PyTrees of arrays are bit-identical."""
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(
        np.array_equal(np.asarray(x), np.asarray(y))
        for x, y in zip(la, lb, strict=True)
    )


def _close(a: Any, b: Any) -> None:
    """Integers equal exactly, floats within the vmap-vs-single tolerances."""
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb)
    for x, y in zip(la, lb, strict=True):
        if np.issubdtype(np.asarray(x).dtype, np.integer):
            assert np.array_equal(np.asarray(x), np.asarray(y))
        else:
            np.testing.assert_allclose(x, y, rtol=RTOL_VMAP_E, atol=ATOL_VMAP_S)


def _lane(res: RelaxResult, i: int) -> RelaxResult:
    return jax.tree_util.tree_map(lambda x: x[i], res)


def _batch(b: int = 5, n: int = 16, seed: int = 0) -> jnp.ndarray:
    return jnp.asarray(np.random.default_rng(seed).uniform(size=(b, n, 2)))


_METHODS: list[Callable[[], Settings]] = [
    lambda: DescentSettings(
        max_evals=300, step_init=3e-2, record_every=10, max_records=8
    ),
    lambda: LBFGSSettings(max_evals_per_pass=20),
]

# ============================================================
# TESTS
# ============================================================


@pytest.mark.parametrize("make", _METHODS, ids=["descent", "lbfgs"])
def test_relax_batched_equals_single(make: Callable[[], Settings]) -> None:
    """relax equals the single-config minimizer lane by lane.

    Integers (status, counts, iteration indices) are exactly equal; floats agree to
    ~1 ulp per step, not bitwise: vmapped XLA reductions are ordered differently.
    """
    settings = make()
    _, _, energy = small_system(16, 2, 4.0)
    s0 = _batch()
    out = relax(energy, s0, settings, chunk_iters=11)
    single = adaptive_descent if isinstance(settings, DescentSettings) else lbfgs
    for i in range(s0.shape[0]):
        _close(_lane(out, i), single(energy, s0[i], settings))
    assert out.s.shape == s0.shape and out.energy.shape == (s0.shape[0],)


def _mixed_batch() -> jnp.ndarray:
    """Lane 0: square lattice (Phi ~ 0 below the first shell); others random."""
    lattice = square_lattice(4)  # n = 16, side 4, first reciprocal shell 2 pi
    rest = np.random.default_rng(2).uniform(size=(3, 16, 2))
    return jnp.asarray(np.concatenate([lattice[None], rest]))


@pytest.mark.parametrize("make", _METHODS, ids=["descent", "lbfgs"])
def test_relax_per_config_stopping(make: Callable[[], Settings]) -> None:
    """A converged lane stops at once and is bit-identical across further rounds."""
    base = make()
    kw = dict(energy_tol=1e-12)
    settings = type(base)(**{**_fields(base), **kw})
    _, _, energy = small_system(16, 2, 5.0)  # window below the 2 pi shell
    s0 = _mixed_batch()
    assert float(energy(s0[0])) < 1e-12 < float(energy(s0[1]))
    short = relax(energy, s0, settings, chunk_iters=3)
    long = relax(energy, s0, settings, chunk_iters=10_000)
    assert int(short.status[0]) == STATUS_ENERGY_TOL and int(short.n_iters[0]) == 0
    assert np.array_equal(np.asarray(short.s[0]), np.asarray(s0[0]) % 1.0)
    assert np.all(np.asarray(short.status[1:]) != STATUS_RUNNING)
    assert int(jnp.max(short.n_iters[1:])) > 0
    assert _same(_lane(short, 0), _lane(long, 0))
    # advance the batch state by hand: the finished lane must not move at all
    init, chunk = _batched_fns(settings)
    state = init(energy, s0, settings)
    s_before = jax.tree_util.tree_map(lambda x: x[0], state)
    for _ in range(3):
        state = chunk(energy, settings, state, 5)
    assert _same(s_before, jax.tree_util.tree_map(lambda x: x[0], state))


def _fields(settings: Settings) -> dict[str, Any]:
    return {k: getattr(settings, k) for k in settings.__dataclass_fields__}


def _batched_fns(settings: Settings) -> tuple[Callable[..., Any], Callable[..., Any]]:
    from xla_cco.minimize import lbfgs as lbfgs_mod

    mod = descent_mod if isinstance(settings, DescentSettings) else lbfgs_mod
    return mod.init_batch, mod.chunk_batch


@pytest.mark.parametrize("make", _METHODS, ids=["descent", "lbfgs"])
def test_relax_compaction_preserves_results(make: Callable[[], Settings]) -> None:
    """chunk_iters=7 (many compactions) equals chunk_iters=10_000 exactly."""
    base = make()
    _, _, energy = small_system(16, 2, 4.0)
    s0 = _batch(b=6, seed=4)
    pre = relax(energy, s0, base, chunk_iters=10_000)
    tol = float(np.median(np.asarray(pre.energy)))  # lanes below it stop early
    settings = type(base)(**{**_fields(base), "energy_tol": tol})
    fine = relax(energy, s0, settings, chunk_iters=7)
    coarse = relax(energy, s0, settings, chunk_iters=10_000)
    assert _same(fine, coarse)
    iters = np.asarray(fine.n_iters)
    assert iters.min() != iters.max(), "lanes must finish at different rounds"


def test_relax_returns_original_order() -> None:
    """Lane i of the result belongs to input i whatever order lanes finish in."""
    _, _, energy = small_system(16, 2, 4.0)
    s0 = _batch(b=4, seed=7)
    cfg = DescentSettings(max_evals=200, step_init=3e-2, energy_tol=0.5)
    perm = np.array([2, 0, 3, 1])
    a = relax(energy, s0, cfg, chunk_iters=5)
    b = relax(energy, s0[perm], cfg, chunk_iters=5)
    assert _same(jax.tree_util.tree_map(lambda x: x[perm], a), b)


def _chunk_compiles(caplog: pytest.LogCaptureFixture) -> int:
    msgs = [r.getMessage() for r in caplog.records]
    return sum(1 for m in msgs if "Compiling" in m and "chunk_one" in m)


def test_relax_compiles_once_per_shape(caplog: pytest.LogCaptureFixture) -> None:
    """A repeated call with the same shapes triggers no new chunk compilation."""
    _, _, energy = small_system(20, 2, 4.0)  # shape unique to this test
    cfg = DescentSettings(max_evals=60, step_init=3e-2)
    with jax.log_compiles(), caplog.at_level(logging.WARNING):
        relax(energy, _batch(b=4, n=20, seed=1), cfg, chunk_iters=4)
        first = _chunk_compiles(caplog)
        relax(energy, _batch(b=4, n=20, seed=2), cfg, chunk_iters=4)
        second = _chunk_compiles(caplog)
    assert first >= 1, "the log filter must see the first compilation"
    assert second == first, "same shapes must not recompile"


def test_relax_odd_batch_with_staggered_finish() -> None:
    """An odd batch with staggered finishes (padding path) completes in full."""
    _, _, energy = small_system(16, 2, 4.0)
    cfg = DescentSettings(max_evals=400, step_init=3e-2, energy_tol=2.0)
    s0 = _batch(b=7, seed=3)
    out = relax(energy, s0, cfg, chunk_iters=2)
    assert out.s.shape[0] == 7 and np.all(np.asarray(out.status) != STATUS_RUNNING)


def test_loop_run_freezes_done_lanes_and_bounds_iterations() -> None:
    """_loop.run: finished lanes are untouched; max_iters bounds the loop."""

    def step(c: jnp.ndarray) -> jnp.ndarray:
        return c + 1.0

    def done(c: jnp.ndarray) -> jnp.ndarray:
        return c >= 3.0

    out = jax.vmap(lambda c: _loop.run(step, done, c, 100))(jnp.array([0.0, 2.0, 9.0]))
    assert np.array_equal(np.asarray(out), [3.0, 3.0, 9.0])
    capped = _loop.run(lambda c: c + 1.0, lambda c: c < 0.0, jnp.asarray(0.0), 4)
    assert float(capped) == 4.0


def test_relax_rejects_non_float64_and_bad_settings() -> None:
    """float32 input raises TypeError; unknown settings raise TypeError."""
    _, _, energy = small_system(8, 2, 4.0)
    with pytest.raises(TypeError):
        relax(energy, jnp.zeros((2, 8, 2), jnp.float32), DescentSettings())
    with pytest.raises(TypeError):
        relax(energy, jnp.zeros((2, 8, 2)), object())  # type: ignore[arg-type]


def test_relax_budget_status_all_lanes() -> None:
    """Out-of-budget lanes finish with STATUS_BUDGET and n_evals = 2 * iterations."""
    _, _, energy = small_system(16, 2, 4.0)
    out = relax(energy, _batch(b=3), DescentSettings(max_evals=21), chunk_iters=4)
    assert np.all(np.asarray(out.status) == STATUS_BUDGET)
    assert np.all(np.asarray(out.n_iters) == 10) and np.all(
        np.asarray(out.n_evals) == 20
    )


# ============================================================
# FIX ROUND 1: bit-exact ground states, output wrap, lane freeze
# ============================================================


def _ground_state(n: int = 16) -> tuple[Any, jnp.ndarray]:
    """Return (energy, s) for a generic non-dyadic Poisson ground state, Phi<1e-20."""
    _, _, energy = small_system(n, 2, 4.0)
    s0 = _batch(b=1, n=n, seed=11)
    res = relax(energy, s0, LBFGSSettings(max_evals_per_pass=300))
    assert float(res.energy[0]) < 1e-20
    s = res.s[0]
    assert not np.array_equal(np.asarray(s) * 2**20, np.round(np.asarray(s) * 2**20))
    return energy, s


@pytest.mark.parametrize("method", ["descent", "lbfgs"])
def test_ground_state_lane_is_returned_bit_exactly(method: str) -> None:
    """A converged lane's s is bit-identical to its input (no s*R/R round trip)."""
    energy, s_in = _ground_state()
    if method == "descent":
        cfg: Settings = DescentSettings(energy_tol=1e-20)
        single = adaptive_descent
    else:
        cfg = LBFGSSettings(energy_tol=1e-20)
        single = lbfgs
    out = single(energy, s_in, cfg)
    assert int(out.status) == STATUS_ENERGY_TOL and int(out.n_iters) == 0
    assert np.array_equal(np.asarray(out.s), np.asarray(s_in))
    batch = relax(energy, jnp.stack([s_in, s_in, s_in]), cfg, chunk_iters=3)
    assert np.array_equal(np.asarray(batch.s), np.asarray(jnp.stack([s_in] * 3)))


def _unwrapped_descent(energy: Any, s0: jnp.ndarray, step0: float, k: int) -> Any:
    """Replay k descent iterations without wrapping (grow = 1, C++ rule)."""
    s, step = s0, step0
    f, g = energy.value_and_grad(s)
    for _ in range(k):
        norm = jnp.sqrt(jnp.sum(g * g))
        xn = s - g / norm * step
        f2, g2 = energy.value_and_grad(xn)
        if abs((f - f2) - step * norm) / (step * norm) > 0.01:
            step *= 0.5
        else:
            s, f, g = xn, f2, g2
    return s


def _edge_config(n: int = 16) -> jnp.ndarray:
    """Random positions with half of the coordinates within 1e-4 of the boundary."""
    s = np.random.default_rng(5).uniform(size=(n, 2))
    s[: n // 2, 0] = 1e-4
    s[n // 2 :, 0] = 1.0 - 1e-4
    return jnp.asarray(s)


def _is_same_mod_1(a: np.ndarray, b: np.ndarray, tol: float = 1e-15) -> bool:
    diff = a - b
    return bool(np.max(np.abs(diff - np.round(diff))) <= tol)


def test_descent_output_is_wrapped_into_unit_cube() -> None:
    """Final s lies in [0, 1) and equals the unwrapped iterate mod 1."""
    _, _, energy = small_system(16, 2, 4.0)
    s0, step0, k = _edge_config(), 0.3, 12
    cfg = DescentSettings(step_init=step0, max_evals=2 * k)
    unwrapped = np.asarray(_unwrapped_descent(energy, s0, step0, k))
    assert unwrapped.min() < 0.0 or unwrapped.max() >= 1.0, "must leave [0, 1)"
    for res in (
        adaptive_descent(energy, s0, cfg),
        _lane(relax(energy, s0[None], cfg), 0),
    ):
        s = np.asarray(res.s)
        assert s.min() >= 0.0 and s.max() < 1.0
        assert _is_same_mod_1(s, unwrapped)


def test_lbfgs_output_is_wrapped_into_unit_cube() -> None:
    """Final s lies in [0, 1) and equals the carried unwrapped s mod 1."""
    from xla_cco.minimize import _loop as loop
    from xla_cco.minimize import lbfgs as lbfgs_mod

    _, _, energy = small_system(16, 2, 4.0)
    s0 = _edge_config()
    cfg = LBFGSSettings(max_evals_per_pass=20)
    state = lbfgs_mod._init_jit(energy, s0, cfg)
    state = loop.drive_single(
        lambda st: lbfgs_mod._chunk_jit(energy, cfg, st, 50), state, 100
    )
    raw = np.asarray(state.s)
    assert raw.min() < 0.0 or raw.max() >= 1.0, "must leave [0, 1)"
    s = np.asarray(lbfgs(energy, s0, cfg).s)
    assert s.min() >= 0.0 and s.max() < 1.0
    assert _is_same_mod_1(s, raw, 1e-12)
    # batched path: compare with its own unwrapped state (vmapped paths may differ)
    bstate = lbfgs_mod.init_batch(energy, s0[None], cfg)
    for _ in range(100):
        if int(bstate.status[0]) != STATUS_RUNNING:
            break
        bstate = lbfgs_mod.chunk_batch(energy, cfg, bstate, 50)
    braw = np.asarray(bstate.s[0])
    assert braw.min() < 0.0 or braw.max() >= 1.0
    sb = np.asarray(relax(energy, s0[None], cfg).s[0])
    assert sb.min() >= 0.0 and sb.max() < 1.0
    assert _is_same_mod_1(sb, braw, 1e-12)


def _freeze_settings(kind: str, s0: jnp.ndarray, energy: Any) -> Settings:
    """Settings whose energy_tol sits just below every start, so lanes finish at
    different iterations (the lowest-energy lane first, mid-chunk)."""
    if kind == "descent":
        base: Settings = DescentSettings(step_init=3e-2, grow=2.0, max_evals=300)
    else:
        base = LBFGSSettings(max_evals_per_pass=20)
    f0 = np.asarray(jax.vmap(energy)(s0))
    return type(base)(**{**_fields(base), "energy_tol": float(0.9 * f0.min())})


@pytest.mark.parametrize("kind", ["descent", "lbfgs"])
def test_finished_lane_with_nonzero_gradient_stays_frozen(kind: str) -> None:
    """A lane that finishes mid-chunk (energy_tol, |g| != 0) is bit-frozen after.

    Also checks it equals the same batch stopped exactly at that iteration, and
    the single-config run (ints exact, floats to the vmap tolerance).
    """
    _, _, energy = small_system(16, 2, 4.0)
    s0 = _batch(b=6, seed=21)
    settings = _freeze_settings(kind, s0, energy)
    long = relax(energy, s0, settings, chunk_iters=10_000)
    iters = np.asarray(long.n_iters)
    lane = int(np.argmin(iters))
    assert 0 < iters[lane] < iters.max(), "lane must finish mid-chunk, others go on"
    assert float(long.grad_norm[lane]) > 1e-8, "finished lane has nonzero gradient"
    assert int(long.status[lane]) == STATUS_ENERGY_TOL
    # same batch, chunk ends exactly where the lane finishes: no idle iterations
    exact = relax(energy, s0, settings, chunk_iters=int(iters[lane]))
    assert _same(_lane(long, lane), _lane(exact, lane))
    single = adaptive_descent if kind == "descent" else lbfgs
    _close(_lane(long, lane), single(energy, s0[lane], settings))
    # later rounds change nothing, including counters
    init, chunk = _batched_fns(settings)
    state = init(energy, s0, settings)
    state = chunk(energy, settings, state, 10_000)
    snap = jax.tree_util.tree_map(lambda x: x[lane], state)
    assert int(snap.status) == STATUS_ENERGY_TOL
    for _ in range(3):
        state = chunk(energy, settings, state, 5)
    assert _same(snap, jax.tree_util.tree_map(lambda x: x[lane], state))
    assert int(snap.it) == int(iters[lane]) and int(snap.n_evals) == int(
        long.n_evals[lane]
    )
