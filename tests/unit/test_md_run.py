#!/usr/bin/env python3

"""
Unit tests of the MD driver (``xla_cco.dynamics.run``).

References are independent of the driver: ``enumerate_kvectors`` and a
brute-force integer grid for the tracked waves, ``jax.grad`` of a direct
Cartesian Phi(r) = (1/|det B|) sum_k |sum_j exp(i k . r_j)|^2 for the force,
Galilean invariance (rigid translation) for the COM rule, and re-runs of
the same lanes for freezing, snapshots and cadences.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import dataclasses
import math
from pathlib import Path
from typing import Any, List, Optional, Tuple

# ----- numerics -----
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

# ----- project -----
from md_states import leaves_bitwise_equal
from strategies import boxes
from systems import stealthy_ground_state
from tolerances import (
    ATOL_MD_COM,
    ATOL_MD_GALILEAN,
    RTOL_MD_FORCE,
    RTOL_MD_KE,
    RTOL_MD_KVEC,
)
from xla_cco.box import Box
from xla_cco.dynamics import run
from xla_cco.dynamics.checkpoint import load_state
from xla_cco.dynamics.state import MDSettings, MDState, Waves
from xla_cco.dynamics.thermal import harmonic_sample, kinetic_energy
from xla_cco.energy import StealthyEnergy, make_stealthy_energy
from xla_cco.kvectors import enumerate_kvectors
from xla_cco.minimize.result import STATUS_DRIFT, STATUS_RUNNING

# ============================================================
# CONSTANTS
# ============================================================

N, D, SIDE = 16, 2, 4.0
K = 2.0 * math.pi / SIDE * 1.6  # |n| < 1.6: (1,0), (0,1), (1,1), (1,-1)
K_TRACK = 2.0 * K
RECT = np.diag([32.0, 16.0 * math.sqrt(3.0)])  # the URL cell (N = 887, density 1)
SHEARED = np.array([[3.0, 0.0], [1.1, 2.6]])  # density 10 / 7.8 != 1
# ground-state system shared with the integration tests (chi = 12 / 62)
GS_N, GS_K = 32, 3.2
FAST = settings(deadline=None, max_examples=40)

# ============================================================
# HELPERS
# ============================================================


def _system(
    basis: Optional[np.ndarray] = None, k: float = K, k_track: float = K_TRACK
) -> Tuple[StealthyEnergy, Waves, jax.Array]:
    """(energy (0 < |k| < k), tracked waves (|k| < k_track), basis)."""
    b = SIDE * np.eye(D) if basis is None else np.asarray(basis, dtype=np.float64)
    kset = enumerate_kvectors(b, 0.0, k)
    energy = make_stealthy_energy(Box(basis=jnp.asarray(b)), kset)
    waves = run.make_waves(b, k, k_track)
    assert waves.n_constrained == kset.n.shape[0], "constrained set = KSet"
    return energy, waves, jnp.asarray(b)


def _settings(**kw: Any) -> MDSettings:
    """Test settings: verlet, dt 0.01, chunks of 4 steps, no drift abort."""
    base = dict(dt=0.01, scheme="verlet", chunk_steps=4, drift_tol=math.inf)
    base.update(kw)
    return MDSettings(**base)  # type: ignore[arg-type]


def _stack(states: List[MDState]) -> MDState:
    """Stack single lanes along a new leading axis."""
    out: MDState = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *states)
    assert out.step.shape == (len(states),), "leading lane axis"
    return out


def _lane(i: int, out: MDState) -> MDState:  # one lane of a batch
    """Lane ``i`` of a batched state."""
    lane: MDState = jax.tree_util.tree_map(lambda x: x[i], out)
    return lane


def _random_lanes(
    seeds: List[int], st: MDSettings, temperature: float = 1e-3,
    v_shift: Optional[np.ndarray] = None,
    system: Optional[Tuple[StealthyEnergy, Waves, jax.Array]] = None,
) -> MDState:  # fmt: skip
    """Batched lanes at uniform random positions with N(0, T) velocities."""
    energy, waves, basis = _system() if system is None else system
    lanes = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        r = rng.uniform(size=(N, D)) @ np.asarray(basis)
        v = math.sqrt(temperature) * rng.standard_normal((N, D))
        if v_shift is not None:
            v = v + v_shift
        lanes.append(run.init_state(
            energy, waves, basis, jnp.asarray(r), jnp.asarray(v),
            jnp.asarray(temperature), jax.random.PRNGKey(seed), st))  # fmt: skip
    return _stack(lanes)


def _gs_lanes(
    seeds: List[int], st: MDSettings, temperature: float
) -> Tuple[MDState, Tuple[StealthyEnergy, Waves, jax.Array]]:
    """Harmonic thermal lanes around the cached N = 32 ground state."""
    s_gs, basis, _ = stealthy_ground_state(GS_N, D, GS_K, 0)
    system = _system(basis, GS_K, 2.0 * GS_K)
    energy, waves = system[0], system[1]
    lanes = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        r = harmonic_sample(rng, energy, s_gs, basis, temperature)
        v = math.sqrt(temperature) * rng.standard_normal((GS_N, D))
        v = v - v.mean(axis=0)
        lanes.append(run.init_state(
            energy, waves, jnp.asarray(basis), jnp.asarray(r), jnp.asarray(v),
            jnp.asarray(temperature), jax.random.PRNGKey(seed), st))  # fmt: skip
    return _stack(lanes), system


def _oracle_phi(r: jax.Array, kvec: jax.Array, volume: float) -> jax.Array:
    """Phi(r) = (1/V) sum_k |sum_j exp(i k . r_j)|^2 (Cartesian, direct)."""
    theta = r @ kvec.T
    re, im = jnp.sum(jnp.cos(theta), axis=0), jnp.sum(jnp.sin(theta), axis=0)
    out: jax.Array = jnp.sum(re * re + im * im) / volume
    return out


def _brute_half_space(basis: np.ndarray, k_lo: float, k_hi: float) -> List[Any]:
    """Integer n (as tuples) with k_lo < |n 2 pi inv(B)^T| < k_hi, half space."""
    d = basis.shape[0]
    recip = 2.0 * math.pi * np.linalg.inv(basis).T
    # |n_i| = |k . a_i| / 2 pi <= k_hi |a_i| / 2 pi
    lim = [int(k_hi * np.linalg.norm(basis[i]) / (2 * math.pi)) + 2 for i in range(d)]
    grid = np.stack(np.meshgrid(*[np.arange(-m, m + 1) for m in lim], indexing="ij"),
                    axis=-1).reshape(-1, d)  # fmt: skip
    first = np.array([row[np.flatnonzero(row)[0]] if row.any() else 0 for row in grid])
    k2 = np.sum((grid @ recip) ** 2, axis=1)
    keep = (first > 0) & (k2 > k_lo**2) & (k2 < k_hi**2)
    return sorted(map(tuple, grid[keep].tolist()))


# ============================================================
# WAVES
# ============================================================


@pytest.mark.critical
@pytest.mark.parametrize("basis", [SIDE * np.eye(2), RECT, SHEARED, np.eye(3) * 3.0])
def test_make_waves_constrained_rows_are_the_kset(basis: np.ndarray) -> None:
    k_c = 2.0 * math.pi / float(np.min(np.abs(np.diag(basis)))) * 1.6
    waves = run.make_waves(basis, k_c, 2.0 * k_c)
    kset = enumerate_kvectors(basis, 0.0, k_c)
    m = kset.n.shape[0]
    assert waves.n_constrained == m and m >= 2
    np.testing.assert_array_equal(np.asarray(waves.n_vectors[:m]), kset.n)
    assert np.asarray(waves.n_vectors).dtype == np.float64
    np.testing.assert_array_equal(np.asarray(waves.constrained),
                                  np.arange(waves.kvec.shape[0]) < m)  # fmt: skip
    kvec = np.asarray(waves.n_vectors) @ (2.0 * math.pi * np.linalg.inv(basis).T)
    np.testing.assert_allclose(np.asarray(waves.kvec), kvec, rtol=0, atol=RTOL_MD_KVEC)
    np.testing.assert_allclose(np.asarray(waves.kvec[:m]), kset.k, rtol=0,
                               atol=RTOL_MD_KVEC * np.abs(kset.k).max())  # fmt: skip


@st.composite
def _wave_cases(draw: Any) -> Tuple[np.ndarray, float, float]:
    basis = draw(boxes(dim=draw(st.sampled_from((2, 3)))))
    k_c = draw(st.floats(2.0, 7.0))
    return basis, k_c, k_c * draw(st.floats(1.0, 2.0))


@pytest.mark.critical
@FAST
@given(case=_wave_cases())
@example(case=(RECT, 0.4, 0.8))
@example(case=(SIDE * np.eye(2), K, K))
@example(case=(2.0 * np.eye(2), 2.0, 4.0))  # |k|_min = pi > 2: M = 0
def test_make_waves_extra_rows_property(case: Tuple[np.ndarray, float, float]) -> None:
    """Extra rows: half space, strict shell, sorted by |k|, no duplicates."""
    basis, k_c, k_t = case
    if not _brute_half_space(basis, 0.0, k_c):
        with pytest.raises(ValueError):  # no constrained wave: no energy
            run.make_waves(basis, k_c, k_t)
        return
    waves = run.make_waves(basis, k_c, k_t)
    m = waves.n_constrained
    extra = np.asarray(waves.n_vectors[m:]).astype(np.int64)
    k_norm = np.linalg.norm(np.asarray(waves.kvec), axis=1)
    assert sorted(map(tuple, extra.tolist())) == _brute_half_space(basis, k_c, k_t)
    assert sorted(map(tuple, np.asarray(waves.n_vectors[:m], np.int64).tolist())) == (
        _brute_half_space(basis, 0.0, k_c)
    )
    assert np.all(np.diff(k_norm[m:]) >= -1e-12 * max(1.0, k_t))
    rows = np.asarray(waves.n_vectors).astype(np.int64)
    both = {tuple(x) for x in rows} | {tuple(-x) for x in rows}
    assert len(both) == 2 * rows.shape[0], "duplicates or +-k pairs"
    assert np.all(k_norm[m:] > k_c) and np.all(k_norm < max(k_t, k_c))


@pytest.mark.medium
def test_make_waves_rejects_bad_cutoffs() -> None:
    for k_c, k_t in [(K, 0.5 * K), (0.0, K), (-1.0, K), (math.nan, K), (K, math.inf)]:
        with pytest.raises(ValueError):
            run.make_waves(SIDE * np.eye(D), k_c, k_t)
    waves = run.make_waves(SIDE * np.eye(D), K, K)
    assert waves.kvec.shape[0] == waves.n_constrained == 4


# ============================================================
# FORCE
# ============================================================


def _force_check(basis: np.ndarray, frac: np.ndarray, k_c: float) -> None:
    """force_fn(r) against -grad of the direct Cartesian Phi (V = |det B|)."""
    energy, waves, b = _system(basis, k_c, k_c)
    m = waves.n_constrained
    r = jnp.asarray(frac @ basis)
    phi, f, re, im = run.make_force_fn(energy, b)(r)
    kvec = waves.kvec[:m]
    vol = abs(float(np.linalg.det(basis)))
    phi_ref, g_ref = jax.value_and_grad(_oracle_phi)(r, kvec, vol)
    scale = max(1.0, float(jnp.abs(g_ref).max()))
    np.testing.assert_allclose(np.asarray(f), -np.asarray(g_ref), rtol=0,
                               atol=RTOL_MD_FORCE * scale)  # fmt: skip
    np.testing.assert_allclose(float(phi), float(phi_ref), rtol=RTOL_MD_FORCE,
                               atol=RTOL_MD_FORCE)  # fmt: skip
    theta = np.asarray(r) @ np.asarray(kvec).T
    n_part = frac.shape[0]
    np.testing.assert_allclose(np.asarray(re[:m]), np.cos(theta).sum(0), rtol=0,
                               atol=RTOL_MD_FORCE * n_part)  # fmt: skip
    np.testing.assert_allclose(np.asarray(im[:m]), np.sin(theta).sum(0), rtol=0,
                               atol=RTOL_MD_FORCE * n_part)  # fmt: skip


@pytest.mark.critical
@pytest.mark.parametrize("basis", [RECT, SHEARED, 3.0 * np.eye(2)])
def test_force_is_minus_cartesian_gradient(basis: np.ndarray) -> None:
    frac = np.random.default_rng(3).uniform(size=(10, 2)) * 1.7 - 0.3  # unwrapped
    k_c = 2.0 * math.pi / float(np.min(np.diag(basis))) * 2.2
    _force_check(basis, frac, k_c)


@pytest.mark.critical
@FAST
@given(basis=boxes(dim=2), seed=st.integers(0, 2**16), n=st.integers(2, 9),
       mult=st.floats(1.1, 2.5))  # fmt: skip
@example(basis=SHEARED, seed=0, n=10, mult=2.0)
def test_force_property(basis: np.ndarray, seed: int, n: int, mult: float) -> None:
    """F = -dPhi/dr on sheared / scaled cells (any density, unwrapped r)."""
    frac = np.random.default_rng(seed).uniform(-1.0, 2.0, size=(n, 2))
    k_c = 2.0 * math.pi / float(np.min(np.diag(basis))) * mult
    _force_check(basis, frac, k_c)


@pytest.mark.medium
def test_force_fn_rejects_bad_basis() -> None:
    energy, _, _ = _system()
    with pytest.raises(ValueError):
        run.make_force_fn(energy, jnp.eye(3))
    with pytest.raises(ValueError):
        run.make_force_fn(energy, jnp.ones((2, 3)))


# ============================================================
# STATE AND SEGMENTS
# ============================================================


@pytest.mark.high
@pytest.mark.parametrize("scheme", ["verlet", "andersen"])
def test_init_state_fields(scheme: str) -> None:
    st_ = _settings(scheme=scheme, record_steps=(2, 5), record_rho_every=2,
                    max_rho_records=3)  # fmt: skip
    energy, waves, basis = _system()
    lane = _lane(0, _random_lanes([4], st_, system=(energy, waves, basis)))
    w = waves.kvec.shape[0]
    r = np.asarray(lane.r)
    rho = np.exp(1j * (r @ np.asarray(waves.kvec).T)).sum(0)
    if scheme != "verlet":
        rho = rho * np.exp(-1j * (np.asarray(waves.kvec) @ r.mean(0)))
    got = np.asarray(lane.rho_prev)
    np.testing.assert_allclose(got[:, 0] + 1j * got[:, 1], rho, atol=1e-12 * N)
    assert lane.step.dtype == jnp.int64 and int(lane.step) == 0
    assert float(lane.e0) == pytest.approx(float(lane.phi + kinetic_energy(lane.v)))
    assert lane.snap.r.shape == (2, N, D) and np.all(np.asarray(lane.snap.step) == -1)
    assert lane.trace.rho.shape == (3, w, 2) and int(lane.trace.count) == 0
    assert int(lane.status) == STATUS_RUNNING and int(lane.snap.count) == 0
    assert np.all(np.asarray(lane.n_valid) == 0) and lane.n_valid.shape == (w,)


@pytest.mark.medium
def test_settings_are_validated() -> None:
    energy, waves, basis = _system()
    bad = [dict(scheme="leapfrog"), dict(dt=0.0), dict(dt=math.nan),
           dict(record_steps=(3, 2)), dict(record_steps=(0, 2)),
           dict(record_steps=(2, 2)), dict(wind_every=0), dict(chunk_steps=0),
           dict(record_rho_every=2, max_rho_records=0), dict(andersen_p=1.5),
           dict(andersen_mode="poisson"), dict(gamma=-1.0), dict(phase_floor=-1.0),
           dict(drift_tol=0.0), dict(record_steps=tuple(range(1, 258)))]  # fmt: skip
    r = jnp.zeros((N, D))
    for kw in bad:
        with pytest.raises(ValueError):
            run.init_state(energy, waves, basis, r, r, jnp.asarray(1e-3),
                           jax.random.PRNGKey(0), _settings(**kw))  # fmt: skip


@pytest.mark.medium
def test_simulate_validates_inputs() -> None:
    st_ = _settings(record_steps=(2,))
    energy, waves, basis = _system()
    lanes = _random_lanes([0], st_)
    with pytest.raises(ValueError):  # not batched
        run.simulate(energy, waves, basis, st_, _lane(0, lanes), 2)
    with pytest.raises(ValueError):
        run.simulate(energy, waves, basis, st_, lanes, -1)
    with pytest.raises(ValueError):  # snapshot capacity differs from record_steps
        run.simulate(energy, waves, basis, _settings(record_steps=(2, 4)), lanes, 2)
    with pytest.raises(ValueError):  # trace capacity differs
        other = _settings(record_steps=(2,), max_rho_records=3)
        run.simulate(energy, waves, basis, other, lanes, 2)
    out = run.simulate(energy, waves, basis, st_, lanes, 0)
    assert leaves_bitwise_equal(out.state, lanes) and out.trace is None


@pytest.mark.high
@FAST
@given(v=hnp.arrays(np.float64, (N, D), elements=st.floats(-3.0, 3.0)),
       flags=st.tuples(st.booleans(), st.booleans(), st.booleans()))  # fmt: skip
@example(v=np.full((N, D), 0.5), flags=(True, True, True))
def test_begin_segment_property(v: np.ndarray, flags: Tuple[bool, bool, bool]) -> None:
    remove_com, reset_sums, reset_winding = flags
    st_ = _settings(remove_com=remove_com)
    lanes = run.simulate(*_system(), st_, _random_lanes([1], st_), 3).state
    lane = dataclasses.replace(_lane(0, lanes), v=jnp.asarray(v))
    out = run.begin_segment(lane, st_, reset_sums, reset_winding)
    v_exp = v - v.mean(0) if remove_com else v
    np.testing.assert_allclose(np.asarray(out.v), v_exp, rtol=0, atol=ATOL_MD_COM * 8)
    ke = float(kinetic_energy(out.v))
    assert float(out.e0) == pytest.approx(float(lane.phi) + ke, rel=RTOL_MD_KE)
    assert float(out.max_drift) == 0.0 and int(out.step) == int(lane.step) == 3
    sums_zero = float(out.sum_phi) == 0.0 and int(out.n_sum) == 0
    assert sums_zero == reset_sums and np.all(np.asarray(out.sum_s) == 0) == reset_sums
    wind_zero = np.all(np.asarray(out.wind_abs) == 0) and np.all(out.n_valid == 0)
    assert bool(wind_zero) == reset_winding
    assert leaves_bitwise_equal(out.snap, lane.snap) and leaves_bitwise_equal(
        (out.r, out.f, out.key, out.trace), (lane.r, lane.f, lane.key, lane.trace)
    )


@pytest.mark.critical
def test_regression_D060_begin_segment_recomputes_rho_prev_on_scheme_switch() -> None:
    """D-060: winding after Andersen -> begin_segment -> NVE uses the raw rho.

    The Andersen phase stores COM-corrected rho; the first NVE step must
    compare raw rho at the segment start with raw rho one step later.
    """
    energy, waves, basis = _system()
    st_a = _settings(scheme="andersen", andersen_p=0.0, remove_com=False)
    st_v = _settings(scheme="verlet", remove_com=False)
    lanes = _random_lanes([2], st_a, v_shift=np.array([0.4, -0.3]))
    lanes = run.simulate(energy, waves, basis, st_a, lanes, 6).state
    lanes = jax.vmap(lambda s: run.begin_segment(s, st_v, True, True))(lanes)
    r0 = np.asarray(lanes.r[0])
    out = run.simulate(energy, waves, basis, st_v, lanes, 1).state
    kv = np.asarray(waves.kvec)
    rho0 = np.exp(1j * (r0 @ kv.T)).sum(0)
    rho1 = np.exp(1j * (np.asarray(out.r[0]) @ kv.T)).sum(0)
    delta = np.angle(rho1 * np.conj(rho0))
    np.testing.assert_allclose(np.asarray(out.wind_net[0]), delta, atol=1e-10)
    np.testing.assert_allclose(np.asarray(out.wind_abs[0]), np.abs(delta), atol=1e-10)
    assert np.all(np.asarray(out.n_valid[0]) == 1)


# ============================================================
# LANES, SNAPSHOTS, CADENCES
# ============================================================


@pytest.mark.critical
def test_drifted_lane_is_frozen_and_does_not_perturb_others() -> None:
    st_ = _settings(drift_tol=1.0, record_steps=(1, 5, 9), record_rho_every=2,
                    max_rho_records=8)  # fmt: skip
    system = _system()
    good = _random_lanes([0, 1], st_, system=system)
    bad = dataclasses.replace(good, e0=good.e0.at[1].add(1e6))
    ref = run.simulate(*system, st_, good, 10).state
    out = run.simulate(*system, st_, bad, 10).state
    one = run.simulate(*system, st_, bad, 1).state
    assert np.asarray(ref.status).tolist() == [STATUS_RUNNING] * 2
    assert np.asarray(out.status).tolist() == [STATUS_RUNNING, STATUS_DRIFT]
    assert leaves_bitwise_equal(_lane(0, out), _lane(0, ref))
    assert int(out.step[1]) == 1 and int(out.snap.count[1]) == 1
    assert leaves_bitwise_equal(_lane(1, out), _lane(1, one))
    more = run.simulate(*system, st_, out, 7).state
    assert leaves_bitwise_equal(_lane(1, more), _lane(1, out))
    assert int(more.step[0]) == 17 and int(more.trace.count[1]) == 0


@pytest.mark.critical
def test_snapshots_at_absolute_record_steps_across_calls() -> None:
    steps = (1, 3, 7, 12, 20)
    st_ = _settings(record_steps=steps, record_rho_every=3, max_rho_records=5)
    system = _system()
    lanes = _random_lanes([0, 1], st_, system=system)
    a = run.simulate(*system, st_, lanes, 10)
    b = run.simulate(*system, st_, a.state, 10)
    assert np.asarray(a.snapshots.count).tolist() == [3, 3]
    assert np.asarray(a.snapshots.step)[:, 3:].tolist() == [[-1, -1]] * 2
    assert np.asarray(b.snapshots.step).tolist() == [list(steps)] * 2
    assert np.asarray(b.snapshots.count).tolist() == [5, 5]
    assert b.snapshots is b.state.snap and b.trace is b.state.trace
    for j, s in enumerate(steps):
        ref = run.simulate(*system, st_, lanes, s).state
        snap = b.snapshots
        pairs = [
            (snap.r[:, j], ref.r),
            (snap.time[:, j], ref.time),
            (snap.wind_abs[:, j], ref.wind_abs),
            (snap.phi[:, j], ref.phi),
            (snap.wind_net[:, j], ref.wind_net),
            (snap.rho[:, j], ref.rho_prev),
        ]
        for got, want in pairs:  # fmt: skip
            assert np.asarray(got).tobytes() == np.asarray(want).tobytes(), (j, s)
        ke = jax.vmap(kinetic_energy)(ref.v)  # eager: another summation order
        np.testing.assert_allclose(np.asarray(snap.kinetic[:, j]), np.asarray(ke),
                                   rtol=RTOL_MD_KE, atol=0)  # fmt: skip


@pytest.mark.high
def test_trace_cadence_and_capacity() -> None:
    st_ = _settings(record_rho_every=3, max_rho_records=4)
    system = _system()
    lanes = _random_lanes([5], st_, system=system)
    out = run.simulate(*system, st_, lanes, 14)
    assert out.trace is not None and int(out.trace.count[0]) == 4
    np.testing.assert_allclose(np.asarray(out.trace.time[0]),
                               0.01 * np.array([3, 6, 9, 12]), rtol=1e-12)  # fmt: skip
    ref = run.simulate(*system, st_, lanes, 9).state
    assert np.asarray(out.trace.rho[0, 2]).tobytes() == np.asarray(
        ref.rho_prev[0]).tobytes()  # fmt: skip
    off = run.simulate(*system, _settings(max_rho_records=4), lanes, 3)
    assert off.trace is None and int(off.state.trace.count[0]) == 0


@pytest.mark.critical
def test_wind_every_constrained_always_unconstrained_on_multiples() -> None:
    st_ = _settings(wind_every=3, chunk_steps=4)
    system = _system()
    nc = system[1].n_constrained
    lanes = _random_lanes([0, 1], st_, system=system)
    a = run.simulate(*system, st_, lanes, 10).state  # steps 1..10: 3, 6, 9
    nv = np.asarray(a.n_valid)
    assert np.all(nv[:, :nc] == 10) and np.all(nv[:, nc:] == 3)
    b = run.simulate(*system, st_, a, 5).state  # steps 11..15: 12, 15
    nv = np.asarray(b.n_valid)
    assert np.all(nv[:, :nc] == 15) and np.all(nv[:, nc:] == 5)
    # lanes at different steps (per-lane cadence): lane 0 at 0, lane 1 at 10
    mixed = _stack([_lane(0, lanes), _lane(1, a)])
    c = run.simulate(*system, st_, mixed, 5).state
    nv = np.asarray(c.n_valid)
    assert np.all(nv[0, nc:] == 1) and np.all(nv[1, nc:] == 3 + 2)
    assert np.all(nv[:, :nc] == np.array([[5], [15]]))
    np.testing.assert_allclose(np.asarray(c.r[1]), np.asarray(b.r[1]), rtol=0,
                               atol=1e-13)  # fmt: skip


@pytest.mark.high
def test_md_chunk_under_vmap_matches_simulate() -> None:
    st_ = _settings(
        scheme="andersen", andersen_p=0.3, wind_every=2, record_steps=(2, 3),
        record_rho_every=1, max_rho_records=4,
    )  # fmt: skip
    energy, waves, basis = _system()
    lanes = _random_lanes([3, 4], st_)

    @jax.jit
    def chunk(s: MDState) -> MDState:
        return run.md_chunk(energy, waves, basis, st_, s, jnp.asarray(3))

    got = jax.vmap(chunk)(lanes)
    ref = run.simulate(energy, waves, basis, st_, lanes, 3).state
    assert np.asarray(got.step).tolist() == [3, 3]
    for a, b in zip(jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(ref),
                    strict=True):  # fmt: skip
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-13,
                                   atol=1e-13)  # fmt: skip
    single = run.md_chunk(energy, waves, basis, st_, _lane(0, lanes), jnp.asarray(9))
    assert int(single.step) == 4  # min(chunk_steps, n_steps)


@pytest.mark.critical
@pytest.mark.parametrize("scheme", ["verlet", "andersen", "baoab"])
def test_com_rule_under_rigid_translation(scheme: str) -> None:
    """Galilean boost V: raw rho gains exp(i k . V t) (verlet); corrected none."""
    st_ = _settings(scheme=scheme, andersen_p=0.0, gamma=0.0, remove_com=False)
    system = _system()
    boost = np.array([0.3, -0.17])
    rest = _random_lanes([7], st_, temperature=0.0, system=system)
    moving = dataclasses.replace(rest, v=rest.v + jnp.asarray(boost))
    moving = jax.vmap(lambda s: run.begin_segment(s, st_, True, True))(moving)
    rest = jax.vmap(lambda s: run.begin_segment(s, st_, True, True))(rest)
    a = run.simulate(*system, st_, rest, 50).state
    b = run.simulate(*system, st_, moving, 50).state
    t = float(b.time[0])
    np.testing.assert_allclose(np.asarray(b.r - a.r)[0], np.tile(boost * t, (N, 1)),
                               atol=ATOL_MD_GALILEAN)  # fmt: skip
    shift = np.asarray(system[1].kvec) @ boost * t
    expect = shift if scheme == "verlet" else np.zeros_like(shift)
    assert np.abs(shift).min() > 1e3 * ATOL_MD_GALILEAN
    np.testing.assert_allclose(np.asarray(b.wind_net - a.wind_net)[0], expect,
                               atol=ATOL_MD_GALILEAN)  # fmt: skip


@pytest.mark.critical
def test_drift_abort_on_bad_dt_only() -> None:
    good = _settings(dt=0.01, drift_tol=1e-3, chunk_steps=500)
    lanes, system = _gs_lanes([0, 1], good, 1e-6)
    ok = run.simulate(*system, good, lanes, 1500).state
    thr = 1e-3 * D * (GS_N - 1) * 1e-6 / 2
    assert np.all(np.asarray(ok.status) == STATUS_RUNNING)
    assert np.all(np.asarray(ok.max_drift) < 0.5 * thr) and np.all(ok.max_drift > 0)
    bad = dataclasses.replace(good, dt=0.3)
    lanes_bad = jax.vmap(lambda s: run.begin_segment(s, bad, True, True))(lanes)
    out = run.simulate(*system, bad, lanes_bad, 200).state
    assert np.all(np.asarray(out.status) == STATUS_DRIFT)
    assert np.all(np.asarray(out.step) < 200) and np.all(out.max_drift > thr)


# ============================================================
# DT SELECTION AND CHECKPOINTS
# ============================================================


def _nve_rel_drift(
    system: Tuple[StealthyEnergy, Waves, jax.Array], lanes: MDState, dt: float,
    n_steps: int,
) -> np.ndarray:  # fmt: skip
    """|E_last - E_first| / <KE> per lane for an independent NVE run."""
    st_ = _settings(dt=dt, chunk_steps=n_steps, remove_com=False)
    fresh = jax.vmap(lambda s: run.begin_segment(s, st_, True, True))(lanes)
    out = run.simulate(*system, st_, fresh, n_steps).state
    e_last = np.asarray(out.phi + jax.vmap(kinetic_energy)(out.v))
    ke = np.asarray(out.sum_kinetic) / np.asarray(out.n_sum)
    return np.asarray(np.abs(e_last - np.asarray(fresh.e0)) / ke)


@pytest.mark.high
def test_select_dt_returns_largest_passing_dt() -> None:
    lanes, system = _gs_lanes([3, 4], _settings(), 1e-6)
    ladder, n_steps, rel = (0.005, 0.01, 0.02, 0.35), 300, 2e-3
    drifts = {dt: _nve_rel_drift(system, lanes, dt, n_steps) for dt in ladder}
    assert np.all(drifts[0.02] < rel) and not np.all(drifts[0.35] < rel), drifts
    got = run.select_dt(*system, lanes, ladder, n_steps, rel)
    assert got == 0.02
    assert run.select_dt(*system, lanes, ladder[::-1], n_steps, rel) == 0.02
    with pytest.raises(ValueError):
        run.select_dt(*system, lanes, (0.35,), n_steps, rel)
    with pytest.raises(ValueError):
        run.select_dt(*system, lanes, (), n_steps, rel)


@pytest.mark.high
def test_simulate_writes_checkpoints_and_never_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    st_ = _settings(record_steps=(2, 9))
    system = _system()
    lanes = _random_lanes([0, 1], st_, system=system)
    path = tmp_path / "c.npz"
    path.write_bytes(b"garbage, not a checkpoint")
    calls: List[int] = []
    real = run.save_state

    def spy(p: Path, state: MDState, meta: Any) -> None:
        calls.append(int(np.max(np.asarray(state.step))))
        real(p, state, meta)

    monkeypatch.setattr(run, "save_state", spy)
    out = run.simulate(*system, st_, lanes, 10, checkpoint=path,
                       checkpoint_every_s=1e-9)  # fmt: skip
    assert calls == [4, 8, 10]
    back, meta = load_state(path, out.state)
    assert leaves_bitwise_equal(back, out.state) and meta["steps_done"] == 10
    calls.clear()
    run.simulate(*system, st_, lanes, 10, checkpoint=path, checkpoint_every_s=1e9)
    assert calls == [10]
    with pytest.raises(ValueError):
        run.simulate(*system, st_, lanes, 1, checkpoint=path, checkpoint_every_s=0.0)
