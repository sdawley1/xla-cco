#!/usr/bin/env python3

"""
MD driver: force function, lane initialization, segments, chunked simulation.

``simulate`` runs a batch of lanes (MDState with a leading axis) for
``n_steps`` steps in jitted chunks of ``settings.chunk_steps`` (a bounded
lax.while_loop per lane, as minimize/flow.py), with lane freezing on
status != STATUS_RUNNING, snapshots at ``settings.record_steps``, the dense
rho trace, energy-drift monitoring ("verlet" only) and optional
checkpointing between chunks. Per step: integrate (scheme), Andersen
resets if "andersen", rho of the tracked waves (constrained ones from the
force pass), COM correction when the scheme is not "verlet", winding,
running sums, snapshot/trace, drift check.

Details:

- The unconstrained waves are evaluated only on steps with
  step % wind_every == 0 (lax.cond). ``simulate`` passes the common step of
  the running lanes as an unbatched argument, so under vmap the cond stays
  a branch and the off-cadence steps skip the extra waves; lanes at
  different steps fall back to the per-lane predicate (a select).
- ``begin_segment`` has no access to the waves, so with ``reset_winding``
  it marks rho_prev pending (NaN); the next ``md_chunk`` recomputes it from
  the current positions with the COM rule of its scheme before the first
  step (D-060). The constrained rows of rho_prev are otherwise always the
  current rho; the unconstrained rows date from the last cadence step.
- Snapshots and the trace store the latest rho (rho_prev after the step),
  COM-corrected when the scheme is not "verlet".

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import dataclasses
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

# ----- numerics -----
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float, UInt32

# ----- project -----
from xla_cco._types import require_float64
from xla_cco.dynamics.checkpoint import save_state
from xla_cco.dynamics.integrators import (
    ForceFn,
    ForceOut,
    andersen_reset,
    baoab_step,
    verlet_step,
)
from xla_cco.dynamics.observables import (
    center_of_mass,
    com_correct,
    wave_densities,
    winding_update,
)
from xla_cco.dynamics.state import (
    ANDERSEN_MODES,
    SCHEMES,
    MDResult,
    MDSettings,
    MDState,
    RhoTrace,
    Snapshots,
    Waves,
)
from xla_cco.dynamics.thermal import kinetic_energy, require_key
from xla_cco.energy import StealthyEnergy
from xla_cco.kvectors import enumerate_kvectors
from xla_cco.minimize.result import STATUS_DRIFT, STATUS_RUNNING

# ============================================================
# CONSTANTS
# ============================================================

MAX_RECORD_STEPS: int = 256
BUFFER_FIELDS: Tuple[str, ...] = ("snap", "trace")
CHECKPOINT_SOURCE: str = "xla_cco.dynamics.run.simulate"

# ============================================================
# VALIDATION
# ============================================================


def _check_settings(settings: MDSettings) -> None:
    """Raise ValueError on settings the driver cannot run (host, static)."""
    s = settings
    if s.scheme not in SCHEMES or s.andersen_mode not in ANDERSEN_MODES:
        raise ValueError(f"unknown scheme {s.scheme!r} or mode {s.andersen_mode!r}")
    if not (math.isfinite(s.dt) and s.dt > 0.0):
        raise ValueError(f"dt must be finite and positive, got {s.dt}")
    if not (0.0 <= s.andersen_p <= 1.0) or not (0.0 <= s.gamma < math.inf):
        raise ValueError(f"need 0 <= andersen_p <= 1 and gamma >= 0: {s}")
    if s.chunk_steps < 1 or s.wind_every < 1:
        raise ValueError("chunk_steps and wind_every must be >= 1")
    rs = tuple(s.record_steps)
    if len(rs) > MAX_RECORD_STEPS or any(int(x) != x or x < 1 for x in rs):
        raise ValueError(f"record_steps: at most {MAX_RECORD_STEPS} integers >= 1")
    if any(b <= a for a, b in zip(rs, rs[1:], strict=False)):
        raise ValueError("record_steps must be strictly increasing")
    if s.record_rho_every < 0 or s.max_rho_records < 0:
        raise ValueError("record_rho_every and max_rho_records must be >= 0")
    if s.record_rho_every > 0 and s.max_rho_records < 1:
        raise ValueError("a rho trace (record_rho_every > 0) needs max_rho_records")
    if not (s.phase_floor >= 0.0) or not (s.drift_tol > 0.0):
        raise ValueError("need phase_floor >= 0 and drift_tol > 0")


def _check_lane(state: MDState, waves: Waves, settings: MDSettings) -> None:
    """Raise ValueError unless one lane's buffers fit the waves and settings."""
    w = int(waves.kvec.shape[0])
    if state.r.ndim != 2 or state.rho_prev.shape != (w, 2):
        raise ValueError(
            f"need one lane with (n, d) positions and ({w}, 2) rho, got "
            f"{state.r.shape} and {state.rho_prev.shape}"
        )
    if state.snap.step.shape != (len(settings.record_steps),):
        raise ValueError(
            f"snapshot capacity {state.snap.step.shape} != "
            f"{len(settings.record_steps)} record_steps"
        )
    if state.trace.rho.shape[:2] != (settings.max_rho_records, w):
        raise ValueError(
            f"trace capacity {state.trace.rho.shape[:2]} != "
            f"({settings.max_rho_records}, {w})"
        )


# ============================================================
# WAVES AND FORCES
# ============================================================


def make_waves(basis: np.ndarray, k_constraint: float, k_track: float) -> Waves:
    """Tracked waves: the constrained set (|k| < k_constraint, KSet order)
    then the half-space waves with k_constraint < |k| < k_track by |k|.

    Both cutoffs are strict (a wave with |k| = k_constraint is in neither
    set); ties in |k| keep the ``enumerate_kvectors`` order (n lexicographic).
    kvec = n @ 2 pi inv(B)^T.

    Raises
    ------
    ValueError
        Unless 0 < k_constraint <= k_track < inf, on a bad basis, or when
        no wave has |k| < k_constraint (no constrained set, no energy).
    """
    b = np.asarray(basis, dtype=np.float64)
    if not (0.0 < k_constraint <= k_track < math.inf):
        raise ValueError(f"need 0 < k_constraint <= k_track < inf, got "
                         f"{k_constraint}, {k_track}")  # fmt: skip
    inner = enumerate_kvectors(b, 0.0, k_constraint)
    if inner.n.shape[0] == 0:
        raise ValueError(f"no constrained wave with |k| < {k_constraint}")
    blocks = [inner.n]
    if k_track > k_constraint:
        blocks.append(enumerate_kvectors(b, k_constraint, k_track).n)
    n_vec = np.concatenate(blocks).astype(np.float64)
    kvec = n_vec @ (2.0 * math.pi * np.linalg.inv(b).T)
    m = int(inner.n.shape[0])
    assert n_vec.shape == kvec.shape and n_vec.shape[1] == b.shape[0], "shapes"
    assert 1 <= m <= n_vec.shape[0], "constrained rows come first"
    return Waves(
        n_vectors=jnp.asarray(n_vec),
        kvec=jnp.asarray(kvec),
        constrained=jnp.asarray(np.arange(n_vec.shape[0]) < m),
        n_constrained=m,
    )


def make_force_fn(energy: StealthyEnergy, basis: Float[Array, "d d"]) -> ForceFn:
    """r -> (Phi, F, re, im) via ``energy.value_grad_density`` at s = r inv(B).

    F = -dPhi/ds inv(B)^T (as Box.cartesian_forces), with the inverse built
    once; re/im are the energy's densities including its padding rows.

    Raises
    ------
    ValueError
        If ``basis`` is not (d, d) with d the energy's dimension.
    """
    b = jnp.asarray(basis, dtype=jnp.float64)
    d = int(energy.n_vectors.shape[1])
    if b.shape != (d, d):
        raise ValueError(f"basis must have shape ({d}, {d}), got {b.shape}")
    inv = jnp.linalg.inv(b)
    inv_t = inv.T

    def force_fn(r: Float[Array, "n d"]) -> ForceOut:
        assert r.ndim == 2 and r.shape[1] == d, "r must have shape (n, d)"
        phi, g, re, im = energy.value_grad_density(r @ inv)
        f = -g @ inv_t
        assert f.shape == r.shape, "force shape"
        return phi, f, re, im

    return force_fn


def _lane_rho(
    r: Array, out: ForceOut, rho_prev: Array, waves: Waves, settings: MDSettings,
    wind: Union[bool, Array], chunk: Optional[int],
) -> Tuple[Array, Array]:  # fmt: skip
    """(rho (w, 2), active (w,)): constrained rows from the force pass, the
    others fresh where ``wind`` (else rho_prev's); COM-corrected unless verlet.
    """
    nc = waves.n_constrained
    corrected = settings.scheme != "verlet"
    r_cm = center_of_mass(r)
    rho_c = jnp.stack([out[2][:nc], out[3][:nc]], axis=-1)
    if corrected:
        rho_c = com_correct(rho_c, waves.kvec[:nc], r_cm)
    w = int(waves.kvec.shape[0])
    assert rho_prev.shape == (w, 2) and nc <= w, "rho_prev must be (w, 2)"
    if w == nc:
        return rho_c, jnp.ones(w, dtype=bool)
    k_u = waves.kvec[nc:]

    def fresh(x: Array) -> Array:
        rho = wave_densities(x, k_u, chunk)
        return com_correct(rho, k_u, r_cm) if corrected else rho

    if isinstance(wind, bool):
        assert wind, "a static wind flag is only ever True"
        return jnp.concatenate([rho_c, fresh(r)]), jnp.ones(w, dtype=bool)
    rho_u = jax.lax.cond(wind, fresh, lambda x: rho_prev[nc:], r)
    return jnp.concatenate([rho_c, rho_u]), waves.constrained | wind


# ============================================================
# STATE
# ============================================================


def _empty_buffers(
    n: int, d: int, w: int, settings: MDSettings
) -> Tuple[Snapshots, RhoTrace]:
    """Snapshot (step -1 = empty) and trace buffers of one lane."""
    n_rec, n_tr = len(settings.record_steps), settings.max_rho_records
    assert n_rec <= MAX_RECORD_STEPS and n_tr >= 0, "validated capacities"
    zi = jnp.zeros((), jnp.int64)
    snap = Snapshots(
        step=jnp.full((n_rec,), -1, jnp.int64), time=jnp.zeros(n_rec),
        wind_abs=jnp.zeros((n_rec, w)), wind_net=jnp.zeros((n_rec, w)),
        rho=jnp.zeros((n_rec, w, 2)), r=jnp.zeros((n_rec, n, d)),
        phi=jnp.zeros(n_rec), kinetic=jnp.zeros(n_rec), count=zi,
    )  # fmt: skip
    trace = RhoTrace(rho=jnp.zeros((n_tr, w, 2)), time=jnp.zeros(n_tr), count=zi)
    assert snap.r.dtype == trace.rho.dtype == jnp.float64, "float64 buffers"
    return snap, trace


def init_state(
    energy: StealthyEnergy, waves: Waves, basis: Float[Array, "d d"],
    r0: Float[Array, "n d"], v0: Float[Array, "n d"],
    temperature: Float[Array, ""], key: UInt32[Array, " 2"], settings: MDSettings,
) -> MDState:  # fmt: skip
    """One lane at step 0: forces, rho_prev, zero winding/sums, empty buffers.

    e0 = Phi + KE(v0) (velocities as given; ``begin_segment`` removes the
    COM momentum). rho_prev follows the COM rule of ``settings.scheme``.
    Only static checks, so it can be vmapped.

    Raises
    ------
    ValueError
        On invalid settings or shapes.
    TypeError
        On non-float64 positions/velocities or a bad PRNG key.
    """
    _check_settings(settings)
    r0, v0 = jnp.asarray(r0), jnp.asarray(v0)
    require_float64(r0, "r0")
    require_float64(v0, "v0")
    require_key(key)
    if r0.ndim != 2 or v0.shape != r0.shape:
        raise ValueError(
            f"r0 and v0 must share an (n, d) shape: {r0.shape}, {v0.shape}"
        )
    n, d = r0.shape
    w = int(waves.kvec.shape[0])
    if waves.kvec.shape != (w, d):
        raise ValueError(f"waves have dimension {waves.kvec.shape}, positions {d}")
    out = make_force_fn(energy, basis)(r0)
    rho, _ = _lane_rho(r0, out, jnp.zeros((w, 2)), waves, settings, True,
                       energy.chunk_size)  # fmt: skip
    snap, trace = _empty_buffers(n, d, w, settings)
    zf, zi = jnp.zeros(()), jnp.zeros((), jnp.int64)
    assert out[1].shape == r0.shape and rho.shape == (w, 2), "force / rho shapes"
    return MDState(
        r=r0, v=v0, f=out[1], phi=out[0],
        temperature=jnp.asarray(temperature, jnp.float64), key=key, step=zi,
        time=zf, reset_acc=zf, rho_prev=rho, wind_abs=jnp.zeros(w),
        wind_net=jnp.zeros(w), n_valid=jnp.zeros(w, jnp.int64),
        e0=out[0] + kinetic_energy(v0), max_drift=zf, sum_kinetic=zf, sum_phi=zf,
        sum_phi2=zf, sum_s=jnp.zeros(w), n_sum=zi,
        status=jnp.asarray(STATUS_RUNNING, jnp.int64), snap=snap, trace=trace,
    )  # fmt: skip


def begin_segment(
    state: MDState, settings: MDSettings, reset_sums: bool, reset_winding: bool
) -> MDState:
    """Start a segment: optional COM momentum removal (settings.remove_com),
    e0 = Phi + KE, max_drift = 0, optional reset of sums and winding.

    With ``reset_winding`` (wind_abs, wind_net, n_valid zeroed) rho_prev is
    marked pending (NaN): the next ``md_chunk`` recomputes it from the
    current positions with the COM rule of its scheme (D-060). Works under
    jax.vmap; positions, forces, key, step and buffers are unchanged.
    """
    assert state.v.ndim == 2, "begin_segment acts on one lane (vmap for a batch)"
    v = state.v - jnp.mean(state.v, axis=0) if settings.remove_com else state.v
    zf = jnp.zeros_like(state.phi)
    new = dataclasses.replace(
        state, v=v, e0=state.phi + kinetic_energy(v), max_drift=zf
    )
    if reset_sums:
        new = dataclasses.replace(
            new, sum_kinetic=zf, sum_phi=zf, sum_phi2=zf,
            sum_s=jnp.zeros_like(state.sum_s), n_sum=jnp.zeros_like(state.n_sum),
        )  # fmt: skip
    if reset_winding:
        new = dataclasses.replace(
            new, wind_abs=jnp.zeros_like(state.wind_abs),
            wind_net=jnp.zeros_like(state.wind_net),
            n_valid=jnp.zeros_like(state.n_valid),
            rho_prev=jnp.full_like(state.rho_prev, jnp.nan),
        )  # fmt: skip
    assert new.v.shape == state.v.shape, "velocities keep their shape"
    return new


# ============================================================
# ONE STEP
# ============================================================


class _Ctx(eqx.Module):
    """Static context of a chunk: force function, waves, settings, k chunk."""

    force_fn: Callable[[Array], ForceOut] = eqx.field(static=True)
    waves: Waves
    settings: MDSettings
    chunk: Optional[int] = eqx.field(static=True)


def _integrate(st: MDState, ctx: _Ctx) -> Tuple[Array, Array, Array, Array, ForceOut]:
    """(key', r', v', reset_acc', force output) of one step of the scheme."""
    s = ctx.settings
    key, acc = st.key, st.reset_acc
    if s.scheme == "baoab":
        key, r, v, out = baoab_step(key, st.r, st.v, st.f, ctx.force_fn, s.dt,
                                    s.gamma, st.temperature)  # fmt: skip
    else:
        r, v, out = verlet_step(st.r, st.v, st.f, ctx.force_fn, s.dt)
        if s.scheme == "andersen":
            key, v, acc, _ = andersen_reset(key, v, acc, st.temperature,
                                            s.andersen_p, s.andersen_mode)  # fmt: skip
    assert key.shape == st.key.shape and acc.shape == (), "key / accumulator"
    assert r.shape == st.r.shape and v.shape == st.v.shape, "step changed shapes"
    return key, r, v, acc, out


def _step(st: MDState, ctx: _Ctx, wind: Union[bool, Array]) -> Tuple[MDState, Array]:
    """One unfrozen step (integrate, rho, winding, sums, drift); returns KE too."""
    s = ctx.settings
    n, d = st.r.shape
    key, r, v, acc, out = _integrate(st, ctx)
    rho, active = _lane_rho(r, out, st.rho_prev, ctx.waves, s, wind, ctx.chunk)
    rho_prev, w_abs, w_net, n_valid = winding_update(
        st.rho_prev, rho, st.wind_abs, st.wind_net, st.n_valid, active, n,
        s.phase_floor,
    )  # fmt: skip
    phi, ke = out[0], kinetic_energy(v)
    assert phi.shape == ke.shape == (), "scalar energies"
    drift = jnp.abs(phi + ke - st.e0)
    status = st.status
    if s.scheme == "verlet":
        limit = s.drift_tol * 0.5 * d * (n - 1) * st.temperature
        status = jnp.where(drift <= limit, status, STATUS_DRIFT)  # NaN aborts
    new = dataclasses.replace(
        st, r=r, v=v, f=out[1], phi=phi, key=key, step=st.step + 1,
        time=st.time + s.dt, reset_acc=acc, rho_prev=rho_prev, wind_abs=w_abs,
        wind_net=w_net, n_valid=n_valid, max_drift=jnp.maximum(st.max_drift, drift),
        sum_kinetic=st.sum_kinetic + ke, sum_phi=st.sum_phi + phi,
        sum_phi2=st.sum_phi2 + phi * phi,
        sum_s=st.sum_s + jnp.sum(rho_prev * rho_prev, axis=-1) / n,
        n_sum=st.n_sum + 1, status=status.astype(st.status.dtype),
    )  # fmt: skip
    assert new.rho_prev.shape == st.rho_prev.shape, "rho_prev keeps its shape"
    return new, ke


def _freeze(old: MDState, new: MDState, running: Array) -> MDState:
    """Keep every non-buffer leaf of ``old`` where the lane was not running."""
    names = [f.name for f in dataclasses.fields(old) if f.name not in BUFFER_FIELDS]
    assert len(names) == len(dataclasses.fields(old)) - 2, "two buffer fields"
    picked: Dict[str, Any] = {
        k: jnp.where(running, getattr(new, k), getattr(old, k)) for k in names
    }
    assert running.shape == (), "one running flag per lane"
    return dataclasses.replace(new, **picked)


def _put(buf: Array, row: Array, take: Array, value: Array) -> Array:
    """buf with row ``row`` replaced by ``value`` where ``take`` (in place)."""
    assert buf.shape[1:] == jnp.shape(value), "row shape"
    assert take.shape == (), "scalar predicate"
    return buf.at[row].set(jnp.where(take, value, buf[row]))


def _record(st: MDState, ke: Array, settings: MDSettings, running: Array) -> MDState:
    """Write the snapshot (step in record_steps) and the trace row if due."""
    snap, trace = st.snap, st.trace
    assert running.shape == ke.shape == (), "one lane"
    if settings.record_steps:
        hit = jnp.asarray(settings.record_steps, st.step.dtype) == st.step
        take, j = jnp.any(hit) & running, jnp.argmax(hit)
        snap = Snapshots(
            step=_put(snap.step, j, take, st.step),
            time=_put(snap.time, j, take, st.time),
            wind_abs=_put(snap.wind_abs, j, take, st.wind_abs),
            wind_net=_put(snap.wind_net, j, take, st.wind_net),
            rho=_put(snap.rho, j, take, st.rho_prev), r=_put(snap.r, j, take, st.r),
            phi=_put(snap.phi, j, take, st.phi),
            kinetic=_put(snap.kinetic, j, take, ke),
            count=snap.count + take.astype(snap.count.dtype),
        )  # fmt: skip
    every, cap = settings.record_rho_every, settings.max_rho_records
    if every > 0 and cap > 0:
        take = (st.step % every == 0) & (trace.count < cap) & running
        row = jnp.minimum(trace.count, cap - 1)
        trace = RhoTrace(
            rho=_put(trace.rho, row, take, st.rho_prev),
            time=_put(trace.time, row, take, st.time),
            count=trace.count + take.astype(trace.count.dtype),
        )
    assert snap.r.shape == st.snap.r.shape, "snapshot buffers keep their shape"
    return dataclasses.replace(st, snap=snap, trace=trace)


# ============================================================
# CHUNKS
# ============================================================


def _repair_pending(st: MDState, ctx: _Ctx) -> MDState:
    """Recompute rho_prev rows marked pending (NaN) by begin_segment."""
    pending = jnp.isnan(st.rho_prev[:, 0]) & (st.status == STATUS_RUNNING)
    out = ctx.force_fn(st.r)
    rho, _ = _lane_rho(st.r, out, st.rho_prev, ctx.waves, ctx.settings, True,
                       ctx.chunk)  # fmt: skip
    assert rho.shape == st.rho_prev.shape, "rho_prev shape"
    assert pending.shape == (rho.shape[0],), "one flag per wave"
    return dataclasses.replace(
        st, rho_prev=jnp.where(pending[:, None], rho, st.rho_prev)
    )


def _chunk(
    energy: StealthyEnergy, waves: Waves, basis: Array, settings: MDSettings,
    state: MDState, n_steps: Array, step0: Array,
) -> MDState:  # fmt: skip
    """md_chunk with the cadence phase from ``step0`` (the lane's step at the
    chunk start; unbatched under vmap when every running lane shares it).
    """
    _check_settings(settings)
    _check_lane(state, waves, settings)
    ctx = _Ctx(make_force_fn(energy, basis), waves, settings, energy.chunk_size)
    state = _repair_pending(state, ctx)
    n_do = jnp.minimum(jnp.asarray(n_steps, jnp.int64), settings.chunk_steps)
    every = settings.wind_every
    assert n_do.shape == () and jnp.shape(step0) == (), "scalar step counts"

    def body(carry: Tuple[Array, MDState]) -> Tuple[Array, MDState]:
        i, st = carry
        wind = True if every == 1 else (step0 + i + 1) % every == 0
        new, ke = _step(st, ctx, wind)
        running = st.status == STATUS_RUNNING
        new = _record(_freeze(st, new, running), ke, settings, running)
        return i + 1, new

    i0 = jnp.zeros((), jnp.int64)
    _, out = jax.lax.while_loop(lambda c: c[0] < n_do, body, (i0, state))
    assert isinstance(out, MDState), "the loop carries an MDState"
    return out


def md_chunk(
    energy: StealthyEnergy,
    waves: Waves,
    basis: Float[Array, "d d"],
    settings: MDSettings,
    state: MDState,
    n_steps: Array,
) -> MDState:
    """Advance one lane by min(settings.chunk_steps, n_steps) steps.

    A lane whose status is not STATUS_RUNNING (or turns STATUS_DRIFT in a
    step) keeps every leaf from then on. Pending rho_prev rows (after
    ``begin_segment``) are recomputed first. jit/vmap compatible.
    """
    return _chunk(energy, waves, basis, settings, state, n_steps, state.step)


_CHUNK_SHARED = eqx.filter_jit(
    eqx.filter_vmap(_chunk, in_axes=(None, None, None, None, 0, None, None))
)
_CHUNK_LANES = eqx.filter_jit(
    eqx.filter_vmap(md_chunk, in_axes=(None, None, None, None, 0, None))
)


def _run_chunk(
    energy: StealthyEnergy, waves: Waves, basis: Array, settings: MDSettings,
    state: MDState, n_steps: int,
) -> MDState:  # fmt: skip
    """One vmapped chunk; the shared-cadence program when the running lanes
    are at one step, else the per-lane one."""
    steps = np.asarray(state.step)[np.asarray(state.status) == STATUS_RUNNING]
    n = jnp.asarray(n_steps, jnp.int64)
    assert steps.size >= 1 and n_steps >= 1, "a chunk needs running lanes and steps"
    if np.unique(steps).size == 1:
        step0 = jnp.asarray(steps[0], jnp.int64)
        out = _CHUNK_SHARED(energy, waves, basis, settings, state, n, step0)
    else:
        out = _CHUNK_LANES(energy, waves, basis, settings, state, n)
    assert isinstance(out, MDState), "chunks return an MDState"
    return out


def _write(path: Path, state: MDState, done: int, n_steps: int) -> None:
    """Atomic in-phase checkpoint (save_state) with progress metadata."""
    assert 0 <= done <= n_steps, "progress within the call"
    meta: Dict[str, Any] = {"source": CHECKPOINT_SOURCE, "steps_done": done,
                            "n_steps": n_steps}  # fmt: skip
    assert state.step.ndim == 1, "checkpoints hold the batched state"
    save_state(Path(path), state, meta)


def simulate(
    energy: StealthyEnergy, waves: Waves, basis: Float[Array, "d d"],
    settings: MDSettings, state: MDState, n_steps: int,
    checkpoint: Optional[Path] = None, checkpoint_every_s: float = 1800.0,
) -> MDResult:  # fmt: skip
    """Run the batched lanes for n_steps (host loop over jitted vmapped chunks).

    Stops early when no lane is running. With ``checkpoint`` the state is
    written there (atomically; any existing file is overwritten, never read)
    whenever ``checkpoint_every_s`` seconds have passed after a chunk, and at
    the end. ``step`` is absolute; snapshots and the trace persist across
    calls.

    Raises
    ------
    ValueError
        On invalid settings, an unbatched state, buffers that do not match
        the settings, n_steps < 0 or checkpoint_every_s <= 0.
    """
    _check_settings(settings)
    if state.step.ndim != 1:
        raise ValueError(f"state must be batched (leading lane axis), "
                         f"got step shape {state.step.shape}")  # fmt: skip
    if int(n_steps) != n_steps or n_steps < 0 or not checkpoint_every_s > 0.0:
        raise ValueError(f"need n_steps >= 0 and checkpoint_every_s > 0, got "
                         f"{n_steps}, {checkpoint_every_s}")  # fmt: skip
    if state.step.shape[0] < 1:
        raise ValueError("need at least one lane")
    _check_lane(jax.tree_util.tree_map(lambda x: x[0], state), waves, settings)
    done, last = 0, time.monotonic()
    for _ in range(-(-int(n_steps) // settings.chunk_steps)):
        if done >= n_steps or not np.any(np.asarray(state.status) == STATUS_RUNNING):
            break
        n_this = min(settings.chunk_steps, int(n_steps) - done)
        state = _run_chunk(energy, waves, basis, settings, state, n_this)
        done += n_this
        due = time.monotonic() - last >= checkpoint_every_s
        if checkpoint is not None and done < n_steps and due:
            _write(checkpoint, state, done, int(n_steps))
            last = time.monotonic()
    if checkpoint is not None:
        _write(checkpoint, state, done, int(n_steps))
    assert done <= n_steps and state.step.ndim == 1, "bounded, batched"
    trace = state.trace if settings.record_rho_every > 0 else None
    return MDResult(state=state, snapshots=state.snap, trace=trace)


# ============================================================
# DT SELECTION
# ============================================================


def _restart_lanes(
    energy: StealthyEnergy, waves: Waves, basis: Array, state: MDState,
    settings: MDSettings,
) -> MDState:  # fmt: skip
    """Fresh batched lanes (init_state) at the r, v, T and key of ``state``."""
    assert state.step.ndim == 1, "batched state"

    def one(r: Array, v: Array, t: Array, k: Array) -> MDState:
        return init_state(energy, waves, basis, r, v, t, k, settings)

    out: MDState = jax.vmap(one)(state.r, state.v, state.temperature, state.key)
    assert out.step.shape == state.step.shape, "one restart per lane"
    return out


def select_dt(
    energy: StealthyEnergy,
    waves: Waves,
    basis: Float[Array, "d d"],
    state: MDState,
    ladder: Sequence[float],
    n_steps: int,
    drift_rel: float,
) -> float:
    """Largest dt in ``ladder`` whose NVE drift |E_last - E_first| / <KE> over
    n_steps stays below ``drift_rel`` in every lane.

    Each trial restarts the lanes (batched ``state``: r, v, temperature,
    key) with ``init_state`` under velocity Verlet (velocities as given, no
    drift abort, no recording); <KE> is the mean over the n_steps steps.
    Non-finite energies fail the criterion.

    Raises
    ------
    ValueError
        On an empty ladder, a non-positive dt, n_steps < 1, drift_rel <= 0,
        an unbatched state, or when no dt meets the criterion.
    """
    dts = sorted({float(x) for x in ladder}, reverse=True)
    if not dts or not all(math.isfinite(x) and x > 0.0 for x in dts):
        raise ValueError(f"ladder must hold finite positive steps, got {ladder}")
    if n_steps < 1 or not drift_rel > 0.0 or state.step.ndim != 1:
        raise ValueError("need n_steps >= 1, drift_rel > 0 and a batched state")
    for dt in dts:
        st = MDSettings(
            dt=dt, scheme="verlet", remove_com=False,
            chunk_steps=min(int(n_steps), 10_000), drift_tol=math.inf,
        )  # fmt: skip
        lanes = _restart_lanes(energy, waves, basis, state, st)
        out = simulate(energy, waves, basis, st, lanes, int(n_steps)).state
        e_last = np.asarray(out.phi + jax.vmap(kinetic_energy)(out.v))
        ke = np.asarray(out.sum_kinetic) / np.maximum(np.asarray(out.n_sum), 1)
        rel = np.abs(e_last - np.asarray(lanes.e0)) / ke
        assert rel.shape == np.shape(state.step), "one drift per lane"
        if np.all(rel < drift_rel):
            return dt
    raise ValueError(f"no dt in {tuple(ladder)} keeps the drift below {drift_rel}")
