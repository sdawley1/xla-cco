#!/usr/bin/env python3

"""
Hand-built MD states and fake MD drivers for the checkpoint and CLI tests.

``random_state`` fills every MDState leaf (including the uint32 PRNG key and
the Andersen accumulator) with seeded values. ``FakeDriver`` stands in for
``xla_cco.dynamics.run`` and ``thermal`` so the CLI protocol, outputs and
resume logic can be tested while the real driver is not available: its
``simulate`` is a deterministic function of the state (it consumes the PRNG
key and the accumulator), records every call, writes the snapshots at
``settings.record_steps`` and can crash on request, optionally after an
in-phase checkpoint.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import dataclasses
from pathlib import Path
from typing import Any, List, Optional, Tuple

# ----- numerics -----
import jax
import jax.numpy as jnp
import numpy as np

# ----- project -----
from xla_cco.dynamics.checkpoint import save_state
from xla_cco.dynamics.state import (
    MDResult,
    MDSettings,
    MDState,
    RhoTrace,
    Snapshots,
    Waves,
)
from xla_cco.kvectors import enumerate_kvectors

# ============================================================
# STATES
# ============================================================


def _floats(rng: np.random.Generator, shape: Tuple[int, ...]) -> np.ndarray:
    """Seeded float64 values with -0.0, inf and nan sprinkled in (bitwise tests)."""
    a = rng.standard_normal(shape)
    if a.size >= 4:
        flat = a.reshape(-1)
        flat[0], flat[1], flat[2] = -0.0, np.inf, np.nan
    return a


def random_state(
    seed: int, n: int, d: int, w: int, n_rec: int, n_trace: int,
    batch: Optional[int] = None, int_dtype: Any = np.int64,
) -> MDState:  # fmt: skip
    """MDState with every leaf seeded; ``batch`` adds a leading lane axis."""
    assert min(n, d) >= 1 and min(w, n_rec, n_trace) >= 0, "bad shapes"
    rng = np.random.default_rng(seed)
    lead: Tuple[int, ...] = () if batch is None else (batch,)

    def f(*shape: int) -> jax.Array:
        return jnp.asarray(_floats(rng, lead + shape))

    def i(*shape: int) -> jax.Array:
        vals = rng.integers(-(2**30), 2**30, size=lead + shape)
        return jnp.asarray(vals.astype(int_dtype))

    key = rng.integers(0, 2**32, size=lead + (2,), dtype=np.uint64)
    snap = Snapshots(
        step=i(n_rec), time=f(n_rec), wind_abs=f(n_rec, w), wind_net=f(n_rec, w),
        rho=f(n_rec, w, 2), r=f(n_rec, n, d), phi=f(n_rec), kinetic=f(n_rec),
        count=i(),
    )  # fmt: skip
    trace = RhoTrace(rho=f(n_trace, w, 2), time=f(n_trace), count=i())
    return MDState(
        r=f(n, d), v=f(n, d), f=f(n, d), phi=f(), temperature=f(),
        key=jnp.asarray(key.astype(np.uint32)), step=i(), time=f(),
        reset_acc=f(), rho_prev=f(w, 2), wind_abs=f(w), wind_net=f(w),
        n_valid=i(w), e0=f(), max_drift=f(), sum_kinetic=f(), sum_phi=f(),
        sum_phi2=f(), sum_s=f(w), n_sum=i(), status=i(), snap=snap, trace=trace,
    )  # fmt: skip


def leaves_bitwise_equal(a: Any, b: Any) -> bool:
    """True when two pytrees have equal structure and bit-identical leaves."""
    la, ta = jax.tree_util.tree_flatten(a)
    lb, tb = jax.tree_util.tree_flatten(b)
    if ta != tb or len(la) != len(lb):
        return False
    for x, y in zip(la, lb, strict=True):
        xa, ya = np.asarray(x), np.asarray(y)
        if xa.dtype != ya.dtype or xa.shape != ya.shape:
            return False
        if xa.tobytes() != ya.tobytes():
            return False
    return True


# ============================================================
# FAKE DRIVER
# ============================================================


def fake_make_waves(basis: np.ndarray, k_constraint: float, k_track: float) -> Waves:
    """Constrained waves (KSet order) then k_constraint < |k| < k_track."""
    inner = enumerate_kvectors(basis, 0.0, k_constraint)
    outer = enumerate_kvectors(basis, k_constraint, k_track)
    n_vec = np.concatenate([inner.n, outer.n]).astype(np.float64)
    kvec = np.concatenate([inner.k, outer.k])
    m = int(inner.n.shape[0])
    mask = np.arange(n_vec.shape[0]) < m
    return Waves(
        n_vectors=jnp.asarray(n_vec),
        kvec=jnp.asarray(kvec),
        constrained=jnp.asarray(mask),
        n_constrained=m,
    )


def fake_init_state(
    energy: Any, waves: Waves, basis: Any, r0: Any, v0: Any, temperature: Any,
    key: Any, settings: MDSettings,
) -> MDState:  # fmt: skip
    """One lane at step 0 with zero winding/sums and empty buffers."""
    n, d = r0.shape
    w = int(waves.n_vectors.shape[0])
    n_rec, n_tr = len(settings.record_steps), settings.max_rho_records
    zf, zi = jnp.zeros((), jnp.float64), jnp.zeros((), jnp.int64)
    snap = Snapshots(
        step=jnp.full((n_rec,), -1, jnp.int64), time=jnp.zeros(n_rec),
        wind_abs=jnp.zeros((n_rec, w)), wind_net=jnp.zeros((n_rec, w)),
        rho=jnp.zeros((n_rec, w, 2)), r=jnp.zeros((n_rec, n, d)),
        phi=jnp.zeros(n_rec), kinetic=jnp.zeros(n_rec), count=zi,
    )  # fmt: skip
    trace = RhoTrace(rho=jnp.zeros((n_tr, w, 2)), time=jnp.zeros(n_tr), count=zi)
    return MDState(
        r=jnp.asarray(r0), v=jnp.asarray(v0), f=jnp.zeros((n, d)), phi=zf,
        temperature=jnp.asarray(temperature, jnp.float64), key=key, step=zi,
        time=zf, reset_acc=zf, rho_prev=jnp.ones((w, 2)), wind_abs=jnp.zeros(w),
        wind_net=jnp.zeros(w), n_valid=jnp.zeros(w, jnp.int64), e0=zf,
        max_drift=zf, sum_kinetic=zf, sum_phi=zf, sum_phi2=zf, sum_s=jnp.zeros(w),
        n_sum=zi, status=zi, snap=snap, trace=trace,
    )  # fmt: skip


def fake_begin_segment(
    state: MDState, settings: MDSettings, reset_sums: bool, reset_winding: bool
) -> MDState:
    """Reset sums / winding as asked and set e0 (jax-traceable)."""
    zf = jnp.zeros_like(state.phi)
    new = dataclasses.replace(state, e0=state.phi + 0.5 * jnp.sum(state.v**2))
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
        )  # fmt: skip
    return new


def fake_harmonic_sample(
    rng: np.random.Generator, energy: Any, s_gs: np.ndarray, basis: np.ndarray,
    temperature: float, zero_rel: float = 1e-9,
) -> np.ndarray:  # fmt: skip
    """Ground state (Cartesian) plus an isotropic N(0, T) displacement."""
    r = np.asarray(s_gs) @ np.asarray(basis)
    return r + np.sqrt(temperature) * rng.standard_normal(r.shape)


def fake_maxwell_boltzmann(
    key: Any, n: int, d: int, temperature: Any, remove_com: bool
) -> jax.Array:
    """N(0, T) velocities with the mean removed when asked."""
    v = jnp.sqrt(temperature) * jax.random.normal(key, (n, d), dtype=jnp.float64)
    return v - jnp.mean(v, axis=0) if remove_com else v


def _record(snap: Snapshots, new: MDState, old_step: np.ndarray, steps: Any) -> Any:
    """Host write of the snapshots whose record step fell in (old, new]."""
    s = {f.name: np.array(getattr(snap, f.name)) for f in dataclasses.fields(snap)}
    for lane in range(old_step.shape[0]):
        for j, rs in enumerate(steps):
            if old_step[lane] < rs <= int(new.step[lane]):
                s["step"][lane, j] = rs
                s["time"][lane, j] = float(new.time[lane])
                s["wind_abs"][lane, j] = np.asarray(new.wind_abs[lane])
                s["wind_net"][lane, j] = np.asarray(new.wind_net[lane])
                s["r"][lane, j] = np.asarray(new.r[lane])
                s["phi"][lane, j] = float(new.phi[lane])
                s["count"][lane] += 1
    return Snapshots(**{k: jnp.asarray(v) for k, v in s.items()})


class FakeDriver:
    """Deterministic stand-in for ``run.simulate`` with call log and crashes.

    ``crash_at`` (call index) raises RuntimeError at that simulate call;
    with ``inphase`` the call first advances half its steps and saves them
    to the ``checkpoint`` path it was given.
    """

    def __init__(self, crash_at: Optional[int] = None, inphase: bool = False):
        self.calls: List[Tuple[str, int]] = []
        self.crash_at = crash_at
        self.inphase = inphase

    def _step(self, settings: MDSettings, state: MDState) -> MDState:
        """One composable step: consumes the key and the accumulator."""
        keys = jax.vmap(lambda k: jax.random.split(k, 2))(state.key)
        noise = jax.vmap(lambda k, x: jax.random.normal(k, x.shape))(
            keys[:, 1], state.v
        )
        acc = state.reset_acc + settings.andersen_p
        v = state.v * jnp.cos(acc)[:, None, None] + 1e-3 * noise
        r = state.r + settings.dt * v
        phi = jnp.sum(jnp.sin(r) ** 2, axis=(1, 2))
        dw = jnp.abs(jnp.sin(jnp.sum(r, axis=(1, 2))))[:, None] * (1.0 + state.sum_s)
        new = dataclasses.replace(
            state, r=r, v=v, phi=phi, key=keys[:, 0], reset_acc=acc % 1.0,
            step=state.step + 1, time=state.time + settings.dt,
            wind_abs=state.wind_abs + dw, wind_net=state.wind_net - dw,
            n_valid=state.n_valid + 1, sum_phi=state.sum_phi + phi,
            sum_kinetic=state.sum_kinetic + 0.5 * jnp.sum(v**2, axis=(1, 2)),
            sum_phi2=state.sum_phi2 + phi**2, n_sum=state.n_sum + 1,
        )  # fmt: skip
        snap = _record(state.snap, new, np.asarray(state.step), settings.record_steps)
        return dataclasses.replace(new, snap=snap)

    def _advance(self, settings: MDSettings, state: MDState, n: int) -> MDState:
        assert 0 <= n <= 10_000, "fake runs are short"
        for _ in range(n):
            state = self._step(settings, state)
        return state

    def simulate(
        self, energy: Any, waves: Waves, basis: Any, settings: MDSettings,
        state: MDState, n_steps: int, checkpoint: Optional[Path] = None,
        checkpoint_every_s: float = 1800.0,
    ) -> MDResult:  # fmt: skip
        """Advance every lane by n_steps (two halves when crashing in-phase)."""
        assert n_steps >= 1, "the CLI never asks for an empty phase"
        index = len(self.calls)
        self.calls.append((settings.scheme, int(n_steps)))
        if index == self.crash_at:
            if self.inphase and checkpoint is not None and n_steps >= 2:
                half = self._advance(settings, state, n_steps // 2)
                save_state(checkpoint, half, {"source": "simulate"})
            raise RuntimeError("simulated crash")
        out = self._advance(settings, state, n_steps)
        return MDResult(state=out, snapshots=out.snap, trace=None)
