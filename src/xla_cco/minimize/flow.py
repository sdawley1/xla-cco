#!/usr/bin/env python3

"""
Gradient flow of an energy landscape (adaptive Dormand-Prince 5(4)).

Integrates the steepest-descent flow dr/dt = -grad_r Phi in Cartesian
coordinates r = s B (rows of B are the lattice vectors), written in the
fractional coordinates the energies use:

    ds/dt = -grad_s Phi (B B^T)^{-1}.

Unlike the adaptive descent rule (descent.py), whose normalized steps can
zigzag across contours, the integrated path follows the streamlines of the
landscape to a local error tolerance. Steps use the Dormand-Prince 5(4)
pair with FSAL and a standard step-size controller. The local error (RMS
Cartesian displacement error) must stay below atol + rtol * |step|, where
|step| is the RMS Cartesian displacement of the step itself: the error is
kept small relative to the motion, so Phi decreases monotonically even as
the flow approaches a minimum. Positions are not wrapped, so recorded paths
are continuous.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import math
from typing import Callable, Optional, Tuple

# ----- numerics -----
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Int

# ----- project -----
from xla_cco._types import Positions, require_float64
from xla_cco.energy import AbstractEnergy
from xla_cco.minimize.result import (
    STATUS_BUDGET,
    STATUS_ENERGY_TOL,
    STATUS_GRAD_TOL,
    STATUS_RUNNING,
    STATUS_STEP_FLOOR,
)

# ============================================================
# CONSTANTS (Dormand-Prince 5(4) tableau)
# ============================================================

_A = (
    (),
    (1 / 5,),
    (3 / 40, 9 / 40),
    (44 / 45, -56 / 15, 32 / 9),
    (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
    (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
)
_C = (0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0)
_B5 = (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0)
_B4 = (
    5179 / 57600,
    0.0,
    7571 / 16695,
    393 / 640,
    -92097 / 339200,
    187 / 2100,
    1 / 40,
)
_E = tuple(b5 - b4 for b5, b4 in zip(_B5, _B4, strict=True))
_SAFETY, _SHRINK_MIN, _GROW_MAX = 0.9, 0.2, 5.0

# ============================================================
# TYPES
# ============================================================


class FlowSettings(eqx.Module):
    """Integration controls (all static).

    Local error tolerance per step: ``atol + rtol * |step|`` (RMS Cartesian
    displacement; length units at density 1).
    Stops: ``energy_tol`` (Phi below it), ``grad_tol`` (|grad_r Phi| below it,
    or an exactly zero gradient), ``t_max``, ``max_steps`` accepted steps,
    or a step below ``dt_min``. ``record_every`` > 0 stores every k-th
    accepted step (plus the start and the final state) in ``max_records`` rows.
    """

    rtol: float = eqx.field(static=True, default=1e-6)
    atol: float = eqx.field(static=True, default=1e-14)
    dt_init: float = eqx.field(static=True, default=1e-3)
    dt_min: float = eqx.field(static=True, default=1e-14)
    dt_max: float = eqx.field(static=True, default=math.inf)
    t_max: float = eqx.field(static=True, default=math.inf)
    max_steps: int = eqx.field(static=True, default=200_000)
    energy_tol: float = eqx.field(static=True, default=0.0)
    grad_tol: float = eqx.field(static=True, default=0.0)
    record_every: int = eqx.field(static=True, default=0)
    max_records: int = eqx.field(static=True, default=0)


class FlowTrajectory(eqx.Module):
    """Recorded flow (fixed capacity, ``count`` valid rows; s unwrapped)."""

    s: Float[Array, "r n d"]
    t: Float[Array, " r"]
    energy: Float[Array, " r"]
    count: Int[Array, ""]


class FlowResult(eqx.Module):
    """Final state of a gradient flow."""

    s: Positions
    energy: Float[Array, ""]
    t: Float[Array, ""]
    n_steps: Int[Array, ""]
    n_evals: Int[Array, ""]
    status: Int[Array, ""]
    trajectory: Optional[FlowTrajectory]


class _State(eqx.Module):
    s: Array
    f: Array
    k1: Array
    g: Array
    t: Array
    dt: Array
    n_steps: Array
    n_tries: Array
    n_evals: Array
    status: Array
    rec_s: Array
    rec_t: Array
    rec_e: Array
    count: Array


# ============================================================
# HELPERS
# ============================================================


def _validate(s0: Positions, basis: Array, settings: FlowSettings) -> None:
    """Raise on bad inputs before tracing."""
    require_float64(s0, "s0")
    if s0.ndim != 2 or basis.shape != (s0.shape[1], s0.shape[1]):
        raise ValueError(f"s0 must be (n, d) with a (d, d) basis, got {s0.shape}")
    positive = (settings.atol, settings.rtol, settings.dt_init, settings.max_steps)
    if not all(x > 0 for x in positive):
        raise ValueError("atol, rtol, dt_init and max_steps must be positive")
    if settings.record_every < 0 or settings.max_records < 0:
        raise ValueError("record_every and max_records must be >= 0")
    if (settings.record_every > 0) != (settings.max_records > 1):
        raise ValueError("recording needs record_every > 0 and max_records >= 2")


def _status(
    f: Array,
    g: Array,
    gnorm: Array,
    t: Array,
    dt: Array,
    steps: Array,
    tries: Array,
    settings: FlowSettings,
) -> Array:
    """Stop code; precedence energy > gradient > time/budget > step floor."""
    code = jnp.where(dt < settings.dt_min, STATUS_STEP_FLOOR, STATUS_RUNNING)
    over = (t >= settings.t_max) | (steps >= settings.max_steps)
    code = jnp.where(over | (tries >= 4 * settings.max_steps), STATUS_BUDGET, code)
    flat = jnp.all(g == 0.0) | (gnorm < settings.grad_tol)
    code = jnp.where(flat, STATUS_GRAD_TOL, code)
    return jnp.where(f < settings.energy_tol, STATUS_ENERGY_TOL, code)


# ============================================================
# PUBLIC API
# ============================================================


def gradient_flow(
    energy: AbstractEnergy,
    s0: Positions,
    basis: Float[Array, "d d"],
    settings: FlowSettings,
) -> FlowResult:
    """Integrate dr/dt = -grad_r Phi from ``s0`` (fractional) until a stop.

    Parameters
    ----------
    energy : AbstractEnergy
        Landscape Phi(s) in fractional coordinates (``value_and_grad`` used).
    s0 : Positions
        (n, d) float64 fractional start.
    basis : Float[Array, "d d"]
        Cell basis (rows = lattice vectors) defining the Cartesian metric.
    settings : FlowSettings
        Tolerances, stops and recording.

    Returns
    -------
    FlowResult
        Final (unwrapped) positions, energy, flow time, step counts, status
        and, if recording, the trajectory. vmap/jit compatible.
    """
    _validate(s0, basis, settings)
    basis = jnp.asarray(basis, dtype=s0.dtype)
    metric_inv = jnp.linalg.inv(basis @ basis.T)
    basis_inv_t = jnp.linalg.inv(basis).T

    def velocity(s: Array) -> Tuple[Array, Array, Array]:
        f, g = energy.value_and_grad(s)
        return f, g, -g @ metric_inv

    def gnorm(g: Array) -> Array:
        return jnp.sqrt(jnp.sum((g @ basis_inv_t) ** 2))

    f0, g0, k0 = velocity(s0)
    rows = max(settings.max_records, 1)
    zero = jnp.asarray(0, jnp.int64)
    state = _State(
        s=s0,
        f=f0,
        k1=k0,
        g=g0,
        t=jnp.asarray(0.0, s0.dtype),
        dt=jnp.asarray(min(settings.dt_init, settings.dt_max), s0.dtype),
        n_steps=zero,
        n_tries=zero,
        n_evals=jnp.asarray(1, jnp.int64),
        status=_status(
            f0, g0, gnorm(g0), jnp.asarray(0.0), jnp.asarray(1.0), zero, zero, settings
        ).astype(jnp.int64),
        rec_s=jnp.zeros((rows,) + s0.shape, s0.dtype).at[0].set(s0),
        rec_t=jnp.zeros(rows, s0.dtype),
        rec_e=jnp.zeros(rows, s0.dtype).at[0].set(f0),
        count=jnp.asarray(1, jnp.int64),
    )
    out = jax.lax.while_loop(
        lambda st: st.status == STATUS_RUNNING,
        lambda st: _step(st, velocity, gnorm, basis, settings),
        state,
    )
    return _finish(out, settings)


_Velocity = Callable[[Array], Tuple[Array, Array, Array]]


def _step(
    st: _State,
    velocity: _Velocity,
    gnorm: Callable[[Array], Array],
    basis: Array,
    settings: FlowSettings,
) -> _State:
    """One attempted Dormand-Prince step with error control (FSAL)."""
    dt = jnp.minimum(st.dt, settings.t_max - st.t)
    ks = [st.k1]
    for i in range(1, 6):
        incr = sum(a * k for a, k in zip(_A[i], ks, strict=True))
        ks.append(velocity(st.s + dt * incr)[2])
    s_new = st.s + dt * sum(b * k for b, k in zip(_B5[:6], ks, strict=True))
    f_new, g_new, k7 = velocity(s_new)
    err_s = dt * sum(e * k for e, k in zip(_E, ks + [k7], strict=True))
    move = jnp.sqrt(jnp.mean(((s_new - st.s) @ basis) ** 2))
    tol = settings.atol + settings.rtol * move
    err = jnp.sqrt(jnp.mean((err_s @ basis) ** 2)) / tol
    accept = err <= 1.0
    factor = jnp.clip(_SAFETY * err ** (-0.2), _SHRINK_MIN, _GROW_MAX)
    factor = jnp.where(jnp.isfinite(factor), factor, _SHRINK_MIN)
    dt_next = jnp.minimum(dt * factor, settings.dt_max)

    def pick(new: Array, old: Array) -> Array:
        return jnp.where(accept, new, old)

    s, f, g, k1 = (
        pick(s_new, st.s),
        pick(f_new, st.f),
        pick(g_new, st.g),
        pick(k7, st.k1),
    )
    t = pick(st.t + dt, st.t)
    steps = st.n_steps + accept.astype(jnp.int64)
    tries = st.n_tries + 1
    rec = (
        accept
        & (settings.record_every > 0)
        & (steps % max(settings.record_every, 1) == 0)
    )
    rec = rec & (st.count < max(settings.max_records, 1))
    idx = jnp.minimum(st.count, max(settings.max_records, 1) - 1)
    status = _status(f, g, gnorm(g), t, dt_next, steps, tries, settings)
    return _State(
        s=s,
        f=f,
        k1=k1,
        g=g,
        t=t,
        dt=dt_next,
        n_steps=steps,
        n_tries=tries,
        n_evals=st.n_evals + 6,
        status=status.astype(jnp.int64),
        rec_s=jnp.where(rec, st.rec_s.at[idx].set(s), st.rec_s),
        rec_t=jnp.where(rec, st.rec_t.at[idx].set(t), st.rec_t),
        rec_e=jnp.where(rec, st.rec_e.at[idx].set(f), st.rec_e),
        count=st.count + rec.astype(jnp.int64),
    )


def _finish(st: _State, settings: FlowSettings) -> FlowResult:
    """Append the final state to the record (overwrite the last row if full)."""
    traj = None
    if settings.record_every > 0:
        last_is_final = st.rec_t[st.count - 1] == st.t
        idx = jnp.where(
            last_is_final, st.count - 1, jnp.minimum(st.count, settings.max_records - 1)
        )
        traj = FlowTrajectory(
            s=st.rec_s.at[idx].set(st.s),
            t=st.rec_t.at[idx].set(st.t),
            energy=st.rec_e.at[idx].set(st.f),
            count=idx + 1,
        )
    return FlowResult(
        s=st.s,
        energy=st.f,
        t=st.t,
        n_steps=st.n_steps,
        n_evals=st.n_evals,
        status=st.status,
        trajectory=traj,
    )
