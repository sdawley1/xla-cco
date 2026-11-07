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
the flow approaches a minimum. The flow time and arclength are integrated
with the same stages under the same relative control. Positions are not
wrapped, so recorded paths are continuous.

The flow slows down as |grad Phi| -> 0, so near a minimum or along a flat
valley the true-time integration needs very many steps. With
``parametrization="arclength"`` the same streamlines are integrated at unit
Cartesian speed, dr/dsigma = -grad_r Phi / |grad_r Phi|; the physical time is
recovered along the path from dt/dsigma = 1 / |grad_r Phi|. Both modes give
the same curve, the same flow time and the same arclength (to tolerance);
only the integration variable differs.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import math
from typing import Callable, NamedTuple, Optional, Tuple

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
_ROS_GAMMA = 1.0 + 1.0 / math.sqrt(2.0)

# ============================================================
# TYPES
# ============================================================


class FlowSettings(eqx.Module):
    """Integration controls (all static).

    Local error tolerance per step: ``atol + rtol * |step|`` (RMS Cartesian
    displacement; length units at density 1). ``parametrization`` is "time"
    (integrate in the physical flow time t) or "arclength" (unit Cartesian
    speed; same path). ``method`` is "dopri5" (explicit Dormand-Prince 5(4))
    or "rosenbrock" (L-stable ROS2 with the exact Hessian; for stiff
    landscapes, time parametrization only; its arclength is the sum of step
    chords). ROS2's error estimate is first order, so with "rosenbrock" set a
    path-accuracy ``atol`` (e.g. 1e-8) and a loose ``rtol`` (e.g. 1e-3); a tiny
    rtol alone forces steps of order rtol. ``dt_*`` bound the step in the
    integration variable.
    Stops: ``energy_tol`` (Phi below it), ``grad_tol`` (|grad_r Phi| below it,
    or an exactly zero gradient), ``t_max`` (physical time; time mode only),
    ``max_steps`` accepted steps, or a step below ``dt_min``.
    ``record_every`` > 0 stores every k-th accepted step (plus the start and
    the final state) in ``max_records`` rows.
    """

    rtol: float = eqx.field(static=True, default=1e-6)
    atol: float = eqx.field(static=True, default=1e-14)
    parametrization: str = eqx.field(static=True, default="time")
    method: str = eqx.field(static=True, default="dopri5")
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
    length: Float[Array, " r"]
    energy: Float[Array, " r"]
    count: Int[Array, ""]


class FlowResult(eqx.Module):
    """Final state of a gradient flow (t: flow time, length: path arclength).

    ``dt`` is the next step size proposed by the controller. The flow is
    autonomous, so ``gradient_flow(energy, s, basis, settings, dt0=dt)``
    continues the same curve (clocks then restart at 0; add them on the host).
    """

    s: Positions
    energy: Float[Array, ""]
    t: Float[Array, ""]
    length: Float[Array, ""]
    dt: Float[Array, ""]
    n_steps: Int[Array, ""]
    n_evals: Int[Array, ""]
    status: Int[Array, ""]
    trajectory: Optional[FlowTrajectory]


class _Eval(NamedTuple):
    """Energy, gradient, velocity and the rates d(t, length)/d(variable)."""

    f: Array
    g: Array
    v: Array
    rate: Array  # (2,): dt/dvar, dlength/dvar


class _State(eqx.Module):
    s: Array
    ev: _Eval
    var: Array
    t: Array
    length: Array
    dt: Array
    n_steps: Array
    n_tries: Array
    n_evals: Array
    status: Array
    rec_s: Array
    rec_tl: Array
    rec_e: Array
    count: Array


_Velocity = Callable[[Array], _Eval]

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
    if settings.parametrization not in ("time", "arclength"):
        raise ValueError(f"unknown parametrization {settings.parametrization!r}")
    if settings.method not in ("dopri5", "rosenbrock"):
        raise ValueError(f"unknown method {settings.method!r}")
    if settings.method == "rosenbrock" and settings.parametrization != "time":
        raise ValueError("method='rosenbrock' integrates the time parametrization")
    if settings.parametrization == "arclength" and math.isfinite(settings.t_max):
        raise ValueError("t_max is supported only with parametrization='time'")
    if settings.record_every < 0 or settings.max_records < 0:
        raise ValueError("record_every and max_records must be >= 0")
    if (settings.record_every > 0) != (settings.max_records > 1):
        raise ValueError("recording needs record_every > 0 and max_records >= 2")


def _status(
    ev: _Eval,
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
    flat = jnp.all(ev.g == 0.0) | (gnorm < settings.grad_tol)
    code = jnp.where(flat, STATUS_GRAD_TOL, code)
    return jnp.where(ev.f < settings.energy_tol, STATUS_ENERGY_TOL, code)


def _make_velocity(
    energy: AbstractEnergy, basis: Array, arclength: bool
) -> Tuple[_Velocity, Callable[[Array], Array]]:
    """Return the vector field of the chosen parametrization and |grad_r Phi|."""
    metric_inv = jnp.linalg.inv(basis @ basis.T)
    basis_inv_t = jnp.linalg.inv(basis).T

    def gnorm(g: Array) -> Array:
        return jnp.sqrt(jnp.sum((g @ basis_inv_t) ** 2))

    def velocity(s: Array) -> _Eval:
        f, g = energy.value_and_grad(s)
        v = -g @ metric_inv
        n = gnorm(g)
        if not arclength:
            return _Eval(f, g, v, jnp.stack([jnp.ones_like(n), n]))
        safe = jnp.where(n > 0.0, n, 1.0)
        inv = jnp.where(n > 0.0, 1.0 / safe, 0.0)
        return _Eval(f, g, v * inv, jnp.stack([inv, jnp.ones_like(n)]))

    return velocity, gnorm


# ============================================================
# PUBLIC API
# ============================================================


def gradient_flow(
    energy: AbstractEnergy,
    s0: Positions,
    basis: Float[Array, "d d"],
    settings: FlowSettings,
    dt0: Optional[Float[Array, ""]] = None,
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
        Tolerances, parametrization, stops and recording.
    dt0 : Float[Array, ""], optional
        First step size, overriding ``settings.dt_init`` (traced, so it can
        differ per vmapped lane; used to resume from ``FlowResult.dt``). A
        value below ``dt_min`` stops at once with STATUS_STEP_FLOOR.

    Returns
    -------
    FlowResult
        Final (unwrapped) positions, energy, flow time, arclength, step counts,
        status and, if recording, the trajectory. vmap/jit compatible.
    """
    _validate(s0, basis, settings)
    basis = jnp.asarray(basis, dtype=s0.dtype)
    velocity, gnorm = _make_velocity(
        energy, basis, settings.parametrization == "arclength"
    )
    ev0 = velocity(s0)
    dt_first = jnp.asarray(min(settings.dt_init, settings.dt_max), s0.dtype)
    if dt0 is not None:
        dt_first = jnp.minimum(jnp.asarray(dt0, s0.dtype), settings.dt_max)
    rows = max(settings.max_records, 1)
    zero = jnp.asarray(0, jnp.int64)
    origin = jnp.asarray(0.0, s0.dtype)
    state = _State(
        s=s0,
        ev=ev0,
        var=origin,
        t=origin,
        length=origin,
        dt=dt_first,
        n_steps=zero,
        n_tries=zero,
        n_evals=jnp.asarray(1, jnp.int64),
        status=_status(
            ev0, gnorm(ev0.g), origin, dt_first, zero, zero, settings
        ).astype(jnp.int64),
        rec_s=jnp.zeros((rows,) + s0.shape, s0.dtype).at[0].set(s0),
        rec_tl=jnp.zeros((rows, 2), s0.dtype),
        rec_e=jnp.zeros(rows, s0.dtype).at[0].set(ev0.f),
        count=jnp.asarray(1, jnp.int64),
    )
    out = jax.lax.while_loop(
        lambda st: st.status == STATUS_RUNNING,
        lambda st: _step(st, velocity, gnorm, energy, basis, settings),
        state,
    )
    return _finish(out, settings)


class _Proposal(NamedTuple):
    """Candidate step: new state, clock increments, scaled error, evaluations."""

    s: Array
    ev: _Eval
    tl: Array  # (2,): increments of flow time and arclength
    err: Array  # error / tolerance (accept if <= 1)
    n_evals: int
    order: int  # local error order of the estimate (for the controller)


def _scaled_error(
    err_s: Array, step: Array, basis: Array, settings: FlowSettings
) -> Array:
    """RMS Cartesian error over atol + rtol * (RMS Cartesian step)."""
    move = jnp.sqrt(jnp.mean((step @ basis) ** 2))
    tol = settings.atol + settings.rtol * move
    return jnp.sqrt(jnp.mean((err_s @ basis) ** 2)) / tol


def _propose_dopri(
    st: _State, velocity: _Velocity, basis: Array, settings: FlowSettings, dt: Array
) -> _Proposal:
    """Dormand-Prince 5(4) candidate (FSAL); clocks error-controlled too."""
    evs = [st.ev]
    for i in range(1, 6):
        incr = sum(a * e.v for a, e in zip(_A[i], evs, strict=True))
        evs.append(velocity(st.s + dt * incr))
    s_new = st.s + dt * sum(b * e.v for b, e in zip(_B5[:6], evs, strict=True))
    tl_new = dt * sum(b * e.rate for b, e in zip(_B5[:6], evs, strict=True))
    ev_new = velocity(s_new)
    err_s = dt * sum(e * x.v for e, x in zip(_E, evs + [ev_new], strict=True))
    err_tl = dt * sum(e * x.rate for e, x in zip(_E, evs + [ev_new], strict=True))
    err = _scaled_error(err_s, s_new - st.s, basis, settings)
    # The clocks (flow time, arclength) are integrated alongside; control
    # their error relative to their own increments (1/|grad| grows near minima).
    tol_tl = settings.atol + settings.rtol * jnp.abs(tl_new)
    err = jnp.maximum(err, jnp.max(jnp.abs(err_tl) / tol_tl))
    return _Proposal(s_new, ev_new, tl_new, err, 6, 5)


def _propose_rosenbrock(
    st: _State,
    velocity: _Velocity,
    energy: AbstractEnergy,
    basis: Array,
    settings: FlowSettings,
    dt: Array,
) -> _Proposal:
    """ROS2 candidate (Verwer et al. 1999): L-stable, exact Hessian Jacobian.

    (I - g h J) k1 = v(y),  (I - g h J) k2 = v(y + h k1) - 2 k1,
    y+ = y + 3/2 h k1 + 1/2 h k2,  g = 1 + 1/sqrt(2); the embedded first-order
    solution y + h k1 gives the error estimate h (k1 + k2) / 2.
    """
    n, d = st.s.shape
    metric_inv = jnp.linalg.inv(basis @ basis.T)
    hess = energy.hessian(st.s)
    jac = -jnp.einsum("ibjc,ba->iajc", hess.reshape(n, d, n, d), metric_inv)
    w = jnp.eye(n * d, dtype=st.s.dtype) - _ROS_GAMMA * dt * jac.reshape(n * d, n * d)
    lu = jax.scipy.linalg.lu_factor(w)
    k1 = jax.scipy.linalg.lu_solve(lu, st.ev.v.reshape(-1)).reshape(n, d)
    ev1 = velocity(st.s + dt * k1)
    k2 = jax.scipy.linalg.lu_solve(lu, (ev1.v - 2.0 * k1).reshape(-1)).reshape(n, d)
    s_new = st.s + dt * (1.5 * k1 + 0.5 * k2)
    ev_new = velocity(s_new)
    chord = jnp.sqrt(jnp.sum(((s_new - st.s) @ basis) ** 2))
    # Shampine's filter: W^{-1} damps the stiff components of the estimate,
    # which the L-stable scheme suppresses anyway (ode23s practice).
    raw = (0.5 * dt * (k1 + k2)).reshape(-1)
    err_f = jax.scipy.linalg.lu_solve(lu, raw).reshape(n, d)
    err = _scaled_error(err_f, s_new - st.s, basis, settings)
    return _Proposal(s_new, ev_new, jnp.stack([dt, chord]), err, 2, 2)


def _step(
    st: _State,
    velocity: _Velocity,
    gnorm: Callable[[Array], Array],
    energy: AbstractEnergy,
    basis: Array,
    settings: FlowSettings,
) -> _State:
    """One attempted step with error control, then bookkeeping and recording."""
    dt = st.dt
    if settings.parametrization == "time":
        dt = jnp.minimum(dt, settings.t_max - st.t)
    if settings.method == "rosenbrock":
        prop = _propose_rosenbrock(st, velocity, energy, basis, settings, dt)
    else:
        prop = _propose_dopri(st, velocity, basis, settings, dt)
    accept = prop.err <= 1.0
    factor = jnp.clip(_SAFETY * prop.err ** (-1.0 / prop.order), _SHRINK_MIN, _GROW_MAX)
    factor = jnp.where(jnp.isfinite(factor), factor, _SHRINK_MIN)
    dt_next = jnp.minimum(dt * factor, settings.dt_max)

    def pick(new: Array, old: Array) -> Array:
        return jnp.where(accept, new, old)

    s = pick(prop.s, st.s)
    ev = _Eval(*(pick(n, o) for n, o in zip(prop.ev, st.ev, strict=True)))
    t = pick(st.t + prop.tl[0], st.t)
    length = pick(st.length + prop.tl[1], st.length)
    steps = st.n_steps + accept.astype(jnp.int64)
    tries = st.n_tries + 1
    every, cap = max(settings.record_every, 1), max(settings.max_records, 1)
    rec = accept & (settings.record_every > 0) & (steps % every == 0)
    rec = rec & (st.count < cap)
    idx = jnp.minimum(st.count, cap - 1)
    status = _status(ev, gnorm(ev.g), t, dt_next, steps, tries, settings)
    return _State(
        s=s,
        ev=ev,
        var=pick(st.var + dt, st.var),
        t=t,
        length=length,
        dt=dt_next,
        n_steps=steps,
        n_tries=tries,
        n_evals=st.n_evals + prop.n_evals,
        status=status.astype(jnp.int64),
        rec_s=jnp.where(rec, st.rec_s.at[idx].set(s), st.rec_s),
        rec_tl=jnp.where(rec, st.rec_tl.at[idx].set(jnp.stack([t, length])), st.rec_tl),
        rec_e=jnp.where(rec, st.rec_e.at[idx].set(ev.f), st.rec_e),
        count=st.count + rec.astype(jnp.int64),
    )


def _finish(st: _State, settings: FlowSettings) -> FlowResult:
    """Append the final state to the record (overwrite the last row if full)."""
    traj = None
    if settings.record_every > 0:
        last = st.count - 1
        last_is_final = (st.rec_tl[last, 1] == st.length) & (st.rec_tl[last, 0] == st.t)
        idx = jnp.where(
            last_is_final, last, jnp.minimum(st.count, settings.max_records - 1)
        )
        tl = st.rec_tl.at[idx].set(jnp.stack([st.t, st.length]))
        traj = FlowTrajectory(
            s=st.rec_s.at[idx].set(st.s),
            t=tl[:, 0],
            length=tl[:, 1],
            energy=st.rec_e.at[idx].set(st.ev.f),
            count=idx + 1,
        )
    return FlowResult(
        s=st.s,
        energy=st.ev.f,
        t=st.t,
        length=st.length,
        dt=st.dt,
        n_steps=st.n_steps,
        n_evals=st.n_evals,
        status=st.status,
        trajectory=traj,
    )
