#!/usr/bin/env python3

"""
C++-faithful adaptive gradient descent in fractional coordinates.

Default rule halves the step on energy increase (grow = 1.0); the PRE 112,
045310 (2025) rule is grow = 2.0. Energies are dimensionless; the step is
in fractional-coordinate units.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from typing import Optional

# ----- numerics -----
import equinox as eqx
import jax.numpy as jnp
from jax import Array

# ----- project -----
from xla_cco._types import Positions, require_float64
from xla_cco.box import wrap_fractional
from xla_cco.energy import AbstractEnergy
from xla_cco.minimize import _loop
from xla_cco.minimize.result import (
    STATUS_BUDGET,
    STATUS_ENERGY_TOL,
    STATUS_GRAD_TOL,
    STATUS_RUNNING,
    STATUS_STEP_FLOOR,
    RelaxResult,
    Trajectory,
)

# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_CHUNK_ITERS: int = 2_000

# ============================================================
# SETTINGS
# ============================================================


class DescentSettings(eqx.Module):
    """Adaptive-descent settings (every field static)."""

    step_init: float = eqx.field(static=True, default=1e-3)
    step_min: float = eqx.field(static=True, default=1e-15)
    shrink: float = eqx.field(static=True, default=0.5)
    grow: float = eqx.field(static=True, default=1.0)
    rel_tol: float = eqx.field(static=True, default=0.01)
    max_evals: int = eqx.field(static=True, default=10_000)
    energy_tol: float = eqx.field(static=True, default=0.0)
    record_every: int = eqx.field(static=True, default=0)
    max_records: int = eqx.field(static=True, default=0)


# ============================================================
# STATE
# ============================================================


class DescentState(eqx.Module):
    """Per-configuration carry: cached (s, f, g), step size and counters.

    ``s`` is NOT wrapped during the iteration (C++ parameters are not wrapped).
    """

    s: Positions
    f: Array
    g: Positions
    step: Array
    it: Array
    n_evals: Array
    status: Array
    traj: Optional[Trajectory]


def _max_iters(settings: DescentSettings) -> int:
    """Iteration budget: ``max_evals // 2`` (odd budgets do not underflow)."""
    return settings.max_evals // 2


def _status(
    f: Array, g: Array, step: Array, it: Array, settings: DescentSettings
) -> Array:
    """Return the status code after an iteration.

    Precedence: energy tol > zero gradient > step floor > budget. An exactly
    zero gradient is a stationary point; the C++ rule would keep accepting
    (0/0 is NaN), and with grow > 1 the step would overflow to NaN (D-012).
    """
    code = jnp.where(it >= _max_iters(settings), STATUS_BUDGET, STATUS_RUNNING)
    code = jnp.where(step <= settings.step_min, STATUS_STEP_FLOOR, code)
    code = jnp.where(jnp.all(g == 0.0), STATUS_GRAD_TOL, code)
    return jnp.where(f < settings.energy_tol, STATUS_ENERGY_TOL, code)


def _recorder(settings: DescentSettings) -> Optional[_loop.Recorder]:
    """Return the trajectory recorder or None when recording is off."""
    if settings.record_every == 0 or settings.max_records == 0:
        return None
    return _loop.Recorder(
        settings.record_every, lambda st: (st.it, st.s, st.f, st.step)
    )


# ============================================================
# ONE-CONFIGURATION PIECES (shared by single and batched paths)
# ============================================================


def init_one(
    energy: AbstractEnergy, s0: Positions, settings: DescentSettings
) -> DescentState:
    """Build the initial state (one value-and-gradient evaluation)."""
    require_float64(s0, "s0")
    assert s0.ndim == 2, "s0 must have shape (n, d)"
    f, g = energy.value_and_grad(s0)
    step = jnp.asarray(settings.step_init, s0.dtype)
    it = jnp.asarray(0, jnp.int64)
    return DescentState(
        s=s0,
        f=f,
        g=g,
        step=step,
        it=it,
        n_evals=jnp.asarray(0, jnp.int64),
        status=_status(f, g, step, it, settings).astype(jnp.int64),
        traj=_loop.init_trajectory(
            s0, f, step, settings.record_every, settings.max_records
        ),
    )


def chunk_one(
    energy: AbstractEnergy,
    settings: DescentSettings,
    state: DescentState,
    max_iters: int,
) -> DescentState:
    """Advance one configuration by at most ``max_iters`` iterations."""
    assert state.s.ndim == 2, "state.s must have shape (n, d)"

    def step_fn(st: DescentState) -> DescentState:
        norm = jnp.sqrt(jnp.sum(st.g * st.g))
        gn = jnp.where(norm > 0.0, st.g / norm, 0.0 * st.g)  # C++ gives NaN at 0
        xn = st.s - gn * st.step
        f2, g2 = energy.value_and_grad(xn)  # same code path as f1 (parity)
        df = st.f - f2
        df2 = st.step * norm
        reject = jnp.abs(df - df2) / df2 > settings.rel_tol  # NaN accepts (C++)
        step = jnp.where(reject, st.step * settings.shrink, st.step * settings.grow)
        s = jnp.where(reject, st.s, xn)
        f = jnp.where(reject, st.f, f2)
        g = jnp.where(reject, st.g, g2)
        it = st.it + 1
        return DescentState(
            s=s,
            f=f,
            g=g,
            step=step,
            it=it,
            n_evals=st.n_evals + 2,
            status=_status(f, g, step, it, settings).astype(jnp.int64),
            traj=st.traj,
        )

    def done_fn(st: DescentState) -> Array:
        return st.status != STATUS_RUNNING

    out: DescentState = _loop.run(
        step_fn, done_fn, state, max_iters, _recorder(settings)
    )
    return out


def finalize_one(state: DescentState) -> RelaxResult:
    """Convert a finished state to a ``RelaxResult`` (wrapped positions)."""
    assert state.s.ndim == 2, "state.s must have shape (n, d)"
    return RelaxResult(
        s=wrap_fractional(state.s),
        energy=state.f,
        grad_norm=jnp.sqrt(jnp.sum(state.g * state.g)),
        n_evals=state.n_evals,
        n_iters=state.it,
        status=state.status,
        trajectory=_loop.trim_trajectory(state.traj),
    )


# ============================================================
# DRIVER
# ============================================================

_init_jit = eqx.filter_jit(init_one)
_chunk_jit = eqx.filter_jit(chunk_one)
_finalize_jit = eqx.filter_jit(finalize_one)
init_batch = eqx.filter_jit(eqx.filter_vmap(init_one, in_axes=(None, 0, None)))
chunk_batch = eqx.filter_jit(eqx.filter_vmap(chunk_one, in_axes=(None, None, 0, None)))
finalize_batch = eqx.filter_jit(eqx.filter_vmap(finalize_one))


def max_rounds(settings: DescentSettings, chunk_iters: int) -> int:
    """Upper bound on host rounds needed for ``settings``."""
    assert chunk_iters >= 1, "chunk_iters must be positive"
    return _max_iters(settings) // chunk_iters + 2


def adaptive_descent(
    energy: AbstractEnergy, s0: Positions, settings: DescentSettings
) -> RelaxResult:
    """Relax one configuration by adaptive gradient descent.

    Equivalent to ``init`` followed by chunked device loops until done, the
    same path ``relax`` uses (vmapped).

    Parameters
    ----------
    energy : AbstractEnergy
        Energy on fractional positions.
    s0 : Positions
        Initial fractional positions, float64, shape (n, d).
    settings : DescentSettings
        Static settings.

    Returns
    -------
    RelaxResult
        Wrapped final positions; ``grad_norm`` is |g| at the final point.
        ``n_evals`` counts 2 per iteration (C++ budget semantics).

    Raises
    ------
    TypeError
        If ``s0`` is not float64.
    """
    require_float64(s0, "s0")
    assert s0.ndim == 2, "s0 must have shape (n, d)"
    chunk = DEFAULT_CHUNK_ITERS
    state = _init_jit(energy, s0, settings)
    state = _loop.drive_single(
        lambda st: _chunk_jit(energy, settings, st, chunk),
        state,
        max_rounds(settings, chunk),
    )
    return _finalize_jit(state)
