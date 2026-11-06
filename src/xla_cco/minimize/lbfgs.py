#!/usr/bin/env python3

"""
Optax L-BFGS minimizer with the C++ three-pass rescale schedule.

Pass with rescale R optimizes x = R * s (R = 1e4, 1, 1e-2 by default,
matching RelaxStructure_NLOPT). Energies are dimensionless; positions are
fractional.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from typing import Any, Optional, Tuple

# ----- numerics -----
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
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
    STATUS_LINESEARCH_FAIL,
    STATUS_RUNNING,
    STATUS_STAGNATED,
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


class LBFGSSettings(eqx.Module):
    """L-BFGS settings (every field static)."""

    memory: int = eqx.field(static=True, default=100)
    rescales: Tuple[float, ...] = eqx.field(static=True, default=(1e4, 1.0, 1e-2))
    max_evals_per_pass: int = eqx.field(static=True, default=500_000)
    energy_tol: float = eqx.field(static=True, default=0.0)
    grad_tol: float = eqx.field(static=True, default=0.0)
    stagnation_patience: int = eqx.field(static=True, default=50)
    max_linesearch_steps: int = eqx.field(static=True, default=30)
    # appended after the frozen fields (additive): trajectory recording
    record_every: int = eqx.field(static=True, default=0)
    max_records: int = eqx.field(static=True, default=0)


# ============================================================
# OBJECTIVE
# ============================================================


@jax.custom_vjp
def _phi_x(energy: AbstractEnergy, rescale: Array, x: Positions) -> Array:
    """Return Phi(x / R); the gradient comes from ``energy.value_and_grad``."""
    value, _ = energy.value_and_grad(x / rescale)
    return value


def _phi_x_fwd(
    energy: AbstractEnergy, rescale: Array, x: Positions
) -> Tuple[Array, Tuple[Positions, AbstractEnergy, Array]]:
    value, grad_s = energy.value_and_grad(x / rescale)
    return value, (grad_s / rescale, energy, rescale)


def _phi_x_bwd(
    res: Tuple[Positions, AbstractEnergy, Array], ct: Array
) -> Tuple[Any, Any, Positions]:
    grad_x, energy, rescale = res
    zeros_e = jax.tree_util.tree_map(jnp.zeros_like, energy)
    return zeros_e, jnp.zeros_like(rescale), ct * grad_x


_phi_x.defvjp(_phi_x_fwd, _phi_x_bwd)


def _optimizer(settings: LBFGSSettings) -> optax.GradientTransformationExtraArgs:
    """Optax L-BFGS with a zoom line search (state reset = ``init`` again)."""
    return optax.lbfgs(
        memory_size=settings.memory,
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=settings.max_linesearch_steps,
            initial_guess_strategy="one",
        ),
    )


# ============================================================
# STATE
# ============================================================


class LBFGSState(eqx.Module):
    """Per-configuration carry across the whole multi-pass schedule.

    ``x = R * s`` for the current pass rescale R; ``gx`` is dPhi/dx cached
    from the previous line search (no re-evaluation, as with
    ``optax.value_and_grad_from_state``).
    """

    s: Positions
    x: Positions
    f: Array
    gx: Positions
    opt: Any
    last_step: Array
    pass_idx: Array
    pass_evals: Array
    stag: Array
    it: Array
    n_evals: Array
    status: Array
    traj: Optional[Trajectory]


def _rescale(settings: LBFGSSettings, pass_idx: Array) -> Array:
    return jnp.asarray(settings.rescales, jnp.float64)[pass_idx]


def _recorder(settings: LBFGSSettings) -> Optional[Any]:
    """Return the trajectory recorder or None when recording is off."""
    if settings.record_every == 0 or settings.max_records == 0:
        return None
    return _loop.Recorder(
        settings.record_every,
        lambda st: (
            st.it,
            st.s,
            st.f,
            st.last_step,
        ),
    )


# ============================================================
# ONE-CONFIGURATION PIECES
# ============================================================


def init_one(
    energy: AbstractEnergy,
    s0: Positions,
    settings: LBFGSSettings,
) -> LBFGSState:
    """Build the initial state (one value-and-gradient evaluation)."""
    require_float64(s0, "s0")
    assert s0.ndim == 2, "s0 must have shape (n, d)"
    assert len(settings.rescales) >= 1, "need at least one rescale pass"
    r0 = jnp.asarray(settings.rescales[0], s0.dtype)
    f, gs = energy.value_and_grad(s0)
    x = s0 * r0
    status = jnp.where(f < settings.energy_tol, STATUS_ENERGY_TOL, STATUS_RUNNING)
    zero = jnp.asarray(0, jnp.int64)
    return LBFGSState(
        s=s0,
        x=x,
        f=f,
        gx=gs / r0,
        opt=_optimizer(settings).init(x),
        last_step=jnp.asarray(0.0, s0.dtype),
        pass_idx=zero,
        pass_evals=zero,
        stag=zero,
        it=zero,
        n_evals=jnp.asarray(1, jnp.int64),
        status=status.astype(jnp.int64),
        traj=_loop.init_trajectory(
            s0,
            f,
            jnp.asarray(0.0, s0.dtype),
            settings.record_every,
            settings.max_records,
        ),
    )


def _line_search(
    energy: AbstractEnergy, settings: LBFGSSettings, st: LBFGSState, r: Array
) -> Tuple[Array, Array, Array, Array, Array, Any, Array, Array]:
    """One L-BFGS update. Returns x, f, gx, fail, evals, opt_state, step, ok.

    Failure = zoom line search ended with ``decrease_error > 0`` (no step with
    sufficient decrease) or a non-finite value. Evaluations = the line-search
    step count (one ``value_and_grad`` per zoom step).
    """
    solver = _optimizer(settings)
    updates, opt_new = solver.update(
        st.gx,
        st.opt,
        st.x,
        value=st.f,
        grad=st.gx,
        value_fn=lambda x: _phi_x(energy, r, x),
    )
    info = optax.tree.get(opt_new, "info")
    f_new = optax.tree.get(opt_new, "value")
    gx_new = optax.tree.get(opt_new, "grad")
    step = optax.tree.get(opt_new, "learning_rate")
    x_new = optax.apply_updates(st.x, updates)
    fail = (info.decrease_error > 0.0) | ~jnp.isfinite(f_new)
    evals = jnp.maximum(info.num_linesearch_steps, 1).astype(jnp.int64)
    return x_new, f_new, gx_new, fail, evals, opt_new, step, info.decrease_error


def update_stagnation(f_new: Array, f_old: Array, stag: Array) -> Array:
    """Consecutive-iterations counter: reset on a strict decrease, else +1."""
    assert f_new.shape == f_old.shape == stag.shape, "scalar counters expected"
    assert jnp.issubdtype(stag.dtype, jnp.integer), "stag must be an integer"
    return jnp.where(f_new < f_old, jnp.zeros_like(stag), stag + 1)


def _pass_reason(
    settings: LBFGSSettings,
    st: LBFGSState,
    f: Array,
    gx: Array,
    r: Array,
    fail: Array,
    stag: Array,
    pass_evals: Array,
) -> Tuple[Array, Array]:
    """Return (global_code, pass_code); 0 = not triggered."""
    gnorm = jnp.sqrt(jnp.sum(gx * gx)) * r
    glob = jnp.where(gnorm < settings.grad_tol, STATUS_GRAD_TOL, STATUS_RUNNING)
    glob = jnp.where(f < settings.energy_tol, STATUS_ENERGY_TOL, glob)
    local = jnp.where(
        pass_evals >= settings.max_evals_per_pass, STATUS_BUDGET, STATUS_RUNNING
    )
    local = jnp.where(stag >= settings.stagnation_patience, STATUS_STAGNATED, local)
    local = jnp.where(fail, STATUS_LINESEARCH_FAIL, local)
    return glob, local


def chunk_one(
    energy: AbstractEnergy,
    settings: LBFGSSettings,
    state: LBFGSState,
    max_iters: int,
) -> LBFGSState:
    """Advance one configuration by at most ``max_iters`` iterations."""
    assert state.x.ndim == 2, "state.x must have shape (n, d)"
    n_pass = len(settings.rescales)

    def step_fn(st: LBFGSState) -> LBFGSState:
        r = _rescale(settings, st.pass_idx)
        x_n, f_n, g_n, fail, evals, opt_n, step, _ = _line_search(
            energy, settings, st, r
        )
        # a failed search keeps the previous point (monotone in Phi)
        x = jnp.where(fail, st.x, x_n)
        s = jnp.where(fail, st.s, x_n / r)  # s carried: no s*R/R round trip
        f = jnp.where(fail, st.f, f_n)
        gx = jnp.where(fail, st.gx, g_n)
        stag = update_stagnation(f, st.f, st.stag)
        pass_evals = st.pass_evals + evals
        glob, local = _pass_reason(settings, st, f, gx, r, fail, stag, pass_evals)
        last = st.pass_idx >= n_pass - 1
        pass_end = (local != STATUS_RUNNING) & (glob == STATUS_RUNNING)
        advance = pass_end & ~last
        status = jnp.where(glob != STATUS_RUNNING, glob, STATUS_RUNNING)
        status = jnp.where(pass_end & last, local, status)
        r_next = _rescale(settings, jnp.minimum(st.pass_idx + 1, n_pass - 1))
        x_adv = s * r_next
        gx_adv = gx * r / r_next
        new = LBFGSState(
            s=s,
            x=jnp.where(advance, x_adv, x),
            f=f,
            gx=jnp.where(advance, gx_adv, gx),
            opt=jax.tree_util.tree_map(
                lambda fresh, cur: jnp.where(advance, fresh, cur),
                _optimizer(settings).init(x_adv),
                opt_n,
            ),
            last_step=step,
            pass_idx=st.pass_idx + advance.astype(jnp.int64),
            pass_evals=jnp.where(advance, 0, pass_evals),
            stag=jnp.where(advance, 0, stag),
            it=st.it + 1,
            n_evals=st.n_evals + evals,
            status=status.astype(jnp.int64),
            traj=st.traj,
        )
        return new

    def done_fn(st: LBFGSState) -> Array:
        return st.status != STATUS_RUNNING

    out: LBFGSState = _loop.run(step_fn, done_fn, state, max_iters, _recorder(settings))
    return out


def finalize_one(settings: LBFGSSettings, state: LBFGSState) -> RelaxResult:
    """Convert a finished state to a ``RelaxResult`` (wrapped positions)."""
    assert state.x.ndim == 2, "state.x must have shape (n, d)"
    r = _rescale(settings, state.pass_idx)
    gnorm = jnp.sqrt(jnp.sum(state.gx * state.gx)) * r
    return RelaxResult(
        s=wrap_fractional(state.s),
        energy=state.f,
        grad_norm=gnorm,
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
finalize_batch = eqx.filter_jit(eqx.filter_vmap(finalize_one, in_axes=(None, 0)))


def max_rounds(settings: LBFGSSettings, chunk_iters: int) -> int:
    """Upper bound on host rounds (every iteration costs >= 1 evaluation)."""
    assert chunk_iters >= 1, "chunk_iters must be positive"
    total = len(settings.rescales) * (
        settings.max_evals_per_pass + settings.max_linesearch_steps + 1
    )
    return total // chunk_iters + 2


def lbfgs(
    energy: AbstractEnergy, s0: Positions, settings: LBFGSSettings
) -> RelaxResult:
    """Relax one configuration with the three-pass L-BFGS schedule.

    Parameters
    ----------
    energy : AbstractEnergy
        Energy on fractional positions.
    s0 : Positions
        Initial fractional positions, float64, shape (n, d).
    settings : LBFGSSettings
        Static settings.

    Returns
    -------
    RelaxResult
        ``n_evals``/``n_iters`` are totals over passes (plus the initial
        evaluation in ``n_evals``); ``status`` is the reason the last pass
        ended, or the global stop (energy / gradient tolerance). Trajectory
        ``s`` records are wrapped into [0, 1), as for descent, although the
        L-BFGS iterates are unwrapped during the run.

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
    return _finalize_jit(settings, state)
