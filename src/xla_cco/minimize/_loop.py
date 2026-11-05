#!/usr/bin/env python3

"""
Bounded lax.while_loop driver shared by both minimizers.

Provides per-lane freezing of converged configurations (bit-exact state,
history and step), the trajectory recorder, and the host-side loop that runs
chunks until a single configuration is done. Every loop condition carries an
explicit iteration counter. Positions are fractional; energies are
dimensionless.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from typing import Any, Callable, NamedTuple, Optional, Tuple

# ----- numerics -----
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

# ----- project -----
from xla_cco.box import wrap_fractional
from xla_cco.minimize.result import STATUS_RUNNING, Trajectory

# ============================================================
# TRAJECTORY RECORDER
# ============================================================


class Recorder(NamedTuple):
    """Recorder spec: ``view(state) -> (iteration, s, energy, step)``.

    ``state.traj`` must be a buffer made by ``init_trajectory`` (capacity
    ``max_records`` + 1; the last row is a scratch row for skipped writes).
    """

    every: int
    view: Callable[[Any], Tuple[Array, Array, Array, Array]]


def init_trajectory(
    s: Array, energy: Array, step: Array, every: int, max_records: int
) -> Optional[Trajectory]:
    """Return a buffer holding record 0 (iteration 0), or None if disabled."""
    assert every >= 0 and max_records >= 0, "record settings must be >= 0"
    assert s.ndim == 2, "s must have shape (n, d)"
    if every == 0 or max_records == 0:
        return None
    cap = max_records + 1  # +1 scratch row
    buf_s = jnp.zeros((cap,) + s.shape, s.dtype).at[0].set(wrap_fractional(s))
    buf_e = jnp.zeros((cap,), s.dtype).at[0].set(energy)
    buf_t = jnp.zeros((cap,), s.dtype).at[0].set(step)
    return Trajectory(
        s=buf_s,
        energy=buf_e,
        step=buf_t,
        iteration=jnp.zeros((cap,), jnp.int64),
        count=jnp.asarray(1, jnp.int64),
    )


def _record(traj: Trajectory, rec: Recorder, state: Any) -> Trajectory:
    """Write one row at ``count`` when due and not full (else the scratch row)."""
    it, s, f, step = rec.view(state)
    cap = traj.s.shape[0] - 1
    do = ((it % rec.every) == 0) & (traj.count < cap)
    row = jnp.where(do, traj.count, cap)
    dus = jax.lax.dynamic_update_slice
    zero = jnp.zeros((), jnp.int64)
    return Trajectory(
        s=dus(traj.s, wrap_fractional(s)[None], (row, zero, zero)),
        energy=dus(traj.energy, f[None], (row,)),
        step=dus(traj.step, step[None], (row,)),
        iteration=dus(traj.iteration, it.astype(jnp.int64)[None], (row,)),
        count=traj.count + do.astype(jnp.int64),
    )


def trim_trajectory(traj: Optional[Trajectory]) -> Optional[Trajectory]:
    """Drop the scratch row (last row along the record axis)."""
    if traj is None:
        return None
    return Trajectory(
        s=traj.s[:-1],
        energy=traj.energy[:-1],
        step=traj.step[:-1],
        iteration=traj.iteration[:-1],
        count=traj.count,
    )


# ============================================================
# DRIVER
# ============================================================


def run(
    step_fn: Callable[[Any], Any],
    done_fn: Callable[[Any], Array],
    state: Any,
    max_iters: int,
    recorder: Optional[Recorder] = None,
) -> Any:
    """Advance ``state`` by at most ``max_iters`` iterations of ``step_fn``.

    The loop stops when ``done_fn(state)`` holds or the counter reaches
    ``max_iters``. Finished lanes are frozen bit-exactly: the whole state
    PyTree is passed through ``jnp.where(done, old, new)``, which matters
    under ``vmap`` where all lanes step until every lane is done.

    Parameters
    ----------
    step_fn : Callable[[Any], Any]
        One iteration.
    done_fn : Callable[[Any], Array]
        Scalar boolean: the lane is finished.
    state : Any
        State PyTree (an Equinox module with a ``traj`` field if recording).
    max_iters : int
        Static hard iteration bound for this call.
    recorder : Optional[Recorder]
        Trajectory recorder, applied after each step.
    """
    assert max_iters >= 0, "max_iters must be non-negative"
    assert callable(step_fn) and callable(done_fn), "step_fn/done_fn callable"

    def cond(carry: Tuple[Array, Any]) -> Array:
        i, st = carry
        return (i < max_iters) & ~done_fn(st)

    def body(carry: Tuple[Array, Any]) -> Tuple[Array, Any]:
        i, st = carry
        new = step_fn(st)
        if recorder is not None:
            traj = _record(new.traj, recorder, new)
            new = eqx.tree_at(lambda x: x.traj, new, traj)
        done = done_fn(st)
        frozen = jax.tree_util.tree_map(lambda o, n: jnp.where(done, o, n), st, new)
        return i + 1, frozen

    _, out = jax.lax.while_loop(cond, body, (jnp.asarray(0, jnp.int32), state))
    return out


def drive_single(chunk: Callable[[Any], Any], state: Any, max_rounds: int) -> Any:
    """Host loop: call ``chunk`` until ``state.status`` is not running.

    Parameters
    ----------
    chunk : Callable[[Any], Any]
        Runs a bounded number of device iterations from the carried state.
    state : Any
        Initial state with a scalar ``status`` field.
    max_rounds : int
        Upper bound on the number of chunk calls (guards against a stall).
    """
    assert max_rounds >= 1, "max_rounds must be positive"
    for _ in range(max_rounds):
        if int(state.status) != STATUS_RUNNING:
            return state
        state = chunk(state)
    raise RuntimeError("minimizer did not finish within the round bound")
