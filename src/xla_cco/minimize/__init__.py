#!/usr/bin/env python3

"""
Batched relaxation with host-side compaction of converged configurations.

Each round runs a jitted, vmapped chunk of device iterations from every
unfinished lane's carried state; between rounds only unfinished lanes are
kept (padded to a power of two to limit recompiles). Finished lanes are
frozen bit-exactly by the device loop and never touched again.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from typing import Any, Callable, NamedTuple, Union

# ----- numerics -----
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

# ----- project -----
from xla_cco._types import BatchPositions, require_float64
from xla_cco.energy import AbstractEnergy
from xla_cco.minimize import descent, lbfgs
from xla_cco.minimize.descent import DescentSettings
from xla_cco.minimize.lbfgs import LBFGSSettings
from xla_cco.minimize.result import STATUS_BUDGET, STATUS_RUNNING, RelaxResult

# ============================================================
# METHOD TABLE
# ============================================================


class _Method(NamedTuple):
    init: Callable[..., Any]
    chunk: Callable[..., Any]
    finalize: Callable[..., Any]
    max_rounds: Callable[[Any, int], int]
    finalize_takes_settings: bool


_DESCENT = _Method(
    descent.init_batch,
    descent.chunk_batch,
    descent.finalize_batch,
    descent.max_rounds,
    False,
)
_LBFGS = _Method(
    lbfgs.init_batch,
    lbfgs.chunk_batch,
    lbfgs.finalize_batch,
    lbfgs.max_rounds,
    True,
)


def _next_pow2(n: int) -> int:
    assert n >= 1, "n must be positive"
    return 1 << (n - 1).bit_length()


# ============================================================
# BATCHED DRIVER
# ============================================================


def _round(
    method: _Method,
    energy: AbstractEnergy,
    settings: Any,
    state: Any,
    active: np.ndarray,
    chunk_iters: int,
) -> Any:
    """Run one chunk on the active lanes only; scatter results back."""
    b = state.status.shape[0]
    n_act = int(active.size)
    padded = _next_pow2(n_act)
    if n_act == b == padded:
        return method.chunk(energy, settings, state, chunk_iters)
    idx = np.concatenate([active, np.full(padded - n_act, active[0])])
    sub = jax.tree_util.tree_map(lambda a: a[idx], state)
    mask = jnp.asarray(np.arange(padded) < n_act)
    # padded duplicates are masked out: marked finished, so they stay frozen
    sub = eqx.tree_at(
        lambda x: x.status, sub, jnp.where(mask, sub.status, STATUS_BUDGET)
    )
    sub = method.chunk(energy, settings, sub, chunk_iters)
    return jax.tree_util.tree_map(
        lambda full, part: full.at[active].set(part[:n_act]), state, sub
    )


def relax(
    energy: AbstractEnergy,
    s0: BatchPositions,
    settings: Union[DescentSettings, LBFGSSettings],
    chunk_iters: int = 2_000,
) -> RelaxResult:
    """Relax a batch of configurations; result leaves have a leading batch axis.

    The result equals running ``adaptive_descent`` / ``lbfgs`` per
    configuration (same init and chunk code, vmapped); results are returned in
    the original order.

    Parameters
    ----------
    energy : AbstractEnergy
        Energy on fractional positions.
    s0 : BatchPositions
        Initial fractional positions, float64, shape (b, n, d).
    settings : Union[DescentSettings, LBFGSSettings]
        Minimizer settings; the type selects the minimizer.
    chunk_iters : int
        Device iterations between host-side compaction steps.

    Raises
    ------
    TypeError
        If ``s0`` is not float64.
    """
    require_float64(s0, "s0")
    assert s0.ndim == 3, "s0 must have shape (b, n, d)"
    assert chunk_iters >= 1, "chunk_iters must be positive"
    assert s0.shape[0] >= 1, "need at least one configuration"
    if isinstance(settings, DescentSettings):
        method = _DESCENT
    elif isinstance(settings, LBFGSSettings):
        method = _LBFGS
    else:
        raise TypeError(f"unsupported settings type {type(settings).__name__}")
    state = method.init(energy, s0, settings)
    for _ in range(method.max_rounds(settings, chunk_iters)):
        active = np.flatnonzero(np.asarray(state.status) == STATUS_RUNNING)
        if active.size == 0:
            break
        state = _round(method, energy, settings, state, active, chunk_iters)
    else:
        if np.any(np.asarray(state.status) == STATUS_RUNNING):
            raise RuntimeError("relax did not finish within the round bound")
    if method.finalize_takes_settings:
        return method.finalize(settings, state)  # type: ignore[no-any-return]
    return method.finalize(state)  # type: ignore[no-any-return]
