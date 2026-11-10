#!/usr/bin/env python3

"""
Single-lane MD integrators: velocity Verlet, Andersen resets, Langevin BAOAB.

All functions are pure, jit/vmap compatible and act on Cartesian arrays
(mass 1). A ``ForceFn`` maps positions r (n, d) to (Phi, F (n, d), re (m,),
im (m,)) where re/im are the energy's collective densities (m may include
padding rows); integrators pass re/im through so the caller can reuse them.

- ``verlet_step``: v += dt/2 F; r += dt v; F = F(r); v += dt/2 F.
- ``andersen_reset``: "count" (acc += p; if acc >= 1 (at most once per step
  for p <= 1): redraw one uniformly chosen particle's velocity from N(0, T),
  acc -= 1; the accumulator is carried across calls) or "bernoulli" (one
  redraw with probability p). No COM removal (as C++). This is not the C++
  MD_System.cpp AndersonEvolve rule, which resets its counter to 0 in every
  call and fires while it is > 0 (501 resets per 5e4-step call from step 1,
  D-050); a caller reproduces it by setting reset_acc = 1 at each call
  (tools/cpp_reference/md_jax_replica.py, D-081).
- ``baoab_step``: B (half kick), A (half drift), O (v = c v + sqrt((1-c^2) T)
  xi, c = exp(-gamma dt)), A, force, B. Exact configurational sampling for
  harmonic potentials at any stable dt (Leimkuhler & Matthews 2013).

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import math
from typing import Callable, Tuple

# ----- numerics -----
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Int, UInt32

# ----- project -----
from xla_cco._types import require_float64
from xla_cco.dynamics.state import ANDERSEN_MODES
from xla_cco.dynamics.thermal import require_key

# ============================================================
# TYPES
# ============================================================

ForceOut = Tuple[Float[Array, ""], Float[Array, "n d"], Float[Array, " m"],
                 Float[Array, " m"]]  # fmt: skip
ForceFn = Callable[[Float[Array, "n d"]], ForceOut]

# ============================================================
# HELPERS
# ============================================================


def _check_phase_space(r: Array, v: Array, f: Array, dt: float) -> None:
    """Static checks: float64 (n, d) arrays of one shape and a positive step."""
    for x, name in ((r, "r"), (v, "v"), (f, "f")):
        require_float64(x, name)
    if r.ndim != 2 or v.shape != r.shape or f.shape != r.shape:
        raise ValueError(f"r, v, f must share an (n, d) shape: {r.shape}, "
                         f"{v.shape}, {f.shape}")  # fmt: skip
    if not (math.isfinite(dt) and dt > 0.0):
        raise ValueError(f"dt must be finite and positive, got {dt}")


def _kick(v: Array, out: ForceOut, half: float) -> Array:
    """Half kick v + half F from a force evaluation (shape checked statically)."""
    assert len(out) == 4, "force_fn must return (Phi, F, re, im)"
    assert out[1].shape == v.shape, f"force shape {out[1].shape} != {v.shape}"
    return v + half * out[1]


# ============================================================
# FUNCTIONS
# ============================================================


def verlet_step(
    r: Float[Array, "n d"], v: Float[Array, "n d"], f: Float[Array, "n d"],
    force_fn: ForceFn, dt: float,
) -> Tuple[Float[Array, "n d"], Float[Array, "n d"], ForceOut]:  # fmt: skip
    """One velocity-Verlet step; returns (r', v', force_fn(r'))."""
    _check_phase_space(r, v, f, dt)
    half = 0.5 * dt
    v_half = v + half * f
    r_new = r + dt * v_half
    assert r_new.shape == r.shape, "drift changed the shape of r"
    out = force_fn(r_new)
    v_new = _kick(v_half, out, half)
    assert v_new.dtype == v.dtype == r_new.dtype, "dtype changed in the step"
    return r_new, v_new, out


def andersen_reset(
    key: UInt32[Array, " 2"], v: Float[Array, "n d"], reset_acc: Float[Array, ""],
    temperature: Float[Array, ""], p: float, mode: str,
) -> Tuple[UInt32[Array, " 2"], Float[Array, "n d"], Float[Array, ""],
           Int[Array, ""]]:  # fmt: skip
    """Andersen resets after a step; returns (key', v', reset_acc', n_reset).

    ``n_reset`` is 0 or 1 (int32). In "bernoulli" mode ``reset_acc`` is
    returned unchanged. Keys: ``key', k_pick, k_draw, k_coin =
    split(key, 4)``; the particle index comes from ``k_pick``, the redrawn
    row sqrt(T) N(0, I_d) from ``k_draw`` (independent of the choice) and
    the Bernoulli coin from ``k_coin``.

    Raises
    ------
    ValueError
        On an unknown ``mode``, ``p`` outside [0, 1] or a bad ``v`` shape.
    """
    if mode not in ANDERSEN_MODES:
        raise ValueError(f"mode must be one of {ANDERSEN_MODES}, got {mode!r}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must lie in [0, 1], got {p}")
    require_key(key)
    require_float64(v, "v")
    if v.ndim != 2:
        raise ValueError(f"v must have shape (n, d), got {v.shape}")
    n, d = v.shape
    key, k_pick, k_draw, k_coin = jax.random.split(key, 4)
    if mode == "count":
        acc = reset_acc + p
        fire = acc >= 1.0
        acc = jnp.where(fire, acc - 1.0, acc)
    else:
        fire = jax.random.uniform(k_coin, (), dtype=jnp.float64) < p
        acc = jnp.asarray(reset_acc, dtype=jnp.float64)
    pick = jax.random.randint(k_pick, (), 0, n)
    t = jnp.asarray(temperature, dtype=jnp.float64)
    row = jnp.sqrt(t) * jax.random.normal(k_draw, (d,), dtype=jnp.float64)
    mask = jnp.logical_and(fire, jnp.arange(n) == pick)[:, None]
    v_new = jnp.where(mask, row[None, :], v)
    return key, v_new, acc, fire.astype(jnp.int32)


def baoab_step(
    key: UInt32[Array, " 2"], r: Float[Array, "n d"], v: Float[Array, "n d"],
    f: Float[Array, "n d"], force_fn: ForceFn, dt: float, gamma: float,
    temperature: Float[Array, ""],
) -> Tuple[UInt32[Array, " 2"], Float[Array, "n d"], Float[Array, "n d"],
           ForceOut]:  # fmt: skip
    """One BAOAB Langevin step; returns (key', r', v', force_fn(r')).

    Keys: ``key', k_xi = split(key)``; xi ~ N(0, I) of shape (n, d) from
    ``k_xi``.

    Raises
    ------
    ValueError
        On a negative or non-finite ``gamma`` or a bad step/shape.
    """
    _check_phase_space(r, v, f, dt)
    if not (math.isfinite(gamma) and gamma >= 0.0):
        raise ValueError(f"gamma must be finite and >= 0, got {gamma}")
    require_key(key)
    c = math.exp(-gamma * dt)
    noise_var = -math.expm1(-2.0 * gamma * dt)  # 1 - c^2 without cancellation
    half = 0.5 * dt
    key, k_xi = jax.random.split(key)
    v = v + half * f
    r = r + half * v
    t = jnp.asarray(temperature, dtype=jnp.float64)
    xi = jax.random.normal(k_xi, v.shape, dtype=jnp.float64)
    v = c * v + jnp.sqrt(noise_var * t) * xi
    r = r + half * v
    assert xi.shape == r.shape and xi.dtype == jnp.float64, "bad noise draw"
    out = force_fn(r)
    v = _kick(v, out, half)
    assert v.shape == r.shape and v.dtype == r.dtype, "step changed v"
    return key, r, v, out
