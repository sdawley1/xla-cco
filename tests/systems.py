#!/usr/bin/env python3

"""
Small stealthy systems shared by the minimizer tests.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import functools
from typing import Any, Tuple

# ----- numerics -----
import jax.numpy as jnp
import numpy as np

# ----- project -----
from xla_cco.box import cubic_box
from xla_cco.energy import StealthyEnergy, make_stealthy_energy
from xla_cco.kvectors import enumerate_kvectors
from xla_cco.minimize.flow import FlowSettings, gradient_flow

# ============================================================
# FACTORIES
# ============================================================


def small_system(n: int, d: int, k_outer: float) -> Tuple[Any, Any, StealthyEnergy]:
    """Return (box, kset, energy) for a cubic box with a (0, k_outer) window."""
    assert n >= 1 and d in (1, 2, 3), "bad system shape"
    box = cubic_box(n, d)
    kset = enumerate_kvectors(np.asarray(box.basis), 0.0, k_outer)
    return box, kset, make_stealthy_energy(box, kset)


def square_lattice(side: int) -> np.ndarray:
    """Return a side x side square lattice in fractional coordinates, (n, 2)."""
    g = np.arange(side) / side
    return np.stack(np.meshgrid(g, g, indexing="ij"), axis=-1).reshape(-1, 2)


@functools.lru_cache(maxsize=4)
def stealthy_ground_state(
    n: int, d: int, k_outer: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, float]:
    """(s_gs (n, d) fractional, basis (d, d), Phi) of a cubic-box ground state.

    Gradient flow (Dormand-Prince) from a seeded uniform start down to
    Phi < 1e-20; cached, so the MD tests share one relaxation per system.
    """
    box, _, energy = small_system(n, d, k_outer)
    s0 = jnp.asarray(np.random.default_rng(seed).uniform(size=(n, d)))
    fs = FlowSettings(energy_tol=1e-21, rtol=1e-6, max_steps=100_000)
    res = gradient_flow(energy, s0, box.basis, fs)
    phi = float(res.energy)
    assert phi < 1e-20, f"flow stopped at Phi = {phi:g} (status {int(res.status)})"
    assert np.all(np.isfinite(np.asarray(res.s))), "non-finite ground state"
    return np.asarray(res.s), np.asarray(box.basis), phi
