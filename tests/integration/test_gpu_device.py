#!/usr/bin/env python3

"""
GPU smoke test: kernel and trajectory parity on the GPU device.

Run through ``slurm/test_gpu.slurm`` (JAX_PLATFORMS=cuda,cpu); deselected on
CPU-only hosts by ``-m "not gpu"``.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from typing import Any, Dict, Tuple

# ----- numerics -----
import jax
import jax.numpy as jnp
import numpy as np
import pytest

# ----- project -----
import golden
from tolerances import RTOL_KERNEL_E, RTOL_KERNEL_F, TRAJ_ATOL_S, TRAJ_RTOL_E
from xla_cco.box import cubic_box
from xla_cco.energy import make_stealthy_energy
from xla_cco.kvectors import enumerate_kvectors
from xla_cco.minimize.descent import DescentSettings, adaptive_descent

# ============================================================
# CONSTANTS
# ============================================================

KERNEL = "kernel_d3_cpp_ic"
TRAJ = "traj_d3_n100_chi0.05"
HORIZON = 20  # C++ accept-stable prefix required by the plan (m <= 20)

pytestmark = pytest.mark.gpu

# ============================================================
# HELPERS
# ============================================================


def _build(name: str) -> Tuple[Dict[str, Any], Any, Any]:
    g = golden.load_golden(name)
    p = g["params"]
    box = cubic_box(p["N"], p["d"])
    kset = enumerate_kvectors(
        np.asarray(box.basis),
        float.fromhex(p["k_inner_hex"]),
        float.fromhex(p["k_outer_hex"]),
    )
    return g, box, make_stealthy_energy(box, kset)


def _value_and_grad(
    energy: Any, s: np.ndarray, device: jax.Device
) -> Tuple[float, np.ndarray]:
    e = jax.device_put(energy, device)
    value, grad = e.value_and_grad(jax.device_put(jnp.asarray(s), device))
    assert value.devices() == {device}
    return float(value), np.asarray(grad)


def _trajectory(device: jax.Device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    g, _, energy = _build(TRAJ)
    settings = DescentSettings(
        grow=1.0, record_every=1, max_records=HORIZON + 1, max_evals=2 * HORIZON
    )
    res = adaptive_descent(
        jax.device_put(energy, device),
        jax.device_put(jnp.asarray(g["s0"][0]), device),
        settings,
    )
    traj = res.trajectory
    assert traj is not None and int(traj.count) >= HORIZON + 1
    s = np.asarray(traj.s)[: HORIZON + 1]
    accepted = np.array([np.any(s[m] != s[m - 1]) for m in range(1, HORIZON + 1)])
    return s, np.asarray(traj.energy)[1 : HORIZON + 1], accepted


def check_trajectory(device: jax.Device) -> None:
    """Assert the descent trajectory on ``device`` follows the C++ golden."""
    g = golden.load_golden(TRAJ)
    s, e, accepted = _trajectory(device)
    assert np.array_equal(accepted, g["accepted"][:HORIZON, 0].astype(bool))
    rel = np.abs(e - g["energy"][:HORIZON, 0]) / g["energy"][:HORIZON, 0]
    assert rel.max() <= TRAJ_RTOL_E, rel.max()
    d = s[1:] - g["s"][:HORIZON, 0]
    assert np.abs(d - np.round(d)).max() <= TRAJ_ATOL_S


def check_kernel_matches_cpu(gpu: jax.Device, cpu: jax.Device) -> None:
    """Assert energy and gradient on ``gpu`` equal those on ``cpu``."""
    g, box, energy = _build(KERNEL)
    for s in g["s"][:2]:
        e_g, grad_g = _value_and_grad(energy, s, gpu)
        e_c, grad_c = _value_and_grad(energy, s, cpu)
        assert abs(e_g - e_c) <= RTOL_KERNEL_E * abs(e_c)
        np.testing.assert_allclose(
            grad_g, grad_c, rtol=RTOL_KERNEL_F, atol=1e-12 * np.abs(grad_c).max()
        )
        assert abs(e_g - float(energy(jnp.asarray(s)))) <= RTOL_KERNEL_E * abs(e_c)


# ============================================================
# TESTS
# ============================================================


def test_gpu_matches_cpu_and_cpp() -> None:
    gpus = [d for d in jax.devices() if d.platform in ("gpu", "cuda")]
    assert gpus, "no GPU device present"
    check_kernel_matches_cpu(gpus[0], jax.devices("cpu")[0])
    check_trajectory(gpus[0])
