#!/usr/bin/env python3

"""
GPU smoke test: energy, gradient and descent on the GPU match the CPU device.

Run through ``slurm/test_gpu.slurm``; deselected on CPU-only hosts by
``-m "not gpu"``. The checks run in a fresh interpreter with
``JAX_PLATFORMS=cuda,cpu``: other tests call ``select_device("cpu")`` in
process, which pins the platform list for the rest of the pytest session
(job 16995049 saw no GPU device for that reason).

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Tuple

# ----- numerics -----
import jax
import jax.numpy as jnp
import numpy as np
import pytest

# ----- project -----
from tolerances import ATOL_VMAP_S, RTOL_KERNEL_E, RTOL_KERNEL_F, TRAJ_RTOL_E
from xla_cco.box import cubic_box
from xla_cco.energy import make_stealthy_energy
from xla_cco.kvectors import enumerate_kvectors, k_outer_from_ka, ka_from_chi
from xla_cco.minimize.descent import DescentSettings, adaptive_descent

# ============================================================
# CONSTANTS
# ============================================================

N, DIM, PHI, CHI = 64, 2, 0.05, 0.25
SEED = 20260611
HORIZON = 10  # adaptive-descent iterations compared between devices

pytestmark = pytest.mark.gpu

# ============================================================
# HELPERS
# ============================================================


def _build() -> Tuple[Any, np.ndarray]:
    """Return the energy and a seeded Poisson configuration (N, DIM)."""
    box = cubic_box(N, DIM)
    k_outer = k_outer_from_ka(ka_from_chi(CHI, PHI, DIM), PHI, DIM)
    kset = enumerate_kvectors(np.asarray(box.basis), 0.0, k_outer)
    s = np.random.default_rng(SEED).uniform(size=(N, DIM))
    return make_stealthy_energy(box, kset), s


def _value_and_grad(
    energy: Any, s: np.ndarray, device: jax.Device
) -> Tuple[float, np.ndarray]:
    e = jax.device_put(energy, device)
    value, grad = e.value_and_grad(jax.device_put(jnp.asarray(s), device))
    assert value.devices() == {device}
    return float(value), np.asarray(grad)


def _trajectory(
    energy: Any, s0: np.ndarray, device: jax.Device
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (s, energy) over the first HORIZON adaptive-descent iterations."""
    settings = DescentSettings(
        grow=1.0, record_every=1, max_records=HORIZON + 1, max_evals=2 * HORIZON
    )
    res = adaptive_descent(
        jax.device_put(energy, device),
        jax.device_put(jnp.asarray(s0), device),
        settings,
    )
    traj = res.trajectory
    assert traj is not None and int(traj.count) >= HORIZON + 1
    return np.asarray(traj.s)[: HORIZON + 1], np.asarray(traj.energy)[: HORIZON + 1]


def check_energy_and_gradient(gpu: jax.Device, cpu: jax.Device) -> None:
    """Assert energy and gradient on ``gpu`` equal those on ``cpu``."""
    energy, s = _build()
    e_g, grad_g = _value_and_grad(energy, s, gpu)
    e_c, grad_c = _value_and_grad(energy, s, cpu)
    assert abs(e_g - e_c) <= RTOL_KERNEL_E * abs(e_c)
    np.testing.assert_allclose(
        grad_g, grad_c, rtol=RTOL_KERNEL_F, atol=1e-12 * np.abs(grad_c).max()
    )


def check_trajectory(gpu: jax.Device, cpu: jax.Device) -> None:
    """Assert the first HORIZON descent iterations on ``gpu`` follow ``cpu``."""
    energy, s0 = _build()
    s_g, e_g = _trajectory(energy, s0, gpu)
    s_c, e_c = _trajectory(energy, s0, cpu)
    acc_g = np.any(s_g[1:] != s_g[:-1], axis=(1, 2))
    acc_c = np.any(s_c[1:] != s_c[:-1], axis=(1, 2))
    assert np.array_equal(acc_g, acc_c), "accept decisions differ"
    assert np.all(np.abs(e_g - e_c) <= TRAJ_RTOL_E * np.abs(e_c))
    d = s_g - s_c
    assert np.abs(d - np.round(d)).max() <= ATOL_VMAP_S


# ============================================================
# TESTS
# ============================================================


def main() -> None:
    """Run the device checks in the current (fresh) interpreter."""
    gpus = [d for d in jax.devices() if d.platform in ("gpu", "cuda")]
    assert gpus, f"no GPU device present: {jax.devices()}"
    cpu = jax.devices("cpu")[0]
    check_energy_and_gradient(gpus[0], cpu)
    check_trajectory(gpus[0], cpu)
    print("GPU-OK")


@pytest.mark.high
def test_gpu_matches_cpu() -> None:
    here = Path(__file__).resolve().parent
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cuda,cpu"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(here), str(here.parent)] + [env.get("PYTHONPATH", "")]
    )
    code = "import test_gpu_device as t; t.main()"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0 and "GPU-OK" in proc.stdout, proc.stderr[-3000:]


if __name__ == "__main__":
    main()
