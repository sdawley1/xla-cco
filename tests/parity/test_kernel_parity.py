#!/usr/bin/env python3

"""
Kernel parity against the C++ goldens: k-set, chi, energy, forces.

The window is replayed bit-exactly from the stored hex; D-006 (the recomputed
window equals the hex) is covered in tests/unit/test_kvectors.py.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from typing import Any, Dict, Optional, Set, Tuple

# ----- numerics -----
import jax.numpy as jnp
import numpy as np
import pytest

# ----- project -----
import golden
from tolerances import C_E, RTOL_KERNEL_E, RTOL_KERNEL_F
from xla_cco.box import Box, cubic_box
from xla_cco.energy import StealthyEnergy, make_stealthy_energy
from xla_cco.kvectors import KSet, chi_from_count, enumerate_kvectors

# ============================================================
# CONSTANTS
# ============================================================

NAMES = (
    "kernel_d1_chi0.05",
    "kernel_d1_chi0.25",
    "kernel_d1_chi0.45",
    "kernel_d2_chi0.05",
    "kernel_d2_chi0.25",
    "kernel_d2_chi0.45",
    "kernel_d3_chi0.05",
    "kernel_d3_chi0.25",
    "kernel_d3_chi0.45",
    "kernel_d3_cpp_ic",
    "kernel_boundary_on_shell",
    "kernel_boundary_above_shell",
)
EPS: float = float(np.finfo(np.float64).eps)
CHUNK_SIZE: int = 5  # a non-divisor of every padded M

# ============================================================
# HELPERS
# ============================================================


def _setup(name: str) -> Tuple[Dict[str, Any], Box, KSet]:
    """Load a golden and build the box and k-set from the stored hex window."""
    g = golden.load_golden(name)
    p = g["params"]
    box = cubic_box(p["N"], p["d"])
    kset = enumerate_kvectors(
        np.asarray(box.basis),
        float.fromhex(p["k_inner_hex"]),
        float.fromhex(p["k_outer_hex"]),
    )
    return g, box, kset


def _rows(n: np.ndarray) -> Set[Tuple[int, ...]]:
    return {tuple(int(x) for x in row) for row in n}


def _energies(
    energy: StealthyEnergy, g: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (``__call__`` values, ``value_and_grad`` values) per stored config."""
    call = [float(energy(jnp.asarray(s))) for s in g["s"]]
    vag = [float(energy.value_and_grad(jnp.asarray(s))[0]) for s in g["s"]]
    return np.asarray(call), np.asarray(vag)


# ============================================================
# TESTS
# ============================================================


@pytest.mark.parametrize("name", NAMES)
def test_kset_parity(name: str) -> None:
    g, _, kset = _setup(name)
    p = g["params"]
    assert _rows(kset.n) == _rows(g["n"])
    assert kset.n.shape[0] == int(g["m"]) == p["M"]
    # Same double expression M / (d (N - 1)) as C++: bit-equal, no 1-ulp slack.
    assert chi_from_count(kset.n.shape[0], p["N"], p["d"]) == float(g["chi"])


@pytest.mark.parametrize("chunk_size", [None, CHUNK_SIZE])
@pytest.mark.parametrize("name", NAMES)
def test_energy_parity(name: str, chunk_size: Optional[int]) -> None:
    g, box, kset = _setup(name)
    energy = make_stealthy_energy(box, kset, chunk_size=chunk_size)
    # floor from golden quantities only (all kernel weights are 1)
    floor = C_E * EPS * np.sqrt(g["params"]["N"] * int(g["m"]) * g["energy"])
    for got in _energies(energy, g):
        err = np.abs(got - g["energy"])
        assert np.all(err <= RTOL_KERNEL_E * np.abs(g["energy"]) + floor), err


@pytest.mark.parametrize("name", NAMES)
def test_force_parity(name: str) -> None:
    g, box, kset = _setup(name)
    energy = make_stealthy_energy(box, kset)
    for s, ref in zip(g["s"], g["forces"], strict=True):
        _, grad = energy.value_and_grad(jnp.asarray(s))
        forces = np.asarray(box.cartesian_forces(grad))
        np.testing.assert_allclose(
            forces, ref, rtol=RTOL_KERNEL_F, atol=1e-12 * float(np.max(np.abs(ref)))
        )


@pytest.mark.parametrize(
    "name, excluded",
    [("kernel_boundary_on_shell", True), ("kernel_boundary_above_shell", False)],
)
def test_boundary_kset_parity(name: str, excluded: bool) -> None:
    g, _, kset = _setup(name)
    p = g["params"]
    assert p["shell_excluded"] is excluded
    in_shell = np.sum(kset.n * kset.n, axis=1) == p["shell_n2"]
    assert bool(np.any(in_shell)) is (not excluded)
    assert _rows(kset.n) == _rows(g["n"])
    assert kset.n.shape[0] == int(g["m"])
