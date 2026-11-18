#!/usr/bin/env python3

"""
Shared pytest configuration: Hypothesis profiles, x64 check, RNG policy, and
the priority tiers (tests/tiers.py: --priority, one tier marker per test).

RNG policy: every random test input uses an explicit seed through
``np.random.default_rng(seed)``; no global seeding.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from typing import Callable, Iterator

# ----- numerics -----
import jax
import numpy as np
import pytest
from hypothesis import HealthCheck, settings

pytest_plugins = ["tiers", "pytester"]  # tiers: tests/tiers.py (on pythonpath)

# ============================================================
# HYPOTHESIS PROFILES
# ============================================================

# deadline=None everywhere: wall-clock deadlines flake on loaded nodes (D-007)
settings.register_profile("dev", max_examples=100, deadline=None)
settings.register_profile(
    "ci", max_examples=300, deadline=None, derandomize=True, print_blob=True
)
settings.register_profile(
    "thorough",
    max_examples=5000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("dev")

# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture(scope="session", autouse=True)
def _require_x64() -> None:
    """Fail the session if float64 is not enabled."""
    import xla_cco  # noqa: F401  (import enables x64)

    assert jax.config.jax_enable_x64, "jax_enable_x64 must be on"


@pytest.fixture(scope="module", autouse=True)
def _clear_jax_caches() -> Iterator[None]:
    """Drop compiled XLA programs after each test module.

    Every CPU-JIT program stays memory-mapped until it is released, and the
    full suite exceeded vm.max_map_count (65530) once the MD tests landed
    (``Failed to materialize symbols``; AUDIT-T2 F1). Clearing per module
    keeps the peak near 1e4 maps.
    """
    assert jax.config.jax_enable_x64, "x64 stays on across modules"
    yield
    jax.clear_caches()


@pytest.fixture(autouse=True)
def _restore_default_device() -> Iterator[None]:
    """Restore jax_default_device after each test.

    The CLIs' select_device sets it process-wide (D-085); without the reset
    one in-process CLI test would move the rest of the GPU suite to the CPU.
    """
    saved = jax.config.jax_default_device
    assert saved is None or hasattr(saved, "platform"), "a device or None"
    yield
    jax.config.update("jax_default_device", saved)
    assert jax.config.jax_default_device == saved, "default device restored"


@pytest.fixture
def poisson_configs() -> Callable[[int, int, int, int], np.ndarray]:
    """Return a factory ``(seed, b, n, d) -> (b, n, d)`` uniform fractional."""

    def make(seed: int, b: int, n: int, d: int) -> np.ndarray:
        assert min(b, n, d) >= 1, "b, n, d must be positive"
        return np.random.default_rng(seed).uniform(size=(b, n, d))

    return make
