#!/usr/bin/env python3

"""
Shared pytest configuration: Hypothesis profiles, x64 check, RNG policy.

RNG policy: every random test input uses an explicit seed through
``np.random.default_rng(seed)``; no global seeding.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from typing import Callable

# ----- numerics -----
import jax
import numpy as np
import pytest
from hypothesis import HealthCheck, settings

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


@pytest.fixture
def poisson_configs() -> Callable[[int, int, int, int], np.ndarray]:
    """Return a factory ``(seed, b, n, d) -> (b, n, d)`` uniform fractional."""

    def make(seed: int, b: int, n: int, d: int) -> np.ndarray:
        assert min(b, n, d) >= 1, "b, n, d must be positive"
        return np.random.default_rng(seed).uniform(size=(b, n, d))

    return make
