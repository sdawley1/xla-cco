#!/usr/bin/env python3

"""
Unit tests of the JAX-ensemble cache key (``ensemble_run.cache_key``).

The key must be stable (tuple vs JSON list), and must change with the source
hash, the relaxation settings and chi.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import json

# ----- numerics -----
import pytest

# ----- project -----
import ensemble_run as run
from xla_cco.minimize.lbfgs import LBFGSSettings

# ============================================================
# TESTS
# ============================================================


def test_key_stable_under_json_roundtrip() -> None:
    """Key equals its own JSON round trip (tuples become lists on disk)."""
    key = run.cache_key(0.25, 0, 1.0, 400)
    assert json.loads(json.dumps(key)) == key
    assert isinstance(key["settings"]["rescales"], list)
    assert run.cache_key(0.25, 0, 1.0, 400) == key


def test_key_changes_with_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """A different source hash gives a different key."""
    base = run.cache_key(0.25, 0, 1.0, 400)
    monkeypatch.setattr(run, "src_hash", lambda: "0" * 64)
    assert run.cache_key(0.25, 0, 1.0, 400) != base


def test_key_changes_with_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Changing a minimizer setting changes the key."""
    base = run.cache_key(0.25, 0, 1.0, 400)
    monkeypatch.setattr(
        "xla_cco.minimize.lbfgs.LBFGSSettings",
        lambda: LBFGSSettings(memory=50),
    )
    assert run.cache_key(0.25, 0, 1.0, 400) != base


@pytest.mark.parametrize(
    "args", [(0.35, 0, 1.0, 400), (0.25, 1, 1.0, 400), (0.25, 0, 0.9, 400)]
)
def test_key_changes_with_chi_task_scale(args: tuple) -> None:  # type: ignore[type-arg]
    """chi, task and k-scale each change the key."""
    assert run.cache_key(*args) != run.cache_key(0.25, 0, 1.0, 400)
    assert run.cache_key(0.25, 0, 1.0, 200) != run.cache_key(0.25, 0, 1.0, 400)
