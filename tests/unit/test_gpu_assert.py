#!/usr/bin/env python3

"""
Unit tests for the cause classification in ``scripts/gpu_assert.py``.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import importlib.util
from pathlib import Path
from types import ModuleType

# ----- testing -----
import pytest

# ============================================================
# HELPERS
# ============================================================


def _load() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / "gpu_assert.py"
    spec = importlib.util.spec_from_file_location("gpu_assert", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ga = _load()

# ============================================================
# CLASSIFICATION
# ============================================================


def test_gpu_present_gives_no_cause() -> None:
    assert ga.classify_failure(["gpu"], {}, False) is None


@pytest.mark.parametrize("env", [{}, {"CUDA_VISIBLE_DEVICES": ""}])
def test_no_gres(env: dict[str, str]) -> None:
    cause = ga.classify_failure(["cpu"], env, True)
    assert cause is not None and "--gres=gpu" in cause


def test_no_plugin_wrong_env() -> None:
    cause = ga.classify_failure(["cpu"], {"CUDA_VISIBLE_DEVICES": "0"}, False)
    assert cause is not None and ".venv-cuda" in cause


def test_plugin_error_is_missing_libraries() -> None:
    err = RuntimeError("libcudnn.so not found")
    cause = ga.classify_failure([], {"CUDA_VISIBLE_DEVICES": "0"}, True, err)
    assert cause is not None
    assert "missing CUDA libraries" in cause and "libcudnn" in cause


def test_plugin_installed_detects_missing_module() -> None:
    assert ga.plugin_installed(("definitely_not_a_module_xyz",)) is False
    assert ga.plugin_installed(("json",)) is True


# ============================================================
# ASSERT_GPU_JAX / MAIN
# ============================================================


def test_assert_gpu_jax_raises_with_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ga, "probe_platforms", lambda: (["cpu"], None))
    monkeypatch.setattr(ga, "plugin_installed", lambda: True)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError, match="--gres=gpu"):
        ga.assert_gpu_jax()


def test_assert_gpu_jax_passes_with_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ga, "probe_platforms", lambda: (["gpu"], None))
    ga.assert_gpu_jax()


def test_main_exits_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(ga, "probe_platforms", lambda: (["cpu"], None))
    monkeypatch.setattr(ga, "plugin_installed", lambda: False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert ga.main() == 1
    assert ".venv-cuda" in capsys.readouterr().err
