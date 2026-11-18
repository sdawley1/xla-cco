#!/usr/bin/env python3

"""
Fail-fast GPU guard shared by ``scripts/gpu_assert.py`` and the CLI.

``require_gpu`` raises ``RuntimeError`` with the most likely cause when JAX
cannot see a GPU, so a billed allocation never silently falls back to the CPU:

- no GPU and ``CUDA_VISIBLE_DEVICES`` unset or empty: probably no ``--gres=gpu``;
- jaxlib has no CUDA plugin: wrong environment, use ``.venv-cuda``;
- plugin import or initialization error: missing CUDA libraries or driver.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import importlib.util
import os
from typing import Mapping, Optional, Sequence

# ============================================================
# CONSTANTS
# ============================================================

PLUGIN_MODULES = (
    "jax_plugins.xla_cuda12",
    "jax_plugins.xla_cuda13",
    "jax_cuda12_plugin",
    "jax_cuda13_plugin",
)

# ============================================================
# CLASSIFICATION
# ============================================================


def plugin_installed(modules: Sequence[str] = PLUGIN_MODULES) -> bool:
    """Return True if any JAX CUDA plugin module is importable.

    Parameters
    ----------
    modules : Sequence[str]
        Candidate plugin module names.

    Returns
    -------
    bool
        Whether at least one module can be found.
    """
    assert len(modules) > 0, "need at least one module name"
    assert all(isinstance(m, str) for m in modules), "module names must be str"
    for name in modules:
        try:
            if importlib.util.find_spec(name) is not None:
                return True
        except (ImportError, ValueError):
            continue
    return False


def classify_failure(
    platforms: Sequence[str],
    env: Mapping[str, str],
    has_plugin: bool,
    error: Optional[BaseException] = None,
) -> Optional[str]:
    """Return the likely cause of a missing GPU, or None if a GPU is visible.

    Parameters
    ----------
    platforms : Sequence[str]
        Platform names of ``jax.devices()`` (empty if enumeration failed).
    env : Mapping[str, str]
        Process environment.
    has_plugin : bool
        Whether a JAX CUDA plugin is installed.
    error : BaseException, optional
        Exception raised while enumerating devices, if any.

    Returns
    -------
    str or None
        Human-readable cause, or None when a GPU is present.
    """
    assert isinstance(platforms, (list, tuple)), "platforms must be a sequence"
    assert isinstance(has_plugin, bool), "has_plugin must be bool"
    if any(p in ("gpu", "cuda") for p in platforms):
        return None
    requested = env.get("JAX_PLATFORMS", "").strip().lower()
    if requested and not any(
        p in ("gpu", "cuda") for p in requested.replace(",", " ").split()
    ):
        return (
            f"JAX_PLATFORMS={env['JAX_PLATFORMS']} hides any GPU: unset it, "
            "or pass --device cpu to run on the CPU."
        )
    if not env.get("CUDA_VISIBLE_DEVICES", ""):
        return (
            "no GPU visible and CUDA_VISIBLE_DEVICES is unset or empty: "
            "probably no --gres=gpu in the job request."
        )
    if not has_plugin:
        return (
            "jaxlib has no CUDA plugin installed: wrong environment; "
            "use the CUDA environment, created on a login node with "
            "UV_PROJECT_ENVIRONMENT=.venv-cuda uv sync --dev --extra cuda --locked "
            "and selected with UV_PROJECT_ENVIRONMENT=.venv-cuda."
        )
    detail = f" ({type(error).__name__}: {error})" if error is not None else ""
    return (
        "CUDA plugin installed but failed to initialize (missing CUDA "
        f"libraries, driver, or device); see the JAX error above{detail}."
    )


def probe_platforms() -> tuple[list[str], Optional[BaseException]]:
    """Enumerate JAX device platforms, capturing any initialization error.

    Returns
    -------
    tuple[list[str], BaseException or None]
        Platform names and the exception raised, if any.
    """
    try:
        import jax

        return [d.platform for d in jax.devices()], None
    except Exception as exc:  # backend init can raise many types
        return [], exc


def require_gpu() -> None:
    """Raise RuntimeError naming the likely cause unless a GPU is visible."""
    platforms, error = probe_platforms()
    cause = classify_failure(platforms, os.environ, plugin_installed(), error)
    if cause is not None:
        raise RuntimeError(cause)
