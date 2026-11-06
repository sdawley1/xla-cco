#!/usr/bin/env python3

"""
Trajectory analysis: unwrapping and the configurational proximity metric.

Host-side NumPy (trajectories are large, usually memory-mapped). The
proximity metric follows Dawley & Torquato, PRE 112, 045310 (2025), Eq. (10):
p'_{t'}(t) = rho^{1/2} [sum_j |r_j(t) - r_j(t')|^2]^{1/2}, evaluated on
unwrapped (continuous) particle paths.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- numerics -----
import numpy as np

# ============================================================
# UNWRAPPING
# ============================================================


def unwrap_fractional(frames: np.ndarray) -> np.ndarray:
    """Return continuous fractional paths from wrapped frames.

    Consecutive frames are joined by the minimum-image increment, so every
    per-particle step between frames must be shorter than half a cell.

    Parameters
    ----------
    frames : np.ndarray
        (T, N, d) float64 fractional positions, wrapped into [0, 1).

    Returns
    -------
    np.ndarray
        (T, N, d) float64 unwrapped fractional positions with
        ``out[0] == frames[0]`` and ``out ≡ frames (mod 1)``.

    Raises
    ------
    TypeError
        If ``frames`` is not float64.
    ValueError
        If ``frames`` is not 3-D or is empty.
    """
    if frames.dtype != np.float64:
        raise TypeError(f"frames must be float64, got {frames.dtype}")
    if frames.ndim != 3 or frames.shape[0] == 0:
        raise ValueError(f"frames must have shape (T>0, N, d), got {frames.shape}")
    out = np.empty(frames.shape, dtype=np.float64)
    out[0] = frames[0]
    if frames.shape[0] > 1:
        step = np.diff(np.asarray(frames), axis=0)
        step -= np.round(step)
        np.cumsum(step, axis=0, out=out[1:])
        out[1:] += frames[0]
    return out


# ============================================================
# PROXIMITY METRIC
# ============================================================


def proximity_metric(
    paths: np.ndarray, reference: np.ndarray, basis: np.ndarray, density: float
) -> np.ndarray:
    """Return p(t) = rho^{1/2} sqrt(sum_j |r_j(t) - r_j(ref)|^2) per frame.

    Parameters
    ----------
    paths : np.ndarray
        (T, N, d) float64 UNWRAPPED fractional positions (see
        ``unwrap_fractional``).
    reference : np.ndarray
        (N, d) float64 unwrapped fractional reference configuration, e.g.
        ``paths[0]`` (initial condition) or ``paths[-1]``.
    basis : np.ndarray
        (d, d) float64 Cartesian box basis, rows = lattice vectors.
    density : float
        Number density rho (> 0).

    Returns
    -------
    np.ndarray
        (T,) float64 proximity metric in units of rho^{-1/2}.

    Raises
    ------
    ValueError
        On inconsistent shapes or non-positive density.
    """
    if paths.ndim != 3 or reference.shape != paths.shape[1:]:
        raise ValueError(f"shapes {paths.shape} and {reference.shape} mismatch")
    if basis.shape != (paths.shape[2], paths.shape[2]) or not density > 0.0:
        raise ValueError("basis must be (d, d) and density must be > 0")
    disp = (np.asarray(paths) - reference) @ basis
    out: np.ndarray = np.sqrt(density * np.sum(disp * disp, axis=(1, 2)))
    return out
