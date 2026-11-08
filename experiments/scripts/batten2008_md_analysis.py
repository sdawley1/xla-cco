#!/usr/bin/env python3

"""
Per-file analysis of the low-temperature MD winding runs (plan E2).

Reads one or more ``md*_chi<tag>.npz`` (+ ``.json``) written by
``python -m xla_cco.cli.md`` (lanes = ground state x temperature, ground
state major; snapshots log-spaced in every production segment, with the
winding in radians accumulated since the segment began; per-cycle sums) and
writes ``md_analysis<rest>_chi<tag>.npz`` beside each (``md_chi1-10`` ->
``md_analysis_chi1-10``, ``md_langevin_g0.03_chi1-10`` ->
``md_analysis_langevin_g0.03_chi1-10``). Per ground state (the statistical
unit) and temperature it stores:

- unsigned and net winding rates of every tracked wave in turns per unit
  time: the least-squares slope of W(t) / 2 pi over each production
  segment, and the windowed rates over [t/4, t/2] and [t/2, t] (t the
  segment length), averaged over segments; 20 |k|/K bins over (0, 1] for
  the constrained waves and 20 over (1, 2] for the unconstrained ones;
- Rice predictions with <S_k> = sum_s / n_sum: |k| sqrt(T / S) / 2 pi
  (constrained) and |k| sqrt(T / (pi S)) / 2 pi (unconstrained), and the
  conditional Rice rate |k| sqrt(N T / pi) <1/|rho_k|>_t / 2 pi (H6b,
  Amendment A3; <1/|rho|>_t over the log-spaced production snapshots);
- <Phi> / (M T), <S_k> / T and <KE> / (d (N - 1) T / 2) from the sums;
- the net-winding MSD <W_net(t)^2> (turns^2, mean over waves, t since the
  segment began), its exponent alpha (log-log least squares over t >= 10
  times the median inverse unsigned rate) and D_theta (MSD = 2 D t there);
- the particle MSD from the snapshot positions (COM removed, reference the
  segment's first snapshot);
- optionally the spectrum of K = (2/V) J J^T of every ground state (from
  ``flows_chi<tag>.npz`` s_final via batten2008_landscape.wave_matrix) and
  its moments, and the power spectrum of the dense rho trace if present.

NumPy only at import (batten2008_landscape, which imports JAX, is loaded
only for the K spectra), so the plot environment can import the constants.

Usage: python -I batten2008_md_analysis.py --md A/md_chi1-10.npz [...]
       [--flows auto|none|PATH] [--out-dir DIR]

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ----- numerics -----
import numpy as np

# ============================================================
# CONSTANTS
# ============================================================

TWO_PI = 2.0 * np.pi
N_BINS = 20  # |k|/K bins per wave class
BIN_EDGES_C = np.linspace(0.0, 1.0, N_BINS + 1)  # constrained: (0, 1]
BIN_EDGES_U = np.linspace(1.0, 2.0, N_BINS + 1)  # unconstrained: (1, 2]
WINDOW_FACTOR = 10.0  # MSD fit: lags >= this x median inverse unsigned rate
MIN_FIT_POINTS = 3
STATUS_RUNNING = 0  # minimize.result.STATUS_RUNNING (no JAX import here)
MOMENT_NAMES: Tuple[str, ...] = (
    "k_mean", "k_cv", "k_min_rel", "k_q10_rel", "k_harm_rel",
)  # fmt: skip
COLLAPSE_GRID = 40  # points of the common t sqrt(T) grid
REQUIRED_KEYS: Tuple[str, ...] = (
    "snap_step", "snap_wind_abs", "snap_wind_net", "snap_rho", "snap_r", "snap_phi",
    "snap_count", "cycle_sum_s", "cycle_n_sum", "cycle_sum_phi",
    "cycle_sum_kinetic", "status", "lane_index", "lane_temperature",
    "temperatures", "indices", "waves_kvec", "waves_constrained",
    "record_steps", "basis",
)  # fmt: skip

# ============================================================
# WINDING AND RATES
# ============================================================


def winding_step(
    new: np.ndarray, old: np.ndarray, n: int, floor: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Phase increment (radians) of complex rho old -> new, and the valid mask.

    Delta = atan2(Im(new conj old), Re(new conj old)); counted only where
    S = |rho|^2 / n exceeds ``floor`` at both ends (the MD rule).
    """
    assert new.shape == old.shape and n >= 1, "matching rho arrays"
    assert floor >= 0.0, "phase floor must be >= 0"
    z = new * np.conj(old)
    ok = (np.abs(new) ** 2 / n > floor) & (np.abs(old) ** 2 / n > floor)
    return np.where(ok, np.arctan2(z.imag, z.real), 0.0), ok


def cumulative_turns(
    rho: np.ndarray, n: int, floor: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Unsigned and net winding (turns) up to each frame of rho (F, W) complex."""
    assert rho.ndim == 2 and rho.shape[0] >= 1, "need frames x waves"
    dth, _ = winding_step(rho[1:], rho[:-1], n, floor)
    zero = np.zeros((1, rho.shape[1]))
    w_abs = np.concatenate([zero, np.cumsum(np.abs(dth), axis=0)]) / TWO_PI
    w_net = np.concatenate([zero, np.cumsum(dth, axis=0)]) / TWO_PI
    assert w_abs.shape == rho.shape and np.all(w_abs >= -1e-15), "turns"
    return w_abs, w_net


def ls_slope(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares slope of y (..., R, W) against t (R,) (with intercept)."""
    assert t.ndim == 1 and t.size >= 2, "need at least two times"
    assert y.shape[-2] == t.size, "y must be (..., R, W)"
    tc = t - t.mean()
    yc = y - y.mean(axis=-2, keepdims=True)
    out: np.ndarray = np.einsum("r,...rw->...w", tc, yc) / float(tc @ tc)
    return out


def interp_axis(xq: np.ndarray, x: np.ndarray, y: np.ndarray, axis: int) -> np.ndarray:
    """Linear interpolation of y along ``axis`` from sorted x (R,) to xq (Q,).

    Outside [x[0], x[-1]] the end values are used (no extrapolation).
    """
    assert x.ndim == 1 and np.all(np.diff(x) > 0), "x must increase"
    assert y.shape[axis] == x.size, "y and x disagree"
    xq = np.clip(np.atleast_1d(np.asarray(xq, dtype=np.float64)), x[0], x[-1])
    i = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, x.size - 2)
    a = (xq - x[i]) / (x[i + 1] - x[i])
    ym = np.moveaxis(y, axis, -1)
    out = ym[..., i] * (1.0 - a) + ym[..., i + 1] * a
    return np.moveaxis(out, -1, axis)


def segment_rates(
    t: np.ndarray, w_abs: np.ndarray, w_net: np.ndarray, t_end: float
) -> Dict[str, np.ndarray]:
    """Rates (turns / time) of one segment from W (..., R, W) in turns.

    ``abs_ls`` / ``net_ls`` least-squares slopes over the snapshots;
    ``abs_w1`` over [t_end/4, t_end/2] and ``abs_w2`` over [t_end/2, t_end]
    (W(0) = 0 at the segment start, linear interpolation between snapshots);
    ``net_end`` = W_net(t_end) / t_end.
    """
    assert t.ndim == 1 and t[0] > 0.0 and np.isclose(t[-1], t_end), "segment times"
    assert w_abs.shape == w_net.shape, "abs and net windings must match"
    x0 = np.concatenate([[0.0], t])
    pad = np.zeros((*w_abs.shape[:-2], 1, w_abs.shape[-1]))
    wa = np.concatenate([pad, w_abs], axis=-2)
    q = interp_axis(np.array([0.25, 0.5, 1.0]) * t_end, x0, wa, axis=-2)
    return {
        "abs_ls": ls_slope(t, w_abs),
        "net_ls": ls_slope(t, w_net),
        "abs_w1": (q[..., 1, :] - q[..., 0, :]) / (0.25 * t_end),
        "abs_w2": (q[..., 2, :] - q[..., 1, :]) / (0.5 * t_end),
        "net_end": w_net[..., -1, :] / t_end,
    }


def conditional_rice(
    kmag: np.ndarray, n: int, temp: np.ndarray, inv_rho: np.ndarray
) -> np.ndarray:
    """Conditional Rice rate (turns / time): |k| sqrt(N T / pi) <1/|rho|>_t / 2 pi.

    Given rho, theta' is Gaussian with variance N |k|^2 T / (2 |rho|^2), so
    E|theta'| = |k| sqrt(N T / pi) E[1/|rho|] for any slowly varying
    amplitude (Amendment A3, H6b). ``temp`` broadcasts against inv_rho (..., W).
    """
    assert kmag.shape == inv_rho.shape[-1:] and n >= 1, "per-wave arrays"
    assert np.all(np.asarray(temp) > 0.0), "temperatures must be positive"
    out: np.ndarray = kmag * np.sqrt(n * temp / np.pi) * inv_rho / TWO_PI
    return out


def mean_inverse_modulus(rho: np.ndarray) -> np.ndarray:
    """<1/|rho|> over the snapshot axis -2 of rho (..., R, W, 2); NaN if any is 0."""
    assert rho.shape[-1] == 2 and rho.ndim >= 3, "rho (..., R, W, 2)"
    mod = np.hypot(rho[..., 0], rho[..., 1])
    inv = np.where(mod > 0.0, 1.0 / np.where(mod > 0.0, mod, 1.0), np.nan)
    out: np.ndarray = np.mean(inv, axis=-2)
    assert out.shape == rho.shape[:-3] + rho.shape[-2:-1], "one value per wave"
    return out


def rice_prediction(
    kmag: np.ndarray, temp: np.ndarray, s_mean: np.ndarray, constrained: np.ndarray
) -> np.ndarray:
    """Rice unsigned rate (turns / time) per wave.

    Constrained: |k| sqrt(T / S) / 2 pi; unconstrained: |k| sqrt(T / (pi S))
    / 2 pi. ``temp`` broadcasts against s_mean (..., W) (e.g. (..., 1)).
    """
    assert kmag.shape == constrained.shape == s_mean.shape[-1:], "per-wave arrays"
    assert np.all(np.asarray(temp) > 0.0), "temperatures must be positive"
    s = np.where(s_mean > 0.0, s_mean, np.nan)
    con = kmag * np.sqrt(temp / s) / TWO_PI
    out: np.ndarray = np.where(constrained, con, con / np.sqrt(np.pi))
    return out


# ============================================================
# BINS, MSD AND SPECTRA
# ============================================================


def bin_ids(kmag: np.ndarray, k_outer: float, constrained: np.ndarray) -> np.ndarray:
    """Bin 0..19 over (0, 1] of |k|/K (constrained), 20..39 over (1, 2] (others)."""
    assert kmag.shape == constrained.shape and k_outer > 0.0, "per-wave arrays"
    x = kmag / k_outer
    ic = np.searchsorted(BIN_EDGES_C, x, side="left") - 1
    iu = np.searchsorted(BIN_EDGES_U, x, side="left") - 1 + N_BINS
    out = np.where(constrained, np.clip(ic, 0, N_BINS - 1),
                   np.clip(iu, N_BINS, 2 * N_BINS - 1))  # fmt: skip
    assert np.all((out >= 0) & (out < 2 * N_BINS)), "bin out of range"
    return out


def bin_mean(values: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Mean over the waves of each of the 2 N_BINS bins: (..., W) -> (..., 40)."""
    assert values.shape[-1] == ids.size, "one bin id per wave"
    assert ids.min() >= 0 and ids.max() < 2 * N_BINS, "bin ids out of range"
    out = np.full((*values.shape[:-1], 2 * N_BINS), np.nan)
    for b in range(2 * N_BINS):
        cols = ids == b
        if np.any(cols):
            out[..., b] = np.mean(values[..., cols], axis=-1)
    return out


def window_start(rate_abs: np.ndarray) -> float:
    """MSD fit start: WINDOW_FACTOR x the median inverse unsigned rate (time)."""
    assert rate_abs.ndim == 1, "one rate per wave"
    ok = np.isfinite(rate_abs) & (rate_abs > 0.0)
    if not np.any(ok):
        return float("nan")
    out = WINDOW_FACTOR * float(np.median(1.0 / rate_abs[ok]))
    assert out > 0.0, "positive window start"
    return out


def fit_msd(t: np.ndarray, msd: np.ndarray, t_min: float) -> Tuple[float, float]:
    """(alpha, D) of MSD(t) for t >= t_min: log-log slope; MSD = 2 D t (LSQ).

    NaN when fewer than MIN_FIT_POINTS positive points lie in the window.
    """
    assert t.shape == msd.shape and t.ndim == 1, "one MSD value per time"
    assert not t_min < 0.0, "t_min must be >= 0 (or NaN)"
    sel = (t >= t_min) & (t > 0.0) & np.isfinite(msd) & (msd > 0.0)
    if not np.isfinite(t_min) or int(sel.sum()) < MIN_FIT_POINTS:
        return float("nan"), float("nan")
    alpha = float(np.polyfit(np.log(t[sel]), np.log(msd[sel]), 1)[0])
    d_theta = float(np.sum(t[sel] * msd[sel]) / (2.0 * np.sum(t[sel] ** 2)))
    return alpha, d_theta


def particle_msd(r: np.ndarray) -> np.ndarray:
    """MSD (..., R) of positions r (..., R, n, d) from frame 0, COM removed."""
    assert r.ndim >= 3 and r.shape[-3] >= 1, "need (..., R, n, d)"
    dr = r - r[..., :1, :, :]
    dr = dr - dr.mean(axis=-2, keepdims=True)
    out: np.ndarray = np.mean(np.sum(dr**2, axis=-1), axis=-1)
    assert out.shape == r.shape[:-2], "one MSD per frame"
    return out


def collapse_metric(lag: np.ndarray, msd: np.ndarray, temps: np.ndarray) -> float:
    """Largest |MSD_T(x) / median_T MSD(x) - 1| on the common range of x = t sqrt(T).

    ``msd`` (T, R) on lags (R,); log-log interpolation onto COLLAPSE_GRID
    points; NaN when the ranges do not overlap.
    """
    assert msd.shape == (temps.size, lag.size) and temps.size >= 2, "shapes"
    assert np.all(temps > 0.0), "temperatures must be positive"
    ok = lag > 0.0
    x = lag[ok][None, :] * np.sqrt(temps)[:, None]
    lo, hi = float(np.max(x[:, 0])), float(np.min(x[:, -1]))
    if not hi > lo or np.any(~(msd[:, ok] > 0.0)):
        return float("nan")
    grid = np.geomspace(lo, hi, COLLAPSE_GRID)
    logm = np.stack([np.interp(np.log(grid), np.log(x[i]), np.log(msd[i, ok]))
                     for i in range(temps.size)])  # fmt: skip
    ratio = np.exp(logm - np.median(logm, axis=0))
    return float(np.max(np.abs(ratio - 1.0)))


def spectrum_moments(k_eig: np.ndarray) -> np.ndarray:
    """MOMENT_NAMES of the K spectrum: mean, CV, min, 10 % quantile, harmonic mean."""
    assert k_eig.ndim == 1 and k_eig.size >= 2, "need a spectrum"
    lam = np.sort(k_eig)
    mean = float(lam.mean())
    assert mean > 0.0, "K is positive semi-definite and non-zero"
    pos = lam[lam > 0.0]
    harm = float(1.0 / np.mean(1.0 / pos)) if pos.size else 0.0
    return np.array([mean, float(lam.std()) / mean, lam[0] / mean,
                     float(np.quantile(lam, 0.1)) / mean, harm / mean])  # fmt: skip


def ground_state_spectra(
    flows: Path, indices: np.ndarray, kvec_c: np.ndarray, basis: np.ndarray
) -> np.ndarray:
    """Eigenvalues (G, 2M) of K = (2/V) J J^T at s_final[indices] of ``flows``."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from batten2008_landscape import wave_matrix  # JAX via its imports: lazy

    with np.load(flows) as z:
        s_final = np.asarray(z["s_final"])
    assert indices.max() < s_final.shape[0], "ground-state index out of range"
    volume = float(abs(np.linalg.det(basis)))
    out = np.empty((indices.size, 2 * kvec_c.shape[0]))
    for i, g in enumerate(indices):
        out[i] = np.linalg.eigvalsh(
            wave_matrix(s_final[g] @ basis, kvec_c, volume)["k"]
        )
    assert np.all(out > -1e-9 * np.abs(out).max()), "K must be PSD"
    return out


def power_spectrum(rho: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """(omega, mean over waves of the normalized power) of rho (Q, W) complex.

    Each wave's |FFT(rho - mean)|^2 is normalized to unit sum, so all waves
    weigh the same; omega = 2 pi f, two-sided, sorted.
    """
    assert np.all(np.isfinite(rho)), "non-finite rho trace"
    assert rho.ndim == 2 and rho.shape[0] >= 8 and dt > 0.0, "need a trace"
    x = rho - rho.mean(axis=0, keepdims=True)
    p = np.abs(np.fft.fft(x, axis=0)) ** 2
    p = p / np.maximum(p.sum(axis=0, keepdims=True), 1e-300)
    omega = TWO_PI * np.fft.fftfreq(rho.shape[0], d=dt)
    order = np.argsort(omega)
    return omega[order], p.mean(axis=1)[order]


# ============================================================
# LOADING AND LAYOUT
# ============================================================


def load_md(path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Arrays of an MD output npz and its JSON metadata (the CLI schema).

    Raises
    ------
    ValueError
        If a required array or the JSON 'phases' / 'settings' is missing.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: np.asarray(z[k]) for k in z.files}
    meta = json.loads(path.with_suffix(".json").read_text())
    missing = [k for k in REQUIRED_KEYS if k not in arrays]
    if missing:
        raise ValueError(f"{path} lacks arrays {missing}")
    for key in ("phases", "settings", "M", "K", "N", "d"):
        if key not in meta:
            raise ValueError(f"{path.with_suffix('.json')} lacks '{key}'")
    assert arrays["snap_wind_abs"].ndim == 3, "snap_wind_abs is (L, R, W)"
    assert arrays["waves_kvec"].shape[0] == arrays["snap_wind_abs"].shape[2], "waves"
    return arrays, meta


def lane_layout(arrays: Dict[str, np.ndarray]) -> Tuple[int, int]:
    """(G, n_T) of the ground-state-major lanes; ValueError if the labels disagree."""
    assert arrays["lane_index"].shape == arrays["lane_temperature"].shape, "labels"
    temps, idx = arrays["temperatures"], arrays["indices"]
    g, n_t = int(idx.size), int(temps.size)
    assert g >= 1 and n_t >= 1, "need ground states and temperatures"
    if not (
        np.array_equal(arrays["lane_temperature"], np.tile(temps, g))
        and np.array_equal(arrays["lane_index"], np.repeat(idx, n_t))
    ):
        raise ValueError("lanes are not (ground state x temperature), state major")
    return g, n_t


def to_grid(x: np.ndarray, g: int, n_t: int) -> np.ndarray:
    """Lane axis 0 (L = g n_t, ground state major) -> axes (g, n_t)."""
    assert x.shape[0] == g * n_t, "leading axis must be the lanes"
    out = x.reshape(g, n_t, *x.shape[1:])
    assert out.shape[:2] == (g, n_t), "grid shape"
    return out


def segments(meta: Dict[str, Any]) -> List[Tuple[int, int]]:
    """(start step, length) of every production segment (cycle >= 0), in order."""
    phases = meta["phases"]
    assert len(phases) >= 1, "no phases"
    starts = np.cumsum([0] + [int(ph["n_steps"]) for ph in phases])[:-1]
    prod = sorted((int(ph["cycle"]), int(s0), int(ph["n_steps"]))
                  for s0, ph in zip(starts, phases, strict=True)
                  if int(ph["cycle"]) >= 0)  # fmt: skip
    assert [c for c, _, _ in prod] == list(range(len(prod))), "cycles 0..C-1"
    return [(s0, n) for _, s0, n in prod]


def output_name(path: Path) -> str:
    """md_chi1-10.npz -> md_analysis_chi1-10.npz (md_<x>_chi.. -> md_analysis_<x>..)."""
    assert str(path), "empty path"
    stem = Path(path).stem
    if not stem.startswith("md_") or "_chi" not in stem:
        raise ValueError(f"{path}: expected md[_<kind>]_chi<tag>.npz")
    out = "md_analysis_" + stem[3:] + ".npz"
    assert out.endswith(".npz") and "_chi" in out, "analysis file name"
    return out


# ============================================================
# ANALYSIS
# ============================================================


def lane_ok(arrays: Dict[str, np.ndarray]) -> np.ndarray:
    """Lanes usable for statistics: STATUS_RUNNING and every snapshot taken."""
    cap = arrays["record_steps"].size
    ok = (arrays["status"] == STATUS_RUNNING) & (arrays["snap_count"] == cap)
    assert ok.shape == arrays["status"].shape, "one flag per lane"
    assert arrays["snap_step"].shape[1] >= cap, "snapshot capacity"
    return ok


def _segment(
    arrays: Dict[str, np.ndarray], seg: Tuple[int, int], dt: float, ids: np.ndarray
) -> Dict[str, np.ndarray]:
    """Rates and MSDs of every lane in one production segment."""
    s0, n = seg
    rec = arrays["record_steps"]
    idx = np.flatnonzero((rec > s0) & (rec <= s0 + n))
    assert idx.size >= 2 and rec[idx[-1]] == s0 + n, "snapshots must end the segment"
    t = (rec[idx] - s0) * dt
    w_abs = arrays["snap_wind_abs"][:, idx, :] / TWO_PI
    w_net = arrays["snap_wind_net"][:, idx, :] / TWO_PI
    out = segment_rates(t, w_abs, w_net, n * dt)
    con = arrays["waves_constrained"].astype(bool)
    out["t"], out["lag"], out["idx"] = t, t - t[0], idx
    out["msd_c"] = np.mean(w_net[..., con] ** 2, axis=-1)
    out["msd_u"] = np.mean(w_net[..., ~con] ** 2, axis=-1) if np.any(~con) else (
        np.full(out["msd_c"].shape, np.nan))  # fmt: skip
    out["msd_bin"] = bin_mean(w_net**2, ids)
    out["part_msd"] = particle_msd(arrays["snap_r"][:, idx])
    out["kymo"] = w_abs
    assert out["msd_c"].shape == (w_net.shape[0], idx.size), "MSD per lane, frame"
    return out


def _combine(parts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Average segments: rates directly, MSDs on segment 0's times (interpolated)."""
    assert len(parts) >= 1, "need a segment"
    t0, lag0 = parts[0]["t"], parts[0]["lag"]
    out = {k: np.stack([p[k] for p in parts])
           for k in ("abs_ls", "net_ls", "abs_w1", "abs_w2", "net_end")}  # fmt: skip
    for key, grid, tk in (
        ("msd_c", t0, "t"),
        ("msd_u", t0, "t"),
        ("msd_bin", t0, "t"),
        ("part_msd", lag0, "lag"),
    ):
        acc = [
            interp_axis(grid, p[tk], p[key], axis=1)
            if p[tk].size > 1 and not np.array_equal(p[tk], grid)
            else p[key]
            for p in parts
        ]
        out[key] = np.mean(np.stack(acc), axis=0)
    out["t0"], out["lag0"] = t0, lag0
    assert out["abs_ls"].shape[0] == len(parts), "segment axis first"
    return out


def _sums(arrays: Dict[str, np.ndarray], meta: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """<S_k>, <Phi>/(M T) per segment and overall, <KE>/(d (N-1) T / 2) per lane."""
    n_sum = arrays["cycle_n_sum"].astype(np.float64)  # (C, L)
    assert np.all(n_sum > 0), "every segment has samples"
    temp = arrays["lane_temperature"]
    m, n, d = int(meta["M"]), int(meta["N"]), int(meta["d"])
    s_mean = arrays["cycle_sum_s"].sum(axis=0) / n_sum.sum(axis=0)[:, None]
    phi_seg = arrays["cycle_sum_phi"] / n_sum / (m * temp)
    phi = arrays["cycle_sum_phi"].sum(axis=0) / n_sum.sum(axis=0) / (m * temp)
    ke = arrays["cycle_sum_kinetic"].sum(axis=0) / n_sum.sum(axis=0)
    out = {"s_mean": s_mean, "phi_ratio": phi, "phi_ratio_seg": phi_seg.T,
           "ke_ratio": ke / (d * (n - 1) * temp / 2.0)}  # fmt: skip
    assert out["s_mean"].shape == arrays["cycle_sum_s"].shape[1:], "S per lane, wave"
    return out


def _fits(
    comb: Dict[str, np.ndarray], rate_abs: np.ndarray, con: np.ndarray
) -> Dict[str, np.ndarray]:
    """Per lane: MSD window start, alpha and D_theta (pooled, and per bin)."""
    lanes = rate_abs.shape[0]
    assert comb["msd_c"].shape[0] == lanes, "one MSD per lane"
    out = {k: np.full(lanes, np.nan) for k in ("t_min", "alpha", "d_theta")}
    out["d_theta_bin"] = np.full((lanes, 2 * N_BINS), np.nan)
    t = comb["t0"]
    for lane in range(lanes):
        t_min = window_start(rate_abs[lane, con])
        out["t_min"][lane] = t_min
        alpha, d_theta = fit_msd(t, comb["msd_c"][lane], t_min)
        out["alpha"][lane], out["d_theta"][lane] = alpha, d_theta
        sel = t >= t_min
        if np.isfinite(t_min) and int(sel.sum()) >= MIN_FIT_POINTS:
            m = comb["msd_bin"][lane][sel]  # (R_sel, 40)
            out["d_theta_bin"][lane] = (t[sel] @ m) / (2.0 * float(t[sel] @ t[sel]))
    assert out["alpha"].shape == (lanes,), "one alpha per lane"
    return out


def _production(arrays: Dict[str, np.ndarray], meta: Dict[str, Any]) -> np.ndarray:
    """Snapshot indices of every production segment, in order."""
    rec = arrays["record_steps"]
    assert rec.ndim == 1 and rec.size >= 2, "record steps"
    idx = [np.flatnonzero((rec > a) & (rec <= a + b)) for a, b in segments(meta)]
    out = np.concatenate(idx)
    assert out.size >= 2 and np.all(np.diff(out) > 0), "production snapshots"
    return out


def lane_quantities(
    arrays: Dict[str, np.ndarray], meta: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """Every per-lane (and per-segment) quantity, lane axis first, bad lanes NaN."""
    assert "phases" in meta and "settings" in meta, "CLI metadata"
    dt = float(meta["settings"]["dt"])
    con = arrays["waves_constrained"].astype(bool)
    kmag = np.linalg.norm(arrays["waves_kvec"], axis=1)
    ids = bin_ids(kmag, float(meta["K"]), con)
    comb = _combine([_segment(arrays, seg, dt, ids) for seg in segments(meta)])
    out: Dict[str, np.ndarray] = _sums(arrays, meta)
    temp = arrays["lane_temperature"][:, None]
    out["rate_abs"] = comb["abs_ls"].mean(axis=0)
    out["rate_net"] = comb["net_ls"].mean(axis=0)
    out["rice"] = rice_prediction(kmag, temp, out["s_mean"], con)
    out["ratio"] = out["rate_abs"] / out["rice"]
    out["norm"] = out["rate_abs"] / (kmag / TWO_PI)  # nu (A1.3), not rate / Rice
    out["inv_rho"] = mean_inverse_modulus(
        arrays["snap_rho"][:, _production(arrays, meta)]
    )
    out["rice_cond"] = conditional_rice(kmag, int(meta["N"]), temp, out["inv_rho"])
    out["ratio_cond"] = out["rate_abs"] / out["rice_cond"]
    for key in ("abs_w1", "abs_w2"):  # (C, L, W) -> (L, C, W)
        out[key] = np.moveaxis(comb[key], 0, 1)
    out["net_rms_u"] = (
        np.sqrt(np.mean(comb["net_end"][..., ~con] ** 2, axis=(0, 2)))
        if np.any(~con)
        else np.full(temp.shape[0], np.nan)
    )
    for key in ("msd_c", "msd_u", "msd_bin", "part_msd"):
        out[key] = comb[key]
    out.update(_fits(comb, out["rate_abs"], con))
    ok = lane_ok(arrays)
    for key, val in out.items():
        shape = (-1,) + (1,) * (val.ndim - 1)
        out[key] = np.where(ok.reshape(shape), val, np.nan)
    out["t0"], out["lag0"], out["kmag"], out["bin_id"] = (
        comb["t0"],
        comb["lag0"],
        kmag,
        ids,
    )
    out["lane_ok"] = ok
    assert out["rate_abs"].shape == arrays["snap_wind_abs"][:, 0].shape, "rates"
    return out


def _aggregates(q: Dict[str, np.ndarray], con: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-lane wave means and bin means of the lane quantities."""
    ids = q["bin_id"]
    temp = q["temp"][:, None]
    out = {
        "nu_c": np.mean(q["norm"][:, con], axis=1),
        "abs_mean_c": np.mean(q["rate_abs"][:, con], axis=1),
        "net_mean_c": np.mean(q["rate_net"][:, con], axis=1),
        "w1_c": np.mean(q["abs_w1"][..., con], axis=-1),
        "w2_c": np.mean(q["abs_w2"][..., con], axis=-1),
        "rate_abs_bin": bin_mean(q["rate_abs"], ids),
        "rate_net_bin": bin_mean(q["rate_net"], ids),
        "ratio_bin": bin_mean(q["ratio"], ids),
        "ratio_cond_bin": bin_mean(q["ratio_cond"], ids),
        "norm_bin": bin_mean(q["norm"], ids),
        "s_over_t_bin": bin_mean(q["s_mean"] / temp, ids),
        "w1_bin": bin_mean(q["abs_w1"], ids),
        "w2_bin": bin_mean(q["abs_w2"], ids),
    }
    assert out["nu_c"].shape == q["temp"].shape, "one value per lane"
    assert out["w1_c"].shape == q["abs_w1"].shape[:2], "per lane and segment"
    return out


PER_LANE_KEYS: Tuple[str, ...] = (
    "rate_abs", "rate_net", "s_mean", "rice", "ratio", "norm", "phi_ratio",
    "phi_ratio_seg", "ke_ratio", "net_rms_u", "msd_c", "msd_u", "part_msd",
    "t_min", "alpha", "d_theta", "d_theta_bin", "lane_ok", "inv_rho",
    "rice_cond",
)  # fmt: skip


def analyze_file(path: Path, flows: Optional[Path]) -> Dict[str, np.ndarray]:
    """All stored quantities of one MD output (grids (G, n_T, ...))."""
    assert Path(path).suffix == ".npz", "need an MD output .npz"
    arrays, meta = load_md(path)
    g, n_t = lane_layout(arrays)
    con = arrays["waves_constrained"].astype(bool)
    q = lane_quantities(arrays, meta)
    q["temp"] = arrays["lane_temperature"]
    agg = _aggregates(q, con)
    out = {k: to_grid(q[k], g, n_t) for k in PER_LANE_KEYS}
    out.update({k: to_grid(v, g, n_t) for k, v in agg.items()})
    args = meta.get("args", {})
    stem = Path(path).stem
    out.update(
        system=np.asarray(Path(path).resolve().parent.parent.name),
        tag=np.asarray(stem.split("_chi")[-1]), chi=np.asarray(float(meta["chi"])),
        N=np.asarray(int(meta["N"])), d=np.asarray(int(meta["d"])),
        M=np.asarray(int(meta["M"])), K=np.asarray(float(meta["K"])),
        dt=np.asarray(float(meta["settings"]["dt"])),
        scheme=np.asarray(str(args.get("scheme", "andersen-nve"))),
        gamma=np.asarray(float(args.get("gamma", 0.0) or 0.0)),
        temps=arrays["temperatures"], indices=arrays["indices"], kmag=q["kmag"],
        constrained=con, bin_id=q["bin_id"],
        bin_edges=np.concatenate([BIN_EDGES_C, BIN_EDGES_U[1:]]),
        msd_lag=q["t0"], part_lag=q["lag0"], status=to_grid(arrays["status"], g, n_t),
        gate=to_grid(arrays.get("equil_gate", np.full(g * n_t, np.nan)), g, n_t),
    )  # fmt: skip
    out.update(_snapshot_extras(arrays, meta, n_t))
    if flows is not None:
        k_eig = ground_state_spectra(
            flows, arrays["indices"], arrays["waves_kvec"][con], arrays["basis"]
        )
        moments = np.stack([spectrum_moments(e) for e in k_eig])
        out.update(
            k_eig=k_eig, k_moments=moments, moment_names=np.asarray(MOMENT_NAMES)
        )
    if "trace_rho" in arrays:
        out.update(_trace_spectra(arrays, g, n_t))
    assert out["nu_c"].shape == (g, n_t), "grids are (G, n_T)"
    return out


def _snapshot_extras(
    arrays: Dict[str, np.ndarray], meta: Dict[str, Any], n_t: int
) -> Dict[str, np.ndarray]:
    """Kymograph (ground state 0, segment 0) and Phi / T at every snapshot."""
    segs = segments(meta)
    rec, dt = arrays["record_steps"], float(meta["settings"]["dt"])
    s0, n = segs[0]
    idx0 = np.flatnonzero((rec > s0) & (rec <= s0 + n))
    prod = _production(arrays, meta)
    g = arrays["lane_index"].size // n_t
    phi = arrays["snap_phi"][:, prod] / arrays["lane_temperature"][:, None]
    out = {
        "kymo_t": (rec[idx0] - s0) * dt,
        "kymo_abs": (arrays["snap_wind_abs"][:n_t, idx0] / TWO_PI).astype(np.float32),
        "phi_over_t": to_grid(phi, g, n_t),
    }
    assert out["kymo_abs"].shape[:2] == (n_t, idx0.size), "kymograph shape"
    assert prod.size >= idx0.size, "production snapshots"
    return out


def _trace_spectra(arrays: Dict[str, np.ndarray], g: int, n_t: int) -> Dict:
    """Power spectrum of the constrained rho in the dense trace, per lane."""
    con = arrays["waves_constrained"].astype(bool)
    q = int(np.min(arrays["trace_count"]))
    assert q >= 8, "dense trace too short"
    times = arrays["trace_time"][:, :q]
    steps = np.diff(times, axis=1)
    if not np.allclose(steps, steps[0, 0], rtol=1e-6):
        raise ValueError("dense rho trace is not uniformly spaced")
    rho = arrays["trace_rho"][:, :q, con, 0] + 1j * arrays["trace_rho"][:, :q, con, 1]
    spec = [power_spectrum(r, float(steps[0, 0])) for r in rho]
    power = np.stack([p for _, p in spec])
    assert power.shape[0] == g * n_t, "one spectrum per lane"
    return {"spec_omega": spec[0][0], "spec_power": to_grid(power, g, n_t)}


def resolve_flows(path: Path, flows: str) -> Optional[Path]:
    """'none' -> None; 'auto' -> sibling flows_chi<tag>.npz if present; else PATH."""
    assert Path(path).suffix == ".npz", "need an MD output .npz"
    assert flows, "need a --flows value"
    if flows == "none":
        return None
    if flows == "auto":
        tag = Path(path).stem.split("_chi")[-1]
        cand = Path(path).parent / f"flows_chi{tag}.npz"
        return cand if cand.is_file() else None
    out = Path(flows)
    if not out.is_file():
        raise ValueError(f"--flows {out} not found")
    return out


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[3])

    # ----- required -----
    parser.add_argument(
        "--md",
        type=Path,
        nargs="+",
        required=True,
        help="md[_<kind>]_chi<tag>.npz files (CLI outputs)",
    )

    # ----- optional -----
    parser.add_argument(
        "--flows",
        default="auto",
        help="'auto' (sibling flows_chi<tag>.npz), 'none' or PATH",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="default: beside each input"
    )
    args = parser.parse_args()

    # =========================================================
    # ANALYZE
    # =========================================================
    for md_path in args.md:
        res = analyze_file(md_path, resolve_flows(md_path, args.flows))
        dest = (args.out_dir or md_path.parent) / output_name(md_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dest, **res)
        med = float(np.nanmedian(res["nu_c"]))
        print(f"wrote {dest} (median normalized rate {med:.4f})", flush=True)
    print("Done.", flush=True)
