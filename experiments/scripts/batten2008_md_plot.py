#!/usr/bin/env python3

"""
Figures F1-F10 of the low-temperature MD winding experiment (plan E3).

Reads the ``md_analysis*_chi<tag>.npz`` files of batten2008_md_analysis.py
(searched recursively under --root) and writes PNGs (batten2008_plot.finish)
to --out:

- F1 ``md_F1_kymograph_chi<tag>``: unsigned winding W_n(t) (turns, log color)
  of every wave against log10 t and |k|/K, ground state 0, first segment;
  rows T, columns starts.
- F2 ``md_F2_rates``: unsigned rate / (|k| / 2 pi) against |k|/K (median and
  10-90 % band over ground states, per T) with the Rice lines (dashed).
- F3 ``md_F3_stationarity``: windowed-rate ratio [t/2, t] / [t/4, t/2] - 1 per
  ground state and segment, with the 3 % band of H3.
- F4 ``md_F4_net_msd``: net-winding MSD (median over ground states) with the
  fit window, and the exponent alpha per ground state with the H4 band.
- F5 ``md_F5_starts``: normalized rate nu and D_theta by start (strip plots),
  with Kruskal-Wallis eta^2_H, the largest median difference and max |Cliff's
  delta| per panel.
- F6 ``md_F6_temperature``: nu, the unconstrained net-rate RMS (with T^1/2)
  and D_theta against T.
- F7 ``md_F7_equipartition``: <S>/T per |k|/K bin, <Phi>/(M T) by start, and
  the histogram of Phi/T at the snapshots against Gamma(M, 1).
- F8 ``md_F8_drift``: particle MSD against t sqrt(T) (collapse spread in the
  legend), and the unconstrained net winding (rad^2) against <|k|^2> MSD.
- F9 ``md_F9_thermostat``: BAOAB against Andersen + NVE nu per ground state
  (needs md_analysis_langevin*; skipped with a note otherwise).
- F10 ``md_F10_spectra``: rho_k power spectra of the dense trace against the
  sqrt(eig K) density (needs md_analysis_spectra*; skipped otherwise).

Render with the plottime environment:
/home/sd2402/torquato-scratch-sd2402/conda_envs/plottime/bin/python -I
batten2008_md_plot.py --root RUNS --out FIGDIR [--dpi 600]

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ----- numerics -----
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batten2008_md_analysis import N_BINS, TWO_PI, collapse_metric  # noqa: E402
from batten2008_md_stats import (  # noqa: E402
    H3_TOL,
    H4_ALPHA,
    cliffs_delta,
    kruskal_eta2,
    load_analysis,
    temp_indices,
)
from batten2008_plot import finish, plt  # noqa: E402  (house rcParams)
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

STARTS: Tuple[str, ...] = ("flows2d", "gau2d", "url2d", "rsa2d", "hip2d")
START_LABELS = {"flows2d": "Poisson", "gau2d": "Gaussian", "url2d": "URL",
                "rsa2d": "RSA", "hip2d": "HIP"}  # fmt: skip
START_COLORS = {"flows2d": "black", "gau2d": "#1f4e9c", "url2d": "#c0392b",
                "rsa2d": "#2e7d32", "hip2d": "#b9770e"}  # fmt: skip
T_COLORS = ("#1f4e9c", "#2e7d32", "#b9770e", "#c0392b")
GAMMA_MARKERS = ("o", "s", "^")
PREFIX = "md"

Data = Dict[str, np.ndarray]

# ============================================================
# HELPERS
# ============================================================


def load_all(roots: List[Path]) -> Dict[str, object]:
    """{'main': {(system, tag): data}, 'lang', 'spec': [data], 'tags', 'chis'}."""
    assert roots, "need roots"
    main: Dict[Tuple[str, str], Data] = {}
    lang: List[Data] = []
    spec: List[Data] = []
    for root in roots:
        for f in sorted(Path(root).rglob("md_analysis*_chi*.npz")):
            z = load_analysis(f)
            if f.name.startswith("md_analysis_chi"):
                main[(str(z["system"]), str(z["tag"]))] = z
            elif f.name.startswith("md_analysis_langevin"):
                lang.append(z)
            elif f.name.startswith("md_analysis_spectra"):
                spec.append(z)
    chis = {str(z["tag"]): float(z["chi"]) for z in main.values()}
    tags = sorted(chis, key=chis.__getitem__)
    assert all(len(k) == 2 for k in main), "keys are (system, tag)"
    return {"main": main, "lang": lang, "spec": spec, "tags": tags, "chis": chis}


def starts_of(main: Dict, tag: str) -> List[str]:
    """Starts present at this chi, in the house order (unknown ones last)."""
    assert tag, "need a chi tag"
    have = [s for s, t in main if t == tag]
    out = [s for s in STARTS if s in have] + sorted(set(have) - set(STARTS))
    assert len(out) == len(have), "every start listed once"
    return out


def chi_label(data: Dict[str, object], tag: str) -> str:
    """Panel label '$\\chi = 0.1$'."""
    assert tag in data["chis"], f"unknown chi tag {tag}"  # type: ignore[operator]
    chi = data["chis"][tag]  # type: ignore[index]
    assert chi > 0, "positive chi"
    return rf"$\chi = {chi:.2g}$"


def t_label(t: float) -> str:
    """'$T = 10^{-6}$'."""
    assert t > 0, "positive temperature"
    e = int(round(np.log10(t)))
    assert np.isclose(t, 10.0**e, rtol=1e-6), "temperatures are powers of ten"
    return rf"$T = 10^{{{e}}}$"


def band(ax: plt.Axes, x: np.ndarray, rows: np.ndarray, color: str, ls: str = "-",
         label: str = "") -> None:  # fmt: skip
    """Median and 10-90 % band over rows (ground states) of rows (G, X)."""
    assert rows.ndim == 2 and rows.shape[1] == x.size, "rows (G, X)"
    ok = np.any(np.isfinite(rows), axis=0)
    if not np.any(ok):
        return
    q10, q50, q90 = np.nanpercentile(rows[:, ok], [10.0, 50.0, 90.0], axis=0)
    ax.fill_between(x[ok], q10, q90, color=color, alpha=0.15, lw=0)
    ax.plot(x[ok], q50, color=color, ls=ls, lw=2.0, label=label)


def strip(ax: plt.Axes, x0: float, y: np.ndarray, color: str, marker: str = "o",
          width: float = 0.25) -> None:  # fmt: skip
    """Jittered points of y around x0 with a median tick."""
    assert width > 0.0, "positive strip width"
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return
    jitter = np.linspace(-width / 2, width / 2, y.size) if y.size > 1 else np.zeros(1)
    ax.plot(x0 + jitter, y, marker, color=color, ms=4, alpha=0.75, ls="none")
    ax.plot([x0 - width / 2, x0 + width / 2], [np.median(y)] * 2, color=color, lw=2.0)


def _median_rows(rows: np.ndarray) -> np.ndarray:
    """Median over axis 0, NaN (silently) where a column has no finite value."""
    assert rows.shape[0] >= 1, "need rows"
    assert rows.ndim == 2, "need (G, X)"
    out = np.full(rows.shape[1], np.nan)
    ok = np.any(np.isfinite(rows), axis=0)
    out[ok] = np.nanmedian(rows[:, ok], axis=0)
    return out


def bin_centers() -> np.ndarray:
    """|k|/K centres of the 2 N_BINS bins."""
    assert N_BINS >= 1, "bins"
    edges = np.concatenate([np.linspace(0, 1, N_BINS + 1),
                            np.linspace(1, 2, N_BINS + 1)[1:]])  # fmt: skip
    out: np.ndarray = 0.5 * (edges[:-1] + edges[1:])
    assert out.size == 2 * N_BINS, "one centre per bin"
    return out


def binned(z: Data, values: np.ndarray) -> np.ndarray:
    """Mean over the waves of each bin: (..., W) -> (..., 2 N_BINS)."""
    assert "bin_id" in z, "analysis data"
    ids = z["bin_id"]
    assert values.shape[-1] == ids.size, "one value per wave"
    out = np.full((*values.shape[:-1], 2 * N_BINS), np.nan)
    for b in range(2 * N_BINS):
        if np.any(ids == b):
            out[..., b] = np.mean(values[..., ids == b], axis=-1)
    return out


def start_axis(ax: plt.Axes, starts: List[str]) -> None:
    """Categorical x axis of starts."""
    assert ax is not None, "need an axis"
    assert starts, "need starts"
    ax.set_xticks(np.arange(len(starts)))
    ax.set_xticklabels([START_LABELS.get(s, s) for s in starts], rotation=30)


# ============================================================
# FIGURES
# ============================================================


def figure_kymograph(data: Dict[str, object], tag: str, out: Path, dpi: int) -> None:
    """F1: W_n(t) of ground state 0 (first segment), rows T, columns starts."""
    assert dpi > 0, "positive dpi"
    main = data["main"]  # type: ignore[assignment]
    starts = starts_of(main, tag)
    n_t = max(main[(s, tag)]["temps"].size for s in starts)
    fig, axes = plt.subplots(n_t, len(starts), figsize=(3.4 * len(starts), 2.6 * n_t),
                             sharex=True, sharey=True, squeeze=False,
                             constrained_layout=True)  # fmt: skip
    vals = np.concatenate([main[(s, tag)]["kymo_abs"].ravel() for s in starts])
    pos = vals[vals > 0]
    norm = LogNorm(np.percentile(pos, 1.0), np.percentile(pos, 99.9))
    im = None
    for j, s in enumerate(starts):
        z = main[(s, tag)]
        order = np.argsort(z["kmag"])
        x_k = z["kmag"][order] / float(z["K"])
        lt = np.log10(z["kymo_t"])
        for i, t in enumerate(z["temps"]):
            ax = axes[i, j]
            w = np.maximum(z["kymo_abs"][i][:, order], norm.vmin)
            im = ax.pcolormesh(lt, x_k, w.T, cmap="viridis", norm=norm,
                               shading="nearest", rasterized=True)  # fmt: skip
            ax.axhline(1.0, color="white", lw=0.8, ls="--")
            ax.set_title(f"{START_LABELS.get(s, s)}, {t_label(t)}", fontsize=12)
            if i == n_t - 1:
                ax.set_xlabel(r"Time, $\log_{10} t$")
            if j == 0:
                ax.set_ylabel(r"Wavenumber, $|\mathbf{k}|/K$")
    assert im is not None, "at least one panel"
    fig.colorbar(im, ax=axes, label=r"Unsigned Winding, $W_n$ (turns)", shrink=0.6)
    fig.suptitle(f"{chi_label(data, tag)}: ground state 0, first NVE segment "
                 "(dashed: $|\\mathbf{k}| = K$)", fontsize=14)  # fmt: skip
    finish(fig, out / f"{PREFIX}_F1_kymograph_chi{tag}", dpi)


def figure_rates(data: Dict[str, object], out: Path, dpi: int) -> None:
    """F2: rate / (|k| / 2 pi) against |k|/K, per T, with the Rice lines."""
    assert dpi > 0, "positive dpi"
    main, tags = data["main"], data["tags"]  # type: ignore[assignment]
    n_s = max(len(starts_of(main, t)) for t in tags)
    fig, axes = plt.subplots(len(tags), n_s, figsize=(3.6 * n_s, 3.4 * len(tags)),
                             sharex=True, sharey=True, squeeze=False)  # fmt: skip
    xc = bin_centers()
    for r, tag in enumerate(tags):
        for j, s in enumerate(starts_of(main, tag)):
            z, ax = main[(s, tag)], axes[r, j]
            rice = binned(z, z["rice"] / (z["kmag"] / TWO_PI))
            for i, t in enumerate(z["temps"]):
                for cls in (slice(0, N_BINS), slice(N_BINS, 2 * N_BINS)):
                    band(ax, xc[cls], z["norm_bin"][:, i, cls], T_COLORS[i % 4],
                         label=t_label(t) if cls.start == 0 else "")  # fmt: skip
                    med = _median_rows(rice[:, i, cls])
                    fin = np.isfinite(med)  # empty bins: join the occupied ones
                    ax.plot(xc[cls][fin], med[fin], color="black", ls="--", lw=1.0,
                            zorder=5)  # fmt: skip
            ax.axvline(1.0, color="0.6", lw=0.8, zorder=0)
            ax.set_yscale("log")
            ax.set_title(f"{START_LABELS.get(s, s)}, {chi_label(data, tag)}",
                         fontsize=12)  # fmt: skip
            if r == len(tags) - 1:
                ax.set_xlabel(r"Wavenumber, $|\mathbf{k}|/K$")
        axes[r, 0].set_ylabel(r"Rate / $(|\mathbf{k}|/2\pi)$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(Line2D([], [], color="black", ls="--", lw=1.0))
    labels.append("Rice prediction")
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 0.0), frameon=False)  # fmt: skip
    fig.tight_layout()
    assert len(labels) >= 2, "legend entries"
    finish(fig, out / f"{PREFIX}_F2_rates", dpi)


def figure_stationarity(data: Dict[str, object], out: Path, dpi: int) -> None:
    """F3: windowed-rate ratio - 1 per ground state and segment (H3 band)."""
    assert dpi > 0, "positive dpi"
    main, tags = data["main"], data["tags"]  # type: ignore[assignment]
    fig, axes = plt.subplots(1, len(tags), figsize=(6.5 * len(tags), 4.2),
                             sharey=True, squeeze=False)  # fmt: skip
    for r, tag in enumerate(tags):
        ax, starts = axes[0, r], starts_of(main, tag)
        for j, s in enumerate(starts):
            z = main[(s, tag)]
            n_t = z["temps"].size
            for i in range(n_t):
                ratio = z["w2_c"][:, i] / z["w1_c"][:, i] - 1.0
                strip(ax, j + (i - (n_t - 1) / 2) * 0.2, ratio.ravel(),
                      T_COLORS[i % 4], width=0.15)  # fmt: skip
        ax.axhspan(-H3_TOL, H3_TOL, color="0.85", zorder=0)
        ax.axhline(0.0, color="0.6", lw=0.8, zorder=0)
        start_axis(ax, starts)
        ax.set_title(chi_label(data, tag), fontsize=13)
    axes[0, 0].set_ylabel(r"Rate$[t/2, t]$ / Rate$[t/4, t/2] - 1$")
    temps = next(iter(main.values()))["temps"]
    handles = [Line2D([], [], color=T_COLORS[i % 4], marker="o", ls="none")
               for i in range(temps.size)]  # fmt: skip
    fig.legend(
        handles,
        [t_label(t) for t in temps],
        loc="upper center",
        ncol=temps.size,
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
    )
    fig.tight_layout()
    assert len(handles) == temps.size, "one legend entry per T"
    finish(fig, out / f"{PREFIX}_F3_stationarity", dpi)


def figure_net_msd(data: Dict[str, object], out: Path, dpi: int, t_idx: int) -> None:
    """F4: net-winding MSD (median over ground states, T index t_idx) and alpha."""
    assert dpi > 0 and t_idx >= 0, "positive dpi, T index"
    main, tags = data["main"], data["tags"]  # type: ignore[assignment]
    fig, axes = plt.subplots(2, len(tags), figsize=(6.5 * len(tags), 8.6),
                             squeeze=False)  # fmt: skip
    for r, tag in enumerate(tags):
        ax, starts = axes[0, r], starts_of(main, tag)
        for s in starts:
            z = main[(s, tag)]
            band(ax, z["msd_lag"], z["msd_c"][:, t_idx], START_COLORS.get(s, "0.5"),
                 label=START_LABELS.get(s, s))  # fmt: skip
            ax.axvline(
                float(np.nanmedian(z["t_min"][:, t_idx])),
                color=START_COLORS.get(s, "0.5"),
                lw=0.8,
                ls=":",
            )
        lag = main[(starts[0], tag)]["msd_lag"]
        ref = (
            np.nanmedian(main[(starts[0], tag)]["msd_c"][:, t_idx, -1]) * lag / lag[-1]
        )
        ax.plot(lag, ref, color="0.6", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Time Since Segment Start, $t$")
        temp = main[(starts[0], tag)]["temps"][t_idx]
        ax.set_title(f"{chi_label(data, tag)}, {t_label(temp)} (dotted: fit start)",
                     fontsize=12)  # fmt: skip
        ax = axes[1, r]
        for j, s in enumerate(starts):
            z = main[(s, tag)]
            n_t = z["temps"].size
            for i in range(n_t):
                strip(ax, j + (i - (n_t - 1) / 2) * 0.2, z["alpha"][:, i],
                      T_COLORS[i % 4], width=0.15)  # fmt: skip
        ax.axhspan(*H4_ALPHA, color="0.85", zorder=0)
        start_axis(ax, starts)
    axes[0, 0].set_ylabel(r"Net-Winding MSD, $\langle W_{\rm net}^2\rangle$")
    axes[1, 0].set_ylabel(r"MSD Exponent, $\alpha$")
    axes[0, 0].legend(frameon=False, fontsize=11)
    temps = next(iter(main.values()))["temps"]
    handles = [Line2D([], [], color=T_COLORS[i % 4], marker="o", ls="none")
               for i in range(temps.size)]  # fmt: skip
    fig.legend(
        handles,
        [t_label(t) for t in temps],
        loc="upper center",
        ncol=temps.size,
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
    )
    fig.tight_layout()
    assert axes.shape == (2, len(tags)), "two rows of panels"
    finish(fig, out / f"{PREFIX}_F4_net_msd", dpi)


def _effect_text(groups: Dict[str, np.ndarray]) -> str:
    """'eta2_H = .., dmed = .., max|delta| = ..' for start groups."""
    assert all(np.ndim(g) == 1 for g in groups.values()), "1D groups"
    assert len(groups) >= 2, "need two groups"
    _, _, eta2 = kruskal_eta2(list(groups.values()))
    meds = [float(np.nanmedian(g)) for g in groups.values()]
    grand = float(np.nanmedian(np.concatenate(list(groups.values()))))
    names = list(groups)
    delta = max(abs(cliffs_delta(groups[a], groups[b]))
                for i, a in enumerate(names) for b in names[i + 1:])  # fmt: skip
    return (
        rf"$\eta^2_H$={eta2:.3f}, $\Delta$med={(max(meds) - min(meds)) / grand:.3f},"
        rf" max$|\delta|$={delta:.2f}"
    )


def figure_starts(data: Dict[str, object], out: Path, dpi: int) -> None:
    """F5: nu and D_theta by start per (chi, T), with the effect sizes."""
    assert dpi > 0, "positive dpi"
    main, tags = data["main"], data["tags"]  # type: ignore[assignment]
    n_t = max(z["temps"].size for z in main.values())
    rows = [(tag, key) for tag in tags for key in ("nu_c", "d_theta")]
    fig, axes = plt.subplots(len(rows), n_t, figsize=(4.6 * n_t, 3.6 * len(rows)),
                             squeeze=False)  # fmt: skip
    ylab = {
        "nu_c": r"Normalized Rate, $\nu$",
        "d_theta": r"$D_\theta$ (turns$^2$/time)",
    }
    for r, (tag, key) in enumerate(rows):
        starts = starts_of(main, tag)
        for i in range(n_t):
            ax = axes[r, i]
            groups = {s: main[(s, tag)][key][:, i] for s in starts
                      if i < main[(s, tag)]["temps"].size}  # fmt: skip
            for j, s in enumerate(groups):
                strip(ax, j, groups[s], START_COLORS.get(s, "0.5"))
            start_axis(ax, list(groups))
            temp = main[(starts[0], tag)]["temps"][i]
            text = _effect_text(groups) if len(groups) >= 2 else ""
            ax.set_title(
                f"{chi_label(data, tag)}, {t_label(temp)}\n{text}", fontsize=10
            )
        axes[r, 0].set_ylabel(ylab[key])
    fig.tight_layout()
    assert len(rows) == 2 * len(tags), "two rows per chi"
    finish(fig, out / f"{PREFIX}_F5_starts", dpi)


def figure_temperature(data: Dict[str, object], out: Path, dpi: int) -> None:
    """F6: nu, the unconstrained net-rate RMS and D_theta against T."""
    assert dpi > 0, "positive dpi"
    main, tags = data["main"], data["tags"]  # type: ignore[assignment]
    keys = (("nu_c", r"Normalized Rate, $\nu$"),
            ("net_rms_u", r"Net Rate, $|\mathbf{k}| > K$"),
            ("d_theta", r"$D_\theta$ (turns$^2$/time)"))  # fmt: skip
    fig, axes = plt.subplots(len(tags), 3, figsize=(15.0, 3.9 * len(tags)),
                             sharex=True, squeeze=False)  # fmt: skip
    for r, tag in enumerate(tags):
        for c, (key, ylab) in enumerate(keys):
            ax = axes[r, c]
            for s in starts_of(main, tag):
                z = main[(s, tag)]
                q25, q50, q75 = np.nanpercentile(z[key], [25.0, 50.0, 75.0], axis=0)
                ax.errorbar(z["temps"], q50, yerr=[q50 - q25, q75 - q50], marker="o",
                            color=START_COLORS.get(s, "0.5"), lw=2.0, capsize=3,
                            label=START_LABELS.get(s, s))  # fmt: skip
            ax.set_xscale("log")
            if key != "nu_c":
                ax.set_yscale("log")
            if key == "net_rms_u":
                z = main[(starts_of(main, tag)[0], tag)]
                y0 = float(np.nanmedian(z[key][:, 0]))
                ax.plot(
                    z["temps"],
                    y0 * np.sqrt(z["temps"] / z["temps"][0]),
                    color="0.6",
                    lw=0.8,
                    ls="--",
                    label=r"$\propto T^{1/2}$",
                )
            ax.set_ylabel(ylab)
            ax.set_title(chi_label(data, tag), fontsize=12)
            if r == len(tags) - 1:
                ax.set_xlabel(r"Temperature, $k_B T$")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 0.0), frameon=False)  # fmt: skip
    fig.tight_layout()
    assert len(handles) >= 1, "legend entries"
    finish(fig, out / f"{PREFIX}_F6_temperature", dpi)


def figure_equipartition(data: Dict[str, object], out: Path, dpi: int) -> None:
    """F7: <S>/T per bin, <Phi>/(M T) by start, Phi/T histogram vs Gamma(M, 1)."""
    assert dpi > 0, "positive dpi"
    main, tags = data["main"], data["tags"]  # type: ignore[assignment]
    fig, axes = plt.subplots(len(tags), 3, figsize=(15.0, 3.9 * len(tags)),
                             squeeze=False)  # fmt: skip
    xc = bin_centers()[:N_BINS]
    for r, tag in enumerate(tags):
        starts = starts_of(main, tag)
        z0 = main[(starts[0], tag)]
        for i, t in enumerate(z0["temps"]):
            band(axes[r, 0], xc, z0["s_over_t_bin"][:, i, :N_BINS], T_COLORS[i % 4],
                 label=t_label(t))  # fmt: skip
        axes[r, 0].axhspan(0.95, 1.05, color="0.85", zorder=0)
        axes[r, 0].set_ylabel(r"$\langle S(\mathbf{k})\rangle / T$")
        axes[r, 0].set_xlabel(r"Wavenumber, $|\mathbf{k}|/K$")
        axes[r, 0].set_title(f"{START_LABELS.get(starts[0])}, {chi_label(data, tag)}",
                             fontsize=12)  # fmt: skip
        ax = axes[r, 1]
        for j, s in enumerate(starts):
            z = main[(s, tag)]
            n_t = z["temps"].size
            for i in range(n_t):
                strip(ax, j + (i - (n_t - 1) / 2) * 0.2, z["phi_ratio"][:, i],
                      T_COLORS[i % 4], width=0.15)  # fmt: skip
        ax.axhspan(0.98, 1.02, color="0.85", zorder=0)
        start_axis(ax, starts)
        ax.set_ylabel(r"$\langle\Phi\rangle / (M T)$")
        ax = axes[r, 2]
        harm = temp_indices(z0["temps"], True)
        x = z0["phi_over_t"][:, harm].ravel()
        x = x[np.isfinite(x)]
        m = int(z0["M"])
        ax.hist(x, bins=40, density=True, color="0.7", label="snapshots")
        grid = np.linspace(x.min(), x.max(), 200)
        ax.plot(grid, stats.gamma.pdf(grid, m), color="black", lw=2.0,
                label=rf"Gamma$({m}, 1)$")  # fmt: skip
        ax.set_xlabel(r"$\Phi / T$")
        ax.set_ylabel("Density")
        ax.legend(frameon=False, fontsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 0.0), frameon=False)  # fmt: skip
    fig.tight_layout()
    assert axes.shape[1] == 3, "three columns"
    finish(fig, out / f"{PREFIX}_F7_equipartition", dpi)


def figure_drift(data: Dict[str, object], out: Path, dpi: int) -> None:
    """F8: particle MSD against t sqrt(T); unconstrained net winding vs <|k|^2> MSD."""
    assert dpi > 0, "positive dpi"
    main, tags = data["main"], data["tags"]  # type: ignore[assignment]
    fig, axes = plt.subplots(len(tags), 2, figsize=(12.0, 4.4 * len(tags)),
                             squeeze=False)  # fmt: skip
    for r, tag in enumerate(tags):
        starts = starts_of(main, tag)
        z0 = main[(starts[0], tag)]
        lag = z0["part_lag"]
        ok = lag > 0
        for i, t in enumerate(z0["temps"]):
            band(axes[r, 0], lag[ok] * np.sqrt(t), z0["part_msd"][:, i][:, ok],
                 T_COLORS[i % 4], label=t_label(t))  # fmt: skip
        harm = temp_indices(z0["temps"], True)
        spread = {s: collapse_metric(main[(s, tag)]["part_lag"], np.nanmedian(
            main[(s, tag)]["part_msd"][:, harm], axis=0), main[(s, tag)]["temps"][harm])
            for s in starts}  # fmt: skip
        txt = ", ".join(f"{START_LABELS.get(s, s)} {v:.2f}" for s, v in spread.items())
        ax = axes[r, 0]
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Scaled Time, $t\sqrt{T}$")
        ax.set_ylabel(r"Particle MSD, $\langle|\Delta\mathbf{r}|^2\rangle$")
        ax.set_title(
            f"{START_LABELS.get(starts[0])}, {chi_label(data, tag)}; "
            f"collapse spread (T$\\leq 10^{{-4}}$):\n{txt}",
            fontsize=10,
        )
        ax.legend(frameon=False, fontsize=11)
        ax = axes[r, 1]
        for s in starts:
            z = main[(s, tag)]
            k2 = float(np.mean(z["kmag"][~z["constrained"]] ** 2))
            for i in range(z["temps"].size):
                x = k2 * z["part_msd"][:, i, -1]
                y = (TWO_PI * z["net_rms_u"][:, i] * z["msd_lag"][-1]) ** 2
                ax.plot(x, y, "o", color=T_COLORS[i % 4], ms=4, alpha=0.7)
        lim = ax.get_xlim()
        ax.plot(lim, lim, color="0.6", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\langle|\mathbf{k}|^2\rangle_u\,$MSD$(t_{\rm end})$")
        ax.set_ylabel(r"Unconstrained $\langle(2\pi W_{\rm net})^2\rangle$")
        ax.set_title(
            f"all starts, {chi_label(data, tag)} (dashed: $y = x$)", fontsize=12
        )
    fig.tight_layout()
    assert axes.shape[1] == 2, "two columns"
    finish(fig, out / f"{PREFIX}_F8_drift", dpi)


def figure_thermostat(data: Dict[str, object], out: Path, dpi: int) -> None:
    """F9: BAOAB against Andersen + NVE nu per ground state (harmonic T)."""
    assert dpi > 0, "positive dpi"
    main, lang, tags = data["main"], data["lang"], data["tags"]  # type: ignore
    if not lang:
        print("F9 skipped: no md_analysis_langevin*_chi*.npz", flush=True)
        return
    fig, axes = plt.subplots(1, len(tags), figsize=(5.8 * len(tags), 5.0),
                             squeeze=False)  # fmt: skip
    gammas = sorted({float(z["gamma"]) for z in lang})
    for c, tag in enumerate(tags):
        ax = axes[0, c]
        for z in [q for q in lang if str(q["tag"]) == tag]:
            m = main.get((str(z["system"]), tag))
            if m is None:
                continue
            common = np.intersect1d(z["indices"], m["indices"])
            iz, im = (
                np.searchsorted(z["indices"], common),
                np.searchsorted(m["indices"], common),
            )
            for ti in temp_indices(z["temps"], True):
                tm = np.flatnonzero(np.isclose(m["temps"], z["temps"][ti]))
                if tm.size:
                    ax.plot(
                        m["nu_c"][im, tm[0]],
                        z["nu_c"][iz, ti],
                        GAMMA_MARKERS[gammas.index(float(z["gamma"])) % 3],
                        color=T_COLORS[int(tm[0]) % 4],
                        ms=5,
                        ls="none",
                    )
        lo, hi = ax.get_xlim()
        grid = np.array([lo, hi])
        ax.plot(grid, grid, color="0.6", lw=0.8)
        ax.fill_between(grid, grid * 0.97, grid * 1.03, color="0.85", zorder=0)
        ax.set_xlabel(r"Andersen + NVE, $\nu$")
        ax.set_ylabel(r"BAOAB, $\nu$")
        ax.set_title(chi_label(data, tag), fontsize=12)
    handles = [Line2D([], [], color="0.3", marker=GAMMA_MARKERS[i % 3], ls="none")
               for i in range(len(gammas))]  # fmt: skip
    fig.legend(
        handles,
        [rf"$\gamma = {g:g}$" for g in gammas],
        loc="upper center",
        ncol=len(gammas),
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
    )
    fig.tight_layout()
    assert gammas, "Langevin runs carry gamma"
    finish(fig, out / f"{PREFIX}_F9_thermostat", dpi)


def figure_spectra(data: Dict[str, object], out: Path, dpi: int) -> None:
    """F10: power spectrum of the constrained rho_k against the sqrt(eig K) density."""
    assert dpi > 0, "positive dpi"
    spec, tags = data["spec"], data["tags"]  # type: ignore[assignment]
    spec = [z for z in spec if "spec_power" in z]
    if not spec:
        print("F10 skipped: no md_analysis_spectra*_chi*.npz with a dense trace",
              flush=True)  # fmt: skip
        return
    starts = [s for s in STARTS if any(str(z["system"]) == s for z in spec)]
    fig, axes = plt.subplots(len(tags), len(starts), figsize=(3.8 * len(starts),
                             3.4 * len(tags)), squeeze=False, sharex="row")  # fmt: skip
    for z in spec:
        if str(z["tag"]) not in tags or str(z["system"]) not in starts:
            continue
        ax = axes[tags.index(str(z["tag"])), starts.index(str(z["system"]))]
        pos = z["spec_omega"] > 0
        ax.plot(z["spec_omega"][pos], z["spec_power"][0, 0][pos], color="black",
                lw=1.0, label=r"$|\tilde\rho_k(\omega)|^2$")  # fmt: skip
        ax.set_yscale("log")
        if "k_eig" in z:
            freq = np.sqrt(np.maximum(z["k_eig"][0], 0.0))
            grid = np.linspace(0.0, 2.5 * float(z["K"]), 300)
            dens = stats.gaussian_kde(freq, bw_method=0.1)(grid)
            tw = ax.twinx()
            tw.set_yscale("linear")  # twinx copies the log scale of ax
            tw.fill_between(grid, 0.0, dens, color="#c0392b", alpha=0.35, lw=0)
            tw.set_ylim(0.0, 1.1 * dens.max())
            tw.set_yticks([])
        ax.set_xlim(0.0, 2.5 * float(z["K"]))
        ax.set_title(
            f"{START_LABELS.get(str(z['system']))}, {chi_label(data, str(z['tag']))}",
            fontsize=11,
        )
        ax.set_xlabel(r"Frequency, $\omega$")
    for r in range(len(tags)):
        axes[r, 0].set_ylabel("Mean Normalized Power")
    handles = [
        Line2D([], [], color="black", lw=1.0),
        plt.Rectangle((0, 0), 1, 1, color="#c0392b", alpha=0.35),
    ]
    fig.legend(handles, [r"$\rho_k$ power (ground state 0)",
                         r"$\sqrt{\mathrm{eig}\,K}$ density"], loc="upper center",
               ncol=2, bbox_to_anchor=(0.5, 0.0), frameon=False)  # fmt: skip
    fig.tight_layout()
    assert starts, "spectra files have systems"
    finish(fig, out / f"{PREFIX}_F10_spectra", dpi)


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[3])

    # ----- required -----
    parser.add_argument(
        "--root",
        type=Path,
        nargs="+",
        required=True,
        help="directories searched for md_analysis*_chi*.npz",
    )
    parser.add_argument("--out", type=Path, required=True)

    # ----- optional -----
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--msd-t-index",
        type=int,
        default=0,
        help="temperature index of the F4 MSD panels",
    )
    args = parser.parse_args()

    # =========================================================
    # FIGURES
    # =========================================================
    runs = load_all(args.root)
    if not runs["main"]:
        sys.exit("no md_analysis_chi*.npz under --root")
    args.out.mkdir(parents=True, exist_ok=True)
    for chi_tag in runs["tags"]:  # type: ignore[attr-defined]
        figure_kymograph(runs, chi_tag, args.out, args.dpi)
    figure_rates(runs, args.out, args.dpi)
    figure_stationarity(runs, args.out, args.dpi)
    figure_net_msd(runs, args.out, args.dpi, args.msd_t_index)
    figure_starts(runs, args.out, args.dpi)
    figure_temperature(runs, args.out, args.dpi)
    figure_equipartition(runs, args.out, args.dpi)
    figure_drift(runs, args.out, args.dpi)
    figure_thermostat(runs, args.out, args.dpi)
    figure_spectra(runs, args.out, args.dpi)
    print("Done.", flush=True)
