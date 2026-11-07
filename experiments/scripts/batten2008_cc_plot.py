#!/usr/bin/env python3

"""
Figures for the collective-coordinate analyses of the N = 1000 flows.

- batten2008_N<N>_modes: (a, b) log10 S(k_n, t) for flow 0, chi = 1/3 and
  2/3 (Phi = sum_n S(k_n) exactly); (c) the last t/T at which S(k_n) exceeds
  1e-2 and 1e-8, median and 10-90 % band over flows, against mode number n,
  with an n^-2 guide; (d) S(k_1, t) of the longest wavelength for every flow
  against time, with the linearized decay exp(-2 k_1^2 t).
- batten2008_N<N>_mode_planes: rho(k_n) in polar form for n = 1, 2, M/2, M:
  angle arg rho, radius log10 S + 20 (the centre is S = 1e-20). Circles are
  exact contours of that mode's energy; a radial line is a fixed phase.
- batten2008_N<N>_mode_planes_full: the same for n = 1-5 and n = 10, M/4,
  M/2, 3M/4, M.
- batten2008_N<N>_mode_phases: per mode, (a) total phase winding, (b) S(k_n)
  when the phase locks (less than 0.1 rad left to turn), (c) the late-time
  exponent p_n in S_n ~ S_1^p_n (median and 10-90 % over flows; shown when
  at least 10 flows resolve it).
- batten2008_N<N>_hessian_detail: (a, e) positive eigenvalues over k_p^2,
  p = ceil(i/2), flow 0 at t/T >= 1e-4 and all flows at the end; (b, f)
  curvature along the density waves over k_n^2; (c, g) number of negative
  eigenvalues and |lambda_min|; (d, h) mean Fourier power of the negative
  eigenvectors (solid) and of as many of the stiffest ones (dotted).
- batten2008_N<N>_hessian: (a, b) Hessian eigenvalues of flow 0 at six
  times (symlog); (c) negative and near-zero (|lambda| < 1e-6 lambda_max)
  eigenvalue counts, median over flows.
- batten2008_N<N>_sections: Phi on the plane through r(t) spanned by the flow
  direction tau and the stiffest Hessian direction v, with the flow's frames.
- batten2008_N<N>_floor: flat-slice and valley-floor contours on the start-end
  and PCA planes, and the floor beneath sampled flow frames against the true
  Phi (if floor_chi<tag>_<view>.npz exist; batten2008_floor.py).

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
from typing import Dict, List

# ----- numerics -----
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batten2008_plot import (  # noqa: E402  (also applies the house rcParams)
    CHI_COLORS,
    CHI_TAGS,
    FLOW_COLOR,
    VIEW_LABELS,
    VIEW_TITLES,
    finish,
    load,
    plt,
)
from matplotlib.lines import Line2D  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

FLOOR = 1e-20  # S, Phi below this are ground-state round-off
EPS_STYLES = {0: "-", 1: "--"}  # t_relax thresholds 1e-2 and 1e-8
SAMPLE_CMAP = plt.get_cmap("Blues")
SIX = (0.0, 1e-5, 1e-3, 1e-2, 1e-1, 1.0)  # t/T of batten2008_N<N>_hessian

# ============================================================
# HELPERS
# ============================================================


def legend_below(fig: plt.Figure, handles: List, labels: List[str]) -> None:
    """One frameless legend row beneath the figure."""
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 0.0), frameon=False)  # fmt: skip


def band(ax: plt.Axes, x: np.ndarray, rows: np.ndarray, color: str, ls: str) -> None:
    """Median line and 10-90 % band of rows (flows, x)."""
    q10, q50, q90 = np.nanpercentile(rows, [10.0, 50.0, 90.0], axis=0)
    ax.fill_between(x, q10, q90, color=color, alpha=0.15, lw=0)
    ax.plot(x, q50, color=color, ls=ls, lw=2.0)


# ============================================================
# FIGURES
# ============================================================


def figure_modes(root: Path, n: int, out: Path, dpi: int) -> None:
    """Mode-resolved energy along the flows (collective coordinates)."""
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0), constrained_layout=True)
    side = float(n)  # density 1
    k1 = 2.0 * np.pi / side
    for col, (tag, label) in enumerate(CHI_TAGS):
        z = load(root / f"modes_chi{tag}.npz")
        m = int(z["m"])
        ax = axes[0, col]
        logs = np.log10(np.maximum(z["sk_grid"][0, 1:], FLOOR)).T  # (M, G)
        mesh = ax.pcolormesh(z["grid"][1:], np.arange(1, m + 1), logs,
                             cmap="Blues_r", vmin=-20, vmax=1, shading="nearest",
                             rasterized=True)  # fmt: skip
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"({'ab'[col]}) {label} (M = {m}), flow 0", loc="left")
        ax.set_xlabel(r"Dimensionless Time, $t/T$")
        ax.set_ylabel(r"Mode Number, $n$ ($k_n = 2\pi n/L$)")
        ns = np.arange(1, m + 1)
        for e, ls in EPS_STYLES.items():
            band(axes[1, 0], ns, z["t_relax"][:, :, e], CHI_COLORS[tag], ls)
        with np.load(root / f"flows_chi{tag}.npz") as f:
            t_all, cnt = f["traj_t"], f["traj_count"]
        s1 = z["rho_re"][:, :, 0] ** 2 + z["rho_im"][:, :, 0] ** 2
        for b in range(s1.shape[0]):
            c = int(cnt[b])
            axes[1, 1].semilogy(t_all[b, :c], np.maximum(s1[b, :c], FLOOR),
                                color=CHI_COLORS[tag], lw=0.6, alpha=0.6)  # fmt: skip
    fig.colorbar(mesh, ax=axes[0, :], label=r"$\log_{10} S(k_n, t)$", shrink=0.9)
    ax = axes[1, 0]
    ns = np.arange(1, 668)
    ax.plot(ns, 7e-2 * ns**-2.0, color="0.6", lw=0.8, ls=":")
    ax.text(20.0, 7e-2 * 20.0**-2.0 / 10.0, r"$\propto n^{-2}$", color="0.4")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-7, 1.5)
    ax.set_title("(c) relaxation time of each mode", loc="left")
    ax.set_xlabel(r"Mode Number, $n$")
    ax.set_ylabel(r"Last Time Above $\epsilon$, $t_n/T$")
    ax = axes[1, 1]
    t_ref = np.array([1e5, 4e5])
    ax.semilogy(t_ref, 1e-6 * np.exp(-2.0 * k1**2 * (t_ref - t_ref[0])),
                color="0.3", lw=2.0, ls="-.")  # fmt: skip
    ax.set_title(r"(d) longest wavelength, $n = 1$", loc="left")
    ax.set_xlabel(r"Time, $t$")
    ax.set_ylabel(r"Structure Factor, $S(k_1, t)$")
    handles = [Line2D([], [], color=CHI_COLORS[t], lw=2.0) for t, _ in CHI_TAGS]
    handles += [Line2D([], [], color="0.3", lw=2.0, ls=ls) for ls in ("-", "--")]
    handles += [Line2D([], [], color="0.3", lw=2.0, ls="-.")]
    labels = [lab for _, lab in CHI_TAGS]
    labels += [r"(c) $\epsilon = 10^{-2}$", r"(c) $\epsilon = 10^{-8}$",
               r"(d) $e^{-2k_1^2 t}$"]  # fmt: skip
    legend_below(fig, handles, labels)
    finish(fig, out / f"batten2008_N{n}_modes", dpi)


def polar_panel(ax: plt.Axes, z: Dict[str, np.ndarray], mode: int, title: str) -> None:
    """rho(k_mode) of every flow: angle arg rho, radius log10 S + 20."""
    j = z["picks"].tolist().index(mode)
    for b in range(z["rho_re"].shape[0])[::-1]:
        re, im = z["rho_re"][b, :, j], z["rho_im"][b, :, j]
        ok = np.isfinite(re)
        r = np.clip(np.log10(np.maximum(re[ok] ** 2 + im[ok] ** 2, FLOOR)) + 20,
                    0.0, None)  # fmt: skip
        theta = np.arctan2(im[ok], re[ok])
        style = (dict(color=FLOW_COLOR, lw=1.4, zorder=3) if b == 0
                 else dict(color="0.55", lw=0.5, alpha=0.5))  # fmt: skip
        ax.plot(theta, r, **style)
        if b == 0:
            ax.plot(theta[0], r[0], "o", color="black", ms=4, zorder=4)
    ax.set_rlim(0, 21)
    ax.set_rticks([0, 10, 20])
    ax.set_yticklabels([r"$10^{-20}$", r"$10^{-10}$", r"$1$"], fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_title(title, loc="left", fontsize=13)


def planes_legend(fig: plt.Figure) -> None:
    """Shared legend of the phase-plane figures."""
    handles = [Line2D([], [], color=FLOW_COLOR, lw=1.5),
               Line2D([], [], color="0.55", lw=1.0),
               Line2D([], [], ls="", marker="o", color="black", ms=4)]  # fmt: skip
    legend_below(fig, handles, ["Flow 0", "Other Flows", "Poisson Start"])


def figure_mode_planes(root: Path, n: int, out: Path, dpi: int) -> None:
    """rho(k_n) in polar form; circles are exact per-mode energy contours."""
    fig, axes = plt.subplots(2, 4, figsize=(15.0, 8.0), constrained_layout=True,
                             subplot_kw={"projection": "polar"})  # fmt: skip
    for row, (tag, label) in enumerate(CHI_TAGS):
        z = load(root / f"modes_chi{tag}.npz")
        m = int(z["m"])
        for col, mode in enumerate((1, 2, round(0.5 * m), m)):
            title = f"({'abcdefgh'[4 * row + col]}) {label}, $n = {mode}$"
            polar_panel(axes[row, col], z, mode, title)
    planes_legend(fig)
    finish(fig, out / f"batten2008_N{n}_mode_planes", dpi)


def figure_mode_planes_full(root: Path, n: int, out: Path, dpi: int) -> None:
    """Phase planes for n = 1-5 and n = 10, M/4, M/2, 3M/4, M at both chi."""
    fig, axes = plt.subplots(4, 5, figsize=(18.0, 15.5), constrained_layout=True,
                             subplot_kw={"projection": "polar"})  # fmt: skip
    letters = "abcdefghijklmnopqrst"
    for block, (tag, label) in enumerate(CHI_TAGS):
        z = load(root / f"modes_chi{tag}.npz")
        m = int(z["m"])
        rows = ((1, 2, 3, 4, 5),
                (10, round(0.25 * m), round(0.5 * m), round(0.75 * m), m))  # fmt: skip
        for r, modes in enumerate(rows):
            for col, mode in enumerate(modes):
                k = 10 * block + 5 * r + col
                polar_panel(axes[2 * block + r, col], z, mode,
                            f"({letters[k]}) {label}, $n = {mode}$")  # fmt: skip
    planes_legend(fig)
    finish(fig, out / f"batten2008_N{n}_mode_planes_full", dpi)


def late_exponent(z: Dict[str, np.ndarray], mode: int) -> np.ndarray:
    """Per flow: slope of log S(k_mode) against log S(k_1) late in the flow.

    Fit over frames with 1e-12 < S_1 < 1e-2 and S_mode > 1e-19 (NaN if fewer
    than 10). Independent linear decay of each mode gives mode^2; rho_mode
    slaved to rho_1^p gives p.
    """
    picks = z["picks"].tolist()
    j1, jn = picks.index(1), picks.index(mode)
    out = np.full(z["rho_re"].shape[0], np.nan)
    for b in range(out.shape[0]):
        s1 = z["rho_re"][b, :, j1] ** 2 + z["rho_im"][b, :, j1] ** 2
        sn = z["rho_re"][b, :, jn] ** 2 + z["rho_im"][b, :, jn] ** 2
        ok = np.isfinite(s1) & (s1 > 1e-12) & (s1 < 1e-2) & (sn > 1e-19)
        if ok.sum() >= 10:
            out[b] = np.polyfit(np.log(s1[ok]), np.log(sn[ok]), 1)[0]
    return out


def figure_mode_phases(root: Path, n: int, out: Path, dpi: int) -> None:
    """Phase turns, lock-in S and late-time exponents of every mode."""
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.8), constrained_layout=True)
    small = (2, 3, 4, 5, 10)
    for tag, _label in CHI_TAGS:
        z = load(root / f"modes_chi{tag}.npz")
        m = int(z["m"])
        ns = np.arange(1, m + 1)
        c = CHI_COLORS[tag]
        band(axes[0], ns, np.maximum(z["turns"], 1e-3), c, "-")
        band(axes[1], ns, np.log10(np.maximum(z["s_lock"], FLOOR)), c, "-")
        p = np.array([late_exponent(z, k) for k in small])  # (modes, flows)
        enough = np.isfinite(p).sum(axis=1) >= 10  # resolved in >= 10 flows
        p = np.where(enough[:, None], p, np.nan)
        q10, q50, q90 = np.nanpercentile(p, [10.0, 50.0, 90.0], axis=1)
        axes[2].errorbar(small, q50, yerr=[q50 - q10, q90 - q50], color=c, lw=2.0,
                         marker="o", capsize=3)  # fmt: skip
    ax = axes[0]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("(a) total phase winding", loc="left")
    ax.set_xlabel(r"Mode Number, $n$")
    ax.set_ylabel(r"Turns, $\sum|\Delta\arg\rho|/2\pi$")
    ax = axes[1]
    ax.set_xscale("log")
    ax.set_title("(b) where the phase locks", loc="left")
    ax.set_xlabel(r"Mode Number, $n$")
    ax.set_ylabel(r"$\log_{10} S(k_n)$ at Lock-In")
    ax = axes[2]
    ks = np.array(small, dtype=float)
    ax.plot(ks, ks**2, color="0.6", lw=0.8, ls=":")
    ax.plot(ks, ks, color="0.6", lw=0.8, ls="--")
    ax.axhline(1.0, color="0.6", lw=0.8, ls="-.")
    ax.text(6.0, 40.0, r"$n^2$: own decay", color="0.4", fontsize=11)
    ax.text(6.5, 7.5, r"$n$: $\rho_n\propto\rho_1^{\,n}$", color="0.4", fontsize=11)
    ax.text(6.5, 1.2, r"$1$: $\rho_n\propto\rho_1$", color="0.4", fontsize=11)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(r"(c) late time: $S_n \propto S_1^{\,p_n}$", loc="left")
    ax.set_xticks(ks, [f"{k:g}" for k in ks])
    ax.minorticks_off()
    ax.set_xlabel(r"Mode Number, $n$")
    ax.set_ylabel(r"Exponent, $p_n$")
    handles = [Line2D([], [], color=CHI_COLORS[t], lw=2.0) for t, _ in CHI_TAGS]
    legend_below(fig, handles, [lab for _, lab in CHI_TAGS])
    finish(fig, out / f"batten2008_N{n}_mode_phases", dpi)


def figure_hessian(root: Path, n: int, out: Path, dpi: int) -> None:
    """Hessian spectra along flow 0 and eigenvalue counts over flows."""
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.6), constrained_layout=True)
    for col, (tag, label) in enumerate(CHI_TAGS):
        z = load(root / f"hessian_chi{tag}.npz")
        sel = [i for i, v in enumerate(z["samples"]) if v in SIX]
        lam, samples = z["lam"][0][sel], z["samples"][sel]
        ax = axes[col]
        for i in range(lam.shape[0]):
            c = SAMPLE_CMAP(0.35 + 0.65 * i / (lam.shape[0] - 1))
            ax.plot(np.arange(1, n + 1) / n, lam[i], color=c, lw=1.6,
                    label=rf"$t/T = {samples[i]:g}$")  # fmt: skip
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.axhline(0.0, color="0.6", lw=0.8, zorder=0)
        ax.set_title(f"({'ab'[col]}) {label}, flow 0", loc="left")
        ax.set_xlabel(r"Eigenvalue Rank, $i/N$")
        lam_all = z["lam"][:, sel]
        n_soft = (np.abs(lam_all) < 1e-6 * lam_all[..., -1:]).sum(axis=-1)
        idx = np.arange(samples.shape[0])
        n_neg = z["n_negative"][:, sel]
        axes[2].plot(idx, np.median(n_neg, axis=0), color=CHI_COLORS[tag],
                     lw=2.0, marker="o")  # fmt: skip
        axes[2].plot(idx, np.median(n_soft, axis=0), color=CHI_COLORS[tag], lw=2.0,
                     ls="--", marker="s")  # fmt: skip
    axes[0].set_ylabel(r"Hessian Eigenvalue, $\lambda_i$")
    axes[0].legend(frameon=False, fontsize=10, loc="upper left")
    ax = axes[2]
    ax.axhline(334, color="0.6", lw=0.8, zorder=0)
    ax.text(0.05, 352, r"$N - 2M = 334$", color="0.4", fontsize=11)
    ax.set_xticks(idx, [f"{s:g}" for s in samples])
    ax.set_title("(c) eigenvalue counts, median over flows", loc="left")
    ax.set_xlabel(r"Dimensionless Time, $t/T$")
    ax.set_ylabel("Number of Eigenvalues")
    handles = [Line2D([], [], color=CHI_COLORS[t], lw=2.0) for t, _ in CHI_TAGS]
    handles += [Line2D([], [], color="0.3", lw=2.0, marker="o"),
                Line2D([], [], color="0.3", lw=2.0, ls="--", marker="s")]  # fmt: skip
    labels = [lab for _, lab in CHI_TAGS]
    labels += [r"(c) $\lambda < 0$", r"(c) $|\lambda| < 10^{-6}\lambda_{\max}$"]
    legend_below(fig, handles, labels)
    finish(fig, out / f"batten2008_N{n}_hessian", dpi)


def pair_ratio(lam: np.ndarray, n: int, pairs: int) -> np.ndarray:
    """Positive eigenvalues (ascending) over k_p^2, p = ceil(i/2), i <= 2 pairs."""
    pos = np.sort(lam[lam > 1e-6 * lam[-1]])[: 2 * pairs]
    p = np.arange(pos.shape[0]) // 2 + 1
    out = np.full(2 * pairs, np.nan)
    out[: pos.shape[0]] = pos / (2.0 * np.pi * p / n) ** 2
    return out


def figure_hessian_detail(root: Path, n: int, out: Path, dpi: int) -> None:
    """Low end of the spectrum vs k_n^2, density-wave stiffness, negative modes."""
    fig, axes = plt.subplots(2, 4, figsize=(21.0, 9.0), constrained_layout=True)
    pairs = 60
    for row, (tag, label) in enumerate(CHI_TAGS):
        z = load(root / f"hessian_chi{tag}.npz")
        samples, lam = z["samples"], z["lam"]  # (flows, samples, N)
        u = np.median(z["u"], axis=0)
        late = [i for i, v in enumerate(samples) if v >= 1e-4]
        letter = "abcdefgh"[4 * row : 4 * row + 4]
        # (a) low end of the positive spectrum over k_p^2
        ax = axes[row, 0]
        pidx = np.arange(1, 2 * pairs + 1) / 2.0
        for c, i in enumerate(late):
            col = SAMPLE_CMAP(0.35 + 0.65 * c / max(len(late) - 1, 1))
            ax.plot(pidx, pair_ratio(lam[0, i], n, pairs), color=col, lw=1.4,
                    label=rf"$t/T = {samples[i]:g}$")  # fmt: skip
        end = np.array([pair_ratio(lam[b, -1], n, pairs) for b in range(lam.shape[0])])
        band(ax, pidx, end, FLOW_COLOR, "--")
        ax.axhline(1.0, color="0.6", lw=0.8, zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"({letter[0]}) {label}: low end, flow 0", loc="left", fontsize=13)
        ax.set_xlabel(r"Pair Index, $p = \lceil i/2 \rceil$")
        ax.set_ylabel(r"$\lambda_i / k_p^2$")
        ax.legend(frameon=False, fontsize=8, ncol=2,
                  loc=("center right", "upper right")[row])  # fmt: skip
        # (b) curvature along density waves, median and band over flows
        ax = axes[row, 1]
        m = z["dw_stiff"].shape[2]
        modes = (1, 2, 3, 4, 5, 10, round(0.5 * m), m)
        keep = u > 0
        for c, mode in enumerate(modes):
            k2 = (2.0 * np.pi * mode / n) ** 2
            ratio = z["dw_stiff"][:, :, mode - 1, :].mean(axis=-1) / k2
            col = plt.get_cmap("viridis")(c / (len(modes) - 1))
            q10, q50, q90 = np.percentile(ratio[:, keep], [10.0, 50.0, 90.0], axis=0)
            ax.fill_between(u[keep], q10, q90, color=col, alpha=0.15, lw=0)
            ax.plot(u[keep], q50, color=col, lw=1.8, label=rf"$n = {mode}$")
        ax.axhline(1.0, color="0.6", lw=0.8, zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-1)
        ax.set_title(f"({letter[1]}) {label}: along density waves", loc="left",
                     fontsize=13)  # fmt: skip
        ax.set_xlabel(r"Dimensionless Time, $t/T$")
        ax.set_ylabel(r"$\hat g_n^{\top} H \hat g_n / k_n^2$")
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
        # (c) negative directions: count and the most negative eigenvalue
        ax = axes[row, 2]
        band(ax, u[keep], z["n_negative"][:, keep].astype(float), "black", "-")
        ax.set_xscale("log")
        ax.set_title(f"({letter[2]}) {label}: negative curvature", loc="left",
                     fontsize=13)  # fmt: skip
        ax.set_xlabel(r"Dimensionless Time, $t/T$")
        ax.set_ylabel(r"Number with $\lambda < 0$")
        twin = ax.twinx()
        lam_min = np.maximum(-lam[:, :, 0], 1e-12)
        q10, q50, q90 = np.percentile(lam_min[:, keep], [10.0, 50.0, 90.0], axis=0)
        twin.fill_between(u[keep], q10, q90, color="#1f4e9c", alpha=0.15, lw=0)
        twin.plot(u[keep], q50, color="#1f4e9c", lw=2.0, ls="--")
        twin.set_yscale("log")
        twin.set_ylabel(r"$|\lambda_{\min}|$", color="#1f4e9c")
        # (d) wavelengths of the negative vs the stiffest eigenvectors
        ax = axes[row, 3]
        nn = np.median(z["n_negative"], axis=0)
        have = [i for i in range(samples.shape[0]) if nn[i] >= 5]
        show = sorted({have[0], have[len(have) // 2], have[-1]}) if have else []
        qm = np.arange(1, n // 2 + 1) / m
        for c, i in enumerate(show):
            col = SAMPLE_CMAP(0.4 + 0.6 * c / max(len(show) - 1, 1))
            neg = np.nanmedian(z["neg_power"][:, i], axis=0)
            pos = np.nanmedian(z["pos_power"][:, i], axis=0)
            ax.plot(qm, neg, color=col, lw=1.4, label=rf"$t/T = {samples[i]:g}$")
            ax.plot(qm, pos, color=col, lw=1.0, ls=":")
        if qm[-1] > 1.0:  # chi < 1/2: wavelengths the potential leaves free
            ax.axvspan(1.0, qm[-1], color="0.9", zorder=0, lw=0)
            ax.text(1.03, 0.95, "unconstrained", transform=ax.get_xaxis_transform(),
                    fontsize=10, color="0.4", va="top")  # fmt: skip
        ax.set_yscale("log")
        ax.set_title(f"({letter[3]}) {label}: wavelengths moved", loc="left",
                     fontsize=13)  # fmt: skip
        ax.set_xlabel(r"Wavevector, $q/K$ ($K = k_M$)")
        ax.set_ylabel("Mean Fourier Power")
        ax.legend(frameon=False, fontsize=9, loc="lower center")
    handles = [Line2D([], [], color=FLOW_COLOR, lw=2.0, ls="--"),
               Line2D([], [], color="black", lw=2.0),
               Line2D([], [], color="#1f4e9c", lw=2.0, ls="--"),
               Line2D([], [], color="0.3", lw=1.4),
               Line2D([], [], color="0.3", lw=1.0, ls=":")]  # fmt: skip
    labels = ["(a) end state, all flows", "(c) count", r"(c) $|\lambda_{\min}|$",
              "(d) negative eigenvectors", "(d) as many stiffest"]  # fmt: skip
    legend_below(fig, handles, labels)
    finish(fig, out / f"batten2008_N{n}_hessian_detail", dpi)


def figure_sections(root: Path, n: int, out: Path, dpi: int) -> None:
    """Valley cross-sections: flow direction x stiffest Hessian direction."""
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.4), constrained_layout=True)
    for row, (tag, label) in enumerate(CHI_TAGS):
        z = load(root / f"hessian_chi{tag}.npz")
        for j in range(3):
            ax = axes[row, j]
            a, b = z[f"sec{j}_a"], z[f"sec{j}_b"]
            logp = np.log10(np.maximum(z[f"sec{j}_phi"], FLOOR))
            cf = ax.contourf(a, b, logp, levels=24, cmap="Blues")
            ax.contour(a, b, logp, levels=12, colors="0.5", linewidths=0.4)
            p = z[f"sec{j}_path"]
            keep = ((p[:, 0] >= a[0]) & (p[:, 0] <= a[-1])
                    & (np.abs(p[:, 1]) <= b[-1]))  # fmt: skip
            ax.plot(p[keep, 0], p[keep, 1], color=FLOW_COLOR, lw=1.4)
            ax.plot(0.0, 0.0, "o", color="black", ms=4)
            fig.colorbar(cf, ax=ax, shrink=0.85)
            ax.set_title(f"({'abcdef'[3 * row + j]}) {label}, "
                         f"$t/T = {float(z[f'sec{j}_u']):.0e}$", loc="left",
                         fontsize=13)  # fmt: skip
            if row == 1:
                ax.set_xlabel(r"Along Flow, $\alpha$")
            if j == 0:
                ax.set_ylabel(r"Stiffest Direction, $\beta$")
    handles = [Line2D([], [], color=FLOW_COLOR, lw=1.5),
               Line2D([], [], ls="", marker="o", color="black", ms=4)]  # fmt: skip
    legend_below(fig, handles, ["Gradient Flow (projected)", r"$r(t)$"])
    fig.suptitle(
        r"$\log_{10}\Phi$ on the plane through $r(t)$ spanned by "
        r"$\tau = -\nabla\Phi/|\nabla\Phi|$ and the stiffest eigenvector"
    )
    finish(fig, out / f"batten2008_N{n}_sections", dpi)


def figure_floor(root: Path, n: int, out: Path, dpi: int) -> None:
    """Flat-slice vs valley-floor contours, and the floor beneath the path."""
    data = {
        (tag, view): load(root / f"floor_chi{tag}_{view}.npz")
        for tag, _ in CHI_TAGS
        for view in ("se", "pca")
        if (root / f"floor_chi{tag}_{view}.npz").exists()
    }
    tags = [t for t, _ in CHI_TAGS if any(k[0] == t for k in data)]
    views = {t: load(root / f"views_chi{t}.npz") for t in tags}  # flow 0 slices
    fig, axes = plt.subplots(len(tags), 5, figsize=(22.0, 4.2 * len(tags)),
                             constrained_layout=True, squeeze=False)  # fmt: skip
    for row, tag in enumerate(tags):
        label = dict(CHI_TAGS)[tag]
        present = [v for v in ("se", "pca") if (tag, v) in data]
        hi = max(np.log10(np.nanmax(views[tag][f"{v}_phi"][0])) for v in ("se", "pca"))
        lows = [np.nanmin(views[tag][f"{v}_phi"][0]) for v in ("se", "pca")]
        lows += [np.nanmin(data[(tag, v)][f"{v}_phi"]) for v in present]
        lo = max(np.log10(FLOOR), np.floor(np.log10(max(min(lows), FLOOR))))
        levels = np.linspace(lo, hi, 23)  # per row: chi = 2/3 floors are O(1)
        for c, (view, kind) in enumerate([(v, k) for v in ("se", "pca")
                                          for k in ("flat", "phi")]):  # fmt: skip
            ax = axes[row, c]
            letter = "abcdefghij"[5 * row + c]
            if (tag, view) not in data:
                ax.set_axis_off()
                continue
            z = data[(tag, view)]
            if kind == "flat":  # the 101^2 slice of the landscape figures
                a, b, phi = (views[tag][f"{view}_{k}"][0] for k in ("a", "b", "phi"))
            else:
                a, b, phi = z[f"{view}_a"], z[f"{view}_b"], z[f"{view}_phi"]
            logp = np.log10(np.maximum(phi, FLOOR))
            cf = ax.contourf(a, b, logp, levels=levels, cmap="Blues", extend="both")
            if kind == "phi":
                aa, bb = np.meshgrid(a, b)
                ax.plot(aa, bb, ls="", marker=".", color="0.4", ms=2)
            ab = z[f"{view}_ab"]
            ax.plot(ab[:, 0], ab[:, 1], color=FLOW_COLOR, lw=1.2)
            name = "flat slice" if kind == "flat" else "valley floor"
            title = f"({letter}) {label}, {VIEW_TITLES[view]}: {name}"
            ax.set_title(title, loc="left", fontsize=12)
            ax.set_xlabel(VIEW_LABELS[view][0], fontsize=12)
            ax.set_ylabel(VIEW_LABELS[view][1], fontsize=12)
        fig.colorbar(cf, ax=axes[row, :4], label=r"$\log_{10}\Phi$", shrink=0.9)
        ax = axes[row, 4]
        z0 = data[(tag, present[0])]
        u, true = z0["path_u"], z0["path_true"]
        ok = u > 0
        ax.loglog(u[ok], np.maximum(true[ok], FLOOR), color="black", lw=2.0)
        for view, mk in (("se", "o"), ("pca", "s")):
            if (tag, view) in data:
                z = data[(tag, view)]
                floor = np.maximum(z[f"{view}_path_phi"][ok], FLOOR)
                ax.loglog(u[ok], floor, ls="", marker=mk, color="#1f4e9c", ms=5,
                          mfc="none")  # fmt: skip
        ax.set_title(f"({'abcdefghij'[5 * row + 4]}) {label}: beneath the path",
                     loc="left", fontsize=12)  # fmt: skip
        ax.set_xlabel(r"Dimensionless Time, $t/T$", fontsize=12)
        ax.set_ylabel(r"Energy, $\Phi$", fontsize=12)
    blue = "#1f4e9c"
    handles = [
        Line2D([], [], color=FLOW_COLOR, lw=1.5),
        Line2D([], [], ls="", marker=".", color="0.4", ms=6),
        Line2D([], [], color="black", lw=2.0),
        Line2D([], [], ls="", marker="o", color=blue, mfc="none"),
        Line2D([], [], ls="", marker="s", color=blue, mfc="none"),
    ]
    labels = [
        "Gradient Flow (projected)",
        "Relaxed Grid Point",
        r"True $\Phi(r(t))$",
        "Floor beneath $r(t)$, start-end",
        "Floor beneath $r(t)$, PCA",
    ]
    legend_below(fig, handles, labels)
    finish(fig, out / f"batten2008_N{n}_floor", dpi)


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--root", type=Path, default=Path("experiments/trajectories/batten2008")
    )
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("experiments/figures"))
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    root = args.root / f"N{args.n}"
    figure_modes(root, args.n, args.out, args.dpi)
    figure_mode_planes(root, args.n, args.out, args.dpi)
    figure_mode_planes_full(root, args.n, args.out, args.dpi)
    figure_mode_phases(root, args.n, args.out, args.dpi)
    figure_hessian(root, args.n, args.out, args.dpi)
    figure_hessian_detail(root, args.n, args.out, args.dpi)
    figure_sections(root, args.n, args.out, args.dpi)
    if list(root.glob("floor_chi*_*.npz")):
        figure_floor(root, args.n, args.out, args.dpi)
    print("Done.", flush=True)
