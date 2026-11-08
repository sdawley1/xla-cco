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
The Hessian figures use the energy as the clock (batten2008_hessian.py
samples every quarter decade of Phi; the x axes run from high to low Phi, so
time runs left to right):

- batten2008_N<N>_hessian_detail: (a, e) positive eigenvalues over k_p^2,
  p = ceil(i/2), flow 0 every two decades below Phi = 1 and all flows at
  the end; (b, f) curvature along the density waves over k_n^2; (c, g)
  number of negative eigenvalues (|lambda| > 1e-9 lambda_max) and
  |lambda_min|, with that cut; (d, h) mean Fourier power
  of the negative eigenvectors (solid) and of as many of the stiffest ones
  (dotted).
- batten2008_N<N>_hessian: (a, b) Hessian eigenvalues of flow 0 at eight
  energies (symlog); (c) negative (below -1e-9 and -1e-13 lambda_max) and
  near-zero (|lambda| < 1e-6 and 1e-9 lambda_max) eigenvalue counts against
  Phi, median and 10-90 % band over flows. Near the manifold the soft
  eigenvalues scale as sqrt(Phi), so where these counts change is set by
  the cuts.
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
from typing import Dict, List, Tuple

# ----- numerics -----
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batten2008_plot import (  # noqa: E402  (also applies the house rcParams)
    CHI_COLORS,
    CHI_TAGS,
    FLOW_COLOR,
    NAMES,
    VIEW_LABELS,
    VIEW_TITLES,
    add_common_arguments,
    configure,
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
ZERO_REL = 1e-9  # |lambda| below this x lambda_max is zero (batten2008_hessian.py)
TIGHT_REL = 1e-13  # a negative-eigenvalue cut just above eigh round-off
SPECTRA = (3.0, 1.0, -1.0, -3.0, -6.0, -10.0, -14.0, -20.0)  # log10 Phi, (a, b)
PHI_LABEL = r"Energy, $\Phi$"

# ============================================================
# HELPERS
# ============================================================


def phi_axis(ax: plt.Axes) -> None:
    """Log energy axis, high to low (flow time runs left to right)."""
    ax.set_xscale("log")
    if not ax.xaxis_inverted():
        ax.invert_xaxis()
    ax.set_xlabel(PHI_LABEL)


def level_label(level: float, phi0: float) -> str:
    """Legend text of a log10 Phi level; levels above the start are the start."""
    if level >= np.log10(phi0):
        return r"start, $\Phi_0$"
    return rf"$\Phi = 10^{{{level:g}}}$"


def in_range(z: Dict[str, np.ndarray]) -> np.ndarray:
    """Levels every flow reaches from above (none falls on its start frame)."""
    start = np.log10(z["phi"][:, 0]).min()
    return np.asarray(z["samples"]) <= start


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


def kmag_of(z: Dict[str, np.ndarray], n: int) -> np.ndarray:
    """|k| of the k-set rows (stored by the analysis; 1D fallback 2 pi n / L)."""
    if "kmag" in z:
        return np.asarray(z["kmag"])
    m = int(z["m"]) if "m" in z else int(np.asarray(z["dw_stiff"]).shape[2])
    return 2.0 * np.pi * np.arange(1, m + 1) / n


def shells(kmag: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Distinct |k| values (a shell each) and each row's shell index."""
    key = np.round(kmag / kmag[0], 9)
    values, inv = np.unique(key, return_inverse=True)
    return values * kmag[0], inv


def shell_mean(arr: np.ndarray, inv: np.ndarray) -> np.ndarray:
    """Average the last axis (rows of the k-set) within each |k| shell."""
    n_shell = int(inv.max()) + 1
    out = np.zeros(arr.shape[:-1] + (n_shell,))
    for sh in range(n_shell):
        out[..., sh] = np.nanmean(arr[..., inv == sh], axis=-1)
    return out


def mode_label(z: Dict[str, np.ndarray], row: int) -> str:
    """'$n = 5$' in 1D, '$n = (1, 0)$' for a 2D wavevector (1-based row)."""
    if "nvec" not in z or np.asarray(z["nvec"]).shape[1] == 1:
        return f"$n = {row}$"
    v = np.asarray(z["nvec"])[row - 1].astype(int)
    return "$n = (" + ", ".join(str(c) for c in v) + ")$"


def figure_modes(root: Path, n: int, out: Path, dpi: int) -> None:
    """Mode-resolved energy along the flows (collective coordinates)."""
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0), constrained_layout=True)
    guide: List[float] = []
    start: List[float] = []
    for col, (tag, label) in enumerate(CHI_TAGS):
        z = load(root / f"modes_chi{tag}.npz")
        m, kmag = int(z["m"]), kmag_of(z, n)
        ks, inv = shells(kmag)
        x = ks / kmag[0]
        ax = axes[0, col]
        logs = np.log10(np.maximum(shell_mean(z["sk_grid"][0, 1:], inv), FLOOR)).T
        mesh = ax.pcolormesh(z["grid"][1:], x, logs, cmap="Blues_r", vmin=-20, vmax=1,
                             shading="nearest", rasterized=True)  # fmt: skip
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"({'ab'[col]}) {label} (M = {m}), flow 0", loc="left")
        ax.set_xlabel(r"Dimensionless Time, $t/T$")
        ax.set_ylabel(r"Wavenumber, $|\mathbf{k}|/k_1$")
        for e, ls in EPS_STYLES.items():
            tr = shell_mean(z["t_relax"][:, :, e], inv)
            band(axes[1, 0], x, tr, CHI_COLORS[tag], ls)
            if e == 0 and not guide:
                guide.append(float(np.median(tr[:, 0])))
        with np.load(root / f"flows_chi{tag}.npz") as f:
            t_all, cnt = f["traj_t"], f["traj_count"]
        s1 = z["rho_re"][:, :, 0] ** 2 + z["rho_im"][:, :, 0] ** 2
        cross = np.flatnonzero(s1[0, : int(cnt[0])] < 1e-3)
        start.append(float(t_all[0, cross[0]]) if cross.size else 0.0)
        for b in range(s1.shape[0]):
            c = int(cnt[b])
            axes[1, 1].semilogy(t_all[b, :c], np.maximum(s1[b, :c], FLOOR),
                                color=CHI_COLORS[tag], lw=0.6, alpha=0.6)  # fmt: skip
        k1 = float(kmag[0])
    cbar_label = r"$\log_{10} S(\mathbf{k}, t)$ (shell mean)"
    fig.colorbar(mesh, ax=axes[0, :], label=cbar_label, shrink=0.9)
    ax = axes[1, 0]
    xs = np.geomspace(1.0, float(x[-1]), 50)
    ax.plot(xs, guide[0] * xs**-2.0, color="0.6", lw=0.8, ls=":")
    ax.text(xs[20], guide[0] * xs[20] ** -2.0 / 10.0, r"$\propto |\mathbf{k}|^{-2}$",
            color="0.4")  # fmt: skip
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-7, 1.5)
    ax.set_title(r"(c) relaxation time per $|\mathbf{k}|$ shell", loc="left")
    ax.set_xlabel(r"Wavenumber, $|\mathbf{k}|/k_1$")
    ax.set_ylabel(r"Last Time Above $\epsilon$, $t_k/T$")
    ax = axes[1, 1]
    # guide from where flow 0 of the first chi crosses S = 1e-3 down to 1e-18
    t0 = start[0]
    t_ref = np.array([t0, t0 + np.log(1e15) / (2.0 * k1**2)])
    ax.semilogy(t_ref, 1e-3 * np.exp(-2.0 * k1**2 * (t_ref - t0)) / 3.0,
                color="0.3", lw=2.0, ls="-.")  # fmt: skip
    ax.set_ylim(bottom=1e-22)
    ax.set_title(r"(d) longest wavelength, $|\mathbf{k}| = k_1$", loc="left")
    ax.set_xlabel(r"Time, $t$")
    ax.set_ylabel(r"Structure Factor, $S(\mathbf{k}_1, t)$")
    handles = [Line2D([], [], color=CHI_COLORS[t], lw=2.0) for t, _ in CHI_TAGS]
    handles += [Line2D([], [], color="0.3", lw=2.0, ls=ls) for ls in ("-", "--")]
    handles += [Line2D([], [], color="0.3", lw=2.0, ls="-.")]
    labels = [lab for _, lab in CHI_TAGS]
    labels += [r"(c) $\epsilon = 10^{-2}$", r"(c) $\epsilon = 10^{-8}$",
               r"(d) $e^{-2k_1^2 t}$"]  # fmt: skip
    legend_below(fig, handles, labels)
    finish(fig, out / f"{NAMES['prefix']}_N{n}_modes", dpi)


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
    legend_below(fig, handles, ["Flow 0", "Other Flows", "Start"])


def figure_mode_planes(root: Path, n: int, out: Path, dpi: int) -> None:
    """rho(k_n) in polar form; circles are exact per-mode energy contours."""
    fig, axes = plt.subplots(2, 4, figsize=(15.0, 8.0), constrained_layout=True,
                             subplot_kw={"projection": "polar"})  # fmt: skip
    for row, (tag, label) in enumerate(CHI_TAGS):
        z = load(root / f"modes_chi{tag}.npz")
        picks = z["picks"].tolist()
        for col, mode in enumerate((picks[0], picks[1], picks[-3], picks[-1])):
            title = f"({'abcdefgh'[4 * row + col]}) {label}, {mode_label(z, mode)}"
            polar_panel(axes[row, col], z, mode, title)
    planes_legend(fig)
    finish(fig, out / f"{NAMES['prefix']}_N{n}_mode_planes", dpi)


def figure_mode_planes_full(root: Path, n: int, out: Path, dpi: int) -> None:
    """Phase planes for the five smallest |k| rows, row 10 and |k| = K/4..K."""
    fig, axes = plt.subplots(4, 5, figsize=(18.0, 15.5), constrained_layout=True,
                             subplot_kw={"projection": "polar"})  # fmt: skip
    letters = "abcdefghijklmnopqrst"
    for block, (tag, label) in enumerate(CHI_TAGS):
        z = load(root / f"modes_chi{tag}.npz")
        picks = z["picks"].tolist()
        assert len(picks) == 10, "expected rows 1-5, 10 and four |k| fractions"
        for r, modes in enumerate((picks[:5], picks[5:])):
            for col, mode in enumerate(modes):
                k = 10 * block + 5 * r + col
                title = f"({letters[k]}) {label}, {mode_label(z, mode)}"
                polar_panel(axes[2 * block + r, col], z, mode, title)
    planes_legend(fig)
    finish(fig, out / f"{NAMES['prefix']}_N{n}_mode_planes_full", dpi)


def late_exponent(z: Dict[str, np.ndarray], mode: int) -> np.ndarray:
    """Per flow: slope of log S(k_mode) against log S(k_row1) late in the flow.

    Fit over frames with 1e-12 < S_1 < 1e-2 and S_mode > 1e-19 (NaN if fewer
    than 10). Independent linear decay gives (|k_mode|/k_1)^2; rho_mode slaved
    linearly to rho_1 gives 1, as its n-th harmonic n.
    """
    picks = z["picks"].tolist()
    j1, jn = 0, picks.index(mode)
    out = np.full(z["rho_re"].shape[0], np.nan)
    for b in range(out.shape[0]):
        s1 = z["rho_re"][b, :, j1] ** 2 + z["rho_im"][b, :, j1] ** 2
        sn = z["rho_re"][b, :, jn] ** 2 + z["rho_im"][b, :, jn] ** 2
        ok = np.isfinite(s1) & (s1 > 1e-12) & (s1 < 1e-2) & (sn > 1e-19)
        if ok.sum() >= 10:
            out[b] = np.polyfit(np.log(s1[ok]), np.log(sn[ok]), 1)[0]
    return out


def figure_mode_phases(root: Path, n: int, out: Path, dpi: int) -> None:
    """Phase turns, lock-in S and late-time exponents per |k| shell."""
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.8), constrained_layout=True)
    marks = ("o", "s", "^")
    for c_i, (tag, _label) in enumerate(CHI_TAGS):
        z = load(root / f"modes_chi{tag}.npz")
        kmag = kmag_of(z, n)
        ks, inv = shells(kmag)
        x = ks / kmag[0]
        c = CHI_COLORS[tag]
        band(axes[0], x, shell_mean(np.maximum(z["turns"], 1e-3), inv), c, "-")
        lock = np.log10(np.maximum(z["s_lock"], FLOOR))
        band(axes[1], x, shell_mean(lock, inv), c, "-")
        picks = z["picks"].tolist()[1:6]
        p = np.array([late_exponent(z, k) for k in picks])  # (modes, flows)
        enough = np.isfinite(p).sum(axis=1) >= 10  # resolved in >= 10 flows
        p = np.where(enough[:, None], p, np.nan)
        q10, q50, q90 = np.nanpercentile(p, [10.0, 50.0, 90.0], axis=1)
        kx = kmag[np.array(picks) - 1] / kmag[0] * (1.0 + 0.03 * c_i)  # dodge
        err = [q50 - q10, q90 - q50]
        axes[2].errorbar(kx, q50, yerr=err, color=c, lw=0, elinewidth=1.5,
                         marker=marks[c_i], ms=7, capsize=3)  # fmt: skip
    ax = axes[0]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("(a) total phase winding", loc="left")
    ax.set_xlabel(r"Wavenumber, $|\mathbf{k}|/k_1$")
    ax.set_ylabel(r"Turns, $\sum|\Delta\arg\rho|/2\pi$")
    ax = axes[1]
    ax.set_xscale("log")
    ax.set_title("(b) where the phase locks", loc="left")
    ax.set_xlabel(r"Wavenumber, $|\mathbf{k}|/k_1$")
    ax.set_ylabel(r"$\log_{10} S(\mathbf{k})$ at Lock-In")
    ax = axes[2]
    kk = np.geomspace(1.0, 10.0, 30)
    ax.plot(kk, kk**2, color="0.6", lw=0.8, ls=":")
    ax.plot(kk, kk, color="0.6", lw=0.8, ls="--")
    ax.axhline(1.0, color="0.6", lw=0.8, ls="-.")
    ax.text(5.0, 40.0, r"$(|\mathbf{k}|/k_1)^2$: own decay", color="0.4", fontsize=11)
    ax.text(5.5, 7.5, r"$|\mathbf{k}|/k_1$: harmonics", color="0.4", fontsize=11)
    ax.text(
        5.5, 1.2, r"$1$: $\rho_{\mathbf{k}}\propto\rho_1$", color="0.4", fontsize=11
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(r"(c) late time: $S_{\mathbf{k}} \propto S_1^{\,p}$", loc="left")
    ax.set_xlabel(r"Wavenumber, $|\mathbf{k}|/k_1$ (rows 2-5, 10)")
    ax.set_ylabel(r"Exponent, $p$")
    handles = [Line2D([], [], color=CHI_COLORS[t], lw=2.0) for t, _ in CHI_TAGS]
    legend_below(fig, handles, [lab for _, lab in CHI_TAGS])
    finish(fig, out / f"{NAMES['prefix']}_N{n}_mode_phases", dpi)


def figure_hessian(root: Path, n: int, out: Path, dpi: int) -> None:
    """Hessian spectra along flow 0 and eigenvalue counts over flows, vs Phi."""
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.6), constrained_layout=True)
    zero_lines = []
    for col, (tag, label) in enumerate(CHI_TAGS):
        z = load(root / f"hessian_chi{tag}.npz")
        levels = np.asarray(z["samples"])
        sel = [int(np.argmin(np.abs(levels - v))) for v in SPECTRA]
        lam = z["lam"][0][sel]
        nd = lam.shape[-1]
        zero_lines.append((nd - 2 * kmag_of(z, n).shape[0], CHI_COLORS[tag]))
        ax = axes[col]
        for i, j in enumerate(sel):
            c = SAMPLE_CMAP(0.3 + 0.7 * i / (len(sel) - 1))
            ax.plot(np.arange(1, nd + 1) / nd, lam[i], color=c, lw=1.6,
                    label=level_label(levels[j], z["phi"][0, 0]))  # fmt: skip
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.axhline(0.0, color="0.6", lw=0.8, zorder=0)
        ax.set_title(f"({'ab'[col]}) {label}, flow 0", loc="left")
        ax.set_xlabel(r"Eigenvalue Rank, $i/(Nd)$")
        keep = in_range(z)
        x = 10.0 ** levels[keep]
        lam_all = z["lam"][:, keep]
        n_soft = (np.abs(lam_all) < 1e-6 * lam_all[..., -1:]).sum(axis=-1)
        band(axes[2], x, z["n_negative"][:, keep].astype(float), CHI_COLORS[tag], "-")
        band(axes[2], x, n_soft.astype(float), CHI_COLORS[tag], "--")
        band(axes[2], x, z["n_zero"][:, keep].astype(float), CHI_COLORS[tag], ":")
        n_neg_tight = (lam_all < -TIGHT_REL * lam_all[..., -1:]).sum(axis=-1)
        band(axes[2], x, n_neg_tight.astype(float), CHI_COLORS[tag], "-.")
    axes[0].set_ylabel(r"Hessian Eigenvalue, $\lambda_i$")
    axes[0].legend(frameon=False, fontsize=10, loc="upper left")
    ax = axes[2]
    for value, color in zero_lines:
        if value > 0:
            ax.axhline(value, color=color, lw=0.8, ls=":", zorder=0)
            ax.text(0.72, value * 1.03, rf"$Nd - 2M = {value}$", color=color,
                    fontsize=11, transform=ax.get_yaxis_transform())  # fmt: skip
    top = max([v for v, _ in zero_lines] + [1])
    ax.set_ylim(top=max(ax.get_ylim()[1], 1.12 * top))  # room for the labels
    phi_axis(ax)
    ax.set_title("(c) eigenvalue counts over flows", loc="left")
    ax.set_ylabel("Number of Eigenvalues")
    handles = [Line2D([], [], color=CHI_COLORS[t], lw=2.0) for t, _ in CHI_TAGS]
    handles += [Line2D([], [], color="0.3", lw=2.0),
                Line2D([], [], color="0.3", lw=2.0, ls="--"),
                Line2D([], [], color="0.3", lw=2.0, ls=":"),
                Line2D([], [], color="0.3", lw=2.0, ls="-.")]  # fmt: skip
    labels = [lab for _, lab in CHI_TAGS]
    labels += [r"(c) $\lambda < -10^{-9}\lambda_{\max}$",
               r"(c) $|\lambda| < 10^{-6}\lambda_{\max}$",
               r"(c) $|\lambda| < 10^{-9}\lambda_{\max}$",
               r"(c) $\lambda < -10^{-13}\lambda_{\max}$"]  # fmt: skip
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.0),
               frameon=False)  # fmt: skip
    finish(fig, out / f"{NAMES['prefix']}_N{n}_hessian", dpi)


def pair_ratio(lam: np.ndarray, kmag: np.ndarray, pairs: int) -> np.ndarray:
    """Positive eigenvalues (ascending) over |k_p|^2, p = ceil(i/2), i <= 2 pairs."""
    pos = np.sort(lam[lam > 1e-6 * lam[-1]])[: 2 * pairs]
    p = np.arange(pos.shape[0]) // 2  # 0-based k-set row
    out = np.full(2 * pairs, np.nan)
    ok = p < kmag.shape[0]
    out[: pos.shape[0]][ok] = pos[ok] / kmag[p[ok]] ** 2
    return out


def figure_hessian_detail(root: Path, n: int, out: Path, dpi: int) -> None:
    """Low end of the spectrum vs |k|^2, density-wave stiffness, negative modes."""
    fig, axes = plt.subplots(2, 4, figsize=(21.0, 9.0), constrained_layout=True)
    pairs = 60
    for row, (tag, label) in enumerate(CHI_TAGS):
        z = load(root / f"hessian_chi{tag}.npz")
        samples, lam = z["samples"], z["lam"]  # (flows, levels, N d)
        kmag = kmag_of(z, n)
        keep = in_range(z)
        u = 10.0 ** np.asarray(samples)  # the clock: Phi at each level
        late = [i for i, v in enumerate(samples) if v <= 0.0 and v % 2 == 0]
        letter = "abcdefgh"[4 * row : 4 * row + 4]
        # (a) low end of the positive spectrum over |k_p|^2
        ax = axes[row, 0]
        pidx = np.arange(1, 2 * pairs + 1) / 2.0
        for c, i in enumerate(late):
            col = SAMPLE_CMAP(0.35 + 0.65 * c / max(len(late) - 1, 1))
            ax.plot(pidx, pair_ratio(lam[0, i], kmag, pairs), color=col, lw=1.4,
                    label=level_label(samples[i], z["phi"][0, 0]))  # fmt: skip
        end = np.array(
            [pair_ratio(lam[b, -1], kmag, pairs) for b in range(lam.shape[0])]
        )
        band(ax, pidx, end, FLOW_COLOR, "--")
        ax.axhline(1.0, color="0.6", lw=0.8, zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"({letter[0]}) {label}: low end, flow 0", loc="left", fontsize=13)
        ax.set_xlabel(r"Pair Index, $p = \lceil i/2 \rceil$")
        ax.set_ylabel(r"$\lambda_i / |\mathbf{k}_p|^2$")
        if row == 0:  # same samples in every row: one legend
            ax.legend(frameon=False, fontsize=8, ncol=2, loc="center right")
        # (b) curvature along density waves, median and band over flows
        ax = axes[row, 1]
        m = z["dw_stiff"].shape[2]
        rows = sorted(
            {1, 2, 3, 4, 5, 10, int(np.argmin(np.abs(kmag - 0.5 * kmag[-1]))) + 1, m}
        )
        for c, mode in enumerate(rows):
            ratio = z["dw_stiff"][:, :, mode - 1, :].mean(axis=-1) / kmag[mode - 1] ** 2
            col = plt.get_cmap("viridis")(c / (len(rows) - 1))
            q10, q50, q90 = np.percentile(ratio[:, keep], [10.0, 50.0, 90.0], axis=0)
            ax.fill_between(u[keep], q10, q90, color=col, alpha=0.15, lw=0)
            ax.plot(u[keep], q50, color=col, lw=1.8, label=mode_label(z, mode))
        ax.axhline(1.0, color="0.6", lw=0.8, zorder=0)
        phi_axis(ax)
        ax.set_yscale("symlog", linthresh=1e-1)
        ax.set_title(f"({letter[1]}) {label}: along density waves", loc="left",
                     fontsize=13)  # fmt: skip
        ax.set_ylabel(
            r"$\hat g_{\mathbf{k}}^{\top} H \hat g_{\mathbf{k}} / |\mathbf{k}|^2$"
        )
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
        # (c) negative directions: count and the most negative eigenvalue
        ax = axes[row, 2]
        band(ax, u[keep], z["n_negative"][:, keep].astype(float), "black", "-")
        phi_axis(ax)
        ax.set_title(f"({letter[2]}) {label}: negative curvature", loc="left",
                     fontsize=13)  # fmt: skip
        ax.set_ylabel(r"Number with $\lambda < 0$")
        twin = ax.twinx()
        lam_min = np.maximum(-lam[:, :, 0], 1e-12)
        q10, q50, q90 = np.percentile(lam_min[:, keep], [10.0, 50.0, 90.0], axis=0)
        twin.fill_between(u[keep], q10, q90, color="#1f4e9c", alpha=0.15, lw=0)
        twin.plot(u[keep], q50, color="#1f4e9c", lw=2.0, ls="--")
        zero_cut = ZERO_REL * np.median(lam[:, keep, -1], axis=0)
        twin.plot(u[keep], zero_cut, color="#1f4e9c", lw=1.0, ls=":")
        twin.set_yscale("log")
        twin.set_ylabel(r"$|\lambda_{\min}|$", color="#1f4e9c")
        # (d) wavelengths of the negative vs the stiffest eigenvectors
        ax = axes[row, 3]
        nn = np.median(z["n_negative"], axis=0)
        have = [i for i in range(samples.shape[0]) if nn[i] >= 5 and keep[i]]
        show = sorted({have[0], have[len(have) // 2], have[-1]}) if have else []
        if "qmag" in z:
            qs, qinv = shells(np.asarray(z["qmag"]))
        else:  # 1D files written before qmag was stored
            qs = 2.0 * np.pi * np.arange(1, n // 2 + 1) / n
            qinv = np.arange(qs.shape[0])
        qk = qs / kmag[-1]
        for c, i in enumerate(show):
            col = SAMPLE_CMAP(0.4 + 0.6 * c / max(len(show) - 1, 1))
            neg = shell_mean(np.nanmedian(z["neg_power"][:, i], axis=0), qinv)
            pos = shell_mean(np.nanmedian(z["pos_power"][:, i], axis=0), qinv)
            ax.plot(qk, neg, color=col, lw=1.4,
                    label=level_label(samples[i], z["phi"][0, 0]))  # fmt: skip
            ax.plot(qk, pos, color=col, lw=1.0, ls=":")
        if qk[-1] > 1.0:  # wavelengths the potential leaves free
            ax.axvspan(1.0, qk[-1], color="0.9", zorder=0, lw=0)
            ax.text(1.03, 0.95, "unconstrained", transform=ax.get_xaxis_transform(),
                    fontsize=10, color="0.4", va="top")  # fmt: skip
        ax.set_yscale("log")
        ax.set_title(f"({letter[3]}) {label}: wavelengths moved", loc="left",
                     fontsize=13)  # fmt: skip
        ax.set_xlabel(r"Wavenumber, $|\mathbf{q}|/K$")
        ax.set_ylabel(r"Mean Fourier Power ($|\mathbf{q}|$ shell)")
        ax.legend(frameon=False, fontsize=9, loc="lower center")
    handles = [Line2D([], [], color=FLOW_COLOR, lw=2.0, ls="--"),
               Line2D([], [], color="black", lw=2.0),
               Line2D([], [], color="#1f4e9c", lw=2.0, ls="--"),
               Line2D([], [], color="#1f4e9c", lw=1.0, ls=":"),
               Line2D([], [], color="0.3", lw=1.4),
               Line2D([], [], color="0.3", lw=1.0, ls=":")]  # fmt: skip
    labels = ["(a) end state, all flows", "(c) count", r"(c) $|\lambda_{\min}|$",
              r"(c) zero cut $10^{-9}\lambda_{\max}$",
              "(d) negative eigenvectors", "(d) as many stiffest"]  # fmt: skip
    legend_below(fig, handles, labels)
    finish(fig, out / f"{NAMES['prefix']}_N{n}_hessian_detail", dpi)


def figure_sections(root: Path, n: int, out: Path, dpi: int) -> None:
    """Valley cross-sections: flow direction x stiffest Hessian direction."""
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.4), constrained_layout=True)
    for row, (tag, label) in enumerate(CHI_TAGS):
        z = load(root / f"hessian_chi{tag}.npz")
        for j in range(3):
            ax = axes[row, j]
            a, b = z[f"sec{j}_a"], z[f"sec{j}_b"]
            # tiny late-time sections: scale the axes, name the scale in the title
            e = int(np.floor(np.log10(max(np.abs(a).max(), np.abs(b).max()))))
            e = e if e < -2 else 0
            a, b = a / 10.0**e, b / 10.0**e
            logp = np.log10(np.maximum(z[f"sec{j}_phi"], FLOOR))
            cf = ax.contourf(a, b, logp, levels=24, cmap="Blues")
            ax.contour(a, b, logp, levels=12, colors="0.5", linewidths=0.4)
            p = z[f"sec{j}_path"] / 10.0**e
            keep = ((p[:, 0] >= a[0]) & (p[:, 0] <= a[-1])
                    & (np.abs(p[:, 1]) <= b[-1]))  # fmt: skip
            ax.plot(p[keep, 0], p[keep, 1], color=FLOW_COLOR, lw=1.4)
            ax.plot(0.0, 0.0, "o", color="black", ms=4)
            fig.colorbar(cf, ax=ax, shrink=0.85)
            phi0 = float(z[f"sec{j}_phi0"])
            ax.set_title(f"({'abcdef'[3 * row + j]}) {label}, "
                         f"$\\Phi = 10^{{{np.log10(phi0):.0f}}}$"
                         + (f", axes $\\times 10^{{{e}}}$" if e else ""),
                         loc="left", fontsize=13)  # fmt: skip
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
    finish(fig, out / f"{NAMES['prefix']}_N{n}_sections", dpi)


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
    finish(fig, out / f"{NAMES['prefix']}_N{n}_floor", dpi)


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
    add_common_arguments(parser)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configure(args.chi, args.prefix, args.dim)
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
