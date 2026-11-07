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
from typing import List

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


def figure_mode_planes(root: Path, n: int, out: Path, dpi: int) -> None:
    """rho(k_n) in polar form; circles are exact per-mode energy contours."""
    fig, axes = plt.subplots(2, 4, figsize=(15.0, 8.0), constrained_layout=True,
                             subplot_kw={"projection": "polar"})  # fmt: skip
    for row, (tag, label) in enumerate(CHI_TAGS):
        z = load(root / f"modes_chi{tag}.npz")
        picks = z["picks"].tolist()
        m = int(z["m"])
        for col, mode in enumerate((1, 2, picks[len(picks) // 2], m)):
            ax = axes[row, col]
            j = picks.index(mode)
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
            ax.set_title(f"({'abcdefgh'[4 * row + col]}) {label}, $n = {mode}$",
                         loc="left", fontsize=13)  # fmt: skip
    handles = [Line2D([], [], color=FLOW_COLOR, lw=1.5),
               Line2D([], [], color="0.55", lw=1.0),
               Line2D([], [], ls="", marker="o", color="black", ms=4)]  # fmt: skip
    legend_below(fig, handles, ["Flow 0", "Other Flows", "Poisson Start"])
    finish(fig, out / f"batten2008_N{n}_mode_planes", dpi)


def figure_hessian(root: Path, n: int, out: Path, dpi: int) -> None:
    """Hessian spectra along flow 0 and eigenvalue counts over flows."""
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.6), constrained_layout=True)
    for col, (tag, label) in enumerate(CHI_TAGS):
        z = load(root / f"hessian_chi{tag}.npz")
        lam, samples = z["lam"][0], z["samples"]
        ax = axes[col]
        for i in range(lam.shape[0]):
            c = SAMPLE_CMAP(0.35 + 0.65 * i / (lam.shape[0] - 1))
            ax.plot(np.arange(1, n + 1) / n, lam[i], color=c, lw=1.6,
                    label=rf"$t/T = {samples[i]:g}$")  # fmt: skip
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.axhline(0.0, color="0.6", lw=0.8, zorder=0)
        ax.set_title(f"({'ab'[col]}) {label}, flow 0", loc="left")
        ax.set_xlabel(r"Eigenvalue Rank, $i/N$")
        lam_all = z["lam"]
        n_soft = (np.abs(lam_all) < 1e-6 * lam_all[..., -1:]).sum(axis=-1)
        idx = np.arange(samples.shape[0])
        axes[2].plot(idx, np.median(z["n_negative"], axis=0), color=CHI_COLORS[tag],
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
    figure_hessian(root, args.n, args.out, args.dpi)
    figure_sections(root, args.n, args.out, args.dpi)
    if list(root.glob("floor_chi*_*.npz")):
        figure_floor(root, args.n, args.out, args.dpi)
    print("Done.", flush=True)
