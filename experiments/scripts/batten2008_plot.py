#!/usr/bin/env python3

"""
Figures for the Batten, Stillinger and Torquato (2008) appendix landscapes.

- <out>/batten2008_N3.{png,pdf}: contour plots of Phi(x2, x3) (x1 = 0) for
  chi = 1/3 and 2/3 (cf. Fig. 15, D = 0), with the gradient flows from
  Poisson starts drawn in relative coordinates (x2 - x1, x3 - x1) mod 1.
- <out>/batten2008_N<N>_chi<tag>_<view>.{png,pdf}: for N > 3, one panel per
  flow on the start-end, PCA or two-particle slice (batten2008_slices.py),
  with log10 Phi contours and the projected flow.
- <out>/batten2008_energy.{png,pdf}: Phi along every flow vs t/T.

Phi is xla_cco's stealthy energy (1/V) sum_k |rho(k)|^2; the paper's Eq.
(A1) with D = 0 is (|rho|^2 / 2)^2, a monotone function of it, so contours
coincide (with relabelled levels) and the flow lines are the same curves.

Needs matplotlib: `uv sync --dev --extra plot --locked` once, then
`uv run --no-sync python -I experiments/scripts/batten2008_plot.py`.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import argparse
import json
from pathlib import Path
from typing import Dict, List

# ----- plotting -----
import matplotlib

# ----- numerics -----
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 14,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": False,
    }
)

CHI_TAGS = (("1-3", r"$\chi = 1/3$"), ("2-3", r"$\chi = 2/3$"))
CHI_COLORS = {"1-3": "black", "2-3": "#1f4e9c"}
FLOW_COLOR = "#c0392b"
VIEW_LABELS = {
    "se": (r"Net Displacement, $a$", r"Largest Excursion, $b$"),
    "pca": (r"First Principal Axis, $p_1$", r"Second Principal Axis, $p_2$"),
    "pair": (r"Position, $x_i$", r"Position, $x_j$"),
}
VIEW_TITLES = {
    "se": "start-end plane",
    "pca": "PCA plane",
    "pair": "two-particle slice",
}

# ============================================================
# HELPERS
# ============================================================


def load(path: Path) -> Dict[str, np.ndarray]:
    """Read an npz into a plain dict."""
    with np.load(path) as npz:
        return {k: npz[k] for k in npz.files}


def periodic_segments(rel: np.ndarray) -> List[np.ndarray]:
    """Split a path in the unit square (mod 1) wherever it wraps around."""
    wrapped = np.mod(rel, 1.0)
    jumps = np.flatnonzero(np.any(np.abs(np.diff(wrapped, axis=0)) > 0.5, axis=1))
    return np.split(wrapped, jumps + 1)


def finish(fig: plt.Figure, out: Path, dpi: int) -> None:
    """Save PNG and PDF."""
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png/.pdf", flush=True)


# ============================================================
# FIGURES
# ============================================================


def figure_n3(root: Path, out: Path, dpi: int) -> None:
    """Contour plots of Phi(x2, x3) with the N = 3 flows overlaid."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), constrained_layout=True)
    for ax, (tag, label) in zip(axes, CHI_TAGS, strict=True):
        land = load(root / f"landscape_chi{tag}.npz")
        flows = load(root / f"flows_chi{tag}.npz")
        x, phi = land["x"], land["phi"]
        cf = ax.contourf(x, x, phi, levels=24, cmap="Blues")
        ax.contour(x, x, phi, levels=12, colors="0.5", linewidths=0.4)
        for b in range(flows["s0"].shape[0]):
            cnt = int(flows["traj_count"][b])
            s = flows["traj_s"][b, :cnt, :, 0]
            rel = s[:, 1:] - s[:, :1]
            for seg in periodic_segments(rel):
                ax.plot(seg[:, 0], seg[:, 1], color=FLOW_COLOR, lw=0.9)
            start, end = np.mod(rel[0], 1.0), np.mod(rel[-1], 1.0)
            ax.plot(*start, "o", color="black", ms=3)
            ax.plot(*end, "*", color="white", mec="black", mew=0.6, ms=9)
        fig.colorbar(cf, ax=ax, label=r"Energy, $\Phi$", shrink=0.9)
        ax.set_title(f"({'ab'[CHI_TAGS.index((tag, label))]}) {label}", loc="left")
        ax.set_xlabel(r"Position, $x_2$")
        ax.set_ylabel(r"Position, $x_3$")
        ax.set_aspect("equal")
    handles = [
        Line2D([], [], color=FLOW_COLOR, lw=1.5),
        Line2D([], [], ls="", marker="o", color="black", ms=4),
        Line2D([], [], ls="", marker="*", color="white", mec="black", ms=10),
    ]
    fig.legend(
        handles,
        ["Gradient Flow", "Poisson Start", "End (ground state)"],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
    )
    finish(fig, out / "batten2008_N3", dpi)


def figure_views(
    root: Path, n: int, tag: str, label: str, view: str, out: Path, dpi: int
) -> None:
    """One panel per flow on a 2D slice, log10 Phi contours plus the path."""
    v = load(root / f"views_chi{tag}.npz")
    phi = v[f"{view}_phi"]
    n_flow = phi.shape[0]
    ncol = 5
    nrow = int(np.ceil(n_flow / ncol))
    logp = np.log10(np.maximum(phi, 1e-30))
    lo, hi = np.percentile(logp, 1.0), np.max(logp)
    levels = np.linspace(lo, hi, 25)
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(3.0 * ncol, 2.8 * nrow),
        constrained_layout=True,
        squeeze=False,
    )
    for k, ax in enumerate(axes.flat):
        if k >= n_flow:
            ax.set_axis_off()
            continue
        a, b = v[f"{view}_a"][k], v[f"{view}_b"][k]
        cf = ax.contourf(a, b, np.clip(logp[k], lo, hi), levels=levels, cmap="Blues")
        path = v[f"{view}_path"][k]
        path = path[np.all(np.isfinite(path), axis=1)]
        ax.plot(path[:, 0], path[:, 1], color=FLOW_COLOR, lw=1.0)
        ax.plot(*path[0], "o", color="black", ms=3)
        ax.plot(*path[-1], "*", color="white", mec="black", mew=0.6, ms=8)
        ax.set_title(f"flow {k}", loc="left", fontsize=11)
        ax.tick_params(labelsize=9)
        if k % ncol == 0:
            ax.set_ylabel(VIEW_LABELS[view][1], fontsize=11)
        if k >= (nrow - 1) * ncol:
            ax.set_xlabel(VIEW_LABELS[view][0], fontsize=11)
    fig.colorbar(cf, ax=axes, label=r"$\log_{10}\Phi$", shrink=0.6)
    fig.suptitle(f"N = {n}, {label}: {VIEW_TITLES[view]}")
    finish(fig, out / f"batten2008_N{n}_chi{tag}_{view}", dpi)


def figure_energy(roots: Dict[int, Path], out: Path, dpi: int) -> None:
    """Phi along every flow vs dimensionless time t/T (log-log)."""
    fig, axes = plt.subplots(
        1,
        len(roots),
        figsize=(5.5 * len(roots), 4.2),
        constrained_layout=True,
        squeeze=False,
    )
    for ax, (n, root) in zip(axes.flat, sorted(roots.items()), strict=True):
        for tag, _label in CHI_TAGS:
            path = root / f"flows_chi{tag}.npz"
            if not path.exists():
                continue
            f = load(path)
            for b in range(f["s0"].shape[0]):
                cnt = int(f["traj_count"][b])
                t, e = f["traj_t"][b, 1:cnt], f["traj_energy"][b, 1:cnt]
                ax.loglog(t / t[-1], e, color=CHI_COLORS[tag], lw=0.6, alpha=0.5)
        ax.set_title(f"N = {n}", loc="left")
        ax.set_xlabel(r"Dimensionless Time, $t/T$")
        ax.set_ylabel(r"Energy, $\Phi(t)$")
    handles = [Line2D([], [], color=CHI_COLORS[t], lw=2.0) for t, _ in CHI_TAGS]
    fig.legend(
        handles,
        [lab for _, lab in CHI_TAGS],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
    )
    finish(fig, out / "batten2008_energy", dpi)


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--root", type=Path, default=Path("experiments/trajectories/batten2008")
    )
    parser.add_argument("--out", type=Path, default=Path("experiments/figures"))
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    roots = {int(p.name[1:]): p for p in args.root.glob("N*") if p.is_dir()}
    if 3 in roots:
        figure_n3(roots[3], args.out, args.dpi)
    for n, root in sorted(roots.items()):
        if n == 3:
            continue
        for tag, label in CHI_TAGS:
            if (root / f"views_chi{tag}.npz").exists():
                meta = json.loads((root / f"flows_chi{tag}.json").read_text())
                full = rf"{label} (M = {meta['M']})"
                for view in ("se", "pca", "pair"):
                    figure_views(root, n, tag, full, view, args.out, args.dpi)
    figure_energy(roots, args.out, args.dpi)
    print("Done.", flush=True)
