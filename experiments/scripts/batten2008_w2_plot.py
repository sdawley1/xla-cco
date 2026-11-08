#!/usr/bin/env python3

"""
Wasserstein trajectories over the Batten, Stillinger & Torquato (2008) flows.

Reads batten2008_wasserstein.py outputs and the landscape data of
batten2008_slices.py, and writes to <out>:

- batten2008_w2_N3.png: Phi(x2, x3) contours (x1 = 0) per chi, with
  the gradient flows and (top) the flows relabelled by their optimal matching
  to r(0) and to r(inf), (bottom) the W_2 geodesics r(0) -> sigma(r(inf)).
- batten2008_w2_N<N>_chi<tag>_<view>.png: the same overlay for every
  N = 1000 flow on the start-end, PCA and two-particle planes.
- batten2008_w2_N<N>_offplane.png: each curve's distance from the
  plane it is drawn on, relative to |r(inf) - r(0)|.
- batten2008_w2_geodesic_energy.png: Phi along the geodesic vs s and
  along the flow vs normalized arclength.

Needs matplotlib: `uv sync --dev --extra plot --locked` once, then
`uv run --no-sync python -I experiments/scripts/batten2008_w2_plot.py`.

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
from typing import Dict, List, Tuple

# ----- numerics -----
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batten2008_plot import (  # noqa: E402  (also applies the house rcParams)
    CHI_COLORS,
    CHI_TAGS,
    NAMES,
    VIEW_LABELS,
    VIEW_TITLES,
    add_common_arguments,
    configure,
    finish,
    load,
    periodic_segments,
    plt,
)
from matplotlib.lines import Line2D  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

FLOW = dict(color="#c0392b", lw=1.6, ls="-")
RELABEL = {
    "w0": dict(color="black", lw=1.0, ls=(0, (4, 3))),
    "winf": dict(color="#1f4e9c", lw=1.0, ls=(0, (1, 2))),
}
GEO = dict(color="black", lw=1.2, ls="-")
CURVE_LABELS = {
    "flow": "Gradient Flow",
    "w0": r"Relabelled to $r(0)$",
    "winf": r"Relabelled to $r(\infty)$",
    "geo": r"$W_2$ Geodesic",
}

# ============================================================
# HELPERS
# ============================================================


def legend_handles(names: Tuple[str, ...]) -> Tuple[List[Line2D], List[str]]:
    """Legend entries for the requested curve types plus the end markers."""
    style = {"flow": FLOW, **RELABEL, "geo": GEO}
    handles = [Line2D([], [], **style[n]) for n in names]
    handles += [
        Line2D([], [], ls="", marker="o", color="black", ms=4),
        Line2D([], [], ls="", marker="*", color="white", mec="black", ms=10),
    ]
    return handles, [CURVE_LABELS[n] for n in names] + [r"$r(0)$", r"$r(\infty)$"]


def valid(path: np.ndarray) -> np.ndarray:
    """Drop NaN padding rows."""
    out: np.ndarray = path[np.all(np.isfinite(path), axis=-1)]
    return out


def draw_wrapped(ax: plt.Axes, rel: np.ndarray, style: Dict) -> None:
    """Draw an unwrapped N = 3 relative path mod 1, split where it wraps."""
    for seg in periodic_segments(valid(rel)):
        ax.plot(seg[:, 0], seg[:, 1], **style)


def ends(ax: plt.Axes, path: np.ndarray, wrap: bool) -> None:
    """Start dot and end star."""
    p = np.mod(path, 1.0) if wrap else path
    ax.plot(*p[0], "o", color="black", ms=3, zorder=5)
    ax.plot(*p[-1], "*", color="white", mec="black", mew=0.6, ms=8, zorder=5)


# ============================================================
# FIGURES
# ============================================================


def figure_n3(root: Path, out: Path, dpi: int) -> None:
    """2 x 2: rows relabelled flows / geodesics, columns chi = 1/3, 2/3."""
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 10.0), constrained_layout=True)
    for col, (tag, label) in enumerate(CHI_TAGS):
        land = load(root / f"landscape_chi{tag}.npz")
        w = load(root / f"wasserstein_chi{tag}.npz")
        for row in range(2):
            ax = axes[row, col]
            cf = ax.contourf(land["x"], land["x"], land["phi"], levels=24, cmap="Blues")
            ax.contour(
                land["x"],
                land["x"],
                land["phi"],
                levels=12,
                colors="0.5",
                linewidths=0.4,
            )
            for b in range(w["rel_flow"].shape[0]):
                draw_wrapped(ax, w["rel_flow"][b], FLOW)
                if row == 0:
                    for name, style in RELABEL.items():
                        draw_wrapped(ax, w[f"rel_{name}"][b], style)
                else:
                    draw_wrapped(ax, w["rel_geo"][b], GEO)
                ends(ax, valid(w["rel_flow"][b]), wrap=True)
            ax.set_aspect("equal")
            ax.set_xlabel(r"Position, $x_2$")
            ax.set_ylabel(r"Position, $x_3$")
            ax.set_title(f"({'abcd'[2 * row + col]}) {label}", loc="left")
        fig.colorbar(cf, ax=axes[:, col], label=r"Energy, $\Phi$", shrink=0.6)
    handles, labels = legend_handles(("flow", "w0", "winf", "geo"))
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
    )
    finish(fig, out / f"{NAMES['prefix']}_w2_N3", dpi)


def figure_views(
    root: Path, n: int, tag: str, label: str, view: str, out: Path, dpi: int
) -> None:
    """One panel per flow: log10 Phi contours, flow, relabelled flows, geodesic."""
    v, w = load(root / f"views_chi{tag}.npz"), load(root / f"wasserstein_chi{tag}.npz")
    phi = v[f"{view}_phi"]
    n_flow, ncol = phi.shape[0], 5
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
        flow = valid(w[f"{view}_flow"][k])
        ax.plot(flow[:, 0], flow[:, 1], **FLOW)
        for name, style in RELABEL.items():
            p = valid(w[f"{view}_{name}"][k])
            ax.plot(p[:, 0], p[:, 1], **style)
        geo = valid(w[f"{view}_geo"][k])
        ax.plot(geo[:, 0], geo[:, 1], **GEO)
        ends(ax, flow, wrap=False)
        ax.set_xlim(a[0], a[-1])
        ax.set_ylim(b[0], b[-1])
        ax.set_title(f"flow {k}", loc="left", fontsize=11)
        ax.tick_params(labelsize=9)
        if k % ncol == 0:
            ax.set_ylabel(VIEW_LABELS[view][1], fontsize=11)
        if k >= (nrow - 1) * ncol:
            ax.set_xlabel(VIEW_LABELS[view][0], fontsize=11)
    fig.colorbar(cf, ax=axes, label=r"$\log_{10}\Phi$", shrink=0.6)
    fig.suptitle(f"N = {n}, {label}: {VIEW_TITLES[view]}")
    handles, labels = legend_handles(("flow", "w0", "winf", "geo"))
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6,
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
    )
    finish(fig, out / f"{NAMES['prefix']}_w2_N{n}_chi{tag}_{view}", dpi)


def median_band(
    ax: plt.Axes,
    xs: List[np.ndarray],
    ys: List[np.ndarray],
    color: str,
    ls: object,
    grid: np.ndarray,
) -> None:
    """Median and 10-90 % band of curves y(x) resampled on ``grid``."""
    arr = np.stack([np.interp(grid, x, y) for x, y in zip(xs, ys, strict=True)])
    q10, q50, q90 = np.quantile(arr, [0.1, 0.5, 0.9], axis=0)
    ax.fill_between(grid, q10, q90, color=color, alpha=0.15, lw=0)
    ax.plot(grid, q50, color=color, ls=ls, lw=2.0)


def figure_offplane(root: Path, n: int, out: Path, dpi: int) -> None:
    """Off-plane distance / |r(inf) - r(0)| along each curve, per view and chi."""
    fig, axes = plt.subplots(
        2, 3, figsize=(14.0, 7.6), constrained_layout=True, sharex=True
    )
    grid = np.linspace(0.0, 1.0, 201)
    for row, (tag, label) in enumerate(CHI_TAGS):
        w = load(root / f"wasserstein_chi{tag}.npz")
        scale = w["p0"][
            np.arange(w["p0"].shape[0]), np.sum(np.isfinite(w["p0"]), axis=1) - 1
        ]
        for col, view in enumerate(("se", "pca", "pair")):
            ax = axes[row, col]
            for name, style in (("flow", FLOW), *RELABEL.items()):
                xs, ys = [], []
                for b in range(scale.shape[0]):
                    ln = w["length"][b][np.isfinite(w["length"][b])]
                    off = w[f"{view}_{name}_off"][b][: ln.shape[0]]
                    xs.append(ln / ln[-1])
                    ys.append(off / scale[b])
                median_band(ax, xs, ys, style["color"], style["ls"], grid)
            ys = [w[f"{view}_geo_off"][b] / scale[b] for b in range(scale.shape[0])]
            median_band(ax, [w["geo_s"]] * len(ys), ys, "0.45", "-.", grid)
            ax.set_title(
                f"({'abcdef'[3 * row + col]}) {label}, {VIEW_TITLES[view]}",
                loc="left",
                fontsize=13,
            )
            if row == 1:
                ax.set_xlabel(r"Fraction of Path, $\ell/\ell(\infty)$, $s$")
            if col == 0:
                ax.set_ylabel(r"Off-Plane Distance, $\delta/|r(\infty)-r(0)|$")
    handles = [
        Line2D([], [], **FLOW),
        *(Line2D([], [], **s) for s in RELABEL.values()),
        Line2D([], [], color="0.45", ls="-.", lw=2.0),
    ]
    labels = [CURVE_LABELS[k] for k in ("flow", "w0", "winf", "geo")]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
    )
    finish(fig, out / f"{NAMES['prefix']}_w2_N{n}_offplane", dpi)


def figure_geodesic_energy(roots: Dict[int, Path], out: Path, dpi: int) -> None:
    """Phi along the W_2 geodesic vs s and along the flow vs ell / ell(inf)."""
    fig, axes = plt.subplots(
        1,
        len(roots),
        figsize=(5.8 * len(roots), 4.4),
        constrained_layout=True,
        squeeze=False,
    )
    grid = np.linspace(0.0, 1.0, 201)
    for ax, (n, root) in zip(axes.flat, sorted(roots.items()), strict=True):
        for tag, _ in CHI_TAGS:
            w = load(root / f"wasserstein_chi{tag}.npz")
            color = CHI_COLORS[tag]
            geo = [np.log10(np.maximum(g, 1e-30)) for g in w["geo_energy"]]
            median_band(ax, [w["geo_s"]] * len(geo), geo, color, "-", grid)
            xs, ys = [], []
            for b in range(w["energy"].shape[0]):
                ok = np.isfinite(w["length"][b])
                ln = w["length"][b][ok]
                xs.append(ln / ln[-1])
                ys.append(np.log10(np.maximum(w["energy"][b][ok], 1e-30)))
            median_band(ax, xs, ys, color, "--", grid)
        ax.set_title(f"N = {n}", loc="left")
        ax.set_xlabel(r"Fraction of Path, $s$, $\ell/\ell(\infty)$")
        ax.set_ylabel(r"Energy, $\log_{10}\Phi$")
    handles = [Line2D([], [], color=CHI_COLORS[t], lw=2.0) for t, _ in CHI_TAGS]
    handles += [Line2D([], [], color="0.3", lw=2.0, ls=ls) for ls in ("-", "--")]
    labels = [lab for _, lab in CHI_TAGS] + [r"$W_2$ Geodesic", "Gradient Flow"]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
    )
    finish(fig, out / f"{NAMES['prefix']}_w2_geodesic_energy", dpi)


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
    add_common_arguments(parser)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configure(args.chi, args.prefix, args.dim)

    roots = {
        int(p.name[1:]): p
        for p in args.root.glob("N*")
        if p.is_dir() and p.name[1:].isdigit() and any(p.glob("wasserstein_chi*.npz"))
    }
    if 3 in roots:
        figure_n3(roots[3], args.out, args.dpi)
    for n, root in sorted(roots.items()):
        if n == 3:
            continue
        for tag, label in CHI_TAGS:
            meta = json.loads((root / f"flows_chi{tag}.json").read_text())
            full = rf"{label} (M = {meta['M']})"
            for view in ("se", "pca", "pair"):
                figure_views(root, n, tag, full, view, args.out, args.dpi)
        figure_offplane(root, n, args.out, args.dpi)
    figure_geodesic_energy(roots, args.out, args.dpi)
    print("Done.", flush=True)
