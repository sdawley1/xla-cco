#!/usr/bin/env python3

"""
Figures comparing the periodic Wasserstein-2 distance with the proximity metric.

Mirror figure (<out>_w2.{png,pdf}), the Wasserstein analogue of
plot_figure.py:
(a) W_2(r(0), r(inf)) vs p_0(inf), one point per trajectory, with y = x;
(b) W_0(inf) - W_0(t) (solid) and W_inf(t) = W_2(r(t), r(inf)) (dashed)
    vs t/T (Sinkhorn, every stored frame);
(c) Phi(t) vs t/T;
(d) Phi vs [W_0(inf) - W_0(t)]^2 (solid) and vs W_inf(t)^2 (dashed).

Overlay figure (<out>_w2_vs_p0.{png,pdf}), exact assignment on ~200
log-spaced frames per trajectory:
(a) W_0(inf) - W_0(t) (solid) and p_0(inf) - p_0(t) (dashed);
(b) W_inf(t) (solid) and q_inf(t) = rho^{1/2}|r(t) - r(inf)| (dashed);
(c) particles re-matched by the optimal map, relative to r(0) (solid) and
    r(inf) (dotted);
(d) relative gaps |1 - W/p| for W_0 vs p_0 (solid) and W_inf vs q_inf
    (dotted).
Panels on t/T use log-log axes (log-linear for counts). Lines are ensemble
means; shading spans the 10-90 % range.

Needs matplotlib: `uv sync --dev --extra plot --locked` once, then
`uv run --no-sync python -I experiments/scripts/<this script>` (any Python
with numpy and matplotlib also works; xla_cco is not imported).

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import argparse
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

COLORS = {
    "POS": "black",
    "GAU": "#1f4e9c",
    "chi1-3": "black",  # Batten 2008 flows (batten2008_wasserstein.py)
    "chi2-3": "#1f4e9c",
    "chi1-10": "black",  # 2D gradient flows (flows2d.py)
    "chi2-5": "#1f4e9c",
}
LABELS = {
    "chi1-3": r"$\chi = 1/3$",
    "chi2-3": r"$\chi = 2/3$",
    "chi1-10": r"$\chi = 0.1$",
    "chi2-5": r"$\chi = 0.4$",
}
TIME_LABEL = r"Dimensionless Time, $t/T$"
FLOOR = 1e-12  # relative round-off floor for distances and gaps

# ============================================================
# HELPERS
# ============================================================


def curve(
    ax: plt.Axes,
    data: Dict[str, np.ndarray],
    e: str,
    key: str,
    ls: str = "-",
    log10: bool = False,
    shade: bool = True,
    positive: bool = True,
) -> None:
    """Plot the ensemble mean of ``key`` vs t/T.

    Drops t/T = 0 and values below ``FLOOR`` times the curve maximum, i.e.
    round-off in differences of equal distances at t/T -> 1 (energies, which
    are not differences, are kept down to zero).
    """
    x = data[f"{e}/grid"]
    m, lo, hi = (data[f"{e}/{key}_{s}"] for s in ("mean", "q10", "q90"))
    if log10:
        m, lo, hi = 10.0**m, 10.0**lo, 10.0**hi
    floor = 0.0 if log10 else FLOOR * float(np.max(m))
    keep = (x > 0.0) & ((m > floor) if positive else np.isfinite(m))
    if shade:
        ax.fill_between(
            x[keep],
            np.clip(lo[keep], m[keep] * 1e-3, None),
            hi[keep],
            color=COLORS[e],
            alpha=0.15,
            lw=0,
        )
    ax.plot(x[keep], m[keep], color=COLORS[e], lw=2.0, ls=ls)


def legend(fig: plt.Figure, ens: List[str], styles: Dict[str, str]) -> None:
    """Shared frameless legend: one entry per ensemble and per line style."""
    handles = [Line2D([], [], color=COLORS[e], lw=2.0) for e in ens]
    labels = [LABELS.get(e, e) for e in ens]
    for name, ls in styles.items():
        handles.append(Line2D([], [], color="0.3", lw=2.0, ls=ls))
        labels.append(name)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
    )


def finish(fig: plt.Figure, axes: np.ndarray, out: str, dpi: int) -> None:
    """Tag panels, lay out and save PNG and PDF."""
    for ax, tag in zip(axes.flat, "abcd", strict=True):
        ax.set_title(f"({tag})", loc="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=dpi, bbox_inches="tight")
    print(f"wrote {out}.png/.pdf", flush=True)


# ============================================================
# FIGURES
# ============================================================


def mirror_figure(
    data: Dict[str, np.ndarray],
    ens: List[str],
    out: str,
    dpi: int,
    prefix: str = "sk_",
) -> None:
    """Wasserstein analogue of the proximity-metric figure.

    ``prefix`` selects the distance curves: "sk_" for the full-resolution
    Sinkhorn curves (PRE trajectories), "" for exact values at every frame.
    """
    assert ens, "need at least one ensemble"
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.6))
    ax = axes[0, 0]
    top = max(float(data[f"{e}/p0_inf"].max()) for e in ens) * 1.05
    ax.plot([0.0, top], [0.0, top], color="0.6", lw=0.8, zorder=0)
    for e in ens:
        ax.plot(
            data[f"{e}/p0_inf"],
            data[f"{e}/w0_inf"],
            "o",
            ms=3.5,
            color=COLORS[e],
            alpha=0.8,
        )
    ax.set_xlabel(r"Proximity Metric, $p_0(\infty)$")
    ax.set_ylabel(r"Wasserstein Distance, $W_0(\infty)$")
    ax.set_xlim(0.0, top)
    ax.set_ylim(0.0, top)
    for e in ens:
        curve(axes[0, 1], data, e, f"{prefix}w0_dist")
        curve(axes[0, 1], data, e, f"{prefix}w_inf", ls="--", shade=False)
        curve(axes[1, 0], data, e, "log10_energy", log10=True)
        phi = 10.0 ** data[f"{e}/log10_energy_mean"]
        for key, ls in ((f"{prefix}w0_dist", "-"), (f"{prefix}w_inf", "--")):
            d2 = data[f"{e}/{key}_mean"] ** 2
            keep = d2 > FLOOR**2 * float(np.max(d2))
            axes[1, 1].loglog(d2[keep], phi[keep], color=COLORS[e], lw=2.0, ls=ls)
    axes[0, 1].set_ylabel(r"Distance, $W_0(\infty)-W_0(t)$, $W_\infty(t)$")
    axes[1, 0].set_ylabel(r"Energy, $\Phi(t)$")
    axes[1, 1].set_xlabel(r"Squared Distance, $[W_0(\infty)-W_0(t)]^2$, $W_\infty^2$")
    axes[1, 1].set_ylabel(r"Energy, $\Phi(t)$")
    for ax in (axes[0, 1], axes[1, 0]):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(TIME_LABEL)
        ax.set_xlim(right=1.0)
    legend(fig, ens, {r"$W_0(\infty)-W_0(t)$": "-", r"$W_\infty(t)$": "--"})
    finish(fig, axes, out, dpi)


def overlay_figure(
    data: Dict[str, np.ndarray], ens: List[str], out: str, dpi: int
) -> None:
    """Exact W_2 against the labelled proximity metrics on the same frames."""
    assert ens, "need at least one ensemble"
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.6))
    gap_drawn = False
    for e in ens:
        curve(axes[0, 0], data, e, "w0_dist")
        curve(axes[0, 0], data, e, "p0_dist", ls="--", shade=False)
        curve(axes[0, 1], data, e, "w_inf")
        curve(axes[0, 1], data, e, "q_inf", ls="--", shade=False)
        curve(axes[1, 0], data, e, "n_re0", shade=False, positive=False)
        curve(axes[1, 0], data, e, "n_re_inf", ls=":", shade=False, positive=False)
        x = data[f"{e}/grid"]
        for num, den, ls in (("w0_dist", "p0_dist", "-"), ("w_inf", "q_inf", ":")):
            w = data[f"{e}/{num}_mean"]
            p = data[f"{e}/{den}_mean"]
            gap = np.abs(1.0 - w / np.where(p > 0.0, p, 1.0))
            keep = (x > 0.0) & (p > 0.0) & (gap > FLOOR)
            # With the identity as the optimal matching at every frame, W and
            # p are the same sums: any gap left is round-off of differences
            # of nearly equal distances, not a physical difference.
            identity = all(
                float(np.max(data[f"{e}/{k}_q90"])) == 0.0
                for k in ("n_re0", "n_re_inf")
            )
            keep &= not identity
            gap_drawn |= bool(keep.any())
            axes[1, 1].plot(x[keep], gap[keep], color=COLORS[e], lw=2.0, ls=ls)
    axes[0, 0].set_ylabel(r"Distance, $W_0(\infty)-W_0(t)$")
    axes[0, 1].set_ylabel(r"Distance to $r(\infty)$, $W_\infty(t)$")
    axes[1, 0].set_ylabel(r"Re-matched Particles, $n_{\mathrm{re}}$")
    axes[1, 1].set_ylabel(r"Relative Gap, $|1 - W/p|$")
    t_min = float(np.min(data[f"{ens[0]}/grid"][data[f"{ens[0]}/grid"] > 0.0]))
    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_xlabel(TIME_LABEL)
        ax.set_xlim(t_min, 1.0)
    for ax in (axes[0, 0], axes[0, 1]):
        ax.set_yscale("log")
    # Panels (c), (d): line style distinguishes the reference, not the metric.
    refs = [
        Line2D([], [], color="0.3", lw=2.0, ls="-"),
        Line2D([], [], color="0.3", lw=2.0, ls=":"),
    ]
    axes[1, 0].legend(refs, [r"vs $r(0)$", r"vs $r(\infty)$"], frameon=False)
    if gap_drawn:
        axes[1, 1].set_yscale("log")
        axes[1, 1].legend(
            refs, [r"$W_0$ vs $p_0$", r"$W_\infty$ vs $q_\infty$"], frameon=False
        )
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "$W = p$ at every frame\n(optimal matching is the identity)",
            transform=axes[1, 1].transAxes,
            ha="center",
            va="center",
        )
        axes[1, 1].set_yticks([])
    legend(fig, ens, {r"Wasserstein, $W$": "-", r"Labelled, $p_0$, $q_\infty$": "--"})
    finish(fig, axes, out, dpi)


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])

    # ----- inputs -----
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("experiments/trajectories/chi0.1/wasserstein/summary.npz"),
    )
    parser.add_argument("--ensembles", nargs="+", default=["POS", "GAU"])
    parser.add_argument(
        "--exact",
        action="store_true",
        help="summary has exact W at every frame (no Sinkhorn curves)",
    )

    # ----- output -----
    parser.add_argument("--out", type=Path, default=Path("experiments/figures/chi0.1"))
    parser.add_argument("--dpi", type=int, default=600)

    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with np.load(args.summary) as npz:
        data = {k: npz[k] for k in npz.files}
    prefix = "" if args.exact else "sk_"
    mirror_figure(data, args.ensembles, f"{args.out}_w2", args.dpi, prefix)
    overlay_figure(data, args.ensembles, f"{args.out}_w2_vs_p0", args.dpi)
    print("Done.", flush=True)
