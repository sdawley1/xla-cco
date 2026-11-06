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
    r(inf) (dashed);
(d) relative gaps 1 - W/p at matching frames.
Panels on t/T use log-log axes (log-linear for counts). Lines are ensemble
means; shading spans the 10-90 % range.

Run with the plottime interpreter:
/home/sd2402/torquato-scratch-sd2402/conda_envs/plottime/bin/python

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

COLORS = {"POS": "black", "GAU": "#1f4e9c"}
TIME_LABEL = r"Dimensionless Time, $t/T$"

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
) -> None:
    """Plot the ensemble mean of ``key`` vs t/T (t/T > 0, positive values)."""
    x = data[f"{e}/grid"]
    m, lo, hi = (data[f"{e}/{key}_{s}"] for s in ("mean", "q10", "q90"))
    if log10:
        m, lo, hi = 10.0**m, 10.0**lo, 10.0**hi
    keep = (x > 0.0) & (m > 0.0)
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
    labels = list(ens)
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
    data: Dict[str, np.ndarray], ens: List[str], out: str, dpi: int
) -> None:
    """Wasserstein analogue of the proximity-metric figure (Sinkhorn curves)."""
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
        curve(axes[0, 1], data, e, "sk_w0_dist")
        curve(axes[0, 1], data, e, "sk_w_inf", ls="--", shade=False)
        curve(axes[1, 0], data, e, "log10_energy", log10=True)
        phi = 10.0 ** data[f"{e}/log10_energy_mean"]
        for key, ls in (("sk_w0_dist", "-"), ("sk_w_inf", "--")):
            d2 = data[f"{e}/{key}_mean"] ** 2
            keep = d2 > 0.0
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
    for e in ens:
        curve(axes[0, 0], data, e, "w0_dist")
        curve(axes[0, 0], data, e, "p0_dist", ls="--", shade=False)
        curve(axes[0, 1], data, e, "w_inf")
        curve(axes[0, 1], data, e, "q_inf", ls="--", shade=False)
        curve(axes[1, 0], data, e, "n_re0", shade=False)
        curve(axes[1, 0], data, e, "n_re_inf", ls="--", shade=False)
        x = data[f"{e}/grid"]
        for num, den, ls in (("w0_dist", "p0_dist", "-"), ("w_inf", "q_inf", "--")):
            w0 = data[f"{e}/{num}_mean"]
            p0 = data[f"{e}/{den}_mean"]
            keep = (
                (x > 0.0)
                & (p0 > 0.0)
                & (np.abs(1.0 - w0 / np.where(p0 > 0, p0, 1)) > 0)
            )
            axes[1, 1].plot(
                x[keep],
                np.abs(1.0 - w0[keep] / p0[keep]),
                color=COLORS[e],
                lw=2.0,
                ls=ls,
            )
    axes[0, 0].set_ylabel(r"Distance, $W_0(\infty)-W_0(t)$, $p_0(\infty)-p_0(t)$")
    axes[0, 1].set_ylabel(r"Distance to $r(\infty)$, $W_\infty(t)$, $q_\infty(t)$")
    axes[1, 0].set_ylabel(r"Re-matched Particles, $n_{\mathrm{re}}$")
    axes[1, 1].set_ylabel(r"Relative Gap, $|1 - W/p|$")
    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_xlabel(TIME_LABEL)
        ax.set_xlim(right=1.0)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 1]):
        ax.set_yscale("log")
    legend(fig, ens, {"Wasserstein": "-", "Labelled": "--"})
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

    # ----- output -----
    parser.add_argument("--out", type=Path, default=Path("experiments/figures/chi0.1"))
    parser.add_argument("--dpi", type=int, default=600)

    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with np.load(args.summary) as npz:
        data = {k: npz[k] for k in npz.files}
    mirror_figure(data, args.ensembles, f"{args.out}_w2", args.dpi)
    overlay_figure(data, args.ensembles, f"{args.out}_w2_vs_p0", args.dpi)
    print("Done.", flush=True)
