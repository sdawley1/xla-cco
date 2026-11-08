#!/usr/bin/env python3

"""
Figure for the chi = 0.1 POS/GAU trajectory experiment.

Panels, for comparison with Dawley & Torquato, PRE 112, 045310 (2025):
(a) initial-condition S(k) with the chi = 0.1 exclusion region shaded
(cf. Fig. 2(a)); (b) configurational distance p_0(inf) - p_0(t) vs t/T
(cf. Fig. 3(a)); (c) Phi(t) vs t/T (cf. Fig. 3(b)); (d) Phi vs
[p_0(inf) - p_0(t)]^2 (cf. Fig. 3(c)). Panels (b)-(d) use log-log axes.
Lines are ensemble means; shading
spans the 10-90 % range. Reads summary.npz from analyze_trajectories.py.

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
LABELS = {"POS": "POS", "GAU": "GAU"}

# ============================================================
# PANELS
# ============================================================


def panel_sk(ax: plt.Axes, data: Dict[str, np.ndarray], ens: List[str]) -> None:
    """(a) Initial-condition S(k) with the exclusion region shaded."""
    assert len(ens) > 0, "need at least one ensemble"
    k_excl = float(data[f"{ens[0]}/K"])
    ax.axvspan(0.0, k_excl, color="#1f4e9c", alpha=0.12, lw=0)
    ax.axhline(1.0, color="0.6", lw=0.8, zorder=0)
    for e in ens:
        ax.plot(
            data[f"{e}/sk_k"],
            data[f"{e}/sk_S"],
            color=COLORS[e],
            lw=2.0,
            label=LABELS[e],
        )
    ax.set_xlabel(r"Wave Number, $k\rho^{-1/2}$")
    ax.set_ylabel(r"Structure Factor, $S(k)$")
    assert k_excl > 0.0, "exclusion radius must be positive"


def _band(
    ax: plt.Axes, data: Dict[str, np.ndarray], e: str, key: str, log: bool
) -> None:
    """Plot the ensemble mean of ``key`` with its 10-90 % band.

    Points with t/T = 0 or a non-positive value are dropped (log axes).
    """
    x = data[f"{e}/grid"]
    m, lo, hi = (data[f"{e}/{key}_{s}"] for s in ("mean", "q10", "q90"))
    if log:
        m, lo, hi = 10.0**m, 10.0**lo, 10.0**hi
    keep = (x > 0.0) & (m > 0.0)
    ax.fill_between(
        x[keep],
        np.clip(lo[keep], m[keep] * 1e-3, None),
        hi[keep],
        color=COLORS[e],
        alpha=0.2,
        lw=0,
    )
    ax.plot(x[keep], m[keep], color=COLORS[e], lw=2.0, label=LABELS[e])


def panel_distance(ax: plt.Axes, data: Dict[str, np.ndarray], ens: List[str]) -> None:
    """(b) Configurational distance from the ground-state manifold vs t/T."""
    for e in ens:
        _band(ax, data, e, "dist", log=False)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Dimensionless Time, $t/T$")
    ax.set_ylabel(r"Distance, $p_0(\infty)-p_0(t)$")
    ax.set_xlim(right=1.0)


def panel_energy(ax: plt.Axes, data: Dict[str, np.ndarray], ens: List[str]) -> None:
    """(c) Energy vs t/T on a log axis."""
    for e in ens:
        _band(ax, data, e, "log10_energy", log=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Dimensionless Time, $t/T$")
    ax.set_ylabel(r"Energy, $\Phi(t)$")
    ax.set_xlim(right=1.0)


def panel_energy_vs_dist(
    ax: plt.Axes, data: Dict[str, np.ndarray], ens: List[str]
) -> None:
    """(d) Mean energy vs squared mean distance, parameterized by t."""
    for e in ens:
        d2 = data[f"{e}/dist_mean"] ** 2
        phi = 10.0 ** data[f"{e}/log10_energy_mean"]
        keep = d2 > 0.0
        ax.loglog(d2[keep], phi[keep], color=COLORS[e], lw=2.0, label=LABELS[e])
    ax.set_xlabel(r"Squared Distance, $[p_0(\infty)-p_0(t)]^2$")
    ax.set_ylabel(r"Energy, $\Phi(t)$")


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])

    # ----- inputs -----
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("experiments/trajectories/chi0.1/summary.npz"),
    )
    parser.add_argument("--ensembles", nargs="+", default=["POS", "GAU"])

    # ----- output -----
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/figures/chi0.1_pos_gau")
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--sk-xmax", type=float, default=8.0)

    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with np.load(args.summary) as npz:
        data = {k: npz[k] for k in npz.files}

    # =========================================================
    # FIGURE
    # =========================================================
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.6))
    panel_sk(axes[0, 0], data, args.ensembles)
    axes[0, 0].set_xlim(0.0, args.sk_xmax)
    panel_distance(axes[0, 1], data, args.ensembles)
    panel_energy(axes[1, 0], data, args.ensembles)
    panel_energy_vs_dist(axes[1, 1], data, args.ensembles)
    for ax, tag in zip(axes.flat, "abcd", strict=True):
        ax.set_title(f"({tag})", loc="left")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=args.dpi, bbox_inches="tight")  # PNG only
    print(f"wrote {args.out}.png", flush=True)
    print("Done.", flush=True)
