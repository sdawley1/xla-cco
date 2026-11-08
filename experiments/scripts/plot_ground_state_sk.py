#!/usr/bin/env python3

"""
Ensemble-averaged structure factors of the chi = 0.1 stealthy ground states
reached from the POS and GAU initial conditions.

(a) S(k) on linear axes; (b) the same on a log S axis, showing the
suppression inside the exclusion region k < K (shaded) and the recovery
beyond it. Error bars are the standard error of the bin mean (shaded).
Reads summary.npz from analyze_trajectories.py.

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

# ============================================================
# PANELS
# ============================================================


def plot_sk(
    ax: plt.Axes, data: Dict[str, np.ndarray], ens: List[str], log: bool
) -> None:
    """Draw the ground-state S(k) of every ensemble on ``ax``."""
    assert len(ens) > 0, "need at least one ensemble"
    k_excl = float(data[f"{ens[0]}/K"])
    assert k_excl > 0.0, "exclusion radius must be positive"
    ax.axvspan(0.0, k_excl, color="#1f4e9c", alpha=0.12, lw=0)
    ax.axhline(1.0, color="0.6", lw=0.8, zorder=0)
    for e in ens:
        k, s, sem = data[f"{e}/gs_k"], data[f"{e}/gs_S"], data[f"{e}/gs_sem"]
        keep = s > 0.0 if log else np.ones_like(s, dtype=bool)
        ax.fill_between(
            k[keep], (s - sem)[keep], (s + sem)[keep], color=COLORS[e], alpha=0.2, lw=0
        )
        chi = float(data[f"{e}/chi_actual"])
        ax.plot(
            k[keep],
            s[keep],
            color=COLORS[e],
            lw=2.0,
            label=rf"{e} ($N={int(data[f'{e}/N'])}$, $\chi={chi:.3f}$)",
        )
    if log:
        ax.set_yscale("log")
    ax.set_xlabel(r"Wave Number, $k\rho^{-1/2}$")
    ax.set_ylabel(r"Structure Factor, $S(k)$")


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
        "--out", type=Path, default=Path("experiments/figures/chi0.1_ground_state_sk")
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--xmax", type=float, default=8.0)

    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with np.load(args.summary) as npz:
        data = {k: npz[k] for k in npz.files}
    missing = [e for e in args.ensembles if f"{e}/gs_S" not in data]
    if missing:
        raise SystemExit(
            f"summary lacks ground-state S(k) for {missing}; "
            "re-run analyze_trajectories.py"
        )

    # =========================================================
    # FIGURE
    # =========================================================
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharex=True)
    plot_sk(axes[0], data, args.ensembles, log=False)
    plot_sk(axes[1], data, args.ensembles, log=True)
    axes[0].set_xlim(0.0, args.xmax)
    for ax, tag in zip(axes, "ab", strict=True):
        ax.set_title(f"({tag})", loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
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
