#!/usr/bin/env python3

"""
Flow-averaged phase planes of rho(k_n), coloured by the energy.

Reads modes_chi<tag>.npz (batten2008_modes.py: rho(k_n) for ten picked
modes at every stored frame, with Phi) under --root and draws the polar
planes of *_mode_planes_full (angle arg rho, radius log10 S + 20) for the
average over flows. The flows start at random phases and turn either way
(about half each), so each flow is first aligned:

  theta_b(Phi) = sign_b [arg rho_b(Phi) - arg rho_b(start)]   (unwrapped),

with sign_b the sign of flow b's net turn, so every flow starts at angle 0
and turns counterclockwise. The phase is followed while S > 1e-16 (as the
winding metric) and held after. theta_b and log10 S_b are interpolated on a
common grid of log10 Phi (the clock: every flow compared at equal energy)
and averaged over flows. The mean curve is coloured by log10 Phi; thin
grey lines are 20 aligned flows. Rows 1-5, 10 and |k| = K/4..K at both chi.

Writes <out>/<prefix>_N<N>_mode_planes_mean.png.

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
from typing import Dict, Tuple

# ----- numerics -----
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batten2008_plot import finish, load, plt  # noqa: E402  (house rcParams)
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

TAGS = (("1-10", r"$\chi = 0.1$"), ("2-5", r"$\chi = 0.4$"))
S_PHASE = 1e-16  # follow the phase while S exceeds this (winding metric)
S_FLOOR = 1e-20  # radius 0 of the planes
N_GREY = 20  # aligned single flows drawn under the mean
CMAP = plt.get_cmap("plasma")

# ============================================================
# HELPERS
# ============================================================


def aligned_flow(
    re: np.ndarray, im: np.ndarray, energy: np.ndarray, grid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """(theta, log10 S) of one flow on the log10 Phi grid, and its net turns.

    theta starts at 0 and is mirrored to turn counterclockwise; it is held
    once S falls below S_PHASE. Grid points below the flow's last energy
    take its last values, and points above its start energy its start
    values (the grid starts at the highest start of any flow).
    """
    ok = np.isfinite(re) & np.isfinite(im) & np.isfinite(energy)
    re, im, e = re[ok], im[ok], np.minimum.accumulate(energy[ok])
    assert e.shape[0] >= 2, "a flow needs two frames"
    s = re**2 + im**2
    raw = np.unwrap(np.arctan2(im, re))
    live = s > S_PHASE
    last = int(np.flatnonzero(live)[-1]) if live.any() else 0
    theta = raw - raw[0]
    theta[last + 1 :] = theta[last]
    net = float(theta[-1])
    theta *= 1.0 if net >= 0.0 else -1.0
    x = np.log10(np.maximum(e, 1e-300))[::-1]  # increasing for interp
    logs = np.log10(np.maximum(s, S_FLOOR))[::-1]
    th = np.interp(grid, x, theta[::-1], left=theta[-1], right=theta[0])
    ls = np.interp(grid, x, logs, left=logs[0], right=logs[-1])
    return th, ls, net / (2.0 * np.pi)


def mean_plane(z: Dict[str, np.ndarray], j: int, grid: np.ndarray) -> Dict:
    """Aligned flows of mode column j, their mean and the net turns."""
    rows = [aligned_flow(z["rho_re"][b, :, j], z["rho_im"][b, :, j],
                         z["energy"][b], grid)
            for b in range(z["rho_re"].shape[0])]  # fmt: skip
    th = np.array([r[0] for r in rows])
    ls = np.array([r[1] for r in rows])
    net = np.array([r[2] for r in rows])
    return {"theta": th, "logs": ls, "net": net, "mean_theta": th.mean(axis=0),
            "mean_r": ls.mean(axis=0) + 20.0}  # fmt: skip


def mode_label(z: Dict[str, np.ndarray], row: int) -> str:
    """'$n = (1, 0)$' for the 1-based k-set row."""
    v = np.asarray(z["nvec"])[row - 1].astype(int)
    return "$n = (" + ", ".join(str(c) for c in v) + ")$"


def draw(ax: plt.Axes, plane: Dict, grid: np.ndarray, norm: Normalize) -> None:
    """Grey aligned flows, then the mean coloured by log10 Phi."""
    for b in range(min(N_GREY, plane["theta"].shape[0])):
        ax.plot(plane["theta"][b], np.clip(plane["logs"][b] + 20.0, 0.0, None),
                color="0.6", lw=0.4, alpha=0.5)  # fmt: skip
    th, r = plane["mean_theta"], np.clip(plane["mean_r"], 0.0, None)
    pts = np.stack([th, r], axis=1)[:, None, :]
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(seg, cmap=CMAP, norm=norm, lw=2.4, zorder=3)
    lc.set_array(0.5 * (grid[:-1] + grid[1:]))
    ax.add_collection(lc)
    ax.plot(th[-1], r[-1], "o", color="black", ms=4, zorder=4)  # start: grid top
    ax.set_rlim(0, 21)
    ax.set_rticks([0, 10, 20])
    ax.set_yticklabels([r"$10^{-20}$", r"$10^{-10}$", r"$1$"], fontsize=10)
    ax.tick_params(axis="x", labelsize=9)


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--out", type=Path, default=Path("experiments/figures"))
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    systems = [(tag, label, load(args.root / f"modes_chi{tag}.npz"))
               for tag, label in TAGS]  # fmt: skip
    top = max(float(np.log10(np.nanmax(z["energy"][:, 0]))) for _, _, z in systems)
    grid = np.linspace(-20.0, top, 1200)  # log10 Phi, low to high
    norm = Normalize(vmin=-20.0, vmax=top)
    fig, axes = plt.subplots(4, 5, figsize=(18.0, 16.0), constrained_layout=True,
                             subplot_kw={"projection": "polar"})  # fmt: skip
    letters = "abcdefghijklmnopqrst"
    for block, (tag, label, z) in enumerate(systems):
        picks = z["picks"].tolist()
        assert len(picks) == 10, "expected rows 1-5, 10 and four |k| fractions"
        for i, mode in enumerate(picks):
            plane = mean_plane(z, i, grid)
            ax = axes[2 * block + i // 5, i % 5]
            draw(ax, plane, grid, norm)
            net = np.mean(np.abs(plane["net"]))
            ax.set_title(f"({letters[10 * block + i]}) {label}, {mode_label(z, mode)}"
                         f"\nmean net turn {net:.2f}", loc="left",
                         fontsize=12)  # fmt: skip
            print(
                f"{args.prefix} chi{tag} n={mode}: mean |net| {net:.3f} turns, "
                f"CCW share {np.mean(plane['net'] > 0):.2f}",
                flush=True,
            )
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    cbar = fig.colorbar(sm, ax=axes, shrink=0.5, pad=0.02)
    cbar.set_label(r"Energy, $\log_{10}\Phi$ (flow time runs downward)")
    handles = [
        Line2D([], [], color=CMAP(0.6), lw=2.4),
        Line2D([], [], color="0.6", lw=0.8),
        Line2D([], [], ls="", marker="o", color="black", ms=4),
    ]
    fig.legend(handles, ["Mean over flows (aligned, coloured by energy)",
                         f"{N_GREY} aligned flows", "Start"], loc="upper center",
               ncol=3, bbox_to_anchor=(0.5, 0.0), frameon=False)  # fmt: skip
    finish(fig, args.out / f"{args.prefix}_mode_planes_mean", args.dpi)
    print("Done.", flush=True)
