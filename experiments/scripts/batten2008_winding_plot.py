#!/usr/bin/env python3

"""
Windiness of the mode-plane trajectories rho(k_n, t), compared across starts.

Each constrained density wave n traces a path in its own complex plane (the
mode_planes figures: angle arg rho, radius log10 S). Its windiness is the
total winding T_n = (1 / 2 pi) sum_t |d arg rho_n| over the frames where
S(k_n) > 1e-16 (phase_stats in batten2008_modes.py). rho(k_n) is a sum over
all particles, so T_n already averages over the particles of a flow; the
figure then takes the median and 10-90 % band over flows. Also shown: the
net winding |sum d arg rho| / 2 pi (coherent rotation only), where in S the
winding happens, and its link to how far the particles move.

Reads, for each system given by --roots (a flows directory N<N>) and
--labels, modes_chi<tag>.npz, offplane_chi<tag>.npz (for |Delta r|) and
flows_chi<tag>.npz (for N), for the two chi tags in --tags. Writes
<out>/<--name>.png:

- (a, e) T_n against |k_n| / K, binned by |k| (median and 10-90 % over flows
  of the bin mean), with the Mann-Whitney p-value of the first two systems
  per bin (20 flows each; 6.8e-8 is complete separation);
- (b, f) the same for the net winding;
- (c, g) the winding of the short waves |k| > 0.6 K split by log10 S at the
  step: above -2, -2 to -6, -6 to -10, below -10;
- (d, h) per flow, the short-wave winding against the per-particle RMS net
  displacement |Delta r| / sqrt(N), with Spearman's rank correlation over all
  flows and within the first system.

Needs matplotlib and scipy (the plottime environment).

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
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batten2008_plot import finish, load, plt  # noqa: E402  (house rcParams)
from matplotlib.lines import Line2D  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

COLORS = ("black", "#1f4e9c", "#c0392b", "#2e7d32", "#b9770e", "#7d3c98")
MARKERS = "osD^vP"
N_BINS = 8  # |k| / K bins
SHORT = 0.6  # short waves: |k| > SHORT K
BANDS = (r"$S > 10^{-2}$", r"$10^{-6}$-$10^{-2}$", r"$10^{-10}$-$10^{-6}$",
         r"$S < 10^{-10}$")  # fmt: skip

# ============================================================
# HELPERS
# ============================================================


def system(root: Path, tag: str) -> Dict[str, np.ndarray]:
    """Winding per flow and mode, |k| / K, and per-particle displacement."""
    z = load(root / f"modes_chi{tag}.npz")
    with np.load(root / f"flows_chi{tag}.npz") as f:
        n = f["s0"].shape[1]
    with np.load(root / f"offplane_chi{tag}.npz") as o:
        chord = np.asarray(o["chord"])
    kmag = z["kmag"]
    assert z["turns"].shape[1] == kmag.shape[0], "one winding per mode"
    return {"x": kmag / kmag[-1], "turns": z["turns"], "net": z["net_turns"],
            "band": z["band_turns"], "ell": chord / np.sqrt(n)}  # fmt: skip


def binned(x: np.ndarray, per_mode: np.ndarray) -> np.ndarray:
    """Mean over the modes in each |k| / K bin, per flow: (flows, N_BINS)."""
    idx = np.clip((x * N_BINS).astype(int), 0, N_BINS - 1)
    return np.stack([np.nanmean(per_mode[:, idx == i], axis=1) if np.any(idx == i)
                     else np.full(per_mode.shape[0], np.nan)
                     for i in range(N_BINS)], axis=1)  # fmt: skip


def band_panel(
    ax: plt.Axes, data: List[Dict[str, np.ndarray]], labels: List[str]
) -> None:
    """Stacked bars: median short-wave winding per log10 S band, per system."""
    for i, d in enumerate(data):
        short = d["x"] > SHORT
        parts = np.median(d["band"][:, short, :].mean(axis=1), axis=0)
        bottom = 0.0
        for j, v in enumerate(parts):
            ax.bar(i, v, bottom=bottom, color=COLORS[i], alpha=0.25 + 0.25 * j,
                   edgecolor="white", width=0.7)  # fmt: skip
            bottom += v
    ax.set_xticks(range(len(data)), [lab.split(",")[0] for lab in labels],
                  fontsize=10)  # fmt: skip


# ============================================================
# FIGURE
# ============================================================


def figure(roots: List[Path], labels: List[str], tags: List[str], chi_names: List[str],
           out: Path, dpi: int) -> None:  # fmt: skip
    """Windiness by mode, net winding, where it happens, and its driver."""
    fig, axes = plt.subplots(2, 4, figsize=(22.0, 10.0), constrained_layout=True)
    centers = (np.arange(N_BINS) + 0.5) / N_BINS
    for row, (tag, chi) in enumerate(zip(tags, chi_names, strict=True)):
        data = [system(r, tag) for r in roots]
        let = "abcdefgh"[4 * row : 4 * row + 4]
        for col, key in ((0, "turns"), (1, "net")):
            ax = axes[row, col]
            per = [binned(d["x"], d[key]) for d in data]
            for i, b in enumerate(per):
                q10, q50, q90 = np.nanpercentile(b, [10.0, 50.0, 90.0], axis=0)
                ax.fill_between(centers, q10, q90, color=COLORS[i], alpha=0.15, lw=0)
                ax.plot(centers, q50, color=COLORS[i], lw=2.0, marker=MARKERS[i], ms=5)
            for j in range(N_BINS):
                p = mannwhitneyu(per[0][:, j], per[1][:, j]).pvalue
                ax.text(centers[j], 0.02, f"{p:.0e}", fontsize=7, ha="center",
                        rotation=90, transform=ax.get_xaxis_transform())  # fmt: skip
            name = "total winding" if key == "turns" else "net (coherent) winding"
            ax.set_title(f"({let[col]}) {chi}: {name}", loc="left", fontsize=13)
            ax.set_xlabel(r"Wavenumber, $|\mathbf{k}_n|/K$")
            ax.set_ylabel("Turns of arg $\\rho(\\mathbf{k}_n)$")
        band_panel(axes[row, 2], data, labels)
        axes[row, 2].set_title(f"({let[2]}) {chi}: short-wave winding by $S$",
                               loc="left", fontsize=13)  # fmt: skip
        axes[row, 2].set_ylabel(r"Turns, $|\mathbf{k}| > 0.6K$ (median)")
        ax = axes[row, 3]
        xs, ys = [], []
        for i, d in enumerate(data):
            y = d["turns"][:, d["x"] > SHORT].mean(axis=1)
            ax.plot(d["ell"], y, ls="", marker=MARKERS[i], ms=6, color=COLORS[i],
                    mfc="none")  # fmt: skip
            xs.append(d["ell"])
            ys.append(y)
        rho_all = spearmanr(np.concatenate(xs), np.concatenate(ys)).statistic
        rho_one = spearmanr(xs[0], ys[0]).statistic
        ax.set_title(f"({let[3]}) {chi}: Spearman {rho_all:.2f} (all), "
                     f"{rho_one:.2f} ({labels[0].split(',')[0]})", loc="left",
                     fontsize=12)  # fmt: skip
        ax.set_xlabel(r"Per-Particle Displacement, $|\Delta r|/\sqrt{N}$")
        ax.set_ylabel(r"Turns, $|\mathbf{k}| > 0.6K$")
    pairs = list(zip(COLORS, MARKERS, strict=False))[: len(roots)]
    handles = [Line2D([], [], color=c, lw=2.0, marker=m) for c, m in pairs]
    names = [lab.split(",")[0] for lab in labels]
    shades = [plt.Rectangle((0, 0), 1, 1, color="0.3", alpha=0.25 + 0.25 * j)
              for j in range(len(BANDS))]  # fmt: skip
    fig.legend(handles + shades, names + [f"(c, g) {b}" for b in BANDS],
               loc="upper center", ncol=len(names) + len(BANDS),
               bbox_to_anchor=(0.5, 0.0), frameon=False)  # fmt: skip
    finish(fig, out, dpi)


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--tags", nargs=2, default=["1-10", "2-5"])
    parser.add_argument("--chi-names", nargs=2, default=[r"$\chi = 0.1$",
                                                          r"$\chi = 0.4$"])  # fmt: skip
    parser.add_argument("--out", type=Path, default=Path("experiments/figures"))
    parser.add_argument("--name", default="winding_compare")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    assert len(args.roots) == len(args.labels) <= len(COLORS), "one label per root"
    args.out.mkdir(parents=True, exist_ok=True)
    figure(args.roots, args.labels, args.tags, args.chi_names, args.out / args.name,
           args.dpi)  # fmt: skip
    print("Done.", flush=True)
