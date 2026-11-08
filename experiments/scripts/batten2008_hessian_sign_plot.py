#!/usr/bin/env python3

"""
Negative and positive Hessian eigenvalue counts along the 2D flows, against Phi.

Reads hessian_chi<tag>.npz (batten2008_hessian.py: all N d eigenvalues of
every flow every quarter decade of Phi) and plateau_chi<tag>.npz
(batten2008_plateau.py: the cut-free tangent-space spectrum P R P of flows
0-9) for each system root (--roots, one --labels entry each; chi tags 1-10
and 2-5). Eigenvalues are counted negative (lambda < -c), zero (|lambda| <=
c) or positive (lambda > c) with the cut c = 1e-9 lambda_max of
batten2008_hessian.py. Near the ground-state manifold the spectrum has 2M
stiff positive eigenvalues (the constrained waves) and N d - 2M soft ones of
size ~ sqrt(Phi) (P R P), half of each sign. Writes to <out>:

- hessian_sign_compare: (a, b) negative (solid) and positive (dashed)
  counts for every start at chi = 0.1 and 0.4, with 2M, (N d - 2M)/2 and
  2M + (N d - 2M)/2 and, as points, the cut-free P R P counts (negative;
  positive + 2M); (c, d) the soft-sector counts n_- and n_+ - 2M over
  (N d - 2M)/2 (both 1 on the plateau).
- hessian_sign_mechanism: (a, b) signed spectral density sign(lambda)
  log10|lambda| against Phi (2D Poisson, every flow, fraction of the N d
  eigenvalues per bin), with the cut (red dashed); (c) the N d directions split into
  stiff (+), soft +, near-zero and soft - against Phi (2D Poisson, both
  chi); (d) median magnitude of the soft negative and soft positive P R P
  eigenvalues against Phi with a sqrt(Phi) guide (every start); (e)
  positive and negative fractions of the P R P spectrum (every start); (f)
  the stiff band: the 2M-th largest and the largest eigenvalue over k_1^2.

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batten2008_plot import finish, load, plt  # noqa: E402  (house rcParams)
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

COLORS = ("black", "#1f4e9c", "#c0392b", "#2e7d32", "#b9770e", "#7d3c98")
TAGS = (("1-10", r"$\chi = 0.1$", "-", "o"), ("2-5", r"$\chi = 0.4$", "--", "s"))
ZERO_REL = 1e-9  # the eigenvalue cut of batten2008_hessian.py
LAM_FLOOR = 1e-14  # |lambda| below this is drawn at 0 in the signed map
GUIDE = "0.6"

# ============================================================
# HELPERS
# ============================================================


def phi_axis(ax: plt.Axes) -> None:
    """Log energy axis, high to low (time runs left to right)."""
    ax.set_xscale("log")
    if not ax.xaxis_inverted():
        ax.invert_xaxis()
    ax.set_xlabel(r"Energy, $\Phi$")


def band(ax: plt.Axes, x: np.ndarray, rows: np.ndarray, color: str, ls: str) -> None:
    """Median and 10-90 % band of rows (flows, x)."""
    q10, q50, q90 = np.nanpercentile(rows, [10.0, 50.0, 90.0], axis=0)
    ax.fill_between(x, q10, q90, color=color, alpha=0.12, lw=0)
    ax.plot(x, q50, color=color, ls=ls, lw=2.0)


def sign_counts(h: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Negative, zero and positive counts (flows, levels) and the sizes."""
    keep = np.asarray(h["samples"]) <= np.log10(h["phi"][:, 0]).min()
    lam = h["lam"][:, keep]
    cut = ZERO_REL * lam[..., -1:]
    nd, m = lam.shape[-1], h["kmag"].shape[0]
    assert nd > 2 * m, "need a soft sector (N d > 2M)"
    out = {
        "phi": 10.0 ** np.asarray(h["samples"])[keep],
        "neg": (lam < -cut).sum(axis=-1).astype(float),
        "pos": (lam > cut).sum(axis=-1).astype(float),
        "nd": np.array(nd, dtype=float),
        "m2": np.array(2 * m, dtype=float),
        "lam": lam,
        "k1": np.array(float(h["kmag"][0])),
    }
    out["zero"] = out["nd"] - out["neg"] - out["pos"]
    return out


def prp_counts(p: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Cut-free soft-sector counts and median magnitudes of P R P (flows, levels)."""
    lam = p["lam_pr"]
    neg, pos = lam < 0.0, lam > 0.0
    mag_neg = np.where(neg, -lam, np.nan)
    mag_pos = np.where(pos, lam, np.nan)
    return {
        "phi": 10.0 ** p["levels"],
        "neg": neg.sum(axis=-1).astype(float),
        "pos": pos.sum(axis=-1).astype(float),
        "fneg": neg.mean(axis=-1),
        "fpos": pos.mean(axis=-1),
        "med_neg": np.nanmedian(mag_neg, axis=-1),
        "med_pos": np.nanmedian(mag_pos, axis=-1),
    }


def load_all(roots: List[Path]) -> List[Dict]:
    """Per root, per chi tag: sign counts of H and of P R P."""
    out = []
    for root in roots:
        sys_d = {}
        for tag, *_ in TAGS:
            sys_d[tag] = {
                "h": sign_counts(load(root / f"hessian_chi{tag}.npz")),
                "p": prp_counts(load(root / f"plateau_chi{tag}.npz")),
            }
        out.append(sys_d)
    assert len(out) == len(roots), "one entry per root"
    return out


def legend(
    fig: plt.Figure, labels: List[str], extra: List[Line2D], names: List[str]
) -> None:
    """Starts plus extra style entries, one row beneath the figure."""
    handles = [Line2D([], [], color=COLORS[i], lw=2.0) for i in range(len(labels))]
    fig.legend(handles + extra, labels + names, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 0.0), frameon=False)  # fmt: skip


# ============================================================
# FIGURE 1: COUNTS ACROSS STARTS
# ============================================================


def figure_compare(data: List, labels: List[str], out: Path, dpi: int) -> None:
    """Negative and positive counts for every start, both chi."""
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 11.0), constrained_layout=True)
    for i, sys_d in enumerate(data):
        color = COLORS[i % len(COLORS)]
        for col, (tag, _, _, _) in enumerate(TAGS):
            h, p = sys_d[tag]["h"], sys_d[tag]["p"]
            half = 0.5 * (h["nd"] - h["m2"])
            band(axes[0, col], h["phi"], h["neg"], color, "-")
            band(axes[0, col], h["phi"], h["pos"], color, "--")
            for value, ls in ((h["m2"], ":"), (half, ":"), (h["m2"] + half, "-.")):
                axes[0, col].axhline(value, color=color, lw=0.7, ls=ls, zorder=0)
            axes[0, col].plot(p["phi"], np.median(p["neg"], axis=0), ls="",
                              marker="o", ms=6, color=color, mfc="none")  # fmt: skip
            axes[0, col].plot(p["phi"], np.median(p["pos"], axis=0) + h["m2"], ls="",
                              marker="s", ms=6, color=color, mfc="none")  # fmt: skip
            band(axes[1, col], h["phi"], h["neg"] / half, color, "-")
            band(axes[1, col], h["phi"], (h["pos"] - h["m2"]) / half, color, "--")
    for col, (_, chi, _, _) in enumerate(TAGS):
        ax = axes[0, col]
        phi_axis(ax)
        ax.set_ylabel(r"Number of Eigenvalues")
        ax.set_title(f"({'ab'[col]}) {chi}: negative and positive counts", loc="left")
        ax = axes[1, col]
        phi_axis(ax)
        ax.axhline(1.0, color=GUIDE, lw=0.8, zorder=0)
        ax.set_ylim(0.0, 2.2)
        ax.set_ylabel(r"Soft Count over $(Nd-2M)/2$")
        ax.set_title(f"({'cd'[col]}) {chi}: soft sector, $n_-$ and $n_+ - 2M$",
                     loc="left")  # fmt: skip
    extra = [Line2D([], [], color="0.3", lw=2.0),
             Line2D([], [], color="0.3", lw=2.0, ls="--"),
             Line2D([], [], color="0.3", lw=0.7, ls=":"),
             Line2D([], [], color="0.3", lw=0.7, ls="-."),
             Line2D([], [], color="0.3", ls="", marker="o", mfc="none"),
             Line2D([], [], color="0.3", ls="", marker="s", mfc="none")]  # fmt: skip
    names = [r"$\lambda < -c$", r"$\lambda > c$", r"$2M$ and $(Nd-2M)/2$",
             r"$2M + (Nd-2M)/2$", r"$PRP < 0$, no cut",
             r"$2M + (PRP > 0)$, no cut"]  # fmt: skip
    legend(fig, labels, extra, names)
    finish(fig, out / "hessian_sign_compare", dpi)


# ============================================================
# FIGURE 2: MECHANISM
# ============================================================


def panel_signed_map(ax: plt.Axes, h: Dict[str, np.ndarray], title: str) -> object:
    """Fraction of eigenvalues per signed log-magnitude bin, at every level."""
    lam = h["lam"]
    y = np.sign(lam) * np.log10(np.maximum(np.abs(lam), LAM_FLOOR) / LAM_FLOOR)
    edges = np.linspace(-15.5, 15.5, 125)
    frac = np.stack([np.histogram(y[:, j].ravel(), bins=edges)[0]
                     for j in range(y.shape[1])], axis=1) / y[:, 0].size  # fmt: skip
    centers = 0.5 * (edges[1:] + edges[:-1])
    mesh = ax.pcolormesh(h["phi"], centers, np.where(frac > 0, frac, np.nan),
                         norm=LogNorm(1e-5, 0.5), cmap="viridis", shading="nearest",
                         rasterized=True)  # fmt: skip
    cut = ZERO_REL * np.median(lam[..., -1], axis=0)
    yc = np.log10(np.maximum(cut, LAM_FLOOR) / LAM_FLOOR)
    ax.plot(h["phi"], yc, color="#c0392b", lw=1.4, ls="--")
    ax.plot(h["phi"], -yc, color="#c0392b", lw=1.4, ls="--")
    phi_axis(ax)
    ticks = np.array([-14, -10, -6, -2, 0, 2, 6, 10, 14])
    ax.set_yticks(ticks, [rf"$-10^{{{-t - 14}}}$" if t < 0 else
                          (r"$0$" if t == 0 else rf"$10^{{{t - 14}}}$")
                          for t in ticks])  # fmt: skip
    ax.set_ylabel(r"Eigenvalue, $\lambda$ (signed log scale)")
    ax.set_title(title, loc="left", fontsize=13)
    return mesh


def panel_split(ax: plt.Axes, h: Dict[str, np.ndarray], ls: str) -> None:
    """Stiff (+), soft +, near-zero and soft - fractions of the N d directions."""
    nd, m2 = float(h["nd"]), float(h["m2"])
    stiff = np.minimum(h["pos"], m2)
    soft_pos = np.maximum(h["pos"] - m2, 0.0)
    parts = (("stiff +", stiff, "#1f4e9c"), ("soft +", soft_pos, "#2e7d32"),
             ("near zero", h["zero"], "0.55"),
             ("soft $-$", h["neg"], "#c0392b"))  # fmt: skip
    for _, rows, color in parts:
        ax.plot(h["phi"], np.median(rows, axis=0) / nd, color=color, lw=2.0, ls=ls)


def figure_mechanism(data: List, labels: List[str], out: Path, dpi: int) -> None:
    """Where the positive and negative eigenvalues sit, and how they move with Phi."""
    fig, axes = plt.subplots(2, 3, figsize=(20.0, 11.5), constrained_layout=True)
    pois = data[0]
    mesh = panel_signed_map(axes[0, 0], pois["1-10"]["h"],
                            f"(a) {labels[0]}, {TAGS[0][1]}: signed spectrum",
                            )  # fmt: skip
    panel_signed_map(axes[0, 1], pois["2-5"]["h"],
                     rf"(b) {labels[0]}, {TAGS[1][1]}: signed spectrum")  # fmt: skip
    fig.colorbar(mesh, ax=axes[0, :2], label="Fraction of the $Nd$ Eigenvalues",
                 shrink=0.85)  # fmt: skip
    for tag, _, ls, _ in TAGS:
        panel_split(axes[0, 2], pois[tag]["h"], ls)
    for i, sys_d in enumerate(data):
        color = COLORS[i % len(COLORS)]
        for tag, _, ls, _ in TAGS:
            p, h = sys_d[tag]["p"], sys_d[tag]["h"]
            axes[1, 0].plot(p["phi"], np.nanmedian(p["med_neg"], axis=0), color=color,
                            ls=ls, lw=2.0)  # fmt: skip
            axes[1, 0].plot(p["phi"], np.nanmedian(p["med_pos"], axis=0), color=color,
                            ls=ls, lw=1.0, marker="^", ms=4)  # fmt: skip
            axes[1, 1].plot(p["phi"], np.median(p["fneg"], axis=0), color=color, ls=ls,
                            lw=2.0)  # fmt: skip
            axes[1, 1].plot(p["phi"], np.median(p["fpos"], axis=0), color=color, ls=ls,
                            lw=1.0, marker="^", ms=4)  # fmt: skip
            m2 = int(h["m2"])
            k1sq = float(h["k1"]) ** 2
            stiff = h["lam"][..., -m2] / k1sq
            band(axes[1, 2], h["phi"], np.where(stiff > 0.0, stiff, np.nan), color, ls)
            axes[1, 2].plot(h["phi"], np.median(h["lam"][..., -1], axis=0) / k1sq,
                            color=color, ls=ls, lw=1.0)  # fmt: skip
    ax = axes[0, 2]
    phi_axis(ax)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Fraction of the $Nd$ Directions")
    ax.set_title(f"(c) {labels[0]}: how the $Nd$ directions split", loc="left",
                 fontsize=13)  # fmt: skip
    ax.legend([Line2D([], [], color=c, lw=2.0) for c in
               ("#1f4e9c", "#2e7d32", "0.55", "#c0392b")],
              [r"stiff $+$ (the $2M$ waves)", r"soft $+$", r"$|\lambda| \leq c$",
               r"soft $-$"], frameon=False, fontsize=10, loc="upper left",
              ncol=2)  # fmt: skip
    ax = axes[1, 0]
    phi_axis(ax)
    ax.set_yscale("log")
    x = np.geomspace(1e-10, 1e2, 20)
    ax.plot(x, 3e-3 * np.sqrt(x / 1e-4), color=GUIDE, lw=1.0, ls="-.")
    ax.set_ylabel(r"Median Soft $|\lambda|$ ($PRP$)")
    ax.set_title(r"(d) soft $-$ and soft $+$ shrink together as $\sqrt{\Phi}$",
                 loc="left", fontsize=13)  # fmt: skip
    ax = axes[1, 1]
    phi_axis(ax)
    ax.axhline(0.5, color=GUIDE, lw=0.8, zorder=0)
    ax.set_ylim(0.3, 0.7)
    ax.set_ylabel(r"Fraction of the Soft Spectrum ($PRP$)")
    ax.set_title("(e) negative and positive halves, no cut", loc="left", fontsize=13)
    ax = axes[1, 2]
    phi_axis(ax)
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 3e3)
    ax.set_ylabel(r"Stiff Eigenvalue, $\lambda/k_1^2$")
    ax.set_title(r"(f) the stiff band: $2M$-th largest and largest", loc="left",
                 fontsize=13)  # fmt: skip
    extra = [Line2D([], [], color="0.3", lw=2.0, ls=ls) for _, _, ls, _ in TAGS]
    extra += [Line2D([], [], color="0.3", lw=1.0, marker="^", ms=4),
              Line2D([], [], color=GUIDE, lw=1.0, ls="-.")]  # fmt: skip
    names = [chi for _, chi, _, _ in TAGS]
    names += [r"(d, e) positive branch; (f) $\lambda_{\max}$", r"$\propto\sqrt{\Phi}$"]
    legend(fig, labels, extra, names)
    finish(fig, out / "hessian_sign_mechanism", dpi)


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("experiments/figures"))
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    assert len(args.roots) == len(args.labels) <= len(COLORS), "one label per root"
    args.out.mkdir(parents=True, exist_ok=True)
    data = load_all(args.roots)
    for sys_d, lab in zip(data, args.labels, strict=True):
        for tag, *_ in TAGS:
            h = sys_d[tag]["h"]
            j = int(np.argmin(np.abs(np.log10(h["phi"]) + 4.0)))
            neg = np.median(h["neg"][:, j])
            print(f"{lab} chi{tag}: at Phi=1e-4 median neg {neg:.0f}"
                  f" pos {np.median(h['pos'][:, j]):.0f} (2M {int(h['m2'])}, "
                  f"(Nd-2M)/2 {int((h['nd'] - h['m2']) / 2)})", flush=True)  # fmt: skip
    figure_compare(data, args.labels, args.out, args.dpi)
    figure_mechanism(data, args.labels, args.out, args.dpi)
    print("Done.", flush=True)
