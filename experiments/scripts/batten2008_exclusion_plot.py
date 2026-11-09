#!/usr/bin/env python3

"""
Can single waves inside the exclusion region be excited independently near the manifold?

Reads exclusion_chi<tag>.npz (batten2008_exclusion.py: per-wave statistics
of K = (2/V) J J^T at ground states), landscape_chi<tag>.npz (K spectra and
residual weights along the flows) and winding_chi<tag>.npz (S(k) along the
flows) per system root. Writes <out>/exclusion_mode_control.png:

- (a) gradient flow: share of the residual |u|^2 in collective modes whose
  stiffness lies within 10 % of the residual mean <K>_u (1: every wave decays
  at one rate, the coherent decay of winding_cv_explained);
- (b) gradient flow: share of sum_k S(k) held by the longest-wave shell
  |k| = k_1 (the waves the rest are slaved to);
- (c) the least displacement that changes one wave alone, |k|^2 (K^-1)_kk
  (1 for independent wells), median over ground states, with the
  Marchenko-Pastur mean 1 / (1 - 2M/(N d));
- (d) thermal statics: <S(k)>/T = (2/V)(J H^+ J^T)_kk of the harmonic
  canonical ensemble, per wave (equipartition: 1 for every k);
- (e) share of a single-wave excitation still in that wave after a time
  |k|^2 t of gradient flow (1 for independent wells), median over waves;
- (f) MD: the collective frequencies sqrt(lambda) one wave's fluctuation
  carries, weighted, for |k| = K/2 (2D Poisson, one ground state).

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
from matplotlib.lines import Line2D  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

COLORS = ("black", "#1f4e9c", "#c0392b", "#2e7d32", "#b9770e", "#7d3c98")
TAGS = (("1-10", r"$\chi = 0.1$", "-", "o"), ("2-5", r"$\chi = 0.4$", "--", "s"))
BAND = 0.1  # (a): modes within this relative distance of <K>_u
BINS = np.linspace(0.0, 1.0, 11)  # |k|/K bins of (c, d)
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
    """Median and 10-90 % band of rows (samples, x)."""
    q10, q50, q90 = np.nanpercentile(rows, [10.0, 50.0, 90.0], axis=0)
    ax.fill_between(x, q10, q90, color=color, alpha=0.12, lw=0)
    ax.plot(x, q50, color=color, ls=ls, lw=2.0)


def near_mean_share(lz: Dict[str, np.ndarray]) -> np.ndarray:
    """(flows, levels) share of |u|^2 in modes within BAND of <K>_u."""
    mean = lz["ray_k"][:, :, None]
    near = np.abs(lz["k_eig"] - mean) <= BAND * mean
    out: np.ndarray = (lz["weight"] * near).sum(axis=2)
    reached = lz["levels"][None, :] <= np.log10(lz["phi"][:, :1])
    return np.where(reached, out, np.nan)


def shell_share(w: Dict[str, np.ndarray]) -> np.ndarray:
    """(flows, levels) share of sum_k S(k) on the longest-wave shell."""
    shell = np.isclose(w["kmag"], w["kmag"][0])
    tot = w["S"].sum(axis=2)
    out: np.ndarray = w["S"][:, :, shell].sum(axis=2) / np.where(tot > 0, tot, np.nan)
    reached = w["levels"][None, :] <= np.log10(w["phi"][:, :1])
    return np.where(reached, out, np.nan)


def binned(xk: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Median of rows (ground states, M) per |k|/K bin."""
    idx = np.clip(np.digitize(xk, BINS) - 1, 0, BINS.shape[0] - 2)
    out = np.array([np.median(rows[:, idx == b]) if np.any(idx == b) else np.nan
                    for b in range(BINS.shape[0] - 1)])  # fmt: skip
    assert out.shape == (BINS.shape[0] - 1,), "one value per bin"
    return out


# ============================================================
# FIGURE
# ============================================================


def figure(data: List, labels: List[str], out: Path, dpi: int) -> None:
    """Coherent decay, equipartition, and the cost and life of one-wave excitations."""
    fig, axes = plt.subplots(2, 3, figsize=(20.0, 11.5), constrained_layout=True)
    centers = 0.5 * (BINS[1:] + BINS[:-1])
    for i, sys_d in enumerate(data):
        color = COLORS[i % len(COLORS)]
        for tag, _, ls, mk in TAGS:
            z, lz, w = sys_d[tag]["x"], sys_d[tag]["l"], sys_d[tag]["w"]
            band(axes[0, 0], 10.0 ** lz["levels"], near_mean_share(lz), color, ls)
            band(axes[0, 1], 10.0 ** w["levels"], shell_share(w), color, ls)
            xk = z["kmag"] / z["kmag"][-1]
            axes[1, 0].plot(xk, np.median(z["s_thermal"], axis=0), ls="", marker=mk,
                            ms=3, color=color, mfc="none", alpha=0.6)  # fmt: skip
            axes[0, 2].plot(centers, binned(xk, z["cost"]), color=color, ls=ls,
                            lw=2.0, marker=mk, ms=5)  # fmt: skip
            ret = np.median(z["retention"], axis=1)  # over waves, per ground state
            band(axes[1, 1], z["x_grid"], ret, color, ls)
    for tag, _, ls, _ in TAGS:
        c = float(data[0][tag]["x"]["c"])
        axes[0, 2].axhline(1.0 / (1.0 - c), color=GUIDE, lw=1.2, ls=ls)
    panel_spectrum(axes[1, 2], data[0])
    style_axes(axes, labels)
    handles = [Line2D([], [], color=COLORS[i], lw=2.0) for i in range(len(labels))]
    handles += [Line2D([], [], color="0.3", lw=2.0, ls=ls, marker=mk, mfc="none")
                for _, _, ls, mk in TAGS]  # fmt: skip
    fig.legend(handles, labels + [chi for _, chi, _, _ in TAGS], loc="upper center",
               ncol=7, bbox_to_anchor=(0.5, 0.0), frameon=False)  # fmt: skip
    finish(fig, out / "exclusion_mode_control", dpi)


def panel_spectrum(ax: plt.Axes, sys_d: Dict) -> None:
    """Frequency weights of the |k| = K/2 wave of flow 0, both chi (2D Poisson)."""
    for tag, chi, ls, _ in TAGS:
        z = sys_d[tag]["x"]
        j = 1  # EXAMPLE_FRACS[1] = 0.5
        row = int(z["ex_rows"][j])
        om = z["ex_om"][0] / z["kmag"][row]
        wts = z["ex_w"][j]
        order = np.argsort(om)
        hist, edges = np.histogram(om[order], bins=np.linspace(0.0, 3.0, 121),
                                   weights=wts[order])  # fmt: skip
        modes = 1.0 / np.sum(wts**2)
        ax.step(0.5 * (edges[1:] + edges[:-1]), hist, where="mid", lw=1.8, ls=ls,
                color="#1f4e9c" if tag == "1-10" else "#c0392b",
                label=f"{chi}: spread over {modes:.0f} modes")  # fmt: skip
    ax.axvline(1.0, color=GUIDE, lw=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=10, loc="upper right")


def style_axes(axes: np.ndarray, labels: List[str]) -> None:
    """Labels and titles of the six panels."""
    ax = axes[0, 0]
    phi_axis(ax)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(r"Share of $|u|^2$ within 10% of $\langle K\rangle_u$")
    ax.set_title("(a) flow: the residual decays at one rate", loc="left", fontsize=13)
    ax = axes[0, 1]
    phi_axis(ax)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(r"Share of $\sum_k S(k)$ on $|\mathbf{k}| = k_1$")
    ax.set_title("(b) flow: the longest waves carry it", loc="left", fontsize=13)
    ax = axes[0, 2]
    ax.set_yscale("log")
    ax.axhline(1.0, color="black", lw=0.8, ls=":")
    ax.set_xlabel(r"Wavenumber, $|\mathbf{k}|/K$")
    ax.set_ylabel(r"One-Wave Cost, $|\mathbf{k}|^2 (K^{-1})_{\mathbf{kk}}$")
    ax.set_title(r"(c) displacement to excite one wave (grey: $1/(1-c)$)", loc="left",
                 fontsize=13)  # fmt: skip
    ax = axes[1, 0]
    ax.set_ylim(0.5, 1.5)
    ax.set_xlabel(r"Wavenumber, $|\mathbf{k}|/K$")
    ax.set_ylabel(r"Thermal $\langle S(\mathbf{k})\rangle / k_B T$")
    ax.set_title(r"(d) equilibrium: every wave at $S = k_BT$", loc="left", fontsize=13)
    note = ("exact to $10^{-12}$ for every wave and start;\n"
            "selective $S(\\mathbf{k})$ needs a weighted potential:\n"
            r"$\langle S(\mathbf{k})\rangle = k_BT/v(\mathbf{k})$")  # fmt: skip
    ax.text(0.03, 0.06, note, transform=ax.transAxes, fontsize=11, va="bottom")
    ax = axes[1, 1]
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.05)
    ax.axhline(1.0, color="black", lw=0.8, ls=":")
    ax.set_xlabel(r"Time, $|\mathbf{k}|^2 t$")
    ax.set_ylabel("Share Left in the Excited Wave")
    ax.set_title("(e) a one-wave excitation leaks to others", loc="left", fontsize=13)
    ax = axes[1, 2]
    ax.set_xlabel(r"Collective Frequency, $\omega/|\mathbf{k}|$")
    ax.set_ylabel("Weight in the Wave's Fluctuation")
    ax.set_title(f"(f) MD: {labels[0]}, $|\\mathbf{{k}}| = K/2$ rings at many "
                 "frequencies", loc="left", fontsize=13)  # fmt: skip


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
    data = [{tag: {"x": load(r / f"exclusion_chi{tag}.npz"),
                   "l": load(r / f"landscape_chi{tag}.npz"),
                   "w": load(r / f"winding_chi{tag}.npz")} for tag, *_ in TAGS}
            for r in args.roots]  # fmt: skip
    figure(data, args.labels, args.out, args.dpi)
    print("Done.", flush=True)
