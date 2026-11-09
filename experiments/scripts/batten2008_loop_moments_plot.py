#!/usr/bin/env python3

"""
Isolating the loop (beyond-S(k)) contribution to the vibrational spectrum.

Reads <root>/loops_chi<tag>.npz (batten2008_loop_moments.py) for 2D ground
states (--roots: the smallest N is the base, larger N are drawn dashed;
Poisson by default, --prefix names the start: gau2d, url2d, rsa2d, hip2d).
With H = D + W in the complex wave basis and w_n = (1/2M) tr (W/K^2)^n, each
moment splits into non-crossing pairings (SCBA), crossing pairings (random-
phase surrogate mean minus SCBA) and irreducible loops (ground state minus
surrogate mean). Writes <out>/[<prefix>_]loop_moments.png:

- (a) the total moments w_n of the ground states against n, every chi;
- (b) the three parts as fractions of w_n against n (even n), largest chi;
- (c) the irreducible share of the S(k)-only (paired) moment, I_n / P_n,
  against even n, every chi; dashed: larger N (odd n are entirely
  irreducible and are printed: n = 3 is the triad term);
- (d) the density of states in x = lambda(K)/K^2 = (omega/K)^2 at the largest
  chi: ground states, random-phase surrogates and the SCBA (S(k)-only)
  theory, with the ground-state gap shaded (y clipped at 2.5: the soft-mode
  spike at x = 0 reaches about 11);
- (e) the loop contribution to the density, rho_GS - rho_surrogate, for the
  three largest chi (dashed: larger N; y clipped, the x = 0 spike is cut);
- (f) the total-variation distance between the ground-state density and
  the surrogate and SCBA densities, against chi.

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
from batten2008_mp_plot import CHI_COLOR, figure_path, start_title  # noqa: E402
from batten2008_plot import finish, load, plt  # noqa: E402  (house rcParams)
from matplotlib.lines import Line2D  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

TAGS = ("1-10", "1-4", "7-20", "2-5", "9-20", "49-100")
BINS = np.linspace(-0.6, 2.4, 241)
GUIDE = "0.6"
PART_COLORS = {"nc": "#1f4e9c", "cross": "#b9770e", "irr": "#c0392b"}

# ============================================================
# HELPERS
# ============================================================


def parts(d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Mean and error of the total, paired, non-crossing and irreducible moments."""
    w, ws, nc = d["w_gs"], d["w_sur"], d["nc"]
    tot, pair, ncm = w.mean(0), ws.mean(0), nc.mean(0)
    err = np.sqrt(w.var(0, ddof=1) / w.shape[0] + ws.var(0, ddof=1) / ws.shape[0])
    out = {"n": d["orders"], "tot": tot, "pair": pair, "nc": ncm,
           "cross": pair - ncm, "irr": tot - pair, "irr_err": err,
           "w2": np.array(tot[0])}  # fmt: skip
    assert out["tot"].shape == out["n"].shape, "one value per order"
    return out


def density(x: np.ndarray) -> np.ndarray:
    """Histogram density of pooled eigenvalues on BINS."""
    h, _ = np.histogram(x.ravel(), bins=BINS, density=True)
    assert h.shape == (BINS.shape[0] - 1,), "one value per bin"
    return h


def gap_x(d: Dict[str, np.ndarray]) -> np.ndarray:
    """Median largest gap of the ground states in x = (omega/K)^2, 0.09 < x < 1."""
    out = []
    for row in d["x_gs"]:
        o = np.sort(row[(row > 0.09) & (row < 1.0)])
        i = int(np.argmax(np.diff(o)))
        out.append((o[i], o[i + 1]))
    res = np.median(np.asarray(out), axis=0)
    assert res.shape == (2,), "lower and upper edge"
    return res


def tv_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Total-variation distance between two densities on BINS."""
    out = 0.5 * float(np.sum(np.abs(a - b)) * (BINS[1] - BINS[0]))
    assert 0.0 <= out <= 1.0 + 1e-9, "a distance"
    return out


def scba_on_bins(d: Dict[str, np.ndarray]) -> np.ndarray:
    """SCBA density averaged onto BINS."""
    centres = 0.5 * (BINS[1:] + BINS[:-1])
    out = np.interp(centres, d["scba_x"], d["scba_dos"], left=0.0, right=0.0)
    assert out.shape == centres.shape, "one value per bin"
    return out


# ============================================================
# FIGURE
# ============================================================


def figure(data: Dict[int, Dict[str, Dict]], out: Path, dpi: int,
           prefix: str = "") -> None:  # fmt: skip
    """Moments by order, their split, and the loop contribution to the density."""
    fig, axes = plt.subplots(2, 3, figsize=(20.0, 11.5), constrained_layout=True)
    base = data[min(data)]
    tags = [t for t in TAGS if t in base]
    big = tags[-1]
    centres = 0.5 * (BINS[1:] + BINS[:-1])
    for tag in tags:
        p = parts(base[tag])
        even = p["n"] % 2 == 0
        axes[0, 0].plot(p["n"][even], p["tot"][even], color=CHI_COLOR[tag],
                        marker="o", lw=2.0)  # fmt: skip
        even = p["n"] % 2 == 0
        axes[0, 2].errorbar(p["n"][even], (p["irr"] / p["pair"])[even],
                            yerr=(p["irr_err"] / np.abs(p["pair"]))[even],
                            color=CHI_COLOR[tag], marker="o", ms=5, lw=1.8,
                            capsize=3)  # fmt: skip
    for n_sys, by_tag in data.items():
        if n_sys == min(data):
            continue
        for tag, d in by_tag.items():
            p = parts(d)
            even = p["n"] % 2 == 0
            axes[0, 2].plot(p["n"][even], (p["irr"] / p["pair"])[even],
                            color=CHI_COLOR[tag], ls="--", lw=1.6, marker="s", ms=5,
                            mfc="none")  # fmt: skip
    p = parts(base[big])
    even = p["n"] % 2 == 0
    for key in ("nc", "cross", "irr"):
        axes[0, 1].plot(p["n"][even], p[key][even] / p["tot"][even], marker="o",
                        lw=2.0, color=PART_COLORS[key])  # fmt: skip
    d = base[big]
    g = gap_x(d)
    ax = axes[1, 0]
    ax.axvspan(g[0], g[1], color="0.88", lw=0, zorder=0)
    ax.step(centres, density(d["x_gs"]), where="mid", color="black", lw=1.8)
    ax.step(centres, density(d["x_sur"]), where="mid", color=PART_COLORS["irr"],
            lw=1.4)  # fmt: skip
    ax.plot(d["scba_x"], d["scba_dos"], color=PART_COLORS["nc"], lw=2.0, ls="--")
    for tag in tags[-3:]:
        for n_sys, by_tag in data.items():
            if tag not in by_tag:
                continue
            dd = by_tag[tag]
            diff = density(dd["x_gs"]) - density(dd["x_sur"])
            axes[1, 1].plot(centres, diff, color=CHI_COLOR[tag], lw=1.6,
                            ls="-" if n_sys == min(data) else "--")  # fmt: skip
    chi = [float(base[t]["chi"]) for t in tags]
    tv_s = [tv_distance(density(base[t]["x_gs"]), density(base[t]["x_sur"]))
            for t in tags]  # fmt: skip
    tv_c = [tv_distance(density(base[t]["x_gs"]), scba_on_bins(base[t])) for t in tags]
    axes[1, 2].plot(chi, tv_s, color=PART_COLORS["irr"], marker="o", lw=2.0,
                    label="ground states vs surrogates")  # fmt: skip
    axes[1, 2].plot(chi, tv_c, color=PART_COLORS["nc"], marker="s", lw=2.0, ls="--",
                    label="ground states vs SCBA")  # fmt: skip
    style(axes, float(d["chi"]), sorted(data)[1:])
    legend(fig, base, tags, sorted(data)[1:])
    report(data, tags)
    start_title(fig, prefix, min(data))
    finish(fig, figure_path(out, prefix, "loop_moments"), dpi)


def style(axes: np.ndarray, chi_big: float, larger: List[int]) -> None:
    """Labels and titles."""
    ax = axes[0, 0]
    ax.set_yscale("log")
    ax.set_xlabel(r"Order, $n$ (even)")
    ax.set_ylabel(r"Total Moment, $w_n = \frac{1}{2M}\mathrm{tr}(W/K^2)^n$")
    ax.set_title("(a) total moments of the coupling", loc="left", fontsize=13)
    ax = axes[0, 1]
    ax.axhline(0.0, color=GUIDE, lw=0.8, zorder=0)
    ax.set_xlabel(r"Order, $n$ (even)")
    ax.set_ylabel(r"Share of $w_n$")
    ax.set_title(rf"(b) $\chi = {chi_big:.2f}$: non-crossing, crossing, "
                 "irreducible", loc="left", fontsize=13)  # fmt: skip
    ax.legend([Line2D([], [], color=PART_COLORS[k], lw=2.0, marker="o")
               for k in ("nc", "cross", "irr")],
              ["non-crossing pairings (SCBA)", "crossing pairings",
               "irreducible loops (beyond $S$)"], frameon=False,
              fontsize=10)  # fmt: skip
    ax = axes[0, 2]
    ax.axhline(0.0, color=GUIDE, lw=0.8, zorder=0)
    ax.set_xlabel(r"Order, $n$ (even)")
    ax.set_ylabel(r"Irreducible Share of the $S$-Only Moment, $I_n / P_n$")
    note = f" (dashed: $N = {larger[0]}$)" if larger else ""
    ax.set_title(f"(c) the loop contribution by order{note}",
                 loc="left", fontsize=13)  # fmt: skip
    ax = axes[1, 0]
    ax.set_xlim(-0.6, 2.4)
    ax.set_ylim(0.0, 2.5)
    ax.set_xlabel(r"$x = \lambda(K)/K^2 = (\omega/K)^2$")
    ax.set_ylabel("Density of States")
    ax.set_title(rf"(d) $\chi = {chi_big:.2f}$: the gap is not an $S(k)$ effect",
                 loc="left", fontsize=13)  # fmt: skip
    ax.legend([Line2D([], [], color="black", lw=1.8),
               Line2D([], [], color=PART_COLORS["irr"], lw=1.4),
               Line2D([], [], color=PART_COLORS["nc"], lw=2.0, ls="--")],
              ["ground states", "random-phase surrogates (same $|\\rho|$)",
               "SCBA, $S(k)$-only theory"], frameon=False, fontsize=10)  # fmt: skip
    ax = axes[1, 1]
    ax.axhline(0.0, color=GUIDE, lw=0.8, zorder=0)
    ax.set_xlim(-0.6, 2.4)
    ax.set_ylim(-1.0, 1.5)
    ax.set_xlabel(r"$x = (\omega/K)^2$")
    ax.set_ylabel(r"$\rho_{\rm GS}(x) - \rho_{\rm surrogate}(x)$")
    ax.set_title("(e) the loop contribution to the density", loc="left",
                 fontsize=13)  # fmt: skip
    ax = axes[1, 2]
    ax.set_ylim(0.0, None)
    ax.set_xlabel(r"Constraint Ratio, $\chi$")
    ax.set_ylabel("Total-Variation Distance")
    ax.legend(frameon=False, fontsize=11)
    ax.set_title("(f) how much of the spectrum $S(k)$ misses", loc="left",
                 fontsize=13)  # fmt: skip


def legend(fig: plt.Figure, base: Dict[str, Dict], tags: List[str],
           larger: List[int]) -> None:  # fmt: skip
    """Shared chi legend beneath the panels (and the larger N, if any)."""
    handles = [Line2D([], [], color=CHI_COLOR[t], lw=2.0) for t in tags]
    names = [rf"$\chi = {float(base[t]['chi']):.2f}$" for t in tags]
    for n_sys in larger:
        handles.append(Line2D([], [], color="0.3", lw=1.6, ls="--"))
        names.append(f"$N = {n_sys}$ (c, e)")
    fig.legend(handles, names, loc="upper center", ncol=len(names),
               bbox_to_anchor=(0.5, 0.0), frameon=False)  # fmt: skip


def report(data: Dict[int, Dict[str, Dict]], tags: List[str]) -> None:
    """Print the decomposition and the gap-window densities."""
    for n_sys, by_tag in sorted(data.items()):
        for tag in [t for t in tags if t in by_tag]:
            d = by_tag[tag]
            p = parts(d)
            std = p["irr"] / p["w2"] ** (p["n"] / 2.0)
            g = gap_x(d)
            win = (BINS[:-1] >= g[0]) & (BINS[1:] <= g[1])
            dg, ds = density(d["x_gs"]), density(d["x_sur"])
            sc = scba_on_bins(d)
            print(f"N={n_sys} chi {float(d['chi']):.3f}: I_n/w2^(n/2) n=3..8 "
                  f"{np.round(std[1:7], 3).tolist()}; even shares n=4 nc/cross/irr "
                  f"{p['nc'][2] / p['tot'][2]:.3f}/{p['cross'][2] / p['tot'][2]:.3f}/"
                  f"{p['irr'][2] / p['tot'][2]:+.3f}; gap x {g[0]:.3f}-{g[1]:.3f}: "
                  f"density GS {dg[win].mean() if win.any() else np.nan:.3f}, "
                  f"surrogate {ds[win].mean() if win.any() else np.nan:.3f}, SCBA "
                  f"{sc[win].mean() if win.any() else np.nan:.3f}; TV(GS,sur) "
                  f"{tv_distance(dg, ds):.3f}, TV(GS,SCBA) {tv_distance(dg, sc):.3f}",
                  flush=True)  # fmt: skip


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--tags", nargs="+", default=list(TAGS),
                        help="default: every chi tag with a loops cache")  # fmt: skip
    parser.add_argument("--out", type=Path, default=Path("experiments/figures"))
    parser.add_argument("--prefix", default="",
                        help="figure name prefix and start, e.g. hip2d")  # fmt: skip
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    assert set(args.tags) <= set(TAGS), "known chi tags"
    args.out.mkdir(parents=True, exist_ok=True)
    data: Dict[int, Dict[str, Dict]] = {}
    for root in args.roots:
        for tag in [t for t in TAGS if t in args.tags]:
            path = root / f"loops_chi{tag}.npz"
            if path.exists():
                d = load(path)
                data.setdefault(int(d["n"]), {})[tag] = d
    assert data, "no loops_chi*.npz found"
    figure(data, args.out, args.dpi, args.prefix)
    print("Done.", flush=True)
