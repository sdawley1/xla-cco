#!/usr/bin/env python3

"""
The coupling and loop statistics of the stealthy ground states, start by start.

Reads the per-start caches of the coupling and loop analyses
(experiments/slurm/coupling_loops_cpu.slurm) in each root (--roots; the
start is the root's parent directory: flows2d = Poisson, gau2d = Gaussian,
url2d = URL, rsa2d = saturated RSA, hip2d = HIP) at every chi tag present,
and writes <out>/coupling_starts_compare.png (one color per start, as in
batten2008_hessian_sign_plot):

- (a) the relative width 2 (omega_+ - omega_-) / (omega_+ + omega_-) of the
  largest spacing of omega/K in 0.3-1 (phonon_chi<tag>.npz): median and
  16-84 % band over ground states; filled markers where the median width
  exceeds twice the uncoupled shell spacing (a gap), open otherwise;
- (b) the gap edges omega_- (solid) and omega_+ (dashed), only where (a)
  finds a gap;
- (c) the irreducible share of the S(k)-only moment, I_n / P_n with
  I_n = <w_n>_GS - <w_n>_sur and P_n = <w_n>_sur, at n = 4 (solid) and
  n = 12 (dashed) (loops_chi<tag>.npz), with the standard error;
- (d) the exact four-wave (quadrilateral) loop-phase bias 2M c_1 of the
  ground states (loopphase_chi<tag>.npz), mean and standard error, against
  the random-phase and Gaussian-field (SCBA) models of every start (grey:
  their envelope, mean +- 2 standard errors);
- (e) the same for triads;
- (f) the Kolmogorov distance of the coupling spectrum lambda(C) from the
  Marchenko-Pastur law with the measured variance, c_eff = Var lambda
  (mp_chi<tag>.npz): ground states (solid) and starts (dashed).

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
from typing import Dict, List, Optional

# ----- numerics -----
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batten2008_lattice import cell_lengths, wave_vectors, waves  # noqa: E402
from batten2008_mp_plot import (  # noqa: E402
    CHI_COLOR,
    START_COLORS,
    START_LABELS,
    ks_eff,
)
from batten2008_phonon_gap_plot import largest_gap  # noqa: E402
from batten2008_plot import finish, load, plt  # noqa: E402  (house rcParams)
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

TRAJ = Path("experiments/trajectories")
DEFAULT_ROOTS = [TRAJ / "flows2d/N1024", TRAJ / "gau2d/N1000", TRAJ / "url2d/N1024",
                 TRAJ / "rsa2d/N999", TRAJ / "hip2d/N1000"]  # fmt: skip
GUIDE = "0.6"
BAND = "0.82"
ORDERS = (4, 12)  # loop orders of panel (c)

# ============================================================
# PER-TAG SUMMARIES
# ============================================================


def gap_summary(root: Path, tag: str) -> Optional[Dict[str, np.ndarray]]:
    """Relative width and edges of the largest omega spacing; None if no cache."""
    path = root / f"phonon_chi{tag}.npz"
    if not path.exists():
        return None
    d = load(path)
    lo, hi = d["gap"][:, 0], d["gap"][:, 1]
    meta = json.loads((root / f"flows_chi{tag}.json").read_text())
    side, k_cut = cell_lengths(meta["basis"]), float(meta["K"])
    kq = np.linalg.norm(wave_vectors(waves(side, k_cut), side), axis=1) / k_cut
    shell = float(np.diff(largest_gap(kq))[0])  # uncoupled spectrum omega = |q|
    rel = 2.0 * (hi - lo) / (hi + lo)
    out = {"chi": float(d["chi"]), "rel": np.percentile(rel, [16.0, 50.0, 84.0]),
           "lo": float(np.median(lo)), "hi": float(np.median(hi)),
           "real": float(np.median(hi - lo)) > 2.0 * shell}  # fmt: skip
    assert out["rel"].shape == (3,) and shell > 0.0, "band and shell spacing"
    return out


def loop_summary(root: Path, tag: str) -> Optional[Dict[str, np.ndarray]]:
    """I_n / P_n and its standard error at ORDERS; None if no cache."""
    path = root / f"loops_chi{tag}.npz"
    if not path.exists():
        return None
    d = load(path)
    w, ws = d["w_gs"], d["w_sur"]
    idx = [int(np.where(d["orders"] == n)[0][0]) for n in ORDERS]
    pair = ws.mean(0)[idx]
    err = np.sqrt(w.var(0, ddof=1) / w.shape[0] + ws.var(0, ddof=1) / ws.shape[0])
    out = {"chi": float(d["chi"]), "share": (w.mean(0)[idx] - pair) / pair,
           "err": err[idx] / np.abs(pair)}  # fmt: skip
    assert out["share"].shape == (len(ORDERS),), "one share per order"
    return out


def phase_summary(root: Path, tag: str) -> Optional[Dict[str, np.ndarray]]:
    """2M c_1 (triad, quad) per model: mean and standard error; None if no cache."""
    path = root / f"loopphase_chi{tag}.npz"
    if not path.exists():
        return None
    d = load(path)
    out: Dict[str, np.ndarray] = {"chi": np.array(float(d["chi"]))}
    for key in ("gs", "sur", "scba"):
        v = d[f"c1_{key}"] * float(d["m2"])  # (realizations, 2)
        out[key] = v.mean(0)
        out[f"{key}_err"] = v.std(0, ddof=1) / np.sqrt(v.shape[0])
    assert out["gs"].shape == (2,), "triad and quadrilateral"
    return out


def ks_summary(root: Path, tag: str) -> Optional[Dict[str, float]]:
    """Kolmogorov distance from MP(c_eff), ground states and starts; None if absent."""
    path = root / f"mp_chi{tag}.npz"
    if not path.exists():
        return None
    d = load(path)
    out = {"chi": float(d["chi"])}
    for key in ("gs", "start"):
        out[key] = ks_eff(d[key])
    assert np.isnan(out["gs"]) or 0.0 <= out["gs"] <= 1.0, "a distance"
    return out


def collect(root: Path) -> Dict[str, Dict[str, Dict]]:
    """Every summary of every chi tag present in root (missing caches skipped)."""
    out: Dict[str, Dict[str, Dict]] = {"gap": {}, "loop": {}, "phase": {}, "ks": {}}
    funcs = {"gap": gap_summary, "loop": loop_summary, "phase": phase_summary,
             "ks": ks_summary}  # fmt: skip
    for tag in CHI_COLOR:
        if not (root / f"flows_chi{tag}.json").exists():
            continue
        for key, func in funcs.items():
            val = func(root, tag)
            if val is not None:
                out[key][tag] = val
    assert set(out) == set(funcs), "one entry per summary"
    return out


# ============================================================
# FIGURE
# ============================================================


def panel_gap(axes: np.ndarray, data: Dict[str, Dict], color: str) -> None:
    """(a) relative gap width with its band, (b) the gap edges."""
    rows = list(data.values())
    if not rows:
        return
    chi = np.array([r["chi"] for r in rows])
    rel = np.array([r["rel"] for r in rows])
    real = np.array([r["real"] for r in rows])
    axes[0].fill_between(chi, rel[:, 0], rel[:, 2], color=color, alpha=0.15, lw=0)
    axes[0].plot(chi, rel[:, 1], color=color, lw=2.0)
    for mask, mfc in ((real, color), (~real, "white")):
        axes[0].plot(chi[mask], rel[mask, 1], ls="", marker="o", ms=8, color=color,
                     mfc=mfc)  # fmt: skip
    # edges only where a gap exists: the largest spacing of a gapless
    # spectrum is a shell spacing whose position jumps from chi to chi
    edges = [np.where(real, [r[key] for r in rows], np.nan) for key in ("lo", "hi")]
    assert len(edges) == 2 and edges[0].shape == chi.shape, "one value per chi"
    axes[1].fill_between(chi, edges[0], edges[1], color=color, alpha=0.15, lw=0)
    for y, ls, mk in ((edges[0], "-", "o"), (edges[1], "--", "s")):
        axes[1].plot(chi, y, color=color, lw=2.0, ls=ls, marker=mk, ms=7)


def panel_loops(ax: plt.Axes, data: Dict[str, Dict], color: str) -> None:
    """(c) I_n / P_n at n = 4 (solid) and 12 (dashed)."""
    rows = list(data.values())
    if not rows:
        return
    chi = np.array([r["chi"] for r in rows])
    for i, (ls, mk) in enumerate((("-", "o"), ("--", "s"))):
        ax.errorbar(chi, [r["share"][i] for r in rows],
                    yerr=[r["err"][i] for r in rows], color=color, ls=ls,
                    marker=mk, ms=7, lw=2.0, capsize=3)  # fmt: skip


def panel_phase(ax: plt.Axes, data: Dict[str, Dict], color: str, col: int) -> None:
    """(d, e) ground-state 2M c_1 (col 1: quadrilaterals, col 0: triads)."""
    rows = list(data.values())
    if not rows:
        return
    chi = np.array([float(r["chi"]) for r in rows])
    ax.errorbar(chi, [r["gs"][col] for r in rows],
                yerr=[r["gs_err"][col] for r in rows], color=color, marker="o",
                ms=7, lw=2.0, capsize=3)  # fmt: skip


def model_band(ax: plt.Axes, starts: List[Dict], col: int) -> List[float]:
    """Grey envelope (mean +- 2 SE) of the random-phase and Gaussian models."""
    x, lo, hi = [], [], []
    for tag in CHI_COLOR:
        rows = [s["phase"][tag] for s in starts if tag in s["phase"]]
        if not rows:
            continue
        vals = [(r[k][col] - 2 * r[f"{k}_err"][col], r[k][col] + 2 * r[f"{k}_err"][col])
                for r in rows for k in ("sur", "scba")]  # fmt: skip
        x.append(float(np.mean([float(r["chi"]) for r in rows])))
        lo.append(min(v[0] for v in vals))
        hi.append(max(v[1] for v in vals))
    if x:
        ax.fill_between(x, lo, hi, color=BAND, lw=0, zorder=0)
    return [min(lo, default=0.0), max(hi, default=0.0)]


def panel_ks(ax: plt.Axes, data: Dict[str, Dict], color: str) -> None:
    """(f) Kolmogorov distance from MP(c_eff): ground states and starts."""
    rows = list(data.values())
    if not rows:
        return
    chi = np.array([r["chi"] for r in rows])
    ax.plot(chi, [r["gs"] for r in rows], color=color, marker="o", ms=7, lw=2.0)
    ax.plot(chi, [r["start"] for r in rows], color=color, marker="s", ms=7, lw=1.4,
            ls="--")  # fmt: skip


def figure(starts: List[Dict], out: Path, dpi: int) -> None:
    """Gap, edges, loop shares, loop-phase biases and MP distance, every start."""
    fig, axes = plt.subplots(2, 3, figsize=(20.0, 11.5), constrained_layout=True)
    for col, ax in ((1, axes[1, 0]), (0, axes[1, 1])):
        model_band(ax, starts, col)
    for s in starts:
        panel_gap(axes[0, :2], s["gap"], s["color"])
        panel_loops(axes[0, 2], s["loop"], s["color"])
        panel_phase(axes[1, 0], s["phase"], s["color"], 1)
        panel_phase(axes[1, 1], s["phase"], s["color"], 0)
        panel_ks(axes[1, 2], s["ks"], s["color"])
    style(axes)
    handles = [Line2D([], [], color=s["color"], lw=2.0, marker="o") for s in starts]
    fig.legend(handles, [s["label"] for s in starts], loc="upper center",
               ncol=len(starts), bbox_to_anchor=(0.5, 0.0), frameon=False)  # fmt: skip
    handles = [Line2D([], [], color="0.3", lw=2.0, marker="o"),
               Line2D([], [], color="0.3", lw=2.0, ls="--", marker="s"),
               Line2D([], [], color="0.3", ls="", marker="o", mfc="white"),
               Patch(color=BAND)]  # fmt: skip
    names = [r"solid: lower edge (b), $n = 4$ (c), ground states (f)",
             r"dashed: upper edge (b), $n = 12$ (c), starts (f)",
             r"open: no gap, width $\leq$ 2 shell spacings (a)",
             "random-phase and Gaussian models, mean $\\pm$ 2 SE (d, e)"]  # fmt: skip
    fig.legend(handles, names, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.04),
               frameon=False)  # fmt: skip
    finish(fig, out / "coupling_starts_compare", dpi)


def style(axes: np.ndarray) -> None:
    """Labels and titles."""
    titles = (("(a) relative width of the largest $\\omega$ spacing",
               r"Relative Width, $\Delta\omega/\omega_{\rm mid}$"),
              ("(b) gap edges (median over ground states)", r"Frequency, $\omega/K$"),
              ("(c) loops beyond $S(k)$: irreducible share",
               r"Irreducible Share, $I_n / P_n$"),
              ("(d) four-wave loop-phase bias, ground states",
               r"Quadrilaterals, $2M c_1$"),
              ("(e) triad loop-phase bias, ground states", r"Triads, $2M c_1$"),
              (r"(f) distance of $\lambda(C)$ from MP ($c_{\rm eff}$)",
               r"Kolmogorov Distance"))  # fmt: skip
    for ax, (title, ylab) in zip(axes.flat, titles, strict=True):
        ax.set_title(title, loc="left", fontsize=13)
        ax.set_ylabel(ylab)
        ax.set_xlabel(r"Constraint Ratio, $\chi$")
    axes[0, 0].set_ylim(0.0, None)
    axes[1, 2].set_ylim(0.0, None)
    for ax in (axes[0, 2], axes[1, 0], axes[1, 1]):
        ax.axhline(0.0, color=GUIDE, lw=0.8, zorder=0)


def report(starts: List[Dict]) -> None:
    """Print the plotted values per start and chi."""
    for s in starts:
        for tag in CHI_COLOR:
            g, lp = s["gap"].get(tag), s["loop"].get(tag)
            ph, ks = s["phase"].get(tag), s["ks"].get(tag)
            parts = [f"{s['label']} chi{tag}:"]
            if g is not None:
                parts.append(f"gap {g['lo']:.3f}-{g['hi']:.3f} rel {g['rel'][1]:.3f} "
                             f"({'gap' if g['real'] else 'no gap'})")  # fmt: skip
            if lp is not None:
                parts.append(f"I4/P4 {lp['share'][0]:+.3f} "
                             f"I12/P12 {lp['share'][1]:+.3f}")  # fmt: skip
            if ph is not None:
                parts.append(f"2Mc1 quad {ph['gs'][1]:+.2f} (models "
                             f"{ph['sur'][1]:+.2f}, {ph['scba'][1]:+.2f}) triad "
                             f"{ph['gs'][0]:+.2f}")  # fmt: skip
            if ks is not None:
                parts.append(f"KS(c_eff) GS {ks['gs']:.3f} start {ks['start']:.3f}")
            if len(parts) > 1:
                print(" ".join(parts), flush=True)


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--roots", nargs="+", type=Path, default=DEFAULT_ROOTS)
    parser.add_argument("--out", type=Path, default=Path("experiments/figures"))
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    starts = []
    for root in args.roots:
        key = root.parent.name
        assert key in START_COLORS, f"unknown start {key} (root {root})"
        starts.append({"label": START_LABELS[key], "color": START_COLORS[key],
                       **collect(root)})  # fmt: skip
    assert any(s["gap"] or s["loop"] for s in starts), "no caches found"
    report(starts)
    figure(starts, args.out, args.dpi)
    print("Done.", flush=True)
