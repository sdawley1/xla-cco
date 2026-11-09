#!/usr/bin/env python3

"""
Do the dips of the coupling spectrum become gaps? Ground states at N = 1024 and 4096.

For 2D Poisson flows to stealthy ground states at the same chi tags under
two roots (--roots, e.g. flows2d/N1024 and flows2d/N4096), this builds the
normalized coupling C = D^-1/2 K D^-1/2 of every ground state from rho on the
2K lattice (batten2008_mp_shoulder_plot.coupling_from_rho, exact) and caches
the spectra in <root>/finite_chi<tag>.npz. A true gap needs the density in a
dip to fall as N grows at fixed chi; a dip whose density is N-independent is
a pseudo-gap. Writes <out>/coupling_mp_finite_size.png:

- (a, b) the eigenvalue density at each chi, N = 1024 and 4096;
- (c) the density near the dips on a log scale, every chi and N;
- (d) the least density in 0.3 <= lambda <= 0.8 and 1.05 <= lambda <= 1.5
  (inside the 5-95 % quantiles) against N, mean and standard error over
  ground states;
- (e) the smallest eigenvalue against N with the MP hard edge (1 - sqrt c)^2
  of each system's own c = 2M/(N d);
- (f) the edges of the largest gap in omega/K (omega^2 = lambda(K), the
  vibrational frequencies) against N; (c) shows the omega density.

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
from typing import Dict, List

# ----- numerics -----
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batten2008_mp_shoulder_plot import (  # noqa: E402
    coupling_from_rho,
    lattice_rho,
    min_density,
    peak_ratio,
    waves,
)
from batten2008_phonon_gap_plot import k_matrix, largest_gap  # noqa: E402
from batten2008_plot import finish, plt  # noqa: E402  (house rcParams)
from matplotlib.lines import Line2D  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

CHI_COLOR = {"9-20": "#2e7d32", "49-100": "#16a085", "2-5": "#c0392b",
             "7-20": "#7d3c98"}  # fmt: skip
N_STYLE = {1024: ("--", "o"), 4096: ("-", "s")}
BINS = np.arange(0.0, 3.0 + 1e-9, 0.025)
WINDOWS = ((0.3, 0.8), (1.05, 1.5))
GUIDE = "0.6"

# ============================================================
# COMPUTE
# ============================================================


def spectra(root: Path, tag: str, n_states: int) -> Dict[str, np.ndarray]:
    """Ground-state spectra of C (n_states, 2M) and the system's c, chi, N."""
    meta = json.loads((root / f"flows_chi{tag}.json").read_text())
    side, k_cut, n = float(meta["basis"][0][0]), float(meta["K"]), int(meta["N"])
    nq = waves(side, k_cut)
    assert nq.shape[0] == int(meta["M"]), "wave count"
    nmax = int(np.ceil(k_cut * side / (2.0 * np.pi)))
    with np.load(root / f"flows_chi{tag}.npz", mmap_mode="r") as f:
        states = np.asarray(f["s_final"][:n_states]) % 1.0 * side
        energy = np.asarray(f["energy_final"][:n_states])
    assert np.all(energy < 1e-12), f"not all at ground states: {energy.max():.1e}"
    ev, om = [], []
    for b, x in enumerate(states):
        rho = lattice_rho(x, side, nmax)
        ev.append(np.linalg.eigvalsh(coupling_from_rho(rho, nq, side)))
        lam = np.linalg.eigvalsh(k_matrix(rho, nq, side))
        om.append(np.sqrt(np.maximum(lam, 0.0)) / k_cut)
        print(f"  {root.name} chi{tag}: state {b + 1}/{len(states)}", flush=True)
    kq = np.linalg.norm(nq * 2.0 * np.pi / side, axis=1) / k_cut
    out = {"ev": np.asarray(ev), "om": np.asarray(om), "ref": largest_gap(kq),
           "c": np.array(2.0 * nq.shape[0] / (2 * n)),
           "chi": np.array(float(meta["chi_actual"])), "n": np.array(n)}  # fmt: skip
    return out


def load_or_compute(root: Path, tag: str, n_states: int) -> Dict[str, np.ndarray]:
    """Cached in <root>/finite_chi<tag>.npz."""
    path = root / f"finite_chi{tag}.npz"
    if path.exists():
        with np.load(path) as z:
            cached = {k: np.asarray(z[k]) for k in z.files}
        if cached["ev"].shape[0] >= n_states and "om" in cached:
            return cached
    res = spectra(root, tag, n_states)
    np.savez(path, **res)
    assert path.exists(), "cache written"
    return res


def per_state(ev: np.ndarray, fn: object) -> np.ndarray:
    """fn applied to each ground state's spectrum (mean and SE taken later)."""
    out = np.array([fn(row[None, :]) for row in ev])  # type: ignore[operator]
    assert out.shape == (ev.shape[0],), "one value per state"
    return out


# ============================================================
# FIGURE
# ============================================================


def figure(data: Dict[str, Dict[int, Dict]], out: Path, dpi: int) -> None:
    """Densities, dips, edge and peak against N at each chi."""
    fig, axes = plt.subplots(2, 3, figsize=(20.0, 11.5), constrained_layout=True)
    tags = list(data)
    for t_i, tag in enumerate(tags[:2]):
        ax = axes[0, t_i]
        for n, d in data[tag].items():
            h, _ = np.histogram(d["ev"].ravel(), bins=BINS, density=True)
            ls, _ = N_STYLE[n]
            ax.step(0.5 * (BINS[1:] + BINS[:-1]), h, where="mid", ls=ls, lw=1.8,
                    color=CHI_COLOR[tag], label=f"$N = {n}$, "
                    rf"$\chi = {float(d['chi']):.3f}$")  # fmt: skip
        ax.set_ylim(0.0, 3.0)
        ax.legend(frameon=False, fontsize=11)
    obins = np.linspace(0.3, 1.0, 141)
    for tag in tags:
        for n, d in data[tag].items():
            h, _ = np.histogram(d["om"].ravel(), bins=obins, density=True)
            ls, _ = N_STYLE[n]
            axes[0, 2].plot(0.5 * (obins[1:] + obins[:-1]), np.maximum(h, 1e-4),
                            ls=ls, lw=1.6, color=CHI_COLOR[tag])  # fmt: skip
    for tag in tags:
        ns = sorted(data[tag])
        for (lo, hi), mk in zip(WINDOWS, ("o", "s"), strict=True):
            vals = [per_state(data[tag][n]["ev"],
                              lambda e, a=lo, b=hi: min_density(e, a, b))
                    for n in ns]  # fmt: skip
            axes[1, 0].errorbar(ns, [v.mean() for v in vals],
                                yerr=[v.std(ddof=1) / np.sqrt(v.size) if v.size > 1
                                      else 0.0 for v in vals],
                                color=CHI_COLOR[tag], marker=mk, ms=8, lw=2.0,
                                capsize=4,
                                mfc="none" if mk == "s" else None)  # fmt: skip
        lam = [np.median(data[tag][n]["ev"][:, 0]) for n in ns]
        edge = [(1.0 - np.sqrt(float(data[tag][n]["c"]))) ** 2 for n in ns]
        axes[1, 1].plot(ns, lam, color=CHI_COLOR[tag], marker="o", ms=8, lw=2.0)
        axes[1, 1].plot(ns, edge, color=CHI_COLOR[tag], marker="x", ms=10, lw=1.0,
                        ls=":")  # fmt: skip
        for col, mk in ((0, "o"), (1, "s")):
            edge_n = [np.array([largest_gap(o)[col] for o in data[tag][n]["om"]])
                      for n in ns]  # fmt: skip
            axes[1, 2].errorbar(ns, [v.mean() for v in edge_n],
                                yerr=[v.std(ddof=1) / np.sqrt(v.size) for v in edge_n],
                                color=CHI_COLOR[tag], marker=mk, ms=8, lw=2.0,
                                capsize=4, mfc="none" if col else None)  # fmt: skip
    style(axes, data)
    finish(fig, out / "coupling_mp_finite_size", dpi)


def style(axes: np.ndarray, data: Dict[str, Dict[int, Dict]]) -> None:
    """Labels, titles and the shared legend."""
    tags = list(data)
    for t_i, tag in enumerate(tags[:2]):
        ax = axes[0, t_i]
        chi = float(next(iter(data[tag].values()))["chi"])
        ax.axvline(1.0, color=GUIDE, lw=0.8, zorder=0)
        ax.set_xlabel(r"Coupling Spectrum, $\lambda(C)$")
        ax.set_ylabel("Eigenvalue Density")
        ax.set_title(f"({'ab'[t_i]}) $\\chi \\approx {chi:.2f}$: density at two $N$",
                     loc="left", fontsize=13)  # fmt: skip
    ax = axes[0, 2]
    ax.set_yscale("log")
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(1e-3, 10.0)
    ax.set_xlabel(r"Frequency, $\omega/K$, $\omega^2 = \lambda(K)$")
    ax.set_ylabel("Density of Nonzero Modes")
    ax.set_title("(c) vibrational DOS near the gap (dashed 1024, solid 4096)",
                 loc="left", fontsize=13)  # fmt: skip
    for ax in axes[1, :]:
        ax.set_xscale("log", base=2)
        ax.set_xlabel(r"Number of Particles, $N$")
    axes[1, 0].set_ylim(0.0, None)
    axes[1, 0].set_ylabel("Least Density in the Window")
    axes[1, 0].set_title("(d) a gap needs this to fall with $N$ (o: 0.3-0.8, "
                         "s: 1.05-1.5)", loc="left", fontsize=13)  # fmt: skip
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel(r"Smallest Eigenvalue, $\lambda_{\min}(C)$")
    axes[1, 1].set_title(r"(e) soft edge vs MP $(1-\sqrt{c})^2$ (x)", loc="left",
                         fontsize=13)  # fmt: skip
    axes[1, 2].set_ylabel(r"Gap Edges, $\omega/K$ (filled: lower, open: upper)")
    axes[1, 2].set_title("(f) the frequency gap at two $N$", loc="left", fontsize=13)
    handles = [Line2D([], [], color=CHI_COLOR[t], lw=2.0) for t in tags]
    handles += [Line2D([], [], color="0.3", lw=1.8, ls=N_STYLE[n][0])
                for n in sorted(N_STYLE)]  # fmt: skip
    names = [rf"$\chi \approx {float(next(iter(data[t].values()))['chi']):.2f}$"
             for t in tags]  # fmt: skip
    names += [f"$N = {n}$" for n in sorted(N_STYLE)]
    axes[0, 0].figure.legend(handles, names, loc="upper center", ncol=len(names),
                             bbox_to_anchor=(0.5, 0.0), frameon=False)  # fmt: skip


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--tags", nargs="+", default=["9-20", "49-100"])
    parser.add_argument("--states", type=int, default=6)
    parser.add_argument("--out", type=Path, default=Path("experiments/figures"))
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    assert set(args.tags) <= set(CHI_COLOR) and args.states >= 2, "tags, states"
    args.out.mkdir(parents=True, exist_ok=True)
    data: Dict[str, Dict[int, Dict]] = {t: {} for t in args.tags}
    for root in args.roots:
        for tag in args.tags:
            d = load_or_compute(root, tag, args.states)
            data[tag][int(d["n"])] = d
    rows: List[str] = []
    for tag, by_n in data.items():
        for n, d in sorted(by_n.items()):
            dips = [
                per_state(d["ev"], lambda e, w=w: min_density(e, *w)) for w in WINDOWS
            ]
            pk = per_state(d["ev"], peak_ratio)
            gaps = np.array([largest_gap(o) for o in d["om"]])
            rows.append(f"chi{tag} N={n} (chi {float(d['chi']):.4f}, "
                        f"c {float(d['c']):.4f},"
                        f" {d['ev'].shape[0]} states): dip 0.3-0.8 {dips[0].mean():.3f}"
                        f" +- {dips[0].std(ddof=1) / np.sqrt(dips[0].size):.3f}, dip"
                        f" 1.05-1.5 {dips[1].mean():.3f} +- "
                        f"{dips[1].std(ddof=1) / np.sqrt(dips[1].size):.3f}, peak "
                        f"{pk.mean():.2f}, lambda_min {np.median(d['ev'][:, 0]):.2e} "
                        f"(MP {(1 - np.sqrt(float(d['c']))) ** 2:.2e}); omega gap "
                        f"{gaps[:, 0].mean():.3f}-{gaps[:, 1].mean():.3f} (width "
                        f"{np.mean(gaps[:, 1] - gaps[:, 0]):.3f} +- "
                        f"{np.std(gaps[:, 1] - gaps[:, 0], ddof=1):.3f}; shell "
                        f"spacing {float(np.diff(d['ref'])[0]):.4f})")  # fmt: skip
    print("\n".join(rows), flush=True)
    figure(data, args.out, args.dpi)
    print("Done.", flush=True)
