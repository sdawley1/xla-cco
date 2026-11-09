#!/usr/bin/env python3

"""
A frequency gap in the vibrational spectrum of stealthy ground states.

At a ground state the nonzero normal modes (mass 1) are the eigenvectors of
the wave coupling K = (2/V) J J^T with omega^2 = lambda(K); K is fixed by the
density at the pair wave vectors,
K(q, p) = (q.p)/V [rho(q -+ p) terms] (batten2008_mp_shoulder_plot), which is
the plane-wave (Galerkin) form of the scalar wave operator -div(eta grad)
with eta a sum of point weights at the particles, restricted to |q| < K. An
uncoupled wave has omega = |q|. The coupling spectrum lambda(C) of
coupling_mp_compare divides each wave by its own |q|^2, so it does not
resolve frequency; omega does. For every chi tag (2D Poisson ground states,
flows 0..B-1, one or more roots for N) this caches omega and the eigenvector
mean |q|/K in <root>/phonon_chi<tag>.npz and writes
<out>/coupling_phonon_gap.png:

- (a) the density of states in omega/K (nonzero modes), every chi, with the
  homogeneous reference (omega = |q|, 2D: 2 omega / K^2 on [0, K]);
- (b) the same on a log scale near the gap;
- (c) each mode's frequency against the mean |q|/K of its eigenvector
  (an effective dispersion): median and 25-75 % band per |q|/K bin, every
  chi (one panel per chi in coupling_phonon_dispersion.png, with 16-84 %
  and 2.5-97.5 % bands and the gap shaded);
- (d) the relative width of the largest spacing in 0.3 < omega/K < 1
  (median over ground states) against the largest spacing of the uncoupled
  spectrum omega = |q| (k-shells); a gap must exceed the latter;
- (e) the densities of (omega/K)^2 = lambda(K)/K^2 and of lambda(C) at the
  largest chi (lambda(C) from mp_shoulder_chi<tag>.npz): C = D^-1/2 K D^-1/2
  is a congruence of K, which keeps only the inertia, not the gaps;
- (f) the gap edges (where a gap) and half the position of the first peak of
  S(k) beyond K (a Bragg-like estimate).

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
from typing import Dict

# ----- numerics -----
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batten2008_mp_plot import CHI_COLOR  # noqa: E402
from batten2008_mp_shoulder_plot import lattice_kmag, lattice_rho, waves  # noqa: E402
from batten2008_plot import finish, plt  # noqa: E402  (house rcParams)

# ============================================================
# CONSTANTS
# ============================================================

GUIDE = "0.6"
BINS = np.linspace(0.0, 1.5, 301)
GAP_WINDOW = (0.3, 1.0)  # omega/K range searched for the largest gap
PEAK_RANGE = (1.0, 2.0)  # |k|/K range searched for the first S(k) peak

# ============================================================
# COMPUTE
# ============================================================


def k_matrix(rho: np.ndarray, nq: np.ndarray, side: float) -> np.ndarray:
    """K = (2/V) J J^T from rho at q -+ p (Re, Im coordinates)."""
    off = (rho.shape[0] - 1) // 2
    kv = nq * 2.0 * np.pi / side
    qp, vol = kv @ kv.T, side * side
    dm = nq[:, None, :] - nq[None] + off
    dp = nq[:, None, :] + nq[None] + off
    rm, rp = rho[dm[..., 0], dm[..., 1]], rho[dp[..., 0], dp[..., 1]]
    ri = -qp / vol * (rp.imag + rm.imag)
    out = np.block([[qp / vol * (rm.real - rp.real), ri],
                    [ri.T, qp / vol * (rm.real + rp.real)]])  # fmt: skip
    assert np.allclose(out, out.T), "symmetric"
    return out


def largest_gap(om: np.ndarray) -> np.ndarray:
    """(lower, upper) of the widest spacing of sorted omega/K in GAP_WINDOW."""
    o = np.sort(om[(om > GAP_WINDOW[0]) & (om < GAP_WINDOW[1])])
    assert o.size > 2, "modes in the window"
    i = int(np.argmax(np.diff(o)))
    return np.array([o[i], o[i + 1]])


def compute(root: Path, tag: str, n_states: int) -> Dict[str, np.ndarray]:
    """omega/K, eigenvector |q|/K, lambda(C), gaps and S(k) peaks per state."""
    meta = json.loads((root / f"flows_chi{tag}.json").read_text())
    side, k_cut, n = float(meta["basis"][0][0]), float(meta["K"]), int(meta["N"])
    nq = waves(side, k_cut)
    m = nq.shape[0]
    assert m == int(meta["M"]), "wave count"
    nmax = int(np.ceil(k_cut * side / (2.0 * np.pi)))
    km = np.linalg.norm(nq * 2.0 * np.pi / side, axis=1) / k_cut
    klat = lattice_kmag(nmax, side) / k_cut
    ring = (klat > PEAK_RANGE[0]) & (klat < PEAK_RANGE[1])
    with np.load(root / f"flows_chi{tag}.npz", mmap_mode="r") as f:
        states = np.asarray(f["s_final"][:n_states]) % 1.0 * side
    rows: Dict[str, list] = {"om": [], "kbar": [], "lamc": [], "gap": [], "peak": []}
    for x in states:
        rho = lattice_rho(x, side, nmax)
        kmat = k_matrix(rho, nq, side)
        lam, vec = np.linalg.eigh(kmat)
        w = vec[:m] ** 2 + vec[m:] ** 2
        om = np.sqrt(np.maximum(lam, 0.0)) / k_cut
        d = np.sqrt(np.diag(kmat))
        rows["om"].append(om)
        rows["kbar"].append(km @ w)
        rows["lamc"].append(lam / np.maximum((d**2) @ (vec**2), 1e-300))
        rows["gap"].append(largest_gap(om))
        s = np.abs(rho[ring]) ** 2 / n
        b = np.round(klat[ring] / 0.02).astype(int)
        mean = np.bincount(b, s) / np.maximum(np.bincount(b), 1)
        rows["peak"].append(np.argmax(mean) * 0.02)
    out = {k: np.asarray(v) for k, v in rows.items()}
    out.update(chi=np.array(float(meta["chi_actual"])), n=np.array(n))
    print(f"{root.name} chi{tag}: gap {np.median(out['gap'], axis=0).round(3)}",
          flush=True)  # fmt: skip
    return out


def load_or_compute(root: Path, tag: str, n_states: int) -> Dict[str, np.ndarray]:
    """Cached in <root>/phonon_chi<tag>.npz."""
    path = root / f"phonon_chi{tag}.npz"
    if path.exists():
        with np.load(path) as z:
            cached = {k: np.asarray(z[k]) for k in z.files}
        if cached["om"].shape[0] >= n_states:
            return cached
    res = compute(root, tag, n_states)
    np.savez(path, **res)
    assert path.exists(), "cache written"
    return res


# ============================================================
# FIGURE
# ============================================================


def figure(data: Dict[str, Dict], out: Path, dpi: int) -> None:
    """DOS in omega, dispersion, gap against chi, omega^2 vs lambda(C), edges."""
    fig, axes = plt.subplots(2, 3, figsize=(20.0, 11.5), constrained_layout=True)
    centre = 0.5 * (BINS[1:] + BINS[:-1])
    tags = list(data)
    for tag, d in data.items():
        h, _ = np.histogram(d["om"].ravel(), bins=BINS, density=True)
        lab = rf"$\chi = {float(d['chi']):.2f}$"
        axes[0, 0].plot(centre, h, color=CHI_COLOR[tag], lw=1.6, label=lab)
        axes[0, 1].plot(centre, np.maximum(h, 1e-3), color=CHI_COLOR[tag], lw=1.6)
    ref = np.where(centre < 1.0, 2.0 * centre, 0.0)
    for ax in axes[0, :2]:
        ax.plot(centre, np.maximum(ref, 1e-3), color=GUIDE, lw=2.0, ls="--",
                label=r"homogeneous, $\omega = |\mathbf{q}|$")  # fmt: skip
    for tag in tags:
        dispersion_line(axes[0, 2], data[tag], CHI_COLOR[tag], band=(25.0, 75.0))
    chi = np.array([float(data[t]["chi"]) for t in tags])
    lo = np.array([np.median(data[t]["gap"][:, 0]) for t in tags])
    hi = np.array([np.median(data[t]["gap"][:, 1]) for t in tags])
    spacing = np.array([float(np.diff(data[t]["ref"])[0]) for t in tags])
    real = (hi - lo) > 2.0 * spacing  # a gap must beat the uncoupled shell spacing
    rel = 2.0 * (hi - lo) / (hi + lo)
    axes[1, 0].plot(chi, rel, color="black", lw=2.0, ls="--")
    axes[1, 0].plot(chi[real], rel[real], color="black", ls="", marker="o", ms=9)
    axes[1, 0].plot(chi[~real], rel[~real], color="black", ls="", marker="o", ms=9,
                    mfc="none")  # fmt: skip
    axes[1, 0].plot(chi, 2.0 * spacing / (hi + lo), color="#1f4e9c", ls=":",
                    marker="x", ms=8, lw=1.4)  # fmt: skip
    panel_lambda(axes[1, 1], data[tags[-1]])
    half_peak = np.array([0.5 * np.median(data[t]["peak"]) for t in tags])
    axes[1, 2].fill_between(chi[real], lo[real], hi[real], color="0.8", lw=0)
    axes[1, 2].plot(chi[real], lo[real], color="black", marker="o", lw=2.0,
                    label="lower gap edge")  # fmt: skip
    axes[1, 2].plot(chi[real], hi[real], color="#c0392b", marker="s", lw=2.0,
                    label="upper gap edge")  # fmt: skip
    axes[1, 2].plot(
        chi,
        half_peak,
        color="#b9770e",
        marker="D",
        lw=1.6,
        ls="--",
        label=r"$\frac{1}{2}k_{\rm peak}/K$ (first $S$ peak)",
    )
    style(axes)
    for t, a, b, s_, r in zip(tags, lo, hi, spacing, real, strict=True):
        pairs = zip(data[t]["om"], data[t]["gap"], strict=True)
        below = np.median([np.mean(o < g[0]) for o, g in pairs])
        print(f"chi{t}: largest spacing {a:.3f}-{b:.3f} (width {b - a:.3f}, "
              f"relative {2 * (b - a) / (a + b):.3f}; uncoupled shell spacing "
              f"{s_:.3f}; gap {'yes' if r else 'no'}), share below {below:.3f}, "
              f"half S peak {0.5 * np.median(data[t]['peak']):.3f}",
              flush=True)  # fmt: skip
    finish(fig, out / "coupling_phonon_gap", dpi)


DISP_BINS = np.arange(0.0, 1.0 + 1e-9, 0.025)  # eigenvector mean |q|/K bins


def dispersion_stats(d: Dict[str, np.ndarray], qs: tuple) -> Dict[str, np.ndarray]:
    """Median and percentile band of omega/K per bin of eigenvector mean |q|/K."""
    kb, om = d["kbar"].ravel(), d["om"].ravel()
    idx = np.digitize(kb, DISP_BINS) - 1
    nb = DISP_BINS.shape[0] - 1
    stats = np.full((3, nb), np.nan)
    count = np.bincount(idx[(idx >= 0) & (idx < nb)], minlength=nb)
    for i in np.where(count >= 10)[0]:
        stats[:, i] = np.percentile(om[idx == i], [qs[0], 50.0, qs[1]])
    out = {"x": 0.5 * (DISP_BINS[1:] + DISP_BINS[:-1]), "lo": stats[0],
           "mid": stats[1], "hi": stats[2], "count": count}  # fmt: skip
    assert out["mid"].shape == (nb,), "one value per bin"
    return out


def dispersion_line(ax: plt.Axes, d: Dict[str, np.ndarray], color: str,
                    band: tuple, light: bool = False) -> None:  # fmt: skip
    """Median omega/K against eigenvector mean |q|/K, with a percentile band."""
    s = dispersion_stats(d, band)
    ax.fill_between(s["x"], s["lo"], s["hi"], color=color, alpha=0.18, lw=0)
    if light:
        wide = dispersion_stats(d, (2.5, 97.5))
        ax.fill_between(wide["x"], wide["lo"], wide["hi"], color=color, alpha=0.08,
                        lw=0)  # fmt: skip
    ax.plot(s["x"], s["mid"], color=color, lw=2.0,
            label=rf"$\chi = {float(d['chi']):.2f}$")  # fmt: skip


def figure_dispersion(data: Dict[str, Dict], out: Path, dpi: int) -> None:
    """Effective dispersion per chi: median and bands, with the gap shaded."""
    tags = list(data)
    fig, axes = plt.subplots(2, 3, figsize=(19.0, 11.0), constrained_layout=True,
                             sharex=True, sharey=True)  # fmt: skip
    letters = iter("abcdef")
    for ax, tag in zip(axes.flat, tags, strict=False):
        d = data[tag]
        dispersion_line(ax, d, CHI_COLOR[tag], band=(16.0, 84.0), light=True)
        ax.plot([0.0, 1.0], [0.0, 1.0], color=GUIDE, lw=1.0, ls="--")
        g = np.median(d["gap"], axis=0)
        if float(g[1] - g[0]) > 2.0 * float(np.diff(d["ref"])[0]):
            ax.axhspan(g[0], g[1], color="0.85", lw=0, zorder=0)
        ax.set_title(f"({next(letters)}) $\\chi = {float(d['chi']):.2f}$",
                     loc="left", fontsize=13)  # fmt: skip
        ax.legend([], [], frameon=False)
    for ax in axes[1, :]:
        ax.set_xlabel(r"Eigenvector Mean $|\mathbf{q}|/K$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Frequency, $\omega/K$")
    axes[0, 0].set_xlim(0.0, 1.0)
    axes[0, 0].set_ylim(0.0, 1.35)
    handles = [plt.Line2D([], [], color="0.3", lw=2.0),
               plt.Rectangle((0, 0), 1, 1, color="0.3", alpha=0.25),
               plt.Rectangle((0, 0), 1, 1, color="0.3", alpha=0.1),
               plt.Line2D([], [], color=GUIDE, lw=1.0, ls="--"),
               plt.Rectangle((0, 0), 1, 1, color="0.85")]  # fmt: skip
    names = ["median", "16-84 %", "2.5-97.5 %", r"uncoupled, $\omega = |\mathbf{q}|$",
             "frequency gap (median edges)"]  # fmt: skip
    fig.legend(handles, names, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.0),
               frameon=False)  # fmt: skip
    finish(fig, out / "coupling_phonon_dispersion", dpi)


def panel_lambda(ax: plt.Axes, d: Dict[str, np.ndarray]) -> None:
    """Densities of (omega/K)^2 = lambda(K)/K^2 and of lambda(C), largest chi."""
    x = np.geomspace(1e-5, 5.0, 241)
    om2 = (d["om"] ** 2).ravel()
    h, _ = np.histogram(om2, bins=x, density=True)
    cen = np.sqrt(x[1:] * x[:-1])
    ax.plot(cen, np.maximum(h * cen, 1e-4), color="black", lw=2.0,
            label=r"$\lambda(K)/K^2 = (\omega/K)^2$")  # fmt: skip
    if "lam_c" in d:
        hc, _ = np.histogram(d["lam_c"].ravel(), bins=x, density=True)
        ax.plot(cen, np.maximum(hc * cen, 1e-4), color="#16a085", lw=2.0,
                label=r"$\lambda(C)$, $C = D^{-1/2} K D^{-1/2}$")  # fmt: skip
    ax.set_title(rf"(e) $\chi = {float(d['chi']):.2f}$: the gap in $\omega^2$ "
                 r"is lost in $\lambda(C)$", loc="left", fontsize=13)  # fmt: skip


def style(axes: np.ndarray) -> None:
    """Labels and titles."""
    ax = axes[0, 0]
    ax.set_xlabel(r"Frequency, $\omega/K$")
    ax.set_ylabel("Density of Nonzero Modes")
    ax.legend(frameon=False, fontsize=10, ncol=2)
    ax.set_title(r"(a) vibrational DOS, $\omega^2 = \lambda(K)$", loc="left",
                 fontsize=13)  # fmt: skip
    ax = axes[0, 1]
    ax.set_yscale("log")
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(1e-3, 10.0)
    ax.set_xlabel(r"Frequency, $\omega/K$")
    ax.set_ylabel("Density of Nonzero Modes")
    ax.set_title("(b) a gap opens as $\\chi \\to 1/2$", loc="left", fontsize=13)
    ax = axes[0, 2]
    ax.plot([0, 1], [0, 1], color=GUIDE, lw=1.0, ls="--")
    ax.set_xlabel(r"Eigenvector Mean $|\mathbf{q}|/K$")
    ax.set_ylabel(r"Frequency, $\omega/K$")
    ax.set_xlim(0.0, 1.0)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    ax.set_title("(c) effective dispersion: median, 25-75 % band", loc="left",
                 fontsize=13)  # fmt: skip
    ax = axes[1, 0]
    ax.set_ylim(0.0, None)
    ax.set_xlabel(r"Constraint Ratio, $\chi$")
    ax.set_ylabel(r"Relative Width, $\Delta\omega/\omega_{\rm mid}$")
    ax.set_title("(d) largest spacing (filled: a gap) vs uncoupled shells (x)",
                 loc="left", fontsize=13)  # fmt: skip
    ax = axes[1, 1]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 10.0)
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel(r"Density per $\ln$ Eigenvalue")
    ax.legend(frameon=False, fontsize=11)
    ax = axes[1, 2]
    ax.set_xlabel(r"Constraint Ratio, $\chi$")
    ax.set_ylabel(r"Frequency, $\omega/K$")
    ax.legend(frameon=False, fontsize=10)
    ax.set_title("(f) gap edges and the Bragg-like estimate", loc="left", fontsize=13)


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    default_root = Path("experiments/trajectories/flows2d/N1024")
    parser.add_argument("--root", type=Path, default=default_root)
    all_tags = ["1-10", "1-4", "7-20", "2-5", "9-20", "49-100"]
    parser.add_argument("--tags", nargs="+", default=all_tags)
    parser.add_argument("--states", type=int, default=6)
    parser.add_argument("--out", type=Path, default=Path("experiments/figures"))
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    assert set(args.tags) <= set(CHI_COLOR) and args.states >= 1, "tags"
    args.out.mkdir(parents=True, exist_ok=True)
    data = {t: load_or_compute(args.root, t, args.states) for t in args.tags}
    for t, d in data.items():
        meta = json.loads((args.root / f"flows_chi{t}.json").read_text())
        side, k_cut = float(meta["basis"][0][0]), float(meta["K"])
        kq = np.linalg.norm(waves(side, k_cut) * 2.0 * np.pi / side, axis=1) / k_cut
        d["ref"] = largest_gap(kq)  # the uncoupled spectrum omega = |q|
        cache = args.root / f"mp_shoulder_chi{t}.npz"
        if cache.exists():
            with np.load(cache) as z:
                d["lam_c"] = np.asarray(z["ground"])
    figure(data, args.out, args.dpi)
    figure_dispersion(data, args.out, args.dpi)
    print("Done.", flush=True)
