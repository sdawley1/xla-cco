#!/usr/bin/env python3

"""
The shoulder of the coupling spectrum at lambda(C) = 1: long waves that cannot couple.

The coupling of two constrained waves q, p (half-space set, |q|, |p| < K)
is fixed exactly by the density at their sum and difference:

  K(Re q, Re p) = (q.p)/V Re[rho(q-p) - rho(q+p)],
  K(Re q, Im p) = -(q.p)/V Im[rho(q+p) + rho(q-p)],
  K(Im q, Im p) = (q.p)/V Re[rho(q-p) + rho(q+p)],

with rho(0) = N on the diagonal; C = D^-1/2 K D^-1/2. At a ground state
rho = 0 for 0 < |k| < K, so every pair with |q -+ p| < K is uncoupled; in
particular all waves with |q| < K/2 are uncoupled from one another (C is the
identity on them) and couple only to the outer waves. This script builds C
from rho on the 2K lattice (checked against C from the gradients J),
for 2D Poisson ground states at every chi tag (flows 0..B-1), and for
random-phase surrogates of rho with the same lattice: phase-randomized (each
|rho(k)| kept), a step S (0 inside K, 1 outside) at N = 1024 and N = 4096,
and an ideal gas (S = 1). Writes <out>/coupling_mp_shoulder.png:

- (a) the density of lambda(C) at the ground states, every chi;
- (b) log10 |C_qp| of one ground state at the largest chi, waves by |q|;
- (c) each eigenvalue against the mean |q|/K of its eigenvector;
- (d) the dips: least density in 0.3 <= lambda <= 0.8 and 1.05 <= lambda
  <= 1.5 (inside the 5-95 % quantiles) against chi (ground states; step-S
  surrogate for reference);
- (e) the density at the largest chi: ground state and surrogates;
- (f) F - F_MP (ratio c) of the full spectrum and of the outer waves only
  (|q| >= K/2), every chi: the shoulder of coupling_mp_compare;
- (g) the peak ratio, density at 1 over the mean density at 0.8 and 1.2;
- (h) the smallest eigenvalue against chi with the MP hard edge
  (1 - sqrt c)^2, c = 2M/(N d) -> 1 as chi -> 1/2.

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
from batten2008_mp_plot import CHI_COLOR, coupling_eigs, half_space_waves, mp_cdf  # noqa: E402
from batten2008_plot import finish, plt  # noqa: E402  (house rcParams)

# ============================================================
# CONSTANTS
# ============================================================

GUIDE = "0.6"
SUR_STYLE = {"ground": ("black", "-"), "phase": ("#c0392b", "--"),
             "step": ("#1f4e9c", "-."), "step4096": ("#1f4e9c", ":"),
             "ideal": ("#b9770e", "-")}  # fmt: skip
SUR_LABEL = {"ground": "ground states", "phase": r"phases randomized, $|\rho|$ kept",
             "step": r"step $S$, random phases", "step4096": r"step $S$, $N = 4096$",
             "ideal": r"ideal gas, $S = 1$"}  # fmt: skip
BINS = np.linspace(0.0, 3.0, 121)
SEED = 7

# ============================================================
# HELPERS
# ============================================================


def lattice_rho(x: np.ndarray, side: float, nmax: int) -> np.ndarray:
    """rho(k) = sum_j exp(i k.r_j) on the (4 nmax + 1)^2 lattice |n_i| <= 2 nmax."""
    k = 2.0 * np.pi * np.arange(-2 * nmax, 2 * nmax + 1) / side
    out: np.ndarray = np.exp(1j * np.outer(x[:, 0], k)).T @ np.exp(
        1j * np.outer(x[:, 1], k)
    )
    assert out.shape == (k.size, k.size) and x.shape[1] == 2, "2D lattice"
    return out


def coupling_from_rho(rho: np.ndarray, nq: np.ndarray, side: float) -> np.ndarray:
    """C from rho at q -+ p (the identity in the module docstring)."""
    off = (rho.shape[0] - 1) // 2
    kv = nq * 2.0 * np.pi / side
    qp, vol = kv @ kv.T, side * side
    dm = nq[:, None, :] - nq[None] + off
    dp = nq[:, None, :] + nq[None] + off
    rm, rp = rho[dm[..., 0], dm[..., 1]], rho[dp[..., 0], dp[..., 1]]
    rr = qp / vol * (rm.real - rp.real)
    ri = -qp / vol * (rp.imag + rm.imag)
    ii = qp / vol * (rm.real + rp.real)
    k = np.block([[rr, ri], [ri.T, ii]])
    d = np.sqrt(np.diag(k))
    assert np.all(d > 0.0) and np.allclose(k, k.T), "symmetric, positive diagonal"
    return k / np.outer(d, d)


def lattice_kmag(nmax: int, side: float) -> np.ndarray:
    """|k| on the rho lattice."""
    n = np.arange(-2 * nmax, 2 * nmax + 1)
    out = 2.0 * np.pi / side * np.hypot(*np.meshgrid(n, n, indexing="ij"))
    assert out.shape == (n.size, n.size), "lattice"
    return out


def surrogate(s: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Hermitian Gaussian rho with E|rho|^2 = N S(k), rho(0) = N."""
    g = s.shape[0]
    z = (rng.normal(size=(g, g)) + 1j * rng.normal(size=(g, g))) * np.sqrt(n * s / 2.0)
    out = (z + np.conj(z[::-1, ::-1])) / np.sqrt(2.0)
    out[g // 2, g // 2] = n
    assert np.allclose(out, np.conj(out[::-1, ::-1])), "Hermitian"
    return out


def phase_randomized(rho: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Same |rho(k)|, independent uniform phases (Hermitian), rho(0) kept."""
    g = rho.shape[0]
    z = np.abs(rho) * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, (g, g)))
    half = (np.arange(g * g) < g * g // 2).reshape(g, g)
    out = np.where(half, z, 0.0)
    out = out + np.conj(out[::-1, ::-1])
    out[g // 2, g // 2] = rho[g // 2, g // 2]
    assert np.allclose(np.abs(out[half]), np.abs(rho[half])), "moduli kept"
    return out


def waves(side: float, k_cut: float) -> np.ndarray:
    """Integer vectors n of the half-space waves 0 < 2 pi |n| / L < K, by |k|."""
    lim = int(np.ceil(k_cut * side / (2.0 * np.pi)))
    g = np.stack(np.meshgrid(*[np.arange(-lim, lim + 1)] * 2, indexing="ij"), -1)
    n = g.reshape(-1, 2)
    n = n[(n[:, 0] > 0) | ((n[:, 0] == 0) & (n[:, 1] > 0))]
    n = n[np.linalg.norm(n * 2.0 * np.pi / side, axis=1) < k_cut]
    out = n[np.argsort(np.linalg.norm(n, axis=1), kind="stable")]
    assert out.ndim == 2 and out.shape[0] > 0, "waves"
    return out


# ============================================================
# COMPUTE
# ============================================================


def compute(root: Path, tag: str, n_states: int) -> Dict[str, np.ndarray]:
    """Spectra (ground states and surrogates), one C and its eigenvectors."""
    meta = json.loads((root / f"flows_chi{tag}.json").read_text())
    side, k_cut, n = float(meta["basis"][0][0]), float(meta["K"]), int(meta["N"])
    nq = waves(side, k_cut)
    assert nq.shape[0] == int(meta["M"]), "wave count"
    kv = half_space_waves(np.array([side, side]), k_cut, nq.shape[0])
    nmax = int(np.ceil(k_cut * side / (2.0 * np.pi)))
    kmag = lattice_kmag(nmax, side)
    rng = np.random.default_rng(SEED)
    with np.load(root / f"flows_chi{tag}.npz", mmap_mode="r") as f:
        states = np.asarray(f["s_final"][:n_states]) % 1.0 * side
    out: Dict[str, List[np.ndarray]] = {k: [] for k in SUR_STYLE}
    outer = np.r_[np.where(np.linalg.norm(kv, axis=1) >= k_cut / 2)[0]]
    outer = np.r_[outer, outer + nq.shape[0]]
    extra: Dict[str, np.ndarray] = {}
    for b, x in enumerate(states):
        rho = lattice_rho(x, side, nmax)
        c = coupling_from_rho(rho, nq, side)
        lam, vec = np.linalg.eigh(c)
        out["ground"].append(lam)
        out.setdefault("outer", []).append(np.linalg.eigvalsh(c[np.ix_(outer, outer)]))
        if b == 0:
            err = float(np.abs(lam - coupling_eigs(x, kv, side * side)).max())
            assert err < 1e-10, f"rho identity fails ({err:.1e})"
            w = vec[: nq.shape[0]] ** 2 + vec[nq.shape[0] :] ** 2
            km = np.linalg.norm(kv, axis=1) / k_cut
            extra = {"c0": c, "lam0": lam, "kbar0": km @ w, "km": km,
                     "identity_err": np.array(err)}  # fmt: skip
        out["phase"].append(np.linalg.eigvalsh(coupling_from_rho(
            phase_randomized(rho, rng), nq, side)))  # fmt: skip
        step = (kmag >= k_cut).astype(float)
        for key, s in (("step", step), ("ideal", np.ones_like(step))):
            out[key].append(np.linalg.eigvalsh(coupling_from_rho(
                surrogate(s, n, rng), nq, side)))  # fmt: skip
    nq4 = waves(2.0 * side, k_cut)
    nmax4 = int(np.ceil(k_cut * 2.0 * side / (2.0 * np.pi)))
    step4 = (lattice_kmag(nmax4, 2.0 * side) >= k_cut).astype(float)
    out["step4096"].append(np.linalg.eigvalsh(coupling_from_rho(
        surrogate(step4, 4 * n, rng), nq4, 2.0 * side)))  # fmt: skip
    res = {k: np.asarray(v) for k, v in out.items()}
    res.update(extra, c=np.array(2.0 * nq.shape[0] / (2 * n)),
               chi=np.array(float(meta["chi_actual"])))  # fmt: skip
    print(f"{tag}: identity err {float(res['identity_err']):.1e}", flush=True)
    return res


def load_or_compute(root: Path, tag: str, n_states: int) -> Dict[str, np.ndarray]:
    """Cached in <root>/mp_shoulder_chi<tag>.npz."""
    path = root / f"mp_shoulder_chi{tag}.npz"
    if path.exists():
        with np.load(path) as z:
            cached = {k: np.asarray(z[k]) for k in z.files}
        if cached["ground"].shape[0] >= n_states:
            return cached
    res = compute(root, tag, n_states)
    np.savez(path, **res)
    assert path.exists(), "cache written"
    return res


def density(ev: np.ndarray, at: float, half: float = 0.02) -> float:
    """Eigenvalue density at `at` from the share within +-half."""
    out = float(np.mean(np.abs(np.ravel(ev) - at) < half)) / (2.0 * half)
    assert out >= 0.0 and half > 0.0, "density"
    return out


def peak_ratio(ev: np.ndarray) -> float:
    """Density at 1 over the mean density at 0.8 and 1.2."""
    side = 0.5 * (density(ev, 0.8) + density(ev, 1.2))
    return density(ev, 1.0) / side if side > 0.0 else float("nan")


# ============================================================
# FIGURE
# ============================================================


def hist(ax: plt.Axes, ev: np.ndarray, color: str, ls: str, label: str) -> None:
    """Step histogram (density) of pooled eigenvalues."""
    h, _ = np.histogram(np.ravel(ev), bins=BINS, density=True)
    ax.step(0.5 * (BINS[1:] + BINS[:-1]), h, where="mid", color=color, ls=ls, lw=1.8,
            label=label)  # fmt: skip


def figure(data: Dict[str, Dict], out: Path, dpi: int) -> None:
    """Density, zero block, eigenvectors, dips, surrogates, F - F_MP, peak, edge."""
    fig, axes = plt.subplots(2, 4, figsize=(26.0, 11.5), constrained_layout=True)
    for tag, d in data.items():
        hist(axes[0, 0], d["ground"], CHI_COLOR[tag], "-",
             rf"$\chi = {float(d['chi']):.2f}$")  # fmt: skip
    hi = data[max(data, key=lambda t: float(data[t]["chi"]))]
    lo = data[min(data, key=lambda t: float(data[t]["chi"]))]
    panel_matrix(axes[0, 1], fig, hi)
    for d, tag_c in ((hi, "#2e7d32"), (lo, "black")):
        axes[0, 2].scatter(d["lam0"], d["kbar0"], s=5, color=tag_c, alpha=0.5,
                           label=rf"$\chi = {float(d['chi']):.2f}$")  # fmt: skip
    for key, (color, ls) in SUR_STYLE.items():
        hist(axes[1, 0], hi[key], color, ls, SUR_LABEL[key])
    panel_dips(axes[0, 3], data)
    panel_edge(axes[1, 3], data)
    x = np.geomspace(1e-3, 5.0, 2000)
    for tag, d in data.items():
        c = float(d["c"])
        for key, ls in (("ground", "-"), ("outer", ":")):
            s = np.sort(np.ravel(d[key]))
            f = np.searchsorted(s, x, side="right") / s.size
            axes[1, 1].plot(x, f - mp_cdf(c if key == "ground" else outer_c(d), x),
                            color=CHI_COLOR[tag], ls=ls,
                            lw=2.0 if ls == "-" else 1.6)  # fmt: skip
    chi = np.array([float(d["chi"]) for d in data.values()])
    for key, (color, ls) in SUR_STYLE.items():
        axes[1, 2].plot(chi, [peak_ratio(d[key]) for d in data.values()], color=color,
                        ls=ls, lw=2.0, marker="o", ms=6,
                        label=SUR_LABEL[key])  # fmt: skip
    style(axes, float(hi["chi"]))
    finish(fig, out / "coupling_mp_shoulder", dpi)


def min_density(ev: np.ndarray, lo: float, hi: float) -> float:
    """Least histogram density (bins of 0.05) in [lo, hi] inside the support.

    The window is clipped to the 5-95 % quantiles of the spectrum, so a window
    reaching past an edge of the support does not read as a gap.
    """
    q05, q95 = np.quantile(np.ravel(ev), [0.05, 0.95])
    lo, hi = max(lo, float(q05)), min(hi, float(q95))
    edges = np.arange(lo, hi + 1e-9, 0.05) if hi > lo else np.array([lo, lo])
    h, _ = np.histogram(np.ravel(ev), bins=np.arange(-1.0, 4.0 + 1e-9, 0.05),
                        density=True)  # fmt: skip
    centres = np.arange(-1.0, 4.0, 0.05) + 0.025
    sel = (centres >= edges[0]) & (centres <= edges[-1])
    assert np.any(sel), "window inside the histogram"
    return float(h[sel].min())


def panel_dips(ax: plt.Axes, data: Dict[str, Dict]) -> None:
    """Least density near 0.5 and near 1.1 against chi."""
    chi = np.array([float(d["chi"]) for d in data.values()])
    for (lo, hi), mk in (((0.3, 0.8), "o"), ((1.05, 1.5), "s")):
        for key in ("ground", "step"):
            color, ls = SUR_STYLE[key]
            ax.plot(chi, [min_density(d[key], lo, hi) for d in data.values()],
                    color=color, ls=ls, lw=2.0, marker=mk, ms=7,
                    mfc=None if key == "ground" else "none",
                    label=f"{SUR_LABEL[key]}, {lo:g}-{hi:g}")  # fmt: skip
    ax.set_ylim(0.0, None)
    ax.set_xlabel(r"Constraint Ratio, $\chi$")
    ax.set_ylabel(r"Least Density in the Window")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("(d) the dips near 0.5 and 1.1", loc="left", fontsize=13)


def panel_edge(ax: plt.Axes, data: Dict[str, Dict]) -> None:
    """Smallest eigenvalue against chi with the MP hard edge."""
    chi = np.array([float(d["chi"]) for d in data.values()])
    c = np.array([float(d["c"]) for d in data.values()])
    lam_min = [float(np.median(d["ground"][:, 0])) for d in data.values()]
    ax.plot(chi, lam_min, color="black", lw=2.0, marker="o", ms=7,
            label="ground states (median)")  # fmt: skip
    x = np.linspace(0.05, 0.4999, 400)
    ax.plot(x, (1.0 - np.sqrt(2.0 * x)) ** 2, color="#b9770e", lw=2.0, ls="--",
            label=r"MP edge $(1-\sqrt{c})^2$, $c \approx 2\chi$")  # fmt: skip
    ax.plot(chi, (1.0 - np.sqrt(c)) ** 2, ls="", marker="x", ms=9, color="#b9770e")
    ax.set_yscale("log")
    ax.set_xlabel(r"Constraint Ratio, $\chi$")
    ax.set_ylabel(r"Smallest Eigenvalue, $\lambda_{\min}(C)$")
    ax.legend(frameon=False, fontsize=10)
    ax.set_title(r"(h) the soft edge closes as $\chi \to 1/2$", loc="left",
                 fontsize=13)  # fmt: skip


def outer_c(d: Dict[str, np.ndarray]) -> float:
    """MP ratio of the outer-wave block (|q| >= K/2): its own 2M' / (N d)."""
    m_outer = d["outer"].shape[1]
    out = float(d["c"]) * m_outer / d["ground"].shape[1]
    assert 0.0 < out < 1.0, "ratio"
    return out


def panel_matrix(ax: plt.Axes, fig: plt.Figure, d: Dict[str, np.ndarray]) -> None:
    """log10 |C_qp| (Re-Re block), waves ordered by |q|."""
    m = d["km"].shape[0]
    blk = np.abs(d["c0"][:m, :m] - np.eye(m))
    img = ax.imshow(np.log10(np.maximum(blk, 1e-16)), origin="lower", cmap="magma",
                    vmin=-16, vmax=0, extent=(0, 1, 0, 1), aspect="auto")  # fmt: skip
    ticks = [0.25, 0.5, 0.75]
    pos = [float(np.searchsorted(d["km"], t)) / m for t in ticks]
    ax.set_xticks(pos, [f"{t:g}" for t in ticks])
    ax.set_yticks(pos, [f"{t:g}" for t in ticks])
    fig.colorbar(img, ax=ax, label=r"$\log_{10}|C_{\mathbf{qp}}|$ (Re, Re)")
    ax.set_xlabel(r"Wave $\mathbf{p}$, ordered by $|\mathbf{p}|/K$")
    ax.set_ylabel(r"Wave $\mathbf{q}$, ordered by $|\mathbf{q}|/K$")
    ax.set_title(rf"(b) $\chi = {float(d['chi']):.2f}$: zero where "
                 r"$|\mathbf{q}\mp\mathbf{p}| < K$", loc="left",
                 fontsize=13)  # fmt: skip


def style(axes: np.ndarray, chi_hi: float) -> None:
    """Labels and titles."""
    ax = axes[0, 0]
    ax.set_xlabel(r"Coupling Spectrum, $\lambda(C)$")
    ax.set_ylabel("Eigenvalue Density")
    ax.axvline(1.0, color=GUIDE, lw=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=11)
    ax.set_title("(a) ground states: a peak at 1 grows with $\\chi$", loc="left",
                 fontsize=13)  # fmt: skip
    ax = axes[0, 2]
    ax.set_xscale("log")
    ax.axvline(1.0, color=GUIDE, lw=0.8, zorder=0)
    ax.set_xlabel(r"Coupling Spectrum, $\lambda(C)$")
    ax.set_ylabel(r"Eigenvector Mean $|\mathbf{q}|/K$")
    ax.legend(frameon=False, fontsize=11, markerscale=3)
    ax.set_title("(c) the eigenvalues at 1 are long waves", loc="left", fontsize=13)
    ax = axes[1, 0]
    ax.set_xlim(0.0, 2.5)
    ax.axvline(1.0, color=GUIDE, lw=0.8, zorder=0)
    ax.set_xlabel(r"Coupling Spectrum, $\lambda(C)$")
    ax.set_ylabel("Eigenvalue Density")
    ax.legend(frameon=False, fontsize=10)
    ax.set_title(rf"(e) $\chi = {chi_hi:.2f}$: ground state and random-phase "
                 "surrogates", loc="left", fontsize=13)  # fmt: skip
    ax = axes[1, 1]
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 5.0)
    ax.axhline(0.0, color=GUIDE, lw=0.8, zorder=0)
    ax.axvline(1.0, color=GUIDE, lw=0.8, zorder=0)
    ax.set_xlabel(r"Coupling Spectrum, $\lambda(C)$")
    ax.set_ylabel(r"$F(\lambda) - F_{\rm MP}(\lambda)$")
    ax.set_title(r"(f) solid: all waves; dotted: $|\mathbf{q}| \geq K/2$ only",
                 loc="left", fontsize=13)  # fmt: skip
    ax = axes[1, 2]
    ax.axhline(1.0, color=GUIDE, lw=0.8, zorder=0)
    ax.set_xlabel(r"Constraint Ratio, $\chi$")
    ax.set_ylabel(r"Peak Ratio, $\rho(1) / \frac{1}{2}[\rho(0.8) + \rho(1.2)]$")
    ax.legend(frameon=False, fontsize=10)
    ax.set_title("(g) the peak needs the exclusion region and large $\\chi$",
                 loc="left", fontsize=13)  # fmt: skip


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    default_root = Path("experiments/trajectories/flows2d/N1024")
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--tags", nargs="+", default=["1-10", "1-4", "2-5", "9-20"])
    parser.add_argument("--states", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("experiments/figures"))
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    assert args.states >= 1 and set(args.tags) <= set(CHI_COLOR), "tags"
    args.out.mkdir(parents=True, exist_ok=True)
    data = {t: load_or_compute(args.root, t, args.states) for t in args.tags}
    for d in data.values():
        km, lam, kb = d["km"], d["lam0"], d["kbar0"]
        near = np.abs(lam - 1.0) < 0.02
        long_ = np.r_[np.where(km < 0.5)[0], km.shape[0] + np.where(km < 0.5)[0]]
        blk = d["c0"][np.ix_(long_, long_)] - np.eye(long_.size)
        print(f"chi {float(d['chi']):.3f}: |q|<K/2 coords {long_.size}/{lam.size} "
              f"({long_.size / lam.size:.2f}), max |C| inside {np.abs(blk).max():.1e}; "
              f"near-1 eigvec |q|/K {np.median(kb[near]):.2f} "
              f"(all {np.median(kb):.2f}); "
              + ", ".join(f"{k} {peak_ratio(d[k]):.2f}" for k in SUR_STYLE)
              + f", outer-only {peak_ratio(d['outer']):.2f}", flush=True)  # fmt: skip
    figure(data, args.out, args.dpi)
    print("Done.", flush=True)
