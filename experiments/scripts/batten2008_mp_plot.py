#!/usr/bin/env python3

"""
The coupling spectrum of the waves against the Marchenko-Pastur law, all chi.

For 2D starts (Poisson by default; --root, --prefix for the others)
evolved to stealthy ground states at every chi tag given (--tags; default
every tag present in the root), the waves
u = (Re rho_q, Im rho_q) over the constrained set (2M numbers) have
gradients J (2M, N d), coupling K = (2/V) J J^T and the normalized coupling
C = D^-1/2 K D^-1/2 (D = diag K, unit diagonal). If the rows of J were
independent random vectors, the spectrum of C would follow
the Marchenko-Pastur (MP) law with ratio c = 2M / (N d): support
[(1 - sqrt c)^2, (1 + sqrt c)^2], mean 1, variance c, <1/lambda> =
1 / (1 - c). The spectrum is computed at the ground states (s_final) and,
as a random-gradient reference, at the starts (s0) of flows 0..F-1
(--flows), and cached in <root>/mp_chi<tag>.npz. Rectangular cells use
k = 2 pi (n_x / L_x, n_y / L_y) and V = L_x L_y. Writes
<out>/[<prefix>_]coupling_mp_compare.png:

The entries of C are set by the density at the pair wave vectors:
C(Re q, Re p) = (q.p)/(V K_qq^1/2 K_pp^1/2) Re[rho(q-p) - rho(q+p)], so at
a ground state every pair with |q -+ p| < K is uncoupled. With random
phases, Var(lambda) / c = <(q^.p^)^2 [S(q-p) + S(q+p)]> / <2 (q^.p^)^2>
over pairs ("pair prediction"; S = 1 everywhere gives MP, S = 0 inside K
and 1 outside the geometric factor g).

- (a-d) cumulative fraction of the eigenvalues of C, one panel per chi:
  ground states, starts, the MP law (ratio c) and MP with the
  measured variance (ratio c_eff = Var lambda);
- (e) the difference of the cumulative fractions from MP (ratio c);
- (f) Var(lambda) / c against the pair prediction (3 ground states) and g;
- (g) the smallest and largest eigenvalues over the MP edges;
- (h) the Kolmogorov distance from MP (ratio c) and from MP (c_eff).

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
from batten2008_lattice import cell_lengths, present_tags  # noqa: E402
from batten2008_plot import finish, plt  # noqa: E402  (house rcParams)
from matplotlib.lines import Line2D  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

CHI_COLOR = {
    "1-10": "black",
    "1-4": "#1f4e9c",
    "7-20": "#7d3c98",
    "2-5": "#c0392b",
    "9-20": "#2e7d32",
    "49-100": "#16a085",
}  # increasing chi
START_LABELS = {"": "Poisson", "flows2d": "Poisson", "gau2d": "Gaussian",
                "url2d": "URL", "rsa2d": "saturated RSA", "hip2d": "HIP"}  # fmt: skip
# per start, as in batten2008_hessian_sign_plot (roots in this order)
START_COLORS = {"flows2d": "black", "gau2d": "#1f4e9c", "url2d": "#c0392b",
                "rsa2d": "#2e7d32", "hip2d": "#b9770e"}  # fmt: skip
START_COLOR = "0.55"
MP_COLOR = "#b9770e"
GUIDE = "0.6"
GRID = np.geomspace(1e-4, 5.0, 2000)  # lambda grid of the cumulative curves
PAIR_STATES = 3  # ground states per chi for the pair prediction (M^2 N cost)
ZERO_MODE = 1e-13  # lambda_min / lambda_max below this is a round-off zero of C

# ============================================================
# HELPERS
# ============================================================


def start_label(prefix: str) -> str:
    """Start name for titles and legends ("" and flows2d: Poisson)."""
    out = START_LABELS.get(prefix, prefix)
    assert isinstance(out, str) and out, "a start name"
    return out


def figure_path(out: Path, prefix: str, name: str) -> Path:
    """<out>/<name> for the default (Poisson) run, else <out>/<prefix>_<name>."""
    assert name and "/" not in prefix, "a figure name and a plain prefix"
    return out / (f"{prefix}_{name}" if prefix else name)


def root_n(root: Path) -> int:
    """Particle count N of the flows in root (from any flows_chi*.json)."""
    path = next(iter(sorted(root.glob("flows_chi*.json"))), None)
    assert path is not None, f"no flows_chi*.json in {root}"
    out = int(json.loads(path.read_text())["N"])
    assert out > 0, "particle count"
    return out


def start_title(fig: plt.Figure, prefix: str, n: int) -> None:
    """Start label above the figure; none for the default (Poisson) run."""
    assert n > 0, "particle count"
    if prefix:
        fig.suptitle(f"{start_label(prefix)} starts, $N = {n}$", fontsize=18)


def default_tags(root: Path, tags: List[str], order: tuple) -> List[str]:
    """--tags if given, else the tags of `order` present in root."""
    out = list(tags) if tags else present_tags(root, order)
    assert out and set(out) <= set(CHI_COLOR), f"no usable chi tags in {root}"
    return out


def half_space_waves(side: np.ndarray, k_cut: float, m: int) -> np.ndarray:
    """(M, 2) wave vectors 2 pi n / L with 0 < |k| < k_cut, one of each +-k."""
    lim = np.ceil(k_cut * side / (2.0 * np.pi)).astype(int)
    g = np.stack(np.meshgrid(*[np.arange(-v, v + 1) for v in lim], indexing="ij"), -1)
    n = g.reshape(-1, 2)
    n = n[(n[:, 0] > 0) | ((n[:, 0] == 0) & (n[:, 1] > 0))]
    k = n * (2.0 * np.pi / side)
    k = k[np.linalg.norm(k, axis=1) < k_cut]
    assert k.shape == (m, 2), f"{k.shape[0]} waves, metadata says M = {m}"
    return k[np.argsort(np.linalg.norm(k, axis=1), kind="stable")]


def coupling_eigs(x: np.ndarray, kvec: np.ndarray, volume: float) -> np.ndarray:
    """Eigenvalues of C = D^-1/2 K D^-1/2 at Cartesian positions x."""
    n, d = x.shape
    phase = x @ kvec.T  # (N, M), the J of batten2008_plateau.hessian_parts
    ja = -(np.sin(phase)[:, :, None] * kvec[None]).transpose(1, 0, 2).reshape(-1, n * d)
    jb = (np.cos(phase)[:, :, None] * kvec[None]).transpose(1, 0, 2).reshape(-1, n * d)
    jac = np.concatenate([ja, jb])
    k = (2.0 / volume) * jac @ jac.T
    d = np.sqrt(np.diag(k))
    out = np.linalg.eigvalsh(k / np.outer(d, d))
    assert out.shape == (kvec.shape[0] * 2,) and abs(out.mean() - 1.0) < 1e-9, "tr C"
    return out


def pair_factor(x: np.ndarray, kvec: np.ndarray, k_cut: float) -> tuple:
    """Pair prediction of Var(lambda)/c with the measured S, and with a step S."""
    n, m = x.shape[0], kvec.shape[0]
    qh = kvec / np.linalg.norm(kvec, axis=1)[:, None]
    cos2 = (qh @ qh.T) ** 2 * (1.0 - np.eye(m))
    num, geo = 0.0, 0.0
    for sign in (-1.0, 1.0):
        pairs = kvec[:, None, :] + sign * kvec[None, :, :]  # (M, M, 2)
        s = np.zeros((m, m))
        for a in range(m):  # one row of pairs at a time (memory)
            ph = x @ pairs[a].T
            s[a] = (np.cos(ph).sum(0) ** 2 + np.sin(ph).sum(0) ** 2) / n
        num += float((cos2 * s).sum())
        geo += float((cos2 * (np.linalg.norm(pairs, axis=-1) > k_cut)).sum())
    norm = 2.0 * float(cos2.sum())
    assert norm > 0.0 and 0.0 < geo <= norm, "pair sums"
    return num / norm, geo / norm


def compute(root: Path, tag: str, n_flows: int) -> Dict[str, np.ndarray]:
    """Spectra of C at the ground states and starts of n_flows flows."""
    meta = json.loads((root / f"flows_chi{tag}.json").read_text())
    side = cell_lengths(meta["basis"])
    kvec = half_space_waves(side, float(meta["K"]), int(meta["M"]))
    volume = float(np.prod(side))
    with np.load(root / f"flows_chi{tag}.npz", mmap_mode="r") as f:
        s0, s1 = np.asarray(f["s0"][:n_flows]), np.asarray(f["s_final"][:n_flows])
    assert s0.shape == s1.shape and s0.shape[0] == n_flows, "flows"
    pair = [pair_factor(s * side, kvec, float(meta["K"]))
            for s in s1[:PAIR_STATES]]  # fmt: skip
    out = {"gs": np.stack([coupling_eigs(s * side, kvec, volume) for s in s1]),
           "start": np.stack([coupling_eigs(s * side, kvec, volume) for s in s0]),
           "pair_s": np.array([v[0] for v in pair]),
           "pair_geo": np.array(pair[0][1]),
           "c": np.array(2.0 * kvec.shape[0] / s0[0].size),
           "chi": np.array(float(meta["chi_actual"]))}  # fmt: skip
    print(f"{tag}: c = {float(out['c']):.3f}, {n_flows} flows", flush=True)
    return out


def load_or_compute(root: Path, tag: str, n_flows: int) -> Dict[str, np.ndarray]:
    """Cached spectra (<root>/mp_chi<tag>.npz), recomputed if too few flows."""
    path = root / f"mp_chi{tag}.npz"
    if path.exists():
        with np.load(path) as z:
            cached = {k: np.asarray(z[k]) for k in z.files}
        if cached["gs"].shape[0] >= n_flows and "pair_s" in cached:
            return cached
    out = compute(root, tag, n_flows)
    np.savez(path, **out)
    assert path.exists(), "cache written"
    return out


def mp_cdf(c: float, x: np.ndarray) -> np.ndarray:
    """Marchenko-Pastur cumulative distribution (ratio c < 1, unit mean)."""
    lo, hi = (1.0 - np.sqrt(c)) ** 2, (1.0 + np.sqrt(c)) ** 2
    t = np.geomspace(lo, hi, 40001)  # log grid: the soft edge is steep as c -> 1
    dens = np.sqrt(np.maximum((hi - t) * (t - lo), 0.0)) / (2.0 * np.pi * c * t)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(t))])
    assert abs(cum[-1] - 1.0) < 1e-3 and 0.0 < c < 1.0, "MP normalization"
    return np.interp(x, t, cum / cum[-1], left=0.0, right=1.0)


def ecdf(ev: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Empirical cumulative fraction of the pooled eigenvalues at x."""
    s = np.sort(ev.ravel())
    out = np.searchsorted(s, x, side="right") / s.shape[0]
    assert out.shape == x.shape, "one value per x"
    return out


def ks_distance(ev: np.ndarray, c: float) -> float:
    """Kolmogorov distance of the pooled eigenvalues from MP with ratio c."""
    s = np.sort(ev.ravel())
    f_emp = np.arange(1, s.shape[0] + 1) / s.shape[0]
    out = float(np.max(np.abs(f_emp - mp_cdf(c, s))))
    assert 0.0 <= out <= 1.0 and s.size > 0, "a distance"
    return out


def ks_eff(ev: np.ndarray) -> float:
    """Kolmogorov distance from MP with c_eff = Var lambda; NaN unless 0 < c_eff < 1.

    MP(c_eff) is undefined for c_eff >= 1 (HIP starts at chi = 0.4, whose
    spectra have zero modes; D-018).
    """
    c_eff = float(np.mean(ev.var(axis=1)))
    out = ks_distance(ev, c_eff) if 0.0 < c_eff < 1.0 else float("nan")
    assert np.isnan(out) or 0.0 <= out <= 1.0, "a distance"
    return out


def stats(ev: np.ndarray, c: float) -> Dict[str, float]:
    """Variance, edges, mean inverse and Kolmogorov distance against MP.

    <1/lambda> is NaN when the spectrum has zero modes (lambda_min at
    round-off, e.g. HIP starts at chi = 0.4: the wave gradients are
    linearly dependent; D-018).
    """
    s = np.sort(ev.ravel())
    f_emp = np.arange(1, s.shape[0] + 1) / s.shape[0]
    ks = float(np.max(np.abs(f_emp - mp_cdf(c, s))))
    positive = float(s[0]) > ZERO_MODE * float(s[-1])
    inv = float(np.mean(1.0 / ev)) * (1.0 - c) if positive else float("nan")
    out = {"var": float(np.mean(ev.var(axis=1))) / c,
           "lo": float(np.median(ev[:, 0])) / (1.0 - np.sqrt(c)) ** 2,
           "hi": float(np.median(ev[:, -1])) / (1.0 + np.sqrt(c)) ** 2,
           "inv": inv, "ks": ks}  # fmt: skip
    assert out["var"] > 0.0 and not inv <= 0.0, "positive spectrum"
    return out


# ============================================================
# FIGURE
# ============================================================


def figure(data: Dict[str, Dict], out: Path, dpi: int, root: Path,
           prefix: str = "") -> None:  # fmt: skip
    """Cumulative spectra per chi against MP, and the summary statistics."""
    n = len(data)
    assert 1 <= n <= 6, "at most six chi panels"
    fig = plt.figure(figsize=(max(5.0 * n + 2.0, 22.0), 11.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 12)
    top = [fig.add_subplot(grid[0, 12 * i // n : 12 * (i + 1) // n]) for i in range(n)]
    axes = np.array([[None] * 4, [fig.add_subplot(grid[1, 3 * i : 3 * i + 3])
                                  for i in range(4)]], dtype=object)  # fmt: skip
    letters = iter("abcdefghijkl")
    rows = []
    for col, (tag, d) in enumerate(data.items()):
        c, chi, color = float(d["c"]), float(d["chi"]), CHI_COLOR[tag]
        ax = top[col]
        mp = mp_cdf(c, GRID)
        if ax is not None:
            ax.plot(GRID, mp, color=MP_COLOR, lw=3.0, ls="--")
            c_eff = float(np.mean(d["gs"].var(axis=1)))
            if 0.0 < c_eff < 1.0:
                ax.plot(GRID, mp_cdf(c_eff, GRID), color=MP_COLOR, lw=2.0, ls=":")
            ax.plot(GRID, ecdf(d["start"], GRID), color=START_COLOR, lw=2.0)
            ax.plot(GRID, ecdf(d["gs"], GRID), color=color, lw=2.0)
            ax.set_xscale("log")
            ax.set_xlim(min(0.1, max(1e-5, 0.1 * (1.0 - np.sqrt(c)) ** 2)), 5.0)
            ax.set_xlabel(r"Coupling Spectrum, $\lambda(C)$")
            ax.set_ylabel("Cumulative Fraction")
            head = f"({next(letters)}) $\\chi = {chi:.2f}$"
            ax.set_title(f"{head}, $c = 2M/Nd = {c:.2f}$",
                         loc="left", fontsize=13)  # fmt: skip
        axes[1, 0].plot(GRID, ecdf(d["gs"], GRID) - mp, color=color, lw=2.0)
        axes[1, 0].plot(GRID, ecdf(d["start"], GRID) - mp, color=color, lw=1.0, ls=":")
        gs = stats(d["gs"], c)
        gs["pair"] = float(np.mean(d["pair_s"]))
        gs["geo"] = float(d["pair_geo"])
        gs["ks_eff"] = ks_eff(d["gs"])
        st = stats(d["start"], c)
        st["ks_eff"] = ks_eff(d["start"])
        rows.append((chi, c, gs, st, color))
    summary(axes[1, 1:], rows)
    style(axes[1, :], letters)
    handles = [Line2D([], [], color=CHI_COLOR[t], lw=2.0) for t in data]
    handles += [Line2D([], [], color=START_COLOR, lw=2.0),
                Line2D([], [], color=MP_COLOR, lw=3.0, ls="--"),
                Line2D([], [], color=MP_COLOR, lw=2.0, ls=":"),
                Line2D([], [], color="0.3", lw=2.0, marker="o"),
                Line2D([], [], color="0.3", lw=1.0, ls=":", marker="s",
                       mfc="none")]  # fmt: skip
    names = [f"ground states, $\\chi = {float(d['chi']):.2f}$" for d in data.values()]
    start = start_label(prefix)
    names += [f"{start} starts (top row)", "MP, ratio $c$",
              r"MP, ratio $c_{\rm eff} = \mathrm{Var}\,\lambda$ (ground states)",
              "ground states (bottom row)",
              f"{start} starts (bottom row)"]  # fmt: skip
    fig.legend(handles, names, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.0),
               frameon=False)  # fmt: skip
    start_title(fig, prefix, root_n(root))
    finish(fig, figure_path(out, prefix, "coupling_mp_compare"), dpi)


def summary(axes: np.ndarray, rows: List) -> None:
    """(f) variance, (g) edges, (h) mean inverse and Kolmogorov distance."""
    chi = np.array([r[0] for r in rows])
    for k, (key, mk) in enumerate((("gs", "o"), ("start", "s"))):
        st = [r[2 + k] for r in rows]
        ls, mfc = ("-", None) if key == "gs" else (":", "none")
        lw = 2.0 if key == "gs" else 1.0
        kw = {"ls": ls, "lw": lw, "marker": mk, "ms": 7, "mfc": mfc}
        axes[0].plot(chi, [s["var"] for s in st], color="black", **kw)
        axes[1].plot(chi, [s["lo"] for s in st], color="#1f4e9c", **kw)
        axes[1].plot(chi, [s["hi"] for s in st], color="#c0392b", **kw)
        axes[2].plot(chi, [s["ks"] for s in st], color="black", **kw)
        axes[2].plot(chi, [s["ks_eff"] for s in st], color=MP_COLOR, **kw)
    gs_rows = [r[2] for r in rows]
    axes[0].plot(chi, [s["pair"] for s in gs_rows], ls="", marker="x", ms=12, mew=2.5,
                 color="#c0392b")  # fmt: skip
    axes[0].plot(chi, [s["geo"] for s in gs_rows], ls="--", lw=1.2, color=GUIDE)
    axes[0].set_ylim(0.0, 1.15)
    axes[1].set_yscale("log")
    axes[2].set_ylim(0.0, None)
    axes[0].text(0.04, 0.5, "red x: pair prediction (measured $S$)\n"
                 "grey dashed: $S = 0$ inside $K$, 1 outside",
                 transform=axes[0].transAxes, fontsize=12, va="top")  # fmt: skip
    for ax in axes:
        ax.axhline(1.0, color=GUIDE, lw=0.8, zorder=0)
        ax.set_xlabel(r"Constraint Ratio, $\chi$")
    axes[1].text(0.45, 0.3, r"blue: $\lambda_{\min}/(1-\sqrt{c})^2$" "\n"
                 r"red: $\lambda_{\max}/(1+\sqrt{c})^2$", transform=axes[1].transAxes,
                 fontsize=12, va="top")  # fmt: skip
    axes[2].text(0.04, 0.95, r"black: MP, ratio $c$" "\n"
                 r"gold: MP, ratio $c_{\rm eff}$", transform=axes[2].transAxes,
                 fontsize=12, va="top")  # fmt: skip
    for chi_i, c, gs, st, _ in rows:
        print(f"chi {chi_i:.3f}: pair {gs['pair']:.3f}, geometric {gs['geo']:.3f}, "
              f"KS(c_eff) ground {gs['ks_eff']:.3f}, start {st['ks_eff']:.3f}",
              flush=True)  # fmt: skip
        print(f"chi {chi_i:.3f} (c {c:.3f}): ground var/c {gs['var']:.3f}, "
              f"lo {gs['lo']:.3f}, hi {gs['hi']:.3f}, <1/l>(1-c) {gs['inv']:.3f}, "
              f"KS {gs['ks']:.3f} | start var/c {st['var']:.3f}, lo {st['lo']:.3f}, "
              f"hi {st['hi']:.3f}, <1/l>(1-c) {st['inv']:.3f}, KS {st['ks']:.3f}",
              flush=True)  # fmt: skip


def style(axes: np.ndarray, letters: object) -> None:
    """Labels and titles of the summary row."""
    ax = axes[0]
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 5.0)
    ax.axhline(0.0, color=GUIDE, lw=0.8, zorder=0)
    ax.set_xlabel(r"Coupling Spectrum, $\lambda(C)$")
    ax.set_ylabel(r"$F(\lambda) - F_{\rm MP}(\lambda)$")
    titles = ("difference from MP (solid: ground, dotted: start)",
              "exclusion pairs set the variance",
              "edges over MP edges",
              r"Kolmogorov distance, $\sup|F - F_{\rm MP}|$")  # fmt: skip
    ylabels = (None, r"$\mathrm{Var}\,\lambda / c$", "Edge / MP Edge", "Distance")
    for ax, title, ylab in zip(axes, titles, ylabels, strict=True):
        ax.set_title(f"({next(letters)}) {title}", loc="left", fontsize=13)
        if ylab is not None:
            ax.set_ylabel(ylab)


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    default_root = Path("experiments/trajectories/flows2d/N1024")
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--tags", nargs="+", default=None,
                        help="default: every chi tag present in the root")  # fmt: skip
    parser.add_argument("--flows", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("experiments/figures"))
    parser.add_argument("--prefix", default="",
                        help="figure name prefix and start, e.g. hip2d")  # fmt: skip
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    tags = default_tags(args.root, args.tags, tuple(CHI_COLOR))
    assert args.flows >= 1, "flows"
    args.out.mkdir(parents=True, exist_ok=True)
    data = {tag: load_or_compute(args.root, tag, args.flows) for tag in tags}
    figure(data, args.out, args.dpi, args.root, args.prefix)
    print("Done.", flush=True)
