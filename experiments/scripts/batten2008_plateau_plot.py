#!/usr/bin/env python3

"""
Negative curvature along the 2D flows, across initial conditions, and why it plateaus.

Reads hessian_chi<tag>.npz (batten2008_hessian.py; every flow, every
quarter decade of Phi) and plateau_chi<tag>.npz (batten2008_plateau.py; the
split H = G + R for flows 0-9) of each system root (--roots, one label each,
both chi tags). Writes to <out>:

- hessian_negative_compare: panels (c) and (g) of *_hessian_detail for every
  start. (a, b) the number of negative eigenvalues (|lambda| above the
  1e-9 lambda_max cut, median and 10-90 % over flows) for chi = 0.1 and 0.4,
  with (N d - 2M) / 2 for each start and, as points, the negative count of
  P R P (the residual curvature on the tangent space, no cut); (c, d) the
  most negative eigenvalue |lambda_min| with a sqrt(Phi) guide and the cut.
- hessian_plateau_mechanism: (a) sketch: Phi = lambda d^2 around a curved
  manifold, whose tangent curvature has the sign of the side the state is
  on; (b) negative count over N d - 2M for every start and both chi, with
  P R P as points; (c) the soft eigenvalues of H against those of P R P
  (first-order perturbation theory), scaled by sqrt(Phi); (d) the
  distribution of the P R P eigenvalues over sqrt(Phi) at every level
  below Phi = 0.1 (2D Poisson, chi = 0.1, flow 0); (e) the sum rule
  sum_j w_jq = |rho_q|^2: median |sum_j w| / sum_j |w| over the waves;
  (f) the fraction of negative per-particle curvatures (eigenvalues of the
  2 x 2 blocks R_j).

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
from typing import Dict, List, Tuple

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
STYLE = {
    "1-10": (r"$\chi = 0.1$", "-", "o"),
    "1-4": (r"$\chi = 0.25$", "-.", "D"),
    "2-5": (r"$\chi = 0.4$", "--", "s"),
    "9-20": (r"$\chi = 0.45$", ":", "^"),
}  # every chi tag (extra tags only for the first root, --extra-chi)
ZERO_REL = 1e-9  # the eigenvalue cut of batten2008_hessian.py
GUIDE = "0.6"
PHI_LABEL = r"Energy, $\Phi$"

# ============================================================
# HELPERS
# ============================================================


def phi_axis(ax: plt.Axes) -> None:
    """Log energy axis, high to low (time runs left to right)."""
    ax.set_xscale("log")
    if not ax.xaxis_inverted():
        ax.invert_xaxis()
    ax.set_xlabel(PHI_LABEL)


def band(ax: plt.Axes, x: np.ndarray, rows: np.ndarray, color: str, ls: str) -> None:
    """Median and 10-90 % band of rows (flows, x)."""
    q10, q50, q90 = np.nanpercentile(rows, [10.0, 50.0, 90.0], axis=0)
    ax.fill_between(x, q10, q90, color=color, alpha=0.12, lw=0)
    ax.plot(x, q50, color=color, ls=ls, lw=2.0)


def hessian_series(z: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Phi levels each flow reaches, negative counts and |lambda_min| (flows, L)."""
    keep = np.asarray(z["samples"]) <= np.log10(z["phi"][:, 0]).min()
    lam = z["lam"][:, keep]
    nd, m = lam.shape[-1], z["kmag"].shape[0]
    return {
        "phi": 10.0 ** np.asarray(z["samples"])[keep],
        "neg": z["n_negative"][:, keep].astype(float),
        "lam_min": np.maximum(-lam[:, :, 0], 1e-14),
        "cut": ZERO_REL * np.median(lam[:, :, -1], axis=0),
        "tangent": np.array(nd - 2 * m, dtype=float),
    }


def plateau_series(p: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Per level (flows, L): negative P R P count, R block fraction, balance."""
    return {
        "phi": 10.0 ** p["levels"],
        "neg_pr": (p["lam_pr"] < 0.0).sum(axis=-1).astype(float),
        "frac_r": (p["lam_r"] < 0.0).mean(axis=-1),
        "balance": p["balance"],
    }


def styled(sys_d: Dict) -> List[Tuple[str, str, str, str]]:
    """(tag, label, line style, marker) of the chi tags one system has."""
    out = [(t, *STYLE[t]) for t in STYLE if t in sys_d]
    assert out and all(len(o) == 4 for o in out), "known chi tags"
    return out


def all_styles(data: List) -> List[Tuple[str, str, str, str]]:
    """Styles of every chi tag present in any system, in increasing chi."""
    out = [(t, *STYLE[t]) for t in STYLE if any(t in s for s in data)]
    assert out and len(out) <= len(STYLE), "known chi tags"
    return out


def load_all(
    roots: List[Path], extra: List[str]
) -> List[Dict[str, Dict[str, np.ndarray]]]:
    """Per root, per chi tag: hessian and plateau series (extra: first root)."""
    out = []
    for i, root in enumerate(roots):
        sys_d = {}
        for tag in [t for t, *_ in TAGS] + (list(extra) if i == 0 else []):
            h = hessian_series(load(root / f"hessian_chi{tag}.npz"))
            p = load(root / f"plateau_chi{tag}.npz")
            sys_d[tag] = {"h": h, "p": plateau_series(p), "raw": p}
        out.append(sys_d)
    return out


# ============================================================
# FIGURES
# ============================================================


def figure_compare(data: List, labels: List[str], out: Path, dpi: int) -> None:
    """Panels (c) and (g) of *_hessian_detail, every start on one set of axes."""
    cols = all_styles(data)
    fig, axes = plt.subplots(2, len(cols), figsize=(7.0 * len(cols), 10.0),
                             constrained_layout=True, squeeze=False)  # fmt: skip
    for i, sys_d in enumerate(data):
        color = COLORS[i % len(COLORS)]
        for col, (tag, _, _, marker) in enumerate(cols):
            if tag not in sys_d:
                continue
            h, p = sys_d[tag]["h"], sys_d[tag]["p"]
            band(axes[0, col], h["phi"], h["neg"], color, "-")
            axes[0, col].axhline(0.5 * h["tangent"], color=color, lw=0.8, ls=":")
            axes[0, col].plot(p["phi"], np.median(p["neg_pr"], axis=0), ls="",
                              marker=marker, ms=6, color=color, mfc="none")  # fmt: skip
            band(axes[1, col], h["phi"], h["lam_min"], color, "-")
            axes[1, col].plot(h["phi"], h["cut"], color=color, lw=0.8, ls=":")
    letters = "abcdefgh"
    n = len(cols)
    for col, (tag, chi, _, _) in enumerate(cols):
        ax = axes[0, col]
        phi_axis(ax)
        ax.set_title(f"({letters[col]}) {chi}: negative curvature", loc="left")
        ax.set_ylabel(r"Number with $\lambda < 0$")
        ax = axes[1, col]
        phi_axis(ax)
        ax.set_yscale("log")
        x = np.geomspace(1e-12, 1e2, 20)
        ref = np.median(data[0][tag]["h"]["lam_min"], axis=0)
        anchor = np.interp(-3.0, np.log10(data[0][tag]["h"]["phi"])[::-1],
                           ref[::-1])  # fmt: skip
        ax.plot(x, 2.0 * anchor * np.sqrt(x / 1e-3), color=GUIDE, lw=1.0, ls="-.")
        ax.set_title(f"({letters[n + col]}) {chi}: most negative eigenvalue",
                     loc="left")  # fmt: skip
        ax.set_ylabel(r"$|\lambda_{\min}|$")
    handles = [Line2D([], [], color=COLORS[i], lw=2.0) for i in range(len(data))]
    handles += [Line2D([], [], color="0.3", lw=0.8, ls=":"),
                Line2D([], [], color="0.3", ls="", marker="o", mfc="none"),
                Line2D([], [], color=GUIDE, lw=1.0, ls="-.")]  # fmt: skip
    names = labels + [
        r"top: $(Nd-2M)/2$; bottom: cut $10^{-9}\lambda_{\max}$",
        r"top: $\lambda(PRP) < 0$, no cut",
        r"$\propto\sqrt{\Phi}$",
    ]
    fig.legend(handles, names, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.0),
               frameon=False)  # fmt: skip
    finish(fig, out / "hessian_negative_compare", dpi)


def panel_sketch(ax: plt.Axes) -> None:
    """Phi = (r - 1)^2 about a circle: tangent curvature changes sign across it."""
    g = np.linspace(-1.6, 1.6, 401)
    xx, yy = np.meshgrid(g, g)
    phi = (np.hypot(xx, yy) - 1.0) ** 2
    ax.contourf(xx, yy, phi, levels=np.linspace(0, 0.45, 10), cmap="Blues_r",
                extend="max")  # fmt: skip
    th = np.linspace(0.15 * np.pi, 0.85 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color="black", lw=2.5)
    for r, sign, va in ((1.38, "+", "bottom"), (0.62, "-", "top")):
        ax.plot(0.0, r, "o", color="#c0392b", ms=7, zorder=5)
        ax.annotate("", xy=(0.45, r), xytext=(0.0, r),
                    arrowprops=dict(arrowstyle="->", color="#c0392b",
                                    lw=1.8))  # fmt: skip
        circ = np.linspace(-0.5, 0.5, 50)
        ax.plot(r * np.sin(circ), r * np.cos(circ), color="#c0392b", lw=1.0, ls=":")
        txt = (r"outside: $\Phi$ rises, $\lambda > 0$" if sign == "+"
               else r"inside: $\Phi$ falls, $\lambda < 0$")  # fmt: skip
        ax.text(0.0, r + (0.07 if va == "bottom" else -0.07), txt, fontsize=11,
                va=va, ha="center", color="black" if sign == "+" else "white",
                bbox=dict(fc="white", ec="none", alpha=0.8) if sign == "+"
                else None)  # fmt: skip
    ax.text(-1.05, 0.12, "ground-state\nmanifold", fontsize=11, ha="center",
            color="white")  # fmt: skip
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(0.0, 1.75)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(r"(a) $\Phi \approx \lambda\, d^2$ near a curved manifold", loc="left")


def panel_check(ax: plt.Axes, data: List) -> None:
    """Soft eigenvalues of H against eigenvalues of P R P, flow 0, over sqrt(Phi)."""
    worst = 0.0
    for i, sys_d in enumerate(data):
        for tag, _, _, marker in styled(sys_d):
            p = sys_d[tag]["raw"]
            for j, level in enumerate(p["levels"]):
                if level > -3.0:  # first order needs sqrt(Phi) << 1
                    continue
                pr = p["lam_pr"][0, j]
                lam = p["lam"][0, j]
                soft = np.sort(lam[np.argsort(np.abs(lam))[: pr.shape[0]]])
                sc = np.sqrt(p["phi"][0, j])
                ax.plot(pr / sc, soft / sc, ls="", marker=marker, ms=2,
                        color=COLORS[i], alpha=0.3, rasterized=True)  # fmt: skip
                big = np.abs(pr) > 1e-3 * np.abs(pr).max()
                dev = np.abs(soft - pr)[big] / np.abs(pr).max()
                worst = max(worst, float(dev.max()))
    lim = ax.get_xlim()
    ax.plot(lim, lim, color=GUIDE, lw=0.8, zorder=0)
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel(r"$\lambda(PRP)/\sqrt{\Phi}$")
    ax.set_ylabel(r"Soft $\lambda(H)/\sqrt{\Phi}$")
    ax.set_title(r"(c) $H \approx PRP$ on the tangent space", loc="left")
    ax.text(0.04, 0.94, f"flow 0, $\\Phi \\leq 10^{{-3}}$: max deviation\n"
            f"{worst:.0e} of $\\max|\\lambda|$, falling as $\\sqrt{{\\Phi}}$",
            transform=ax.transAxes, va="top", fontsize=11)  # fmt: skip
    print(f"(c) worst |soft H - PRP| / max |PRP| = {worst:.2e}", flush=True)


def panel_collapse(ax: plt.Axes, p: Dict[str, np.ndarray]) -> None:
    """CDF of lambda(P R P)/sqrt(Phi) at every level below 0.1, flow 0."""
    deep = [j for j, v in enumerate(p["levels"]) if v <= -1.0]
    for c, j in enumerate(deep):
        x = 1e3 * np.sort(p["lam_pr"][0, j]) / np.sqrt(p["phi"][0, j])
        col = plt.get_cmap("viridis")(c / max(len(deep) - 1, 1))
        ax.plot(x, np.arange(1, x.shape[0] + 1) / x.shape[0], color=col, lw=1.6,
                label=rf"$10^{{{p['levels'][j]:g}}}$")  # fmt: skip
    ax.axhline(0.5, color=GUIDE, lw=0.8, zorder=0)
    ax.axvline(0.0, color=GUIDE, lw=0.8, zorder=0)
    ax.set_xlabel(r"$10^3\,\lambda(PRP)/\sqrt{\Phi}$")
    ax.set_ylabel("Cumulative Fraction")
    ax.set_title(r"(d) one shape at every $\Phi$ (Poisson, $\chi=0.1$)", loc="left")
    ax.legend(frameon=False, fontsize=9, title=r"$\Phi$", title_fontsize=10,
              loc="upper left")  # fmt: skip


def figure_mechanism(data: List, labels: List[str], out: Path, dpi: int) -> None:
    """Why half the tangent directions curve down, for every start."""
    fig, axes = plt.subplots(2, 3, figsize=(19.0, 11.0), constrained_layout=True)
    panel_sketch(axes[0, 0])
    for i, sys_d in enumerate(data):
        color = COLORS[i % len(COLORS)]
        for tag, _, ls, marker in styled(sys_d):
            h, p = sys_d[tag]["h"], sys_d[tag]["p"]
            band(axes[0, 1], h["phi"], h["neg"] / h["tangent"], color, ls)
            axes[0, 1].plot(p["phi"], np.median(p["neg_pr"], axis=0) / h["tangent"],
                            ls="", marker=marker, ms=5, color=color,
                            mfc="none")  # fmt: skip
            band(axes[1, 1], p["phi"], p["balance"], color, ls)
            band(axes[1, 2], p["phi"], p["frac_r"], color, ls)
    ax = axes[0, 1]
    phi_axis(ax)
    ax.axhline(0.5, color=GUIDE, lw=0.8, zorder=0)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(r"Negative Fraction of the $Nd - 2M$ Soft Directions")
    ax.set_title("(b) the plateau is half, for every start", loc="left")
    panel_check(axes[0, 2], data)
    panel_collapse(axes[1, 0], data[0]["1-10"]["raw"])
    ax = axes[1, 1]
    phi_axis(ax)
    ax.set_yscale("log")
    x = np.geomspace(1e-10, 1e2, 20)
    ax.plot(x, 1.6e-5 * np.sqrt(x / 1e-2), color=GUIDE, lw=1.0, ls="-.")
    ax.set_ylabel(r"$|\sum_j w_{jq}| / \sum_j |w_{jq}|$ (median over $q$)")
    ax.set_title(r"(e) sum rule $\sum_j w_{jq} = |\rho_q|^2 \to 0$", loc="left")
    ax = axes[1, 2]
    phi_axis(ax)
    ax.axhline(0.5, color=GUIDE, lw=0.8, zorder=0)
    ax.set_ylabel(r"Negative Fraction of Particle Curvatures, $\lambda(R_j)$")
    ax.set_title("(f) per-particle curvature signs", loc="left")
    handles = [Line2D([], [], color=COLORS[i], lw=2.0) for i in range(len(data))]
    handles += [Line2D([], [], color="0.3", lw=2.0, ls=ls)
                for _, _, ls, _ in all_styles(data)]  # fmt: skip
    handles += [Line2D([], [], color="0.3", ls="", marker="o", mfc="none"),
                Line2D([], [], color=GUIDE, lw=1.0, ls="-.")]  # fmt: skip
    names = labels + [chi for _, chi, _, _ in all_styles(data)]
    names += [r"(b) $PRP$, no cut", r"(e) $\propto\sqrt{\Phi}$"]
    fig.legend(handles, names, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.0),
               frameon=False)  # fmt: skip
    finish(fig, out / "hessian_plateau_mechanism", dpi)


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("experiments/figures"))
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--extra-chi", nargs="*", default=[],
                        help="extra chi tags of the first root")  # fmt: skip
    args = parser.parse_args()
    assert len(args.roots) == len(args.labels) <= len(COLORS), "one label per root"
    args.out.mkdir(parents=True, exist_ok=True)
    data = load_all(args.roots, args.extra_chi)
    figure_compare(data, args.labels, args.out, args.dpi)
    figure_mechanism(data, args.labels, args.out, args.dpi)
    print("Done.", flush=True)
