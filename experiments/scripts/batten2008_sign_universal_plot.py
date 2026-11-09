#!/usr/bin/env python3

"""
Why the signed Hessian counts change at the same energies for every 2D start.

Reads, per system root (--roots, one --labels entry each; chi tags 1-10 and
2-5): hessian_chi<tag>.npz (all eigenvalues of every flow every quarter
decade of Phi, with the flow time t), plateau_chi<tag>.npz (tr R_j per
particle, flows 0-9), winding_chi<tag>.npz (S(k) along the flows),
landscape_chi<tag>.npz (the coupling spectrum C = D^-1/2 K D^-1/2) and
flows_chi<tag>.npz (the ground states). The count of eigenvalues below -c
(c = 1e-9 lambda_max) is written as n_- / ((N d - 2M)/2) - 1, the negative
excess over the plateau. Writes <out>/hessian_sign_universal.png:

- (a) the excess against Phi: within one chi the starts leave the plateau
  within one decade of Phi, although they start between Phi = 13 and 1000;
- (b) the same against the flow time t / tau_1: the starts reach those
  energies at times up to 100x apart;
- (c) the bias of the soft spectrum, mu = -mean_j tr R_j / std_j tr R_j,
  against the parameter-free prediction f sqrt(2 Phi / N), with the shape
  factor f = sum |q|^2 S / sqrt(sum |q|^4 S sum S) <= 1 of the residual;
- (d) the excess against mu / (1 - c), c = 2M / (N d): the bias spread over
  all N d directions lands in the (1 - c) N d soft ones;
- (e) the largest soft eigenvalue over the smallest stiff one (the 2M-th
  largest): where the two bands separate;
- (f) the final drop against Phi / Phi_x, Phi_x = (c lambda_max / a)^2,
  where a is the median soft |lambda| / sqrt(Phi) (the cut crossing);
- (g) the coupling spectrum (eigenvalues of C) at the ground states;
- (h) the end states' S(k) just beyond the exclusion region.

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
ZERO_REL = 1e-9  # the eigenvalue cut of batten2008_hessian.py
GUIDE = "0.6"
SK_FLOWS = 10  # ground states per start for S(k)
SK_MAX = {"1-10": 3.0, "2-5": 2.0}  # |k|/K range of S(k) per chi (cost)

# ============================================================
# HELPERS
# ============================================================


def phi_axis(ax: plt.Axes) -> None:
    """Log energy axis, high to low (time runs left to right)."""
    ax.set_xscale("log")
    if not ax.xaxis_inverted():
        ax.invert_xaxis()
    ax.set_xlabel(r"Energy, $\Phi$")


def scales(h: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Counts, soft scale a, stiff edge, Phi_x and t/tau_1 of one system."""
    lam, lv = h["lam"], np.asarray(h["samples"])
    keep = lv <= np.log10(h["phi"][:, 0]).min()
    lam, lv = lam[:, keep], lv[keep]
    phi = 10.0**lv
    m2, nd = 2 * h["kmag"].shape[0], lam.shape[-1]
    half = 0.5 * (nd - m2)
    cut = ZERO_REL * lam[..., -1:]
    plate = (lv <= -2.0) & (lv >= -8.0)
    soft = np.abs(lam[:, plate, : nd - m2]) / np.sqrt(phi[plate])[None, :, None]
    a = float(np.median(soft))
    lam_s, lam_max = float(np.median(lam[:, -1, -m2])), float(np.median(lam[:, -1, -1]))
    tau1 = 1.0 / (2.0 * float(h["kmag"][0]) ** 2)
    out = {
        "phi": phi,
        "excess": np.median((lam < -cut).sum(axis=-1), axis=0) / half - 1.0,
        "frac": np.median((lam < -cut).sum(axis=-1), axis=0) / half,
        "t": np.median(h["t"][:, keep], axis=0) / tau1,
        "eta": a * np.sqrt(phi) / lam_s,
        "phi_x": np.array((ZERO_REL * lam_max / a) ** 2),
        "gap": gap_ratio(lam, m2),
        "a": np.array(a),
    }
    assert out["excess"].shape == phi.shape, "one value per level"
    return out


def gap_ratio(lam: np.ndarray, m2: int) -> np.ndarray:
    """Median largest soft |lambda| over the 2M-th largest (NaN while <= 0)."""
    soft = np.abs(lam[..., : lam.shape[-1] - m2]).max(axis=-1)
    stiff = lam[..., -m2]
    out = np.nanmedian(np.where(stiff > 0.0, soft / np.where(stiff > 0, stiff, 1.0),
                                np.nan), axis=0)  # fmt: skip
    assert out.shape == lam.shape[1:2], "one value per level"
    return np.asarray(out)


def bias(p: Dict[str, np.ndarray], w: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Measured trace bias mu and its prediction f sqrt(2 Phi / N) per level."""
    tj, n = p["trace_j"], float(p["n_part"])
    mu = np.median(-tj.mean(axis=-1) / tj.std(axis=-1), axis=0)
    k2 = np.asarray(w["kmag"]) ** 2
    s = np.asarray(w["S"])
    f = (s * k2).sum(-1) / np.sqrt((s * k2**2).sum(-1) * s.sum(-1))
    f_med = np.nanmedian(f, axis=0)
    ok = np.isfinite(f_med)
    f_lv = np.interp(p["levels"], w["levels"][ok][::-1], f_med[ok][::-1])
    phi = np.median(p["phi"], axis=0)
    out = {"mu": mu, "pred": f_lv * np.sqrt(2.0 * phi / n), "phi": phi,
           "f_levels": w["levels"][ok][::-1], "f": f_med[ok][::-1],
           "n": np.array(n)}  # fmt: skip
    assert out["mu"].shape == out["pred"].shape, "one value per level"
    return out


def shell_sk(flows: Path, k_outer: float, kmax_frac: float) -> Dict[str, np.ndarray]:
    """S(k) of SK_FLOWS ground states, averaged in |k|/K shells up to kmax_frac."""
    with np.load(flows) as f:
        s, basis = np.asarray(f["s_final"][:SK_FLOWS]), np.asarray(f["basis"])
    side = np.diag(basis)
    lim = np.ceil(kmax_frac * k_outer * side / (2.0 * np.pi)).astype(int)
    g = np.stack(np.meshgrid(*[np.arange(-v, v + 1) for v in lim], indexing="ij"), -1)
    n = g.reshape(-1, 2)
    n = n[(n[:, 0] > 0) | ((n[:, 0] == 0) & (n[:, 1] > 0))]
    kvec = n * (2.0 * np.pi / side)
    km = np.linalg.norm(kvec, axis=1) / k_outer
    sel = km < kmax_frac
    kvec, km = kvec[sel], km[sel]
    sk = np.zeros((s.shape[0], km.shape[0]))
    for b in range(s.shape[0]):
        x = s[b] * side
        for a0 in range(0, km.shape[0], 512):
            ph = x @ kvec[a0 : a0 + 512].T
            sk[b, a0 : a0 + 512] = np.cos(ph).sum(0) ** 2 + np.sin(ph).sum(0) ** 2
        sk[b] /= x.shape[0]
    edges = np.linspace(0.0, kmax_frac, int(round(kmax_frac / 0.05)) + 1)
    idx = np.digitize(km, edges) - 1
    mean = np.array([sk[:, idx == i].mean() if np.any(idx == i) else np.nan
                     for i in range(edges.shape[0] - 1)])  # fmt: skip
    assert mean.shape == (edges.shape[0] - 1,), "one value per shell"
    return {"k": 0.5 * (edges[1:] + edges[:-1]), "sk": mean}


def load_all(roots: List[Path]) -> List[Dict]:
    """Per root, per chi tag: Hessian scales, coupling spectra and S(k)."""
    out = []
    for root in roots:
        sys_d = {}
        for tag, *_ in TAGS:
            h = load(root / f"hessian_chi{tag}.npz")
            lz = load(root / f"landscape_chi{tag}.npz")
            k_outer = float(h["kmag"][-1])
            b = bias(load(root / f"plateau_chi{tag}.npz"),
                     load(root / f"winding_chi{tag}.npz"))  # fmt: skip
            s = scales(h)
            f_h = np.interp(np.log10(s["phi"]), b["f_levels"], b["f"])
            soft = 1.0 - 2.0 * h["kmag"].shape[0] / h["lam"].shape[-1]
            s["pred"] = f_h * np.sqrt(2.0 * s["phi"] / float(b["n"])) / soft
            sys_d[tag] = {"s": s, "b": b, "c_eig": lz["c_eig"][:, -1].ravel(),
                          "sk": shell_sk(root / f"flows_chi{tag}.npz", k_outer,
                                         SK_MAX[tag])}  # fmt: skip
        out.append(sys_d)
    assert len(out) == len(roots), "one entry per root"
    return out


# ============================================================
# FIGURE
# ============================================================


def figure(data: List, labels: List[str], out: Path, dpi: int) -> None:
    """Shared energies, different times, the bias law, and different end states."""
    fig, axes = plt.subplots(2, 4, figsize=(25.0, 11.5), constrained_layout=True)
    for i, sys_d in enumerate(data):
        color = COLORS[i % len(COLORS)]
        for tag, _, ls, mk in TAGS:
            s, b = sys_d[tag]["s"], sys_d[tag]["b"]
            pos = s["excess"] > 0.0
            for ax, x in ((axes[0, 0], s["phi"]), (axes[0, 1], s["t"]),
                          (axes[0, 3], s["pred"])):  # fmt: skip
                ax.plot(x[pos], s["excess"][pos], color=color, ls=ls, lw=2.0)
            good = (b["mu"] > 0.0) & (b["pred"] > 0.0)
            axes[0, 2].plot(b["pred"][good], b["mu"][good], ls="", marker=mk, ms=5,
                            color=color, mfc="none")  # fmt: skip
            axes[1, 0].plot(s["phi"], s["gap"], color=color, ls=ls, lw=2.0)
            axes[1, 1].plot(s["phi"] / float(s["phi_x"]), s["frac"], color=color,
                            ls=ls, lw=2.0)  # fmt: skip
            ev = np.sort(sys_d[tag]["c_eig"])
            axes[1, 2].plot(ev, np.linspace(0.0, 1.0, ev.shape[0]), color=color,
                            ls=ls, lw=1.6)  # fmt: skip
            sk = sys_d[tag]["sk"]
            beyond = sk["k"] > 1.0
            axes[1, 3].plot(sk["k"][beyond], sk["sk"][beyond], color=color, ls=ls,
                            lw=1.8)  # fmt: skip
    x = np.geomspace(1e-7, 1.0, 10)
    axes[0, 2].plot(x, x, color=GUIDE, lw=1.0)
    style(axes)
    handles = [Line2D([], [], color=COLORS[i], lw=2.0) for i in range(len(labels))]
    handles += [Line2D([], [], color="0.3", lw=2.0, ls=ls, marker=mk, mfc="none")
                for _, _, ls, mk in TAGS]  # fmt: skip
    fig.legend(handles, labels + [c for _, c, _, _ in TAGS], loc="upper center",
               ncol=7, bbox_to_anchor=(0.5, 0.0), frameon=False)  # fmt: skip
    finish(fig, out / "hessian_sign_universal", dpi)


def style(axes: np.ndarray) -> None:
    """Labels and titles of the eight panels."""
    ylab = r"Negative Excess, $n_-/[(Nd-2M)/2] - 1$"
    loglog = (axes[0, 0], axes[0, 1], axes[0, 2], axes[0, 3], axes[1, 0])
    for ax in loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")
    for ax in (axes[0, 0], axes[0, 1], axes[0, 3]):
        ax.set_ylabel(ylab)
    for ax in (axes[0, 0], axes[0, 2], axes[0, 3], axes[1, 0], axes[1, 1]):
        ax.invert_xaxis()
    axes[0, 0].set_xlabel(r"Energy, $\Phi$")
    axes[0, 1].set_xlabel(r"Time, $t/\tau_1$")
    axes[0, 2].set_xlabel(r"Prediction, $f\sqrt{2\Phi/N}$")
    axes[0, 2].set_ylabel(r"Soft-Spectrum Bias, $\mu = -\overline{\mathrm{tr}R_j}"
                          r"/\sigma(\mathrm{tr}R_j)$")  # fmt: skip
    axes[0, 3].set_xlabel(r"Soft-Subspace Bias, $f\sqrt{2\Phi/N}/(1-c)$")
    axes[1, 0].set_xlabel(r"Energy, $\Phi$")
    axes[1, 0].set_ylabel(r"Soft Top / Stiff Bottom, $\max|\lambda_{\rm soft}|"
                          r"/\lambda_{2M}$")  # fmt: skip
    axes[1, 0].axhline(1.0, color=GUIDE, lw=0.8, zorder=0)
    axes[1, 0].set_ylim(1e-11, 1e3)
    ax = axes[1, 1]
    ax.set_xscale("log")
    ax.set_xlim(1e4, 1e-4)
    ax.set_ylim(0.0, 1.2)
    ax.axvline(1.0, color=GUIDE, lw=0.8, zorder=0)
    ax.set_xlabel(r"Scaled Energy, $\Phi/\Phi_\times$, $\Phi_\times = "
                  r"(c\,\lambda_{\max}/a)^2$")  # fmt: skip
    ax.set_ylabel(r"$n_-/[(Nd-2M)/2]$")
    axes[1, 2].set_xscale("log")
    axes[1, 2].set_xlabel(r"Coupling Spectrum, $\lambda(C)$")
    axes[1, 2].set_ylabel("Cumulative Fraction")
    axes[1, 3].set_xlabel(r"Wavenumber, $|\mathbf{k}|/K$")
    axes[1, 3].set_ylabel(r"End-State $S(\mathbf{k})$ (shell mean)")
    axes[1, 3].axhline(1.0, color=GUIDE, lw=0.8, zorder=0)
    titles = ("(a) each $\\chi$ leaves the plateau at one energy",
              "(b) ... reached at times 100$\\times$ apart",
              "(c) the bias law, no free parameter",
              "(d) one curve: the excess follows the bias",
              "(e) the soft band leaves the stiff band",
              "(f) the final drop is the cut crossing",
              "(g) ground states: local landscape",
              "(h) ground states: not one place")  # fmt: skip
    for ax, title in zip(axes.flat, titles, strict=True):
        ax.set_title(title, loc="left", fontsize=13)


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
            s, sk = sys_d[tag]["s"], sys_d[tag]["sk"]
            above = (sk["k"] > 1.05) & (sk["k"] < 1.5)
            s_above = np.nanmean(sk["sk"][above])
            f_one = np.interp(0.0, sys_d[tag]["b"]["f_levels"], sys_d[tag]["b"]["f"])
            print(f"{lab} chi{tag}: a {float(s['a']):.2e}, Phi_x "
                  f"{float(s['phi_x']):.1e}; S(k) at 1.05-1.5 K {s_above:.3f}; "
                  f"median lambda(C) {np.median(sys_d[tag]['c_eig']):.3f}; "
                  f"10-90% lambda(C) {np.percentile(sys_d[tag]['c_eig'], 10):.2f}-"
                  f"{np.percentile(sys_d[tag]['c_eig'], 90):.2f}; f at Phi=1 "
                  f"{f_one:.2f}",
                  flush=True)  # fmt: skip
    figure(data, args.labels, args.out, args.dpi)
    print("Done.", flush=True)
