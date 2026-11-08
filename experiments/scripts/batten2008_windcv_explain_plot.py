#!/usr/bin/env python3

"""
Why winding_cv's winding rate, spiral number and force alignment behave as they do.

With u = (Re rho_q, Im rho_q) over the constrained waves, du/dt = -K u
exactly (K = (2/V) J J^T; batten2008_landscape.py). Per wave,
|d rho_k/dt|^2 = |rho_k|^2 (l_k'^2 + theta_k'^2) with l_k = ln|rho_k|, so
the S-weighted mean square speed of the residual splits exactly into

  <l'^2 + theta'^2>_S = <l'>_S^2 + Var_S(l') + Omega^2
                         coherent   differential  rotation

(<.>_S: weights S_k over the waves considered). With all constrained
waves, <l'^2 + theta'^2>_S = <K^2>_u and -<l'>_S = <K>_u = -1/2 d ln Phi/dt,
so Omega^2 <= <K^2>_u - <K>_u^2 and the spiral number obeys
sigma = Omega / <K>_u <= CV_u(K), the coefficient of variation of the
collective stiffness over the residual. The force alignment of wave k
factorizes exactly as |cos alpha_k| = (|theta_k'|/|k|) sqrt(S_k/Phi)
/ sqrt(<K>_u) (|grad Phi|^2 = 2 <K>_u Phi, |grad theta_k| = |k|/sqrt(2 S_k)).

Reads winding_chi<tag>.npz (rates at every quarter decade, every flow) and
landscape_chi<tag>.npz (K spectra and residual weights, flows 0-9, every
half decade) per root. Panels (a-c, e, f) use the waves with S > 1e-8, as
winding_cv; panel (d) uses every constrained wave on both sides. Writes
<out>/winding_cv_explained.png:

- (a, b) 2D Poisson, chi = 0.1 and 0.4: the three parts as fractions of the
  mean square speed (median over flows);
- (c) rotation share Omega^2 / <l'^2 + theta'^2>_S, every start;
- (d) per flow and level: sigma (all waves) against CV_u(K), with sigma =
  CV_u(K) (the bound);
- (e) 2D Poisson: the three factors of |cos alpha| (median over waves) and
  their product against Phi, with the measured median |cos alpha|;
- (f) every start: fraction of constrained waves with S > 1e-8 and the
  median energy share S_k / Phi among them.

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
FLOOR = 1e-8  # as winding_cv: instantaneous rates only where S exceeds this
PART_COLORS = ("#1f4e9c", "#2e7d32", "#c0392b")  # coherent, differential, rotation
GUIDE = "0.6"
PHI_MIN = 1e-9  # below this no wave has S > FLOOR (S_k <= Phi)

# ============================================================
# HELPERS
# ============================================================


def phi_axis(ax: plt.Axes) -> None:
    """Log energy axis, high to low (time runs left to right)."""
    ax.set_xscale("log")
    if not ax.xaxis_inverted():
        ax.invert_xaxis()
    ax.set_xlabel(r"Energy, $\Phi$")


def partition(w: Dict[str, np.ndarray], floor: float) -> Dict[str, np.ndarray]:
    """Coherent, differential and rotation parts (flows, levels) over S > floor."""
    ok = w["S"] > floor
    s = np.where(ok, w["S"], 0.0)
    th = np.where(ok, w["theta_rate"], 0.0)
    lr = np.where(ok, w["lnrho_rate"], 0.0)
    tot_s = s.sum(axis=2)
    tot_s = np.where(tot_s > 0.0, tot_s, np.nan)
    mean_l = (s * lr).sum(axis=2) / tot_s
    out = {
        "coherent": mean_l**2,
        "differential": (s * lr**2).sum(axis=2) / tot_s - mean_l**2,
        "rotation": (s * th**2).sum(axis=2) / tot_s,
    }
    out["total"] = out["coherent"] + out["differential"] + out["rotation"]
    assert out["total"].shape == w["phi"].shape, "one value per flow and level"
    return out


def cos_factors(w: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """|cos alpha|, its three factors (medians over waves) and the survivors."""
    ok = w["S"] > FLOOR
    phi = w["phi"][:, :, None]
    k_mean = w["grad_phi"] ** 2 / (2.0 * w["phi"])  # <K>_u exactly
    rate = np.abs(w["theta_rate"]) / w["kmag"][None, None, :]
    share = w["S"] / phi
    cos = np.abs(w["theta_rate"]) / (w["grad_theta"] * w["grad_phi"][:, :, None])
    med = lambda x: np.nanmedian(np.where(ok, x, np.nan), axis=2)  # noqa: E731
    return {
        "rate": med(rate),
        "share": med(np.sqrt(share)),
        "kfac": 1.0 / np.sqrt(k_mean),
        "product": med(rate * np.sqrt(share) / np.sqrt(k_mean)[:, :, None]),
        "cos": med(cos),
        "alive": ok.mean(axis=2),
        "share_alive": med(share),
    }


def bound_points(w: Dict[str, np.ndarray], lz: Dict[str, np.ndarray]) -> np.ndarray:
    """(sigma over all waves, CV_u(K)) at the landscape levels, flows 0-9."""
    part = partition(w, -1.0)  # every constrained wave
    wl = list(np.round(w["levels"], 2))
    rows = []
    for j, lev in enumerate(np.round(lz["levels"], 2)):
        if lev not in wl:
            continue
        i = wl.index(lev)
        for b in range(lz["ray_k"].shape[0]):
            if lz["levels"][j] > np.log10(lz["phi"][b, 0]):
                continue
            k1, k2 = (
                lz["ray_k"][b, j],
                (lz["weight"][b, j] * lz["k_eig"][b, j] ** 2).sum(),
            )
            sigma = np.sqrt(part["rotation"][b, i]) / np.sqrt(part["coherent"][b, i])
            rows.append((sigma, np.sqrt(max(k2 - k1**2, 0.0)) / k1, lev))
    out = np.array(rows)
    assert out.ndim == 2 and out.shape[1] == 3, "(sigma, CV, level) rows"
    return out


def band(ax: plt.Axes, x: np.ndarray, rows: np.ndarray, color: str, ls: str) -> None:
    """Median and 10-90 % band of rows (flows, x)."""
    q10, q50, q90 = np.nanpercentile(rows, [10.0, 50.0, 90.0], axis=0)
    ax.fill_between(x, q10, q90, color=color, alpha=0.12, lw=0)
    ax.plot(x, q50, color=color, ls=ls, lw=2.0)


# ============================================================
# FIGURE
# ============================================================


def panel_parts(ax: plt.Axes, w: Dict[str, np.ndarray], title: str) -> None:
    """Stacked median fractions of the mean square speed (S > FLOOR waves)."""
    part = partition(w, FLOOR)
    phi = 10.0 ** w["levels"]
    keep = w["levels"] <= np.log10(np.min(w["phi"][:, 0]))
    fr = [np.nanmedian(part[k] / part["total"], axis=0)[keep]
          for k in ("coherent", "differential", "rotation")]  # fmt: skip
    fr = np.array(fr) / np.maximum(np.nansum(fr, axis=0), 1e-300)
    ax.stackplot(phi[keep], np.nan_to_num(fr), colors=PART_COLORS, alpha=0.75,
                 labels=[r"coherent decay $\langle \dot l\rangle_S^2$",
                         r"differential decay $\mathrm{Var}_S(\dot l)$",
                         r"rotation $\Omega^2$"])  # fmt: skip
    phi_axis(ax)
    ax.set_xlim(phi[keep].max(), PHI_MIN)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(r"Share of $\langle \dot l^2 + \dot\theta^2\rangle_S$")
    ax.set_title(title, loc="left", fontsize=13)


def figure(data: List, labels: List[str], out: Path, dpi: int) -> None:
    """Exact decomposition of the winding-rate diagnostics of winding_cv."""
    fig, axes = plt.subplots(2, 3, figsize=(20.0, 11.5), constrained_layout=True)
    panel_parts(axes[0, 0], data[0]["1-10"]["w"],
                f"(a) {labels[0]}, {TAGS[0][1]}: where the speed goes")  # fmt: skip
    panel_parts(axes[0, 1], data[0]["2-5"]["w"],
                f"(b) {labels[0]}, {TAGS[1][1]}: where the speed goes")  # fmt: skip
    axes[0, 0].legend(frameon=False, fontsize=10, loc="center right")
    for i, sys_d in enumerate(data):
        color = COLORS[i % len(COLORS)]
        for tag, _, ls, mk in TAGS:
            w = sys_d[tag]["w"]
            part = partition(w, FLOOR)
            phi = 10.0 ** w["levels"]
            ok = w["levels"][None, :] <= np.log10(w["phi"][:, :1])
            band(axes[0, 2], phi, np.where(ok, part["rotation"] / part["total"],
                                           np.nan), color, ls)  # fmt: skip
            pts = bound_points(w, sys_d[tag]["l"])
            axes[1, 0].plot(pts[:, 1], pts[:, 0], ls="", marker=mk, ms=4,
                            color=color, mfc="none", alpha=0.6)  # fmt: skip
            cf = cos_factors(w)
            band(axes[1, 2], phi, np.where(ok, cf["alive"], np.nan), color, ls)
            if i == 0:
                for key, c in (("rate", "#1f4e9c"), ("share", "#2e7d32"),
                               ("kfac", "#b9770e"), ("product", "black")):  # fmt: skip
                    axes[1, 1].plot(phi, np.nanmedian(np.where(ok, cf[key], np.nan),
                                                      axis=0), color=c, ls=ls,
                                    lw=2.0)  # fmt: skip
                axes[1, 1].plot(phi, np.nanmedian(np.where(ok, cf["cos"], np.nan),
                                                  axis=0), ls="", marker=mk, ms=4,
                                color="#c0392b", mfc="none")  # fmt: skip
    ax = axes[0, 2]
    phi_axis(ax)
    ax.set_yscale("log")
    ax.set_ylabel(r"Rotation Share, $\Omega^2/\langle \dot l^2+\dot\theta^2\rangle_S$")
    ax.set_title("(c) rotation share of the residual speed", loc="left", fontsize=13)
    ax = axes[1, 0]
    ax.set_xscale("log")
    ax.set_yscale("log")
    lim = np.array([1e-4, 3.0])
    ax.plot(lim, lim, color=GUIDE, lw=1.0)
    ax.fill_between(lim, lim, 10.0, color="0.92", zorder=0, lw=0)
    ax.set_xlim(*lim)
    ax.set_ylim(1e-4, 3.0)
    ax.set_xlabel(r"Stiffness Spread, $\mathrm{CV}_u(K)$")
    ax.set_ylabel(r"Spiral Number, $\sigma$ (all waves)")
    ax.set_title(r"(d) $\sigma \leq \mathrm{CV}_u(K)$ (grey: forbidden)", loc="left",
                 fontsize=13)  # fmt: skip
    ax = axes[1, 1]
    phi_axis(ax)
    ax.set_yscale("log")
    ax.set_xlim(right=PHI_MIN)
    ax.set_ylabel("Median over Waves")
    ax.set_title(f"(e) {labels[0]}: the three factors of $|\\cos\\alpha_k|$",
                 loc="left", fontsize=13)  # fmt: skip
    ax.legend([Line2D([], [], color=c, lw=2.0) for c in
               ("#1f4e9c", "#2e7d32", "#b9770e", "black")] +
              [Line2D([], [], color="#c0392b", ls="", marker="o", mfc="none")],
              [r"$|\dot\theta_k|/|\mathbf{k}|$", r"$\sqrt{S_k/\Phi}$",
               r"$1/\sqrt{\langle K\rangle_u}$", "product",
               r"measured $|\cos\alpha_k|$"], frameon=False, fontsize=10,
              loc="lower left", ncol=2)  # fmt: skip
    ax = axes[1, 2]
    phi_axis(ax)
    ax.set_xlim(right=PHI_MIN)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(r"Fraction of Waves with $S_k > 10^{-8}$")
    ax.set_title("(f) which waves panels (a, b, f) of winding_cv see", loc="left",
                 fontsize=13)  # fmt: skip
    handles = [Line2D([], [], color=COLORS[i], lw=2.0) for i in range(len(labels))]
    handles += [Line2D([], [], color="0.3", lw=2.0, ls=ls, marker=mk, mfc="none")
                for _, _, ls, mk in TAGS]  # fmt: skip
    fig.legend(handles, labels + [chi for _, chi, _, _ in TAGS], loc="upper center",
               ncol=7, bbox_to_anchor=(0.5, 0.0), frameon=False)  # fmt: skip
    finish(fig, out / "winding_cv_explained", dpi)


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
    data = [{tag: {"w": load(r / f"winding_chi{tag}.npz"),
                   "l": load(r / f"landscape_chi{tag}.npz")} for tag, *_ in TAGS}
            for r in args.roots]  # fmt: skip
    for sys_d, lab in zip(data, args.labels, strict=True):
        for tag, *_ in TAGS:
            pts = bound_points(sys_d[tag]["w"], sys_d[tag]["l"])
            eff = pts[:, 0] / pts[:, 1]
            print(f"{lab} chi{tag}: sigma/CV max {np.nanmax(eff):.3f}, median "
                  f"{np.nanmedian(eff):.3f}; late (<=1e-4) median "
                  f"{np.nanmedian(eff[pts[:, 2] <= -4]):.3f}", flush=True)  # fmt: skip
    figure(data, args.labels, args.out, args.dpi)
    print("Done.", flush=True)
