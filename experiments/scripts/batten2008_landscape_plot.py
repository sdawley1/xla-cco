#!/usr/bin/env python3

"""
The 2D stealthy landscape: independent wells, a moving floor, a slow curved valley.

Along the flow the constrained waves u = (Re rho_q, Im rho_q) obey exactly
du/dt = -K u, K = (2/V) J J^T (batten2008_landscape.py). K diagonal
(K_qq ~ |q|^2) means independent quadratic wells. Reads, per system root
(--roots, one --labels entry each; chi tags 1-10 and 2-5):
landscape_chi<tag>.npz, modes_chi<tag>.npz, flows_chi<tag>.npz,
winding_chi<tag>.npz, hessian_chi<tag>.npz, offplane_chi<tag>.npz. Writes:

- landscape_wells: (a, b) sketches of independent wells and a coupled
  curved valley; (c, d) S(k, t)/S(k, 0) against 2|k|^2 t by |k| band
  (2D Poisson; e^-x for an independent well); (e) |k|^2 t at which S(k)
  has fallen e^2-fold (1 for an independent well), every start; (f)
  cumulative distribution of the eigenvalues of C = D^-1/2 K D^-1/2 at
  Phi = 1e-4 with the Marchenko-Pastur law for ratio 2M/(N d); (g) the
  smallest eigenvalue of C against Phi with the Marchenko-Pastur soft edge
  (1 - sqrt(2M/(N d)))^2; (h) overlap of the soft subspace of K with the
  longest waves' own coordinates (1 for independent wells).
- landscape_floor: (a, b) the own-well ratio Gamma_k/|k|^2 (Gamma_k =
  -d ln|rho_k|/dt; 1 in an independent well, << 1 when the wave sits on a
  floor set by the other waves) over Phi and |k| (2D Poisson), with the
  contour where S(k) = 1e-2; (c) S(k)/S(k, 0) when each wave first drops
  below half its own rate; (d) soft-subspace overlap between successive
  half decades (1: the floor is still); (e) overlap with the final soft
  subspace against the path fraction s; (f) share
  of each wave's winding accrued while slaved.
- landscape_valley: (a) u^T K u / u^T D u (1 for independent wells); (b)
  the energy decay rate -1/2 d ln Phi/dt = u^T K u/|u|^2 over k_1^2 (the
  slowest single well); (c) cumulative share of |u|^2 over the eigenvalues
  of K (/k_1^2), 2D Gaussian; (d) floor curvature |lambda_min|/sqrt(Phi);
  (e) path length and clock lag against the linear model, per system;
  (f) the clock lag against the valley's softness 1 - u^T K u/u^T D u.

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
from batten2008_kl_plot import flow_metrics  # noqa: E402
from batten2008_plot import finish, load, plt  # noqa: E402  (house rcParams)
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

COLORS = ("black", "#1f4e9c", "#c0392b", "#2e7d32", "#b9770e", "#7d3c98")
MARKERS = "osD^vP"
TAGS = (("1-10", r"$\chi = 0.1$", "-", "o"), ("2-5", r"$\chi = 0.4$", "--", "s"))
GUIDE = "0.6"
BANDS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01))  # |k|/K
FLOOR_S = 1e-8  # rates shown only above this S (integrator error below)
SLAVED = 0.5  # Gamma_k/|k|^2 below this: the wave sits on the moving floor
REF_LEVEL = -4.0  # log10 Phi of the spectrum panels

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
    """Median and 10-90 % band of rows (flows, x)."""
    q10, q50, q90 = np.nanpercentile(rows, [10.0, 50.0, 90.0], axis=0)
    ax.fill_between(x, q10, q90, color=color, alpha=0.12, lw=0)
    ax.plot(x, q50, color=color, ls=ls, lw=2.0)


def reached(lz: Dict[str, np.ndarray]) -> np.ndarray:
    """(flows, levels) mask: the level lies below the flow's start energy."""
    start = np.log10(lz["phi"][:, 0])[:, None]
    out: np.ndarray = lz["levels"][None, :] <= start
    return out


def mp_cdf(c: float, x: np.ndarray) -> np.ndarray:
    """Marchenko-Pastur cumulative distribution (ratio c < 1, unit variance)."""
    lo, hi = (1.0 - np.sqrt(c)) ** 2, (1.0 + np.sqrt(c)) ** 2
    g = np.linspace(lo, hi, 4001)
    dens = np.sqrt(np.maximum((hi - g) * (g - lo), 0.0)) / (2.0 * np.pi * c * g)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(g))])
    out: np.ndarray = np.interp(x, g, cum / cum[-1], left=0.0, right=1.0)
    return out


def ratio_c(lz: Dict[str, np.ndarray]) -> float:
    """2M / (N d): constrained coordinates over degrees of freedom."""
    m, d = lz["nvec"].shape
    return 2.0 * m / (float(lz["n_part"]) * d)


def two_efold(mz: Dict[str, np.ndarray], fz: Dict[str, np.ndarray]) -> np.ndarray:
    """|k|^2 t at which S(k) first falls below S(k, 0) e^-2, (flows, M)."""
    sk, km = mz["sk_grid"], mz["kmag"]
    t = mz["grid"][None, :] * fz["t_final"][: sk.shape[0], None]
    below = sk <= sk[:, :1, :] * np.exp(-2.0)
    first = np.where(below.any(axis=1), np.argmax(below, axis=1), sk.shape[1] - 1)
    te = np.take_along_axis(t, first, axis=1)
    out: np.ndarray = te * km[None, :] ** 2
    return out


def load_all(roots: List[Path]) -> List[Dict]:
    """Per root, per chi tag: every input file."""
    names = ("landscape", "modes", "flows", "winding", "hessian", "offplane")
    out = []
    for root in roots:
        sys_d = {}
        for tag, *_ in TAGS:
            z = {n: load(root / f"{n}_chi{tag}.npz") for n in names if n != "flows"}
            with np.load(root / f"flows_chi{tag}.npz") as f:
                z["flows"] = {"t_final": np.asarray(f["t_final"])}
            z["metrics"] = flow_metrics(z["offplane"], z["winding"])
            sys_d[tag] = z
        out.append(sys_d)
    return out


# ============================================================
# FIGURE 1: INDEPENDENT WELLS OR COUPLED WAVES
# ============================================================


def sketch(ax: plt.Axes, coupled: bool) -> None:
    """Contours of a 2-wave landscape and gradient-flow lines from three starts."""
    g = np.linspace(-2.2, 2.2, 301)
    u1, u2 = np.meshgrid(g, g)

    def phi(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if coupled:  # a soft, curved valley b = 0.45 a^2 - 0.6
            return 6.0 * (b - 0.45 * a**2 + 0.6) ** 2 + 0.15 * a**2
        return 0.6 * a**2 + 3.0 * b**2

    ax.contourf(u1, u2, phi(u1, u2), levels=np.geomspace(0.02, 40, 14),
                cmap="Blues_r", extend="both")  # fmt: skip
    for x0 in ((-2.0, 1.9), (1.4, 2.0), (2.0, -1.6)):
        p = np.array(x0, dtype=float)
        path = [p.copy()]
        for _ in range(20000):
            h = 1e-5
            gr = np.array(
                [
                    (phi(p[0] + h, p[1]) - phi(p[0] - h, p[1])) / (2 * h),
                    (phi(p[0], p[1] + h) - phi(p[0], p[1] - h)) / (2 * h),
                ]
            )
            p = p - 2e-3 * gr
            path.append(p.copy())
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], color="#c0392b", lw=1.8)
        ax.plot(*x0, "o", color="#c0392b", ms=5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_xlabel(r"long wave, $\rho_{\mathbf{k}_1}$")
    ax.set_ylabel(r"short wave, $\rho_{\mathbf{k}}$")
    title = ("(b) coupled: a soft, curved valley" if coupled
             else "(a) independent quadratic wells")  # fmt: skip
    ax.set_title(title, loc="left", fontsize=13)


def panel_collapse(ax: plt.Axes, z: Dict, letter: str, chi: str) -> None:
    """S(k, t)/S(k, 0) against 2|k|^2 t, median per |k| band, 2D Poisson."""
    mz = z["modes"]
    sk, km = mz["sk_grid"], mz["kmag"]
    t = mz["grid"][None, :] * z["flows"]["t_final"][: sk.shape[0], None]
    xk = km / km[-1]
    grid = np.geomspace(1e-2, 30.0, 120)
    for c, (lo, hi) in enumerate(BANDS):
        cols = np.flatnonzero((xk > lo) & (xk <= hi))
        rows = []
        for b in range(sk.shape[0]):
            for q in cols[:: max(1, cols.shape[0] // 40)]:
                x = 2.0 * km[q] ** 2 * t[b]
                y = sk[b, :, q] / max(sk[b, 0, q], 1e-300)
                rows.append(np.interp(grid, x, y, left=np.nan, right=np.nan))
        col = plt.get_cmap("viridis")(c / (len(BANDS) - 1))
        ax.plot(grid, np.nanmedian(rows, axis=0), color=col, lw=2.0,
                label=rf"$|\mathbf{{k}}|/K \in ({lo:.1f}, {min(hi, 1.0):.1f}]$",
                )  # fmt: skip
    ax.plot(grid, np.exp(-grid), color="black", lw=1.2, ls="--", label=r"$e^{-x}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 3.0)
    ax.set_xlabel(r"Scaled Time, $x = 2|\mathbf{k}|^2 t$")
    ax.set_ylabel(r"$S(\mathbf{k}, t)/S(\mathbf{k}, 0)$ (median)")
    ax.set_title(f"({letter}) 2D Poisson, {chi}: each wave's decay", loc="left",
                 fontsize=13)  # fmt: skip


def figure_wells(data: List, labels: List[str], out: Path, dpi: int) -> None:
    """Independent wells at chi = 0.1, coupled waves at chi = 0.4."""
    fig, axes = plt.subplots(2, 4, figsize=(24.0, 11.0), constrained_layout=True)
    sketch(axes[0, 0], False)
    sketch(axes[0, 1], True)
    panel_collapse(axes[0, 2], data[0]["1-10"], "c", TAGS[0][1])
    panel_collapse(axes[0, 3], data[0]["2-5"], "d", TAGS[1][1])
    axes[0, 3].legend(frameon=False, fontsize=10, loc="lower left")
    for i, sys_d in enumerate(data):
        for tag, _, ls, mk in TAGS:
            z, lz = sys_d[tag], sys_d[tag]["landscape"]
            x = two_efold(z["modes"], z["flows"])
            xk = z["modes"]["kmag"] / z["modes"]["kmag"][-1]
            mids = [np.median(x[:, (xk > a) & (xk <= b)]) for a, b in BANDS]
            axes[1, 0].plot([0.1, 0.3, 0.5, 0.7, 0.9], mids, color=COLORS[i], ls=ls,
                            lw=2.0, marker=mk, ms=6)  # fmt: skip
            j = list(lz["levels"]).index(REF_LEVEL)
            ev = np.sort(lz["c_eig"][:, j].ravel())
            axes[1, 1].plot(ev, np.linspace(0, 1, ev.shape[0]), color=COLORS[i],
                            ls=ls, lw=1.6)  # fmt: skip
            ok = reached(lz)
            phi = 10.0 ** lz["levels"]
            band(axes[1, 2], phi, np.where(ok, lz["c_eig"][:, :, 0], np.nan),
                 COLORS[i], ls)  # fmt: skip
            band(axes[1, 3], phi, np.where(ok, lz["o_wells"], np.nan), COLORS[i], ls)
    c_vals = {tag: ratio_c(data[0][tag]["landscape"]) for tag, *_ in TAGS}
    ax = axes[1, 0]
    ax.axhline(1.0, color=GUIDE, lw=0.8, zorder=0)
    ax.set_xlabel(r"Wavenumber, $|\mathbf{k}|/K$ (band centre)")
    ax.set_ylabel(r"$|\mathbf{k}|^2 t$ at $S(\mathbf{k}) = S(\mathbf{k}, 0)\,e^{-2}$")
    ax.set_title("(e) first two e-folds: own well = 1", loc="left", fontsize=13)
    ax = axes[1, 1]
    x = np.geomspace(1e-3, 5.0, 400)
    for tag, _, ls, _ in TAGS:
        ax.plot(x, mp_cdf(c_vals[tag], x), color=GUIDE, lw=3.0, ls=ls, alpha=0.7)
    ax.axvline(1.0, color=GUIDE, lw=0.8, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel(r"Eigenvalue of $C = D^{-1/2} K D^{-1/2}$")
    ax.set_ylabel("Cumulative Fraction")
    ax.set_title(r"(f) coupled-wave stiffness at $\Phi = 10^{-4}$", loc="left",
                 fontsize=13)  # fmt: skip
    ax = axes[1, 2]
    phi_axis(ax)
    ax.set_yscale("log")
    for tag, _, ls, _ in TAGS:
        ax.axhline((1.0 - np.sqrt(c_vals[tag])) ** 2, color=GUIDE, lw=1.2, ls=ls)
    ax.set_ylabel(r"Softest Coupled Mode, $\min \lambda(C)$")
    ax.set_title(r"(g) soft edge vs $(1-\sqrt{2M/Nd})^2$", loc="left", fontsize=13)
    ax = axes[1, 3]
    phi_axis(ax)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Soft Modes on the Longest Waves' Own Axes")
    ax.set_title("(h) soft modes = single waves?", loc="left", fontsize=13)
    legend(fig, labels, [r"Marchenko–Pastur, $c = 2M/(Nd)$"], GUIDE)
    finish(fig, out / "landscape_wells", dpi)


def legend(fig: plt.Figure, labels: List[str], extra: List[str], color: str) -> None:
    """Starts, the chi styles and extra grey entries, in one row beneath."""
    handles = [Line2D([], [], color=COLORS[i], lw=2.0, marker=MARKERS[i])
               for i in range(len(labels))]  # fmt: skip
    handles += [Line2D([], [], color="0.3", lw=2.0, ls=ls, marker=mk, mfc="none")
                for _, _, ls, mk in TAGS]  # fmt: skip
    handles += [Line2D([], [], color=color, lw=3.0, alpha=0.7) for _ in extra]
    names = labels + [chi for _, chi, _, _ in TAGS] + extra
    fig.legend(handles, names, loc="upper center", ncol=min(len(names), 8),
               bbox_to_anchor=(0.5, 0.0), frameon=False)  # fmt: skip


# ============================================================
# FIGURE 2: THE FLOOR MOVES WITH THE LONG WAVES
# ============================================================


def own_well_ratio(wz: Dict[str, np.ndarray]) -> np.ndarray:
    """Gamma_k/|k|^2 (flows, levels, M), NaN where S < FLOOR_S."""
    r = -wz["lnrho_rate"] / (wz["kmag"] ** 2)[None, None, :]
    out: np.ndarray = np.where(wz["S"] > FLOOR_S, r, np.nan)
    return out


def panel_map(ax: plt.Axes, wz: Dict, letter: str, chi: str) -> plt.cm.ScalarMappable:
    """Median own-well ratio over (Phi, |k|), with the S(k) = 1e-2 contour."""
    r = own_well_ratio(wz)
    lv = wz["levels"]
    keep = lv <= np.log10(np.min(wz["phi"][:, 0]))
    xk = wz["kmag"] / wz["kmag"][-1]
    order = np.argsort(xk)
    med = np.nanmedian(r[:, keep][:, :, order], axis=0).T
    phi = 10.0 ** lv[keep]
    mesh = ax.pcolormesh(phi, xk[order], np.clip(med, 1e-3, 2.0), shading="nearest",
                         norm=LogNorm(1e-2, 2.0), cmap="viridis",
                         rasterized=True)  # fmt: skip
    s_med = np.nanmedian(wz["S"][:, keep][:, :, order], axis=0).T
    ax.contour(phi, xk[order], np.log10(np.maximum(s_med, 1e-30)), levels=[-2.0],
               colors="white", linewidths=1.6)  # fmt: skip
    phi_axis(ax)
    ax.set_xlim(phi.max(), 1e-8)
    ax.set_ylabel(r"Wavenumber, $|\mathbf{k}|/K$")
    ax.set_title(f"({letter}) 2D Poisson, {chi}: own-well ratio "
                 r"$\Gamma_k/|\mathbf{k}|^2$", loc="left", fontsize=13)  # fmt: skip
    return mesh


def path_fraction(wz: Dict, lz: Dict) -> np.ndarray:
    """Path fraction s at each landscape level, flows 0..B-1, from the frames."""
    out = np.full(lz["o_end"].shape, np.nan)
    for b in range(out.shape[0]):
        fp, fl = wz["frame_phi"][b], wz["frame_length"][b]
        ok = np.isfinite(fp) & np.isfinite(fl)
        x = np.log10(np.maximum(np.minimum.accumulate(fp[ok]), 1e-300))[::-1]
        out[b] = np.interp(lz["levels"], x, (fl[ok] / fl[ok][-1])[::-1])
    return out


def onset_and_share(wz: Dict) -> Dict[str, np.ndarray]:
    """Per wave: S/S0 at the first slaved level, and the winding share while slaved."""
    r = own_well_ratio(wz)
    s, wind = wz["S"], wz["wind"]
    start = np.argmax(np.isfinite(r), axis=1)  # first level below the start
    s0 = np.take_along_axis(s, start[:, None, :], axis=1)[:, 0]
    slaved = r < SLAVED
    first = np.where(slaved.any(axis=1), np.argmax(slaved, axis=1), -1)
    s_on = np.where(first >= 0, np.take_along_axis(s, np.maximum(first, 0)[:, None, :],
                                                   axis=1)[:, 0], np.nan)  # fmt: skip
    dw = np.diff(wind, axis=1, prepend=0.0)
    late = s <= FLOOR_S  # rates unreliable: count with the wave's last state
    mask = slaved | (late & np.cumsum(slaved, axis=1).astype(bool))
    share = (dw * mask).sum(axis=1) / np.maximum(wind[:, -1], 1e-300)
    return {"s_on": s_on / s0, "share": share}


def figure_floor(data: List, labels: List[str], out: Path, dpi: int) -> None:
    """Waves leave their own wells and ride a floor set by the long waves."""
    fig, axes = plt.subplots(2, 3, figsize=(20.0, 11.0), constrained_layout=True)
    mesh = panel_map(axes[0, 0], data[0]["1-10"]["winding"], "a", TAGS[0][1])
    panel_map(axes[0, 1], data[0]["2-5"]["winding"], "b", TAGS[1][1])
    fig.colorbar(mesh, ax=axes[0, :2], label=r"$\Gamma_k/|\mathbf{k}|^2$ (median)",
                 shrink=0.9)  # fmt: skip
    centres = [0.1, 0.3, 0.5, 0.7, 0.9]
    for i, sys_d in enumerate(data):
        for t_i, (tag, _, ls, mk) in enumerate(TAGS):
            z = sys_d[tag]
            wz, lz = z["winding"], z["landscape"]
            os_ = onset_and_share(wz)
            xk = wz["kmag"] / wz["kmag"][-1]
            meds = [np.nanmedian(os_["s_on"][:, (xk > a) & (xk <= b)])
                    for a, b in BANDS]  # fmt: skip
            axes[0, 2].plot(centres, meds, color=COLORS[i], ls=ls, lw=2.0, marker=mk,
                            ms=6)  # fmt: skip
            ok = reached(lz)
            phi = 10.0 ** lz["levels"]
            band(axes[1, 0], phi, np.where(ok, lz["o_next"], np.nan), COLORS[i], ls)
            x = path_fraction(wz, lz)
            axes[1, 1].plot(np.where(ok, x, np.nan).ravel(), lz["o_end"].ravel(),
                            ls="", marker=mk, ms=3, color=COLORS[i],
                            alpha=0.5)  # fmt: skip
            sh = np.median(os_["share"][:, xk > 0.6])
            axes[1, 2].bar(i + 0.4 * t_i - 0.2, sh, width=0.38, color=COLORS[i],
                           alpha=1.0 if t_i else 0.45, hatch="//" if t_i else None,
                           edgecolor="white")  # fmt: skip
    ax = axes[0, 2]
    ax.set_yscale("log")
    ax.set_xlabel(r"Wavenumber, $|\mathbf{k}|/K$ (band centre)")
    ax.set_ylabel(r"$S(\mathbf{k})/S(\mathbf{k}, 0)$ when slaved")
    ax.set_title(r"(c) how far a wave relaxed before $\Gamma_k < |\mathbf{k}|^2/2$",
                 loc="left", fontsize=13)  # fmt: skip
    ax = axes[1, 0]
    phi_axis(ax)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Soft Subspace Overlap, Next Half Decade")
    ax.set_title("(d) the floor moves: soft modes rotate", loc="left", fontsize=13)
    ax = axes[1, 1]
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(r"Fraction of Path, $s$")
    ax.set_ylabel("Overlap with the Final Soft Subspace")
    ax.set_title("(e) the floor reaches its place as the path ends", loc="left",
                 fontsize=13)  # fmt: skip
    ax = axes[1, 2]
    ax.set_xticks(range(len(labels)), [lab.replace("2D ", "") for lab in labels])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(r"Winding Share While Slaved ($|\mathbf{k}| > 0.6K$)")
    ax.set_title(r"(f) waves wind on the floor (light $\chi=0.1$, hatched 0.4)",
                 loc="left", fontsize=13)  # fmt: skip
    legend(fig, labels, [r"(a, b) $S(\mathbf{k}) = 10^{-2}$ (white)"], GUIDE)
    finish(fig, out / "landscape_floor", dpi)


# ============================================================
# FIGURE 3: A SLOW, CURVED VALLEY
# ============================================================


def figure_valley(data: List, labels: List[str], out: Path, dpi: int) -> None:
    """The residual sits in soft collective modes; the floor is curved."""
    fig, axes = plt.subplots(2, 3, figsize=(20.0, 11.0), constrained_layout=True)
    sys_pts = []
    for i, sys_d in enumerate(data):
        for tag, _, ls, mk in TAGS:
            z = sys_d[tag]
            lz, k1 = z["landscape"], float(z["landscape"]["kmag"][0])
            ok = reached(lz)
            phi = 10.0 ** lz["levels"]
            ratio = np.where(ok, lz["ray_k"] / lz["ray_d"], np.nan)
            band(axes[0, 0], phi, ratio, COLORS[i], ls)
            band(axes[0, 1], phi, np.where(ok, lz["ray_k"] / k1**2, np.nan),
                 COLORS[i], ls)  # fmt: skip
            hz = z["hessian"]
            lv = list(np.round(hz["samples"], 2))
            jj = [lv.index(v) for v in (-2.0, -4.0, -6.0)]
            curv = -hz["lam"][:, jj, 0] / np.sqrt(hz["phi"][:, jj])
            axes[1, 0].plot(i + (0.15 if tag == "2-5" else -0.15),
                            np.median(curv), marker=mk, ms=10, ls="",
                            color=COLORS[i])  # fmt: skip
            m = z["metrics"]
            axes[1, 1].plot(np.median(m["lag"]), np.median(m["excess"]), marker=mk,
                            ms=11, ls="", color=COLORS[i])  # fmt: skip
            j = list(lz["levels"]).index(0.0)
            soft = 1.0 - np.nanmedian(ratio[:, j])
            sys_pts.append((soft, float(np.median(m["lag"]))))
            axes[1, 2].plot(soft, np.median(m["lag"]), marker=mk, ms=11, ls="",
                            color=COLORS[i])  # fmt: skip
            if i == 1:  # spectral weight of the residual, 2D Gaussian
                for c, lev in enumerate((1.0, -1.0, -4.0)):
                    jl = list(lz["levels"]).index(lev)
                    ev, w = lz["k_eig"][0, jl] / k1**2, lz["weight"][0, jl]
                    o = np.argsort(ev)
                    col = plt.get_cmap("plasma")(0.15 + 0.35 * c)
                    axes[0, 2].plot(ev[o], np.cumsum(w[o]), color=col, ls=ls, lw=2.0,
                                    label=f"{TAGS[0 if tag == '1-10' else 1][1]}, "
                                    rf"$\Phi = 10^{{{lev:g}}}$")  # fmt: skip
    ax = axes[0, 0]
    phi_axis(ax)
    ax.axhline(1.0, color=GUIDE, lw=0.8, zorder=0)
    ax.set_ylim(0.0, 1.15)
    ax.set_ylabel(r"$u^\top K u \,/\, u^\top D u$")
    ax.set_title("(a) residual decay rate / independent-well rate", loc="left",
                 fontsize=13)  # fmt: skip
    ax = axes[0, 1]
    phi_axis(ax)
    ax.set_yscale("log")
    ax.axhline(1.0, color=GUIDE, lw=0.8, zorder=0)
    ax.set_ylabel(r"$-\frac{1}{2}\,d\ln\Phi/dt \;/\; k_1^2$")
    ax.set_title("(b) energy decay vs the slowest single well", loc="left",
                 fontsize=13)  # fmt: skip
    ax = axes[0, 2]
    ax.set_xscale("log")
    ax.axvline(1.0, color=GUIDE, lw=0.8, zorder=0)
    ax.set_xlabel(r"Collective Stiffness, $\lambda(K)/k_1^2$")
    ax.set_ylabel(r"Cumulative Share of $|u|^2$")
    ax.set_title("(c) where the residual sits (2D Gaussian, flow 0)", loc="left",
                 fontsize=13)  # fmt: skip
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax = axes[1, 0]
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)), [lab.replace("2D ", "") for lab in labels])
    ax.set_ylabel(r"Floor Curvature, $|\lambda_{\min}|/\sqrt{\Phi}$")
    ax.set_title("(d) the valley floor is curved (half the tangents)", loc="left",
                 fontsize=13)  # fmt: skip
    ax = axes[1, 1]
    ax.axhline(1.0, color=GUIDE, lw=0.8, zorder=0)
    ax.axvline(1.0, color=GUIDE, lw=0.8, zorder=0)
    ax.set_xlabel(r"Clock Lag, $t_{1/2}/t_{1/2}^{\mathrm{model}}$")
    ax.set_ylabel(r"Path Length, $\ell/\ell^{\mathrm{model}}$")
    ax.set_title("(e) slower and longer than the separable funnel", loc="left",
                 fontsize=13)  # fmt: skip
    ax = axes[1, 2]
    xs, ys = np.array(sys_pts).T
    rho = float(
        np.corrcoef(np.argsort(np.argsort(xs)), np.argsort(np.argsort(ys)))[0, 1]
    )
    ax.set_xlabel(r"Valley Softness at $\Phi = 1$, $1 - u^\top K u/u^\top D u$")
    ax.set_ylabel(r"Clock Lag, $t_{1/2}/t_{1/2}^{\mathrm{model}}$")
    ax.set_title(f"(f) softer valley, slower flow (Spearman {rho:.2f})", loc="left",
                 fontsize=13)  # fmt: skip
    print(f"valley (f) Spearman {rho:.2f}", flush=True)
    legend(fig, labels, [], GUIDE)
    finish(fig, out / "landscape_valley", dpi)


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
    figure_wells(data, args.labels, args.out, args.dpi)
    figure_floor(data, args.labels, args.out, args.dpi)
    figure_valley(data, args.labels, args.out, args.dpi)
    print("Done.", flush=True)
