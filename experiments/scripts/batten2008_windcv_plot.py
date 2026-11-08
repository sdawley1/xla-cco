#!/usr/bin/env python3

"""
Winding of the density waves in time, and winding as a collective variable.

Reads winding_chi<tag>.npz (batten2008_winding.py). Instantaneous rates
(theta', the pitch, Omega, sigma, cos alpha) use only waves with S > 1e-8:
|grad arg rho_k| ~ |k| / sqrt(2 S_k) diverges as a wave relaxes, so for small
S the rate at a stored frame is dominated by the integrator's error (a
displacement delta turns the phase by ~ |k| delta / sqrt(S); following the
exact flow from such a frame, these rates decay 25-80 fold within 2 time
units at fixed Phi). The cumulative winding W_n counts phase changes between
frames and is not affected in the same way.
Two kinds of figure:

- <prefix>_N<N>_winding_kymograph (one system; rows chi): (a) flow 0, the
  cumulative winding W_n(t) of every constrained wave (rows sorted by |k|)
  against log10 t / tau_1, with each wave's relaxation front (the last time
  S(k_n) exceeds 1e-2 and 1e-8); (b) the median over flows of W_n against
  the energy clock log10 Phi; (c) the median spiral pitch |theta_n'| /
  |d ln|rho_n| / dt|, the angle of the path to the radial direction in the
  wave's own (log |rho|, arg rho) plane; (d) W_n(t) of flow 0 as a surface.
- winding_cv (several systems, --compare): (a) Omega, the energy-weighted
  RMS winding rate sqrt(sum S theta'^2 / sum S), against Phi; (b) the spiral
  number sigma = Omega / (-d ln Phi / dt / 2), Omega over the energy-weighted
  decay rate (0 for separable, harmonic relaxation); (c) the mean winding
  over waves against Phi; (d) the triad identity: theta' from the triad sum
  against theta' from the gradient, flow 0 of the first system; (e) the share
  of each wave's phase torque carried by triads through unconstrained
  wavevectors |k +- q| > K, against |k| / K; (f) the alignment |cos alpha_k|
  of the force with each wave's phase direction, theta_k' = -|grad theta_k|
  |grad Phi| cos alpha_k (median over waves; dotted: 1 / sqrt(N d), a random
  direction); (g) per flow, the final winding of the short waves
  |k| > 0.6 K against the slip K ell / 2 pi, the phase a wave at K gains
  when its particles are carried by the per-particle RMS displacement
  ell = |Delta r| / sqrt(N) (from offplane_chi<tag>.npz beside the file).

- with --total (and --compare): the total winding sum_n W_n over all
  constrained waves |k| <= K (top) and per wave, divided by M (bottom),
  against log10 Phi, the path fraction s = ell / ell(inf) and log10 t / tau_1;
  median and 10-90 % band over flows (where at least half of them reach).

Needs matplotlib (the plottime environment).

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
from batten2008_plot import (  # noqa: E402  (also applies the house rcParams)
    CHI_TAGS,
    NAMES,
    add_common_arguments,
    configure,
    finish,
    load,
    plt,
)
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

COLORS = ("black", "#1f4e9c", "#c0392b", "#2e7d32", "#b9770e", "#7d3c98")
FRONTS = (1e-2, 1e-8)  # S thresholds of the relaxation fronts
FLOOR = 1e-8  # instantaneous rates only for S above this (see the docstring)
SHORT = 0.6  # short waves: |k| > SHORT K

# ============================================================
# HELPERS
# ============================================================


def cv_curves(z: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Omega, sigma and the mean winding per flow and level (flows, levels).

    Waves with S <= FLOOR are left out: their instantaneous phase rate at a
    stored frame is dominated by integration error.
    """
    ok = z["S"] > FLOOR
    s = np.where(ok, z["S"], 0.0)
    th, lr = np.where(ok, z["theta_rate"], 0.0), np.where(ok, z["lnrho_rate"], 0.0)
    weight = s.sum(axis=2)
    weight = np.where(weight > 0.0, weight, np.nan)  # no resolved wave: no rate
    omega = np.sqrt((s * th**2).sum(axis=2) / weight)
    decay = np.maximum(-(s * lr).sum(axis=2) / weight, 1e-300)  # NaN stays NaN
    return {"omega": omega, "sigma": omega / decay, "wind": z["wind"].mean(axis=2)}


def band(ax: plt.Axes, x: np.ndarray, rows: np.ndarray, color: str, ls: str) -> None:
    """Median and 10-90 % band over flows of rows (flows, x)."""
    q10, q50, q90 = np.nanpercentile(rows, [10.0, 50.0, 90.0], axis=0)
    ax.fill_between(x, q10, q90, color=color, alpha=0.15, lw=0)
    ax.plot(x, q50, color=color, ls=ls, lw=2.0)


def phi_axis(ax: plt.Axes) -> None:
    """Log Phi axis, high to low."""
    ax.set_xscale("log")
    if not ax.xaxis_inverted():
        ax.invert_xaxis()
    ax.set_xlabel(r"Energy, $\Phi$")


def fronts(t: np.ndarray, sk: np.ndarray, thr: float) -> np.ndarray:
    """Last time each wave's S exceeds thr (NaN if never), per column of sk."""
    above = sk > thr
    last = sk.shape[0] - 1 - np.argmax(above[::-1], axis=0)
    out: np.ndarray = np.where(above.any(axis=0), t[last], np.nan)
    return out


# ============================================================
# FIGURES
# ============================================================


def figure_kymograph(root: Path, n: int, out: Path, dpi: int) -> None:
    """W_n(t) for every wave: flow 0 in time, median on the Phi clock, pitch, 3D."""
    fig = plt.figure(figsize=(23.0, 10.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 4)
    for row, (tag, label) in enumerate(CHI_TAGS):
        z = load(root / f"winding_chi{tag}.npz")
        x_k = z["kmag"] / z["kmag"][-1]
        let = "abcdefgh"[4 * row : 4 * row + 4]
        t = z["flow0_t"] / float(z["tau1"])
        keep = t > 0
        lt = np.log10(t[keep])
        wind = z["flow0_wind"][keep]
        vmax = np.percentile(wind[-1], 99.0)
        ax = fig.add_subplot(grid[row, 0])
        im = ax.pcolormesh(lt, x_k, wind.T, cmap="viridis", shading="nearest",
                           vmin=0.0, vmax=vmax, rasterized=True)  # fmt: skip
        for thr, ls in zip(FRONTS, ("-", "--"), strict=True):
            ax.plot(np.log10(fronts(t, z["flow0_S"], thr)), x_k, color="white",
                    lw=1.0, ls=ls)  # fmt: skip
        fig.colorbar(im, ax=ax, label="Turns So Far, $W_n$", shrink=0.9)
        ax.set_title(f"({let[0]}) {label}, flow 0: winding in time", loc="left",
                     fontsize=12)  # fmt: skip
        ax.set_xlabel(r"Time, $\log_{10}(t/\tau_1)$")
        ax.set_ylabel(r"Wavenumber, $|\mathbf{k}_n|/K$")
        ax = fig.add_subplot(grid[row, 1])
        lv = 10.0 ** np.asarray(z["levels"])
        reach = lv <= np.min(z["phi"][:, 0])
        med = np.median(z["wind"], axis=0)[reach]
        im = ax.pcolormesh(lv[reach], x_k, med.T, cmap="viridis", shading="nearest",
                           vmin=0.0, vmax=vmax, rasterized=True)  # fmt: skip
        phi_axis(ax)
        fig.colorbar(im, ax=ax, label=r"Turns So Far, median", shrink=0.9)
        ax.set_title(f"({let[1]}) {label}: median over flows", loc="left",
                     fontsize=12)  # fmt: skip
        ax.set_ylabel(r"Wavenumber, $|\mathbf{k}_n|/K$")
        ax = fig.add_subplot(grid[row, 2])
        pitch = np.abs(z["theta_rate"]) / np.maximum(np.abs(z["lnrho_rate"]), 1e-300)
        pitch = np.where(z["S"] > FLOOR, pitch, np.nan)  # error-dominated rates
        med = np.nanmedian(pitch, axis=0)[reach]
        im = ax.pcolormesh(lv[reach], x_k, med.T, cmap="magma", shading="nearest",
                           norm=LogNorm(1e-2, 1e2), rasterized=True)  # fmt: skip
        phi_axis(ax)
        fig.colorbar(im, ax=ax, label=r"$|\dot\theta_n| / |d\ln|\rho_n|/dt|$",
                     shrink=0.9)  # fmt: skip
        ax.set_title(f"({let[2]}) {label}: spiral pitch (S > 1e-8)", loc="left",
                     fontsize=12)  # fmt: skip
        ax.set_ylabel(r"Wavenumber, $|\mathbf{k}_n|/K$")
        ax = fig.add_subplot(grid[row, 3], projection="3d")
        step = max(1, lt.shape[0] // 150)
        tt, kk = np.meshgrid(lt[::step], x_k, indexing="ij")
        ax.plot_surface(tt, kk, wind[::step], cmap="viridis", lw=0, vmin=0.0,
                        vmax=vmax, rcount=tt.shape[0], ccount=min(kk.shape[1], 200),
                        antialiased=False)  # fmt: skip
        ax.set_xlabel(r"$\log_{10}(t/\tau_1)$", fontsize=11, labelpad=4)
        ax.set_ylabel(r"$|\mathbf{k}_n|/K$", fontsize=11, labelpad=4)
        ax.set_zlabel(r"$W_n$", fontsize=11, labelpad=4)
        ax.tick_params(labelsize=9, pad=1)
        ax.view_init(elev=28, azim=-55)
        ax.set_title(f"({let[3]}) {label}, flow 0", loc="left", fontsize=12)
    handles = [Line2D([], [], color="0.3", lw=1.0, ls=ls) for ls in ("-", "--")]
    fig.legend(handles, [r"(a, e) last $S(k_n) > 10^{-2}$",
                         r"(a, e) last $S(k_n) > 10^{-8}$"],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.0),
               frameon=False)  # fmt: skip
    finish(fig, out / f"{NAMES['prefix']}_N{n}_winding_kymograph", dpi)


def figure_cv(paths: List[Path], labels: List[str], out: Path, name: str,
              dpi: int) -> None:  # fmt: skip
    """Winding rate and spiral number against Phi across systems; triad check."""
    fig, axes = plt.subplots(2, 4, figsize=(23.0, 10.4), constrained_layout=True)
    handles = []
    for i, path in enumerate(paths):
        z = load(path)
        color = COLORS[(i // 2) % len(COLORS)]
        ls = "-" if i % 2 == 0 else "--"
        handles.append(Line2D([], [], color=color, ls=ls, lw=2.0))
        lv = 10.0 ** np.asarray(z["levels"])
        reach = lv <= np.min(z["phi"][:, 0])
        cv = cv_curves(z)
        band(axes[0, 0], lv[reach], cv["omega"][:, reach], color, ls)
        band(axes[0, 1], lv[reach], cv["sigma"][:, reach], color, ls)
        band(axes[0, 2], lv[reach], cv["wind"][:, reach], color, ls)
        if "grad_theta" in z:  # |cos alpha| = |theta'| / (|grad theta| |grad Phi|)
            denom = z["grad_theta"] * z["grad_phi"][:, :, None]
            cos = np.abs(z["theta_rate"]) / np.maximum(denom, 1e-300)
            cos = np.where(z["S"] > FLOOR, cos, np.nan)  # error-dominated rates
            band(axes[1, 2], lv[reach], np.nanmedian(cos, axis=2)[:, reach], color, ls)
            nd = int(z["nvec"].shape[1]) * int(z["n_part"]) if "n_part" in z else 0
            if nd:
                axes[1, 2].axhline(1.0 / np.sqrt(nd), color=color, lw=0.6, ls=":")
        x_k = z["kmag"] / z["kmag"][-1]
        off_path = path.with_name(path.name.replace("winding_", "offplane_"))
        if off_path.exists() and "n_part" in z:
            with np.load(off_path) as o:
                ell = np.asarray(o["chord"]) / np.sqrt(float(z["n_part"]))
            short = z["wind"][:, -1, x_k > SHORT].mean(axis=1)
            axes[1, 3].plot(float(z["kmag"][-1]) * ell / (2.0 * np.pi), short, ls="",
                            marker="o" if i % 2 == 0 else "s", ms=5, color=color,
                            mfc="none")  # fmt: skip
        shares = [z[f"triad{j}_unc_share"] for j in range(len(z["triad_levels"]))
                  if f"triad{j}_unc_share" in z]  # fmt: skip
        if shares:
            order = np.argsort(x_k)
            share = np.median(np.stack(shares), axis=0)
            k_bins = np.linspace(0.0, 1.0, 11)
            idx = np.clip(np.digitize(x_k[order], k_bins) - 1, 0, 9)
            means = [np.mean(share[order][idx == b]) if np.any(idx == b) else np.nan
                     for b in range(10)]  # fmt: skip
            axes[1, 1].plot(0.5 * (k_bins[1:] + k_bins[:-1]), means, color=color,
                            ls=ls, lw=2.0, marker="o", ms=4)  # fmt: skip
        if i == 0:
            for j in range(len(z["triad_levels"])):
                if f"triad{j}_pred" not in z:
                    continue
                rho_s = np.sqrt(np.maximum(z[f"triad{j}_S"], 1e-300))
                ex, pr = z[f"triad{j}_exact"], z[f"triad{j}_pred"]
                axes[1, 0].plot(np.abs(ex) / rho_s, np.abs(pr) / rho_s, ls="",
                                marker=".", ms=3,
                                color=plt.get_cmap("viridis")(j / 4))  # fmt: skip
    for ax in axes[0]:
        phi_axis(ax)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel(r"Winding Rate, $\Omega$")
    axes[0, 0].set_title(r"(a) $\Omega = (\sum S\dot\theta^2/\sum S)^{1/2}$",
                         loc="left", fontsize=13)  # fmt: skip
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel(r"Spiral Number, $\sigma$")
    axes[0, 1].set_title(r"(b) $\sigma = \Omega / (-\frac{1}{2}d\ln\Phi/dt)$",
                         loc="left", fontsize=13)  # fmt: skip
    axes[0, 2].set_ylabel(r"Mean Turns So Far, $\langle W_n\rangle$")
    axes[0, 2].set_title("(c) winding accumulated, mean over waves", loc="left",
                         fontsize=13)  # fmt: skip
    ax = axes[1, 0]
    ax.set_xscale("log")
    ax.set_yscale("log")
    lim = ax.get_xlim()
    ax.plot(lim, lim, color="0.6", lw=0.8, zorder=0)
    ax.set_title(f"(d) triad identity, {labels[0]}, flow 0", loc="left",
                 fontsize=13)  # fmt: skip
    ax.set_xlabel(r"$|d\rho_k/dt|/|\rho_k|$ from $-\nabla\Phi$")
    ax.set_ylabel(r"$|d\rho_k/dt|/|\rho_k|$ from the triad sum")
    ax = axes[1, 1]
    ax.set_ylim(0.0, 1.05)
    ax.set_title("(e) torque through unconstrained $\\mathbf{k}\\pm\\mathbf{q}$",
                 loc="left", fontsize=13)  # fmt: skip
    ax.set_xlabel(r"Wavenumber, $|\mathbf{k}|/K$")
    ax.set_ylabel("Share of Phase Torque")
    ax = axes[1, 2]
    phi_axis(ax)
    ax.set_xlim(axes[0, 0].get_xlim())
    ax.set_yscale("log")
    ax.set_ylabel(r"Alignment, median $|\cos\alpha_k|$")
    ax.set_title(r"(f) $\dot\theta_k = -|\nabla\theta_k||\nabla\Phi|\cos\alpha_k$",
                 loc="left", fontsize=13)  # fmt: skip
    ax = axes[1, 3]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(r"(g) winding against the slip $K\ell/2\pi$, per flow", loc="left",
                 fontsize=13)  # fmt: skip
    ax.set_xlabel(r"Slip, $K\ell/2\pi$ ($\ell = |\Delta r|/\sqrt{N}$)")
    ax.set_ylabel(r"Turns, $|\mathbf{k}| > 0.6K$")
    axes[0, 3].set_axis_off()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4),
               bbox_to_anchor=(0.5, 0.0), frameon=False)  # fmt: skip
    finish(fig, out / name, dpi)


def on_grid(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """y(x) of each flow (rows, NaN-padded) interpolated on grid; NaN outside."""
    out = np.full((x.shape[0], grid.shape[0]), np.nan)
    for b in range(x.shape[0]):
        ok = np.isfinite(x[b]) & np.isfinite(y[b])
        xb, yb = x[b][ok], y[b][ok]
        order = np.argsort(xb, kind="stable")
        xb, yb = xb[order], yb[order]
        inside = (grid >= xb[0]) & (grid <= xb[-1])
        out[b, inside] = np.interp(grid[inside], xb, yb)
    return out


def figure_total(paths: List[Path], labels: List[str], out: Path, name: str,
                 dpi: int) -> None:  # fmt: skip
    """Total winding sum_n W_n against log10 Phi, path fraction and time."""
    fig, axes = plt.subplots(2, 3, figsize=(19.0, 10.0), constrained_layout=True)
    handles = []
    g_phi = np.linspace(-12.0, 3.0, 151)  # winding has stopped well before 1e-12
    g_s = np.linspace(0.0, 1.0, 201)
    g_t = np.linspace(-6.0, 2.0, 161)
    for i, path in enumerate(paths):
        z = load(path)
        color = COLORS[(i // 2) % len(COLORS)]
        ls = "-" if i % 2 == 0 else "--"
        handles.append(Line2D([], [], color=color, ls=ls, lw=2.0))
        total, m = z["total_wind"], float(z["kmag"].shape[0])
        logphi = np.log10(np.maximum(z["frame_phi"], 1e-300))
        frac = z["frame_length"] / np.nanmax(z["frame_length"], axis=1, keepdims=True)
        logt = np.log10(
            np.where(z["frame_t"] > 0, z["frame_t"], np.nan) / float(z["tau1"])
        )
        for col, (x, grid) in enumerate(((logphi, g_phi), (frac, g_s), (logt, g_t))):
            curves = on_grid(x, total, grid)
            keep = np.mean(np.isfinite(curves), axis=0) >= 0.5  # most flows present
            for row, scale in enumerate((1.0, m)):
                band(axes[row, col], grid[keep], curves[:, keep] / scale, color, ls)
    for row, ylab in enumerate((r"Total Turns, $\sum_n W_n$",
                                r"Turns per Wave, $\sum_n W_n / M$")):  # fmt: skip
        for col in range(3):
            axes[row, col].set_ylabel(ylab)
            axes[row, col].set_yscale("log" if row == 0 else "linear")
        ax = axes[row, 0]
        ax.invert_xaxis()
        ax.set_xlabel(r"Energy, $\log_{10}\Phi$")
        axes[row, 1].set_xlabel(r"Fraction of Path, $s = \ell/\ell(\infty)$")
        axes[row, 2].set_xlabel(r"Time, $\log_{10}(t/\tau_1)$")
    names = ("energy", "path fraction", "time")
    for col in range(3):
        title = f"({'abc'[col]}) total over all constrained waves vs {names[col]}"
        axes[0, col].set_title(title, loc="left", fontsize=13)
        axes[1, col].set_title(f"({'def'[col]}) per wave vs {names[col]}", loc="left",
                               fontsize=13)  # fmt: skip
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4),
               bbox_to_anchor=(0.5, 0.0), frameon=False)  # fmt: skip
    finish(fig, out / name, dpi)


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--root", type=Path, default=Path("experiments/trajectories/batten2008")
    )
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("experiments/figures"))
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--compare", nargs="+", type=Path, default=None)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--compare-name", default="winding_cv")
    parser.add_argument(
        "--total",
        action="store_true",
        help="with --compare: total winding vs Phi, s and time",
    )
    add_common_arguments(parser)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.compare:
        assert args.labels and len(args.labels) == len(args.compare), "one label each"
        maker = figure_total if args.total else figure_cv
        maker(args.compare, args.labels, args.out, args.compare_name, args.dpi)
    else:
        configure(args.chi, args.prefix, args.dim)
        figure_kymograph(args.root / f"N{args.n}", args.n, args.out, args.dpi)
    print("Done.", flush=True)
