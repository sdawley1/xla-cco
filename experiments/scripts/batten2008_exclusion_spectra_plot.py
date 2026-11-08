#!/usr/bin/env python3

"""
Spread of a one-wave excitation over collective modes and waves; drive selectivity.

Reads exclusion_chi<tag>.npz (batten2008_exclusion.py) of each system root;
the full eigensystem (lambda_m, q_m) of K = (2/V) J J^T at ground state 0
and per-wave statistics over 5 ground states. Near a ground state the
constrained waves u = (Re rho_k, Im rho_k) obey d^2u/dt^2 = -K u (MD, mass
1): exciting wave k alone puts weight w_m(k) = (q_m(a_k)^2 + q_m(b_k)^2)/2 on
collective mode m (frequency sqrt(lambda_m)), and over long times wave q
holds the share W(q|k) = sum_m w_m(k) w'_m(q) of its energy (w'_m(q) =
q_m(a_q)^2 + q_m(b_q)^2). Writes to <out>:

- exclusion_mode_spectra (rows chi = 0.1, 0.4; 2D Poisson, ground state 0):
  (a, e) w_m(k) for every constrained wave k against omega/K, the line
  omega = |k| for independent wells; (b, f) the same against omega/|k|;
  (c, g) the long-time share W(q|k) over the receiving waves |q|/K; (d)
  number of collective modes per excited wave and (h) the mean and spread
  of omega/|k| it carries, every start (median over 5 ground states).
- exclusion_drive: (a) selectivity (share of the steady response |u_hat|^2
  in the driven wave) of a travelling-wave field eps cos(k.r - Omega t) with
  Langevin friction gamma, against Omega/|k| (median over waves; Omega = 0
  is the static field, selectivity 1); (b) the same at Omega = |k|/2 and
  Omega = |k| against |k|/K, every start; (c) a heat bath acting only along
  wave k (overdamped): share of the excess covariance in wave k.

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
from typing import Dict, List, Sequence

# ----- numerics -----
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batten2008_plot import finish, load, plt  # noqa: E402  (house rcParams)
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

COLORS = ("black", "#1f4e9c", "#c0392b", "#2e7d32", "#b9770e", "#7d3c98")
TAGS = (("1-10", r"$\chi = 0.1$", "-", "o"), ("2-5", r"$\chi = 0.4$", "--", "s"))
BINS = np.linspace(0.0, 1.0, 11)  # |k|/K bins of the summaries
OMEGA_FRACS = np.linspace(0.0, 2.0, 41)  # Omega/|k| of the drive scan
GAMMAS = (0.05, 0.5)  # friction over |k|
N_SAMPLE = 80  # driven waves sampled per system
GUIDE = "0.6"
CHI_COLOR = {"1-10": "#1f4e9c", "2-5": "#c0392b"}

# ============================================================
# HELPERS
# ============================================================


def wave_weights(q: np.ndarray, m: int) -> np.ndarray:
    """(M, 2M) weight of each collective mode in each wave (quadratures summed)."""
    assert q.shape == (2 * m, 2 * m), "q must be (2M, 2M)"
    sq = q**2
    out: np.ndarray = sq[:m] + sq[m:]
    return out


def binned(xk: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Median of rows (ground states, M) per |k|/K bin."""
    idx = np.clip(np.digitize(xk, BINS) - 1, 0, BINS.shape[0] - 2)
    out = np.array([np.median(rows[:, idx == b]) if np.any(idx == b) else np.nan
                    for b in range(BINS.shape[0] - 1)])  # fmt: skip
    assert out.shape == (BINS.shape[0] - 1,), "one value per bin"
    return out


def drive_selectivity(z: Dict[str, np.ndarray], omega_fracs: Sequence[float],
                      gamma: float, waves: np.ndarray) -> np.ndarray:  # fmt: skip
    """(waves, omegas) share of |u_hat|^2 in the driven wave, travelling field.

    u'' + g u' + K u = (V eps/2) K (e_a + i e_b) e^{-i W t}; the steady
    response is u_hat ~ (K - W^2 - i g W)^-1 K (e_a + i e_b), g = gamma |k|.
    """
    lam, q, km = z["eig_lam"][0], z["eig_q"][0], z["kmag"]
    m = km.shape[0]
    out = np.zeros((waves.shape[0], len(omega_fracs)))
    for i, k in enumerate(waves):
        proj = lam * (q[k] + 1j * q[k + m])  # q^T K (e_a + i e_b)
        for j, frac in enumerate(omega_fracs):
            w = frac * km[k]
            u = q @ (proj / (lam - w**2 - 1j * gamma * km[k] * w))
            p = np.abs(u[:m]) ** 2 + np.abs(u[m:]) ** 2
            out[i, j] = p[k] / p.sum()
    assert np.all((out >= 0.0) & (out <= 1.0 + 1e-12)), "shares lie in [0, 1]"
    return out


def bath_selectivity(z: Dict[str, np.ndarray], waves: np.ndarray) -> np.ndarray:
    """Share of the excess covariance in wave k when only wave k is heated.

    Overdamped: K S + S K = 2 D, D = sum_c (K e_c)(K e_c)^T / K_cc over the
    two quadratures c of wave k; in the eigenbasis S_mn = 2 D_mn/(l_m + l_n).
    """
    lam, q, km = z["eig_lam"][0], z["eig_q"][0], z["kmag"]
    m = km.shape[0]
    kdiag = (q**2 * lam).sum(axis=1)
    denom = 2.0 / (lam[:, None] + lam[None, :])
    out = np.zeros(waves.shape[0])
    for i, k in enumerate(waves):
        own, total = 0.0, 0.0
        for c in (k, k + m):
            d = lam * q[c] / np.sqrt(kdiag[c])
            cmat = np.outer(d, d) * denom
            total += float(np.trace(cmat))
            own += sum(float(q[cc] @ cmat @ q[cc]) for cc in (k, k + m))
        out[i] = own / total
    assert np.all((out >= 0.0) & (out <= 1.0 + 1e-9)), "shares lie in [0, 1]"
    return out


# ============================================================
# FIGURE 1: SPECTRA
# ============================================================


def panel_freq(ax: plt.Axes, z: Dict[str, np.ndarray], scaled: bool) -> object:
    """Heat map: collective-frequency weights of every excited wave."""
    km = z["kmag"]
    m = km.shape[0]
    om = np.sqrt(z["eig_lam"][0])
    wts = 0.5 * wave_weights(z["eig_q"][0], m)
    edges = np.linspace(0.0, 3.0 if scaled else 2.2, 121)
    rows = [np.histogram(om / (km[k] if scaled else km[-1]), bins=edges,
                         weights=wts[k])[0] for k in range(m)]  # fmt: skip
    img = np.array(rows)
    mesh = ax.pcolormesh(edges, np.arange(m + 1), np.where(img > 0, img, np.nan),
                         norm=LogNorm(1e-4, 1.0), cmap="viridis",
                         rasterized=True)  # fmt: skip
    if scaled:
        ax.axvline(1.0, color="white", lw=1.0, ls="--")
    else:
        ax.plot(km / km[-1], np.arange(m) + 0.5, color="white", lw=1.0, ls="--")
    ticks = [int(np.argmin(np.abs(km / km[-1] - f))) for f in (0.2, 0.4, 0.6, 0.8, 1.0)]
    ax.set_yticks(ticks, [f"{km[t] / km[-1]:.1f}" for t in ticks])
    ax.set_ylabel(r"Excited Wave, $|\mathbf{k}|/K$")
    ax.set_xlabel(r"Collective Frequency, $\omega/|\mathbf{k}|$" if scaled
                  else r"Collective Frequency, $\omega/K$")  # fmt: skip
    return mesh


def panel_spread(ax: plt.Axes, z: Dict[str, np.ndarray]) -> object:
    """Heat map: long-time share of the excitation energy in each receiving wave."""
    km = z["kmag"]
    m = km.shape[0]
    wts = wave_weights(z["eig_q"][0], m)
    share = 0.5 * wts @ wts.T  # (excited k, receiving q)
    edges = np.linspace(0.0, 1.0, 41)
    idx = np.clip(np.digitize(km / km[-1], edges) - 1, 0, edges.shape[0] - 2)
    img = np.zeros((m, edges.shape[0] - 1))
    for b in range(edges.shape[0] - 1):
        img[:, b] = share[:, idx == b].sum(axis=1)
    mesh = ax.pcolormesh(edges, np.arange(m + 1), np.where(img > 0, img, np.nan),
                         norm=LogNorm(1e-4, 1.0), cmap="magma",
                         rasterized=True)  # fmt: skip
    ax.plot(km / km[-1], np.arange(m) + 0.5, color="white", lw=0.8, ls=":")
    ticks = [int(np.argmin(np.abs(km / km[-1] - f))) for f in (0.2, 0.4, 0.6, 0.8, 1.0)]
    ax.set_yticks(ticks, [f"{km[t] / km[-1]:.1f}" for t in ticks])
    ax.set_ylabel(r"Excited Wave, $|\mathbf{k}|/K$")
    ax.set_xlabel(r"Receiving Wave, $|\mathbf{q}|/K$")
    self_share = np.median(np.diag(share))
    print(f"  long-time self share median {self_share:.3f}", flush=True)
    return mesh


def figure_spectra(data: List, labels: List[str], out: Path, dpi: int) -> None:
    """Frequency and wave-space spread of one-wave excitations, every excited wave."""
    fig, axes = plt.subplots(2, 4, figsize=(25.0, 11.5), constrained_layout=True)
    centers = 0.5 * (BINS[1:] + BINS[:-1])
    letters = iter("abcdefgh")
    for row, (tag, chi, _, _) in enumerate(TAGS):
        z = data[0][tag]
        m1 = panel_freq(axes[row, 0], z, False)
        panel_freq(axes[row, 1], z, True)
        m2 = panel_spread(axes[row, 2], z)
        for col, title in enumerate(("collective frequencies of each wave",
                                     r"the same, $\omega/|\mathbf{k}|$",
                                     "where its energy ends up")):  # fmt: skip
            axes[row, col].set_title(f"({next(letters)}) {labels[0]}, {chi}: {title}",
                                     loc="left", fontsize=13)  # fmt: skip
        next(letters) if row == 0 else None
    for i, sys_d in enumerate(data):
        for tag, _, ls, mk in TAGS:
            z = sys_d[tag]
            xk = z["kmag"] / z["kmag"][-1]
            axes[0, 3].plot(centers, binned(xk, z["modes"]), color=COLORS[i], ls=ls,
                            lw=2.0, marker=mk, ms=5)  # fmt: skip
            axes[1, 3].plot(centers, binned(xk, z["om_std"]), color=COLORS[i], ls=ls,
                            lw=2.0, marker=mk, ms=5)  # fmt: skip
            axes[1, 3].plot(centers, binned(xk, z["om_mean"]), color=COLORS[i], ls=ls,
                            lw=0.9)  # fmt: skip
    fig.colorbar(m1, ax=axes[:, :2], label="Weight on the Collective Mode",
                 shrink=0.6, location="bottom")  # fmt: skip
    fig.colorbar(m2, ax=axes[:, 2], label="Long-Time Energy Share",
                 shrink=0.8, location="bottom")  # fmt: skip
    ax = axes[0, 3]
    ax.set_yscale("log")
    ax.set_xlabel(r"Excited Wave, $|\mathbf{k}|/K$")
    ax.set_ylabel("Collective Modes per Wave")
    ax.set_title("(d) how many modes one wave excites", loc="left", fontsize=13)
    ax = axes[1, 3]
    ax.axhline(1.0, color=GUIDE, lw=0.8, zorder=0)
    ax.set_xlabel(r"Excited Wave, $|\mathbf{k}|/K$")
    ax.set_ylabel(r"$\omega/|\mathbf{k}|$: spread (thick), mean (thin)")
    ax.set_title("(h) frequency spread and mean", loc="left", fontsize=13)
    handles = [Line2D([], [], color=COLORS[i], lw=2.0) for i in range(len(labels))]
    handles += [Line2D([], [], color="0.3", lw=2.0, ls=ls, marker=mk)
                for _, _, ls, mk in TAGS]  # fmt: skip
    axes[1, 3].legend(handles, labels + [c for _, c, _, _ in TAGS], frameon=False,
                      fontsize=9, loc="center left", bbox_to_anchor=(0.02, 0.68),
                      ncol=2)  # fmt: skip
    finish(fig, out / "exclusion_mode_spectra", dpi)


# ============================================================
# FIGURE 2: DRIVES
# ============================================================


def figure_drive(data: List, labels: List[str], out: Path, dpi: int) -> None:
    """Selectivity of a periodic travelling field and of a mode-selective bath."""
    fig, axes = plt.subplots(1, 3, figsize=(21.0, 6.4), constrained_layout=True)
    centers = 0.5 * (BINS[1:] + BINS[:-1])
    for i, sys_d in enumerate(data):
        for tag, _, ls, mk in TAGS:
            z = sys_d[tag]
            m = z["kmag"].shape[0]
            waves = np.unique(np.linspace(0, m - 1, N_SAMPLE).astype(int))
            xk = z["kmag"][waves] / z["kmag"][-1]
            if i == 0:
                for gfac, lw in zip(GAMMAS, (2.0, 1.0), strict=True):
                    sel = drive_selectivity(z, OMEGA_FRACS, gfac, waves)
                    axes[0].plot(OMEGA_FRACS, np.median(sel, axis=0), ls=ls, lw=lw,
                                 color=CHI_COLOR[tag])  # fmt: skip
            sel = drive_selectivity(z, (0.5, 1.0), GAMMAS[0], waves)
            for j, mkr in enumerate(("v", "^")):
                axes[1].plot(centers, binned(xk, sel[None, :, j]), color=COLORS[i],
                             ls=ls, lw=1.6, marker=mkr, ms=5)  # fmt: skip
            bath = bath_selectivity(z, waves)
            axes[2].plot(centers, binned(xk, bath[None, :]), color=COLORS[i], ls=ls,
                         lw=2.0, marker=mk, ms=5)  # fmt: skip
            half, full = np.median(sel[:, 0]), np.median(sel[:, 1])
            print(f"{labels[i]} chi{tag}: bath median {np.median(bath):.3f}; drive "
                  f"at |k|/2 {half:.3f}, at |k| {full:.3f}", flush=True)  # fmt: skip
    ax = axes[0]
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"Drive Frequency, $\Omega/|\mathbf{k}|$")
    ax.set_ylabel("Share of the Response in the Driven Wave")
    title = (f"(a) {labels[0]}: periodic field (thick: "
             r"$\gamma=0.05|\mathbf{k}|$; thin: $0.5|\mathbf{k}|$)")  # fmt: skip
    ax.set_title(title, loc="left", fontsize=13)
    ax.legend([Line2D([], [], color=c, lw=2.0, ls=ls) for c, ls in
               (("#1f4e9c", "-"), ("#c0392b", "--"))],
              [c for _, c, _, _ in TAGS], frameon=False, loc="lower left")  # fmt: skip
    ax = axes[1]
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"Driven Wave, $|\mathbf{k}|/K$")
    ax.set_ylabel("Share in the Driven Wave")
    ax.set_title(r"(b) $\Omega = |\mathbf{k}|/2$ ($\nabla$), $\Omega = |\mathbf{k}|$ "
                 r"($\Delta$), $\gamma = 0.05|\mathbf{k}|$", loc="left",
                 fontsize=13)  # fmt: skip
    ax = axes[2]
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"Heated Wave, $|\mathbf{k}|/K$")
    ax.set_ylabel("Share of the Excess Fluctuation in It")
    ax.set_title("(c) heat bath acting on one wave only", loc="left", fontsize=13)
    handles = [Line2D([], [], color=COLORS[i], lw=2.0) for i in range(len(labels))]
    handles += [Line2D([], [], color="0.3", lw=2.0, ls=ls) for _, _, ls, _ in TAGS]
    fig.legend(handles, labels + [c for _, c, _, _ in TAGS], loc="upper center",
               ncol=7, bbox_to_anchor=(0.5, 0.0), frameon=False)  # fmt: skip
    finish(fig, out / "exclusion_drive", dpi)


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
    data = [{tag: load(r / f"exclusion_chi{tag}.npz") for tag, *_ in TAGS}
            for r in args.roots]  # fmt: skip
    figure_spectra(data, args.labels, args.out, args.dpi)
    figure_drive(data, args.labels, args.out, args.dpi)
    print("Done.", flush=True)
