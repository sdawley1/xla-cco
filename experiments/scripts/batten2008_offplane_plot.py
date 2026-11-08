#!/usr/bin/env python3

"""
Figures of the two-hump off-plane shape, per particle, per flow and in 3D.

Reads offplane_chi<tag>.npz (batten2008_offplane.py); s is the fraction of
path length, |Delta r| = |r(inf) - r(0)| the chord. Writes to <out>:

- <prefix>_N<N>_offplane_particles: (a, e) flow 0: 25 random particles'
  off-plane amplitudes |W_j| sqrt(N) / |Delta r|, their root-mean-square
  (which is the off-plane curve) and their mean; (b, f) every particle of
  every flow, each scaled to its own maximum: density per s, with the mean
  and median scaled profile; (c, g) every flow's off-plane curve and their
  mean; (d, h) the plane-free residual (sin(pi s) removed from every
  coordinate, no pinned zeros) per flow, its mean and the mean scaled
  per-particle profile.
- <prefix>_N<N>_offplane_modes: (a, e) the signed coordinate along e3, the
  principal off-plane direction: a positive and a negative lobe, whose
  magnitude is the two-hump curve; (b, f) flow 0, every particle's signed
  share, sorted by amplitude; (c, g) the first three Karhunen-Loeve
  eigenfunctions against sin(k pi s); (d, h) their variance fractions with
  the 1/k^2 (random-walk bridge) and 1/k^4 laws.
- <prefix>_N<N>_offplane_3d: (a, e) every flow in the (e1, e2, e3) frame,
  the start-end plane drawn, flow 0 coloured by s with drops to the plane;
  (b, f) the same paths on their first three KL directions; (c, g) flow 0,
  scaled per-particle profiles as a surface over s and the particles' rank
  by first-hump weight; (d, h)
  every flow's off-plane curve as a waterfall, the mean in front.

Needs matplotlib (plot extra, or the plottime environment).

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
from typing import Dict

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
from matplotlib.lines import Line2D  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

BLUE, RED, GUIDE = "#1f4e9c", "#c0392b", "0.6"
THIN = dict(color="0.55", lw=0.6, alpha=0.6)
N_SHOW = 25  # random particles drawn in panel (a)
S_CMAP = plt.get_cmap("viridis")
OFF_LABEL = r"Off-Plane Distance, $|W|/|\Delta r|$"

# ============================================================
# HELPERS
# ============================================================


def mark_star(ax: plt.Axes, z: Dict[str, np.ndarray], flow0: bool = False) -> None:
    """Dotted line at s* (where the plane is pinned): flow 0's, or the median."""
    star = float(z["s_star"][0] if flow0 else np.median(z["s_star"]))
    ax.axvline(star, color=GUIDE, lw=0.8, ls=":", zorder=0)


def colored_path(ax: plt.Axes, xyz: np.ndarray, lw: float) -> None:
    """3D line coloured by its parameter s in [0, 1]."""
    seg = np.stack([xyz[:-1], xyz[1:]], axis=1)
    lc = Line3DCollection(seg, cmap=S_CMAP, lw=lw)
    lc.set_array(np.linspace(0.0, 1.0, seg.shape[0]))
    ax.add_collection(lc)


def label_3d(ax: plt.Axes, x: str, y: str, z: str) -> None:
    """Axis labels and a lighter pane for a 3D panel."""
    ax.set_xlabel(x, fontsize=11, labelpad=4)
    ax.set_ylabel(y, fontsize=11, labelpad=4)
    ax.set_zlabel(z, fontsize=11, labelpad=8)
    ax.tick_params(labelsize=9, pad=1)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.0)


def legend_below(fig: plt.Figure, handles: list, labels: list, ncol: int) -> None:
    """One frameless legend beneath the figure."""
    fig.legend(handles, labels, loc="upper center", ncol=ncol,
               bbox_to_anchor=(0.5, 0.0), frameon=False)  # fmt: skip


# ============================================================
# FIGURES
# ============================================================


def figure_particles(root: Path, n: int, out: Path, dpi: int) -> None:
    """Per-particle amplitudes, their mean shape, flows, plane-free residual."""
    fig, axes = plt.subplots(2, 4, figsize=(21.0, 8.6), constrained_layout=True,
                             sharex=True)  # fmt: skip
    rng = np.random.default_rng(0)
    for row, (tag, label) in enumerate(CHI_TAGS):
        z = load(root / f"offplane_chi{tag}.npz")
        s, off = z["s"], z["off"]
        let = "abcdefgh"[4 * row : 4 * row + 4]
        # (a) individual particles of flow 0, their RMS and mean
        ax = axes[row, 0]
        amp = z["flow0_amp"]  # (g, N)
        for j in rng.choice(amp.shape[1], N_SHOW, replace=False):
            ax.plot(s, amp[:, j], **THIN)
        rms = np.sqrt(np.mean(amp**2, axis=1))
        assert np.allclose(rms, off[0]), "RMS over particles must be the curve"
        ax.plot(s, rms, color="black", lw=2.0)
        ax.plot(s, amp.mean(axis=1), color=BLUE, lw=2.0, ls="--")
        ax.set_ylim(0.0, 3.0 * rms.max())
        mark_star(ax, z, flow0=True)
        ax.set_title(f"({let[0]}) {label}: particles of flow 0", loc="left",
                     fontsize=13)  # fmt: skip
        ax.set_ylabel(r"Particle Share, $|W_j|\sqrt{N}/|\Delta r|$")
        # (b) every particle of every flow, scaled to its own maximum
        ax = axes[row, 1]
        hist = z["pp_hist"].sum(axis=0).astype(float)  # (g, bins)
        hist /= hist.sum(axis=1, keepdims=True)
        centers = (np.arange(hist.shape[1]) + 0.5) / hist.shape[1]
        ax.pcolormesh(s, centers, hist.T, cmap="Greys", shading="nearest",
                      vmax=np.percentile(hist, 99.0))  # fmt: skip
        ax.plot(s, z["pp_mean"].mean(axis=0), color=RED, lw=2.0)
        ax.plot(s, np.median(z["pp_q"][:, 1], axis=0), color=RED, lw=1.4, ls="--")
        ax.plot(s, off.mean(axis=0) / off.mean(axis=0).max(), color=BLUE, lw=2.0,
                ls=":")  # fmt: skip
        mark_star(ax, z)
        ax.set_title(f"({let[1]}) {label}: all {amp.shape[1]} particles "
                     f"$\\times$ {off.shape[0]} flows", loc="left",
                     fontsize=13)  # fmt: skip
        ax.set_ylabel(r"Scaled Share, $|W_j(s)|/\max_s|W_j|$")
        # (c) every flow and the mean over flows
        ax = axes[row, 2]
        for b in range(off.shape[0]):
            ax.plot(s, off[b], **THIN)
        ax.plot(s, off.mean(axis=0), color="black", lw=2.0)
        mark_star(ax, z)
        ax.set_title(f"({let[2]}) {label}: every flow", loc="left", fontsize=13)
        ax.set_ylabel(OFF_LABEL)
        # (d) plane-free residual: sin(pi s) removed, nothing pinned
        ax = axes[row, 3]
        bf = z["bowfree"]
        for b in range(bf.shape[0]):
            ax.plot(s, bf[b], **THIN)
        mean_bf = bf.mean(axis=0)
        ax.plot(s, mean_bf, color="black", lw=2.0)
        shape = z["pp_bowfree_mean"].mean(axis=0)
        ax.plot(s, shape / shape.max() * mean_bf.max(), color=RED, lw=2.0, ls="--")
        ax.plot(s, np.abs(np.sin(2.0 * np.pi * s)) * mean_bf.max(), color=GUIDE,
                lw=1.2, ls="-.")  # fmt: skip
        ax.set_title(f"({let[3]}) {label}: bow removed, no plane", loc="left",
                     fontsize=13)  # fmt: skip
        ax.set_ylabel(r"Residual, $|P - \mathrm{bow}|/|\Delta r|$")
    for ax in axes[1]:
        ax.set_xlabel(r"Fraction of Path, $s$")
    handles = [Line2D([], [], **THIN), Line2D([], [], color="black", lw=2.0),
               Line2D([], [], color=BLUE, lw=2.0, ls="--"),
               Line2D([], [], color=RED, lw=2.0),
               Line2D([], [], color=RED, lw=1.4, ls="--"),
               Line2D([], [], color=BLUE, lw=2.0, ls=":"),
               Line2D([], [], color=GUIDE, lw=1.2, ls="-."),
               Line2D([], [], color=GUIDE, lw=0.8, ls=":")]  # fmt: skip
    labels = ["one particle (a, e) / one flow (c, d, g, h)",
              "(a, e) RMS over particles = off-plane curve; (c, d) mean over flows",
              "(a, e) mean over particles",
              "(b, f) mean scaled profile; (d, h) scaled to the black curve",
              "(b, f) median scaled profile",
              "(b, f) off-plane curve, scaled",
              r"$|\sin 2\pi s|$", r"$s^*$ (plane pinned)"]  # fmt: skip
    legend_below(fig, handles, labels, 3)
    finish(fig, out / f"{NAMES['prefix']}_N{n}_offplane_particles", dpi)


def figure_modes(root: Path, n: int, out: Path, dpi: int) -> None:
    """Signed lobes, per-particle signs, KL eigenfunctions and spectrum."""
    fig, axes = plt.subplots(2, 4, figsize=(21.0, 8.6), constrained_layout=True)
    for row, (tag, label) in enumerate(CHI_TAGS):
        z = load(root / f"offplane_chi{tag}.npz")
        s = z["s"]
        let = "abcdefgh"[4 * row : 4 * row + 4]
        # (a) signed coordinate along e3, and what e3 misses
        ax = axes[row, 0]
        c3 = z["coords"][:, :, 2]
        for b in range(c3.shape[0]):
            ax.plot(s, c3[b], **THIN)
        ax.plot(s, c3.mean(axis=0), color="black", lw=2.0)
        ax.plot(s, z["off"].mean(axis=0), color=BLUE, lw=1.6, ls="--")
        ax.plot(s, z["rest"].mean(axis=0), color=RED, lw=1.6, ls=":")
        ax.axhline(0.0, color=GUIDE, lw=0.8, zorder=0)
        mark_star(ax, z)
        ax.set_title(f"({let[0]}) {label}: along $e_3$, signed", loc="left",
                     fontsize=13)  # fmt: skip
        ax.set_ylabel(r"Coordinate, $W\cdot e_3/|\Delta r|$")
        ax.set_xlabel(r"Fraction of Path, $s$")
        # (b) every particle's signed share in flow 0
        ax = axes[row, 1]
        sig = z["flow0_signed"]  # (g, N)
        order = np.argsort(-np.abs(sig).max(axis=0))
        vmax = np.percentile(np.abs(sig), 99.0)
        im = ax.imshow(sig[:, order].T, aspect="auto", cmap="RdBu_r", vmin=-vmax,
                       vmax=vmax, origin="lower",
                       extent=(0.0, 1.0, 0.0, sig.shape[1]))  # fmt: skip
        fig.colorbar(im, ax=ax, shrink=0.85,
                     label=r"$W_j\cdot\hat e_{3,j}\sqrt{N}/|\Delta r|$")  # fmt: skip
        ax.set_title(f"({let[1]}) {label}: particles of flow 0", loc="left",
                     fontsize=13)  # fmt: skip
        ax.set_ylabel(r"Particle Rank (by amplitude), $j$")
        ax.set_xlabel(r"Fraction of Path, $s$")
        # (c) KL eigenfunctions against sin(k pi s)
        ax = axes[row, 2]
        g = s.shape[0]
        for k, color in zip(range(3), ("black", BLUE, RED), strict=True):
            phi = z["kl_phi"][:, :, k] * np.sqrt(g / 2.0)  # ~ sin(k pi s)
            ax.fill_between(s, *np.percentile(phi, [10.0, 90.0], axis=0),
                            color=color, alpha=0.15, lw=0)  # fmt: skip
            ax.plot(s, np.median(phi, axis=0), color=color, lw=2.0,
                    label=rf"$k = {k + 1}$")  # fmt: skip
            ax.plot(s, np.sin((k + 1) * np.pi * s), color=color, lw=1.0, ls="--")
        ax.axhline(0.0, color=GUIDE, lw=0.8, zorder=0)
        ax.legend(frameon=False, fontsize=10, loc="lower left")
        ov = np.median(z["kl_overlap"], axis=0)[:3]
        ax.set_title(f"({let[2]}) {label}: KL modes (overlap "
                     + ", ".join(f"{v:.2f}" for v in ov) + ")", loc="left",
                     fontsize=13)  # fmt: skip
        ax.set_ylabel(r"Eigenfunction, $\phi_k(s)$")
        ax.set_xlabel(r"Fraction of Path, $s$")
        # (d) KL variance fractions
        ax = axes[row, 3]
        frac = z["kl_frac"]
        k = np.arange(1, frac.shape[1] + 1)
        ax.fill_between(k, *np.percentile(frac, [10.0, 90.0], axis=0),
                        color="black", alpha=0.15, lw=0)  # fmt: skip
        ax.plot(k, np.median(frac, axis=0), color="black", lw=2.0, marker="o")
        ax.plot(k, 6.0 / (np.pi**2 * k**2), color=BLUE, lw=1.4, ls="--")
        ax.plot(k, 90.0 / (np.pi**4 * k**4), color=RED, lw=1.4, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"({let[3]}) {label}: KL spectrum", loc="left", fontsize=13)
        ax.set_xlabel(r"Mode, $k$")
        ax.set_ylabel(r"Variance Fraction, $\lambda_k/\sum\lambda$")
    handles = [Line2D([], [], **THIN), Line2D([], [], color="black", lw=2.0),
               Line2D([], [], color=BLUE, lw=1.6, ls="--"),
               Line2D([], [], color=RED, lw=1.6, ls=":"),
               Line2D([], [], color="0.3", lw=1.0, ls="--"),
               Line2D([], [], color=BLUE, lw=1.4, ls="--"),
               Line2D([], [], color=RED, lw=1.4, ls=":")]  # fmt: skip
    labels = ["(a, e) one flow", "(a, e) mean; (d, h) median",
              "(a, e) off-plane curve (mean)", r"(a, e) part off $e_3$ (mean)",
              r"(c, g) $\sin k\pi s$", r"(d, h) $1/k^2$ (random-walk bridge)",
              r"(d, h) $1/k^4$"]  # fmt: skip
    legend_below(fig, handles, labels, 4)
    finish(fig, out / f"{NAMES['prefix']}_N{n}_offplane_modes", dpi)


def panel_plane_frame(ax: plt.Axes, z: Dict[str, np.ndarray]) -> None:
    """Every flow in (e1, e2, e3); the plane; flow 0 coloured, dropped to it."""
    c = z["coords"]  # (flows, g, 3)
    for b in range(1, c.shape[0]):
        ax.plot(c[b, :, 0], c[b, :, 1], c[b, :, 2], **THIN)
    lo, hi = c[:, :, 1].min(), c[:, :, 1].max()
    xx, yy = np.meshgrid([0.0, 1.0], [lo, hi])
    ax.plot_surface(xx, yy, np.zeros_like(xx), color=BLUE, alpha=0.12, lw=0)
    colored_path(ax, c[0], 2.4)
    for i in range(0, c.shape[1], 16):
        p = c[0, i]
        ax.plot([p[0], p[0]], [p[1], p[1]], [0.0, p[2]], color=RED, lw=0.8)
    star = int(round(float(z["s_star"][0]) * (c.shape[1] - 1)))
    for i, mk in ((0, "o"), (star, "D"), (c.shape[1] - 1, "*")):
        ax.plot(*c[0, i], marker=mk, color="black", ms=6, ls="")


def panel_waterfall(ax: plt.Axes, z: Dict[str, np.ndarray]) -> None:
    """Every flow's off-plane curve at depth b; the mean in front at -2."""
    s, off = z["s"], z["off"]
    for b in range(off.shape[0]):
        col = plt.get_cmap("Greys")(0.35 + 0.5 * b / off.shape[0])
        ax.plot(s, np.full_like(s, b), off[b], color=col, lw=1.0)
    ax.plot(s, np.full_like(s, -2.0), off.mean(axis=0), color=RED, lw=2.4)


def panel_surface(ax: plt.Axes, z: Dict[str, np.ndarray]) -> None:
    """Flow 0: scaled per-particle profiles, ordered by their first-hump weight.

    Particles are sorted by the share of their profile's area before s*, and
    120 evenly spaced ranks are drawn.
    """
    amp = z["flow0_amp"]  # (g, N)
    star = int(round(float(z["s_star"][0]) * (amp.shape[0] - 1)))
    first = amp[:star].sum(axis=0) / np.maximum(amp.sum(axis=0), 1e-300)
    order = np.argsort(first)
    pick = order[np.linspace(0, order.shape[0] - 1, 120).astype(int)]
    scaled = amp[:, pick] / amp[:, pick].max(axis=0, keepdims=True)
    s = z["s"][::4]
    ss, rr = np.meshgrid(s, np.arange(pick.shape[0]) / pick.shape[0], indexing="ij")
    ax.plot_surface(ss, rr, scaled[::4], cmap="Blues", lw=0, antialiased=False,
                    rcount=s.shape[0], ccount=pick.shape[0], vmin=-0.2,
                    vmax=1.0)  # fmt: skip


def figure_3d(root: Path, n: int, out: Path, dpi: int) -> None:
    """Paths in the plane frame and KL frame, particle surface, waterfall."""
    fig = plt.figure(figsize=(22.0, 10.0))
    fig.subplots_adjust(left=0.0, right=0.9, bottom=0.06, top=0.95, wspace=0.08,
                        hspace=0.12)  # fmt: skip
    for row, (tag, label) in enumerate(CHI_TAGS):
        z = load(root / f"offplane_chi{tag}.npz")
        let = "abcdefgh"[4 * row : 4 * row + 4]
        ax = fig.add_subplot(2, 4, 4 * row + 1, projection="3d")
        panel_plane_frame(ax, z)
        label_3d(ax, r"Chord, $e_1$", r"In Plane, $e_2$", r"Off Plane, $e_3$")
        ax.view_init(elev=22, azim=-60)
        ax.set_title(f"({let[0]}) {label}: start-end frame", loc="left", fontsize=13)
        ax = fig.add_subplot(2, 4, 4 * row + 2, projection="3d")
        kc = z["kl_coord"]
        for b in range(1, kc.shape[0]):
            ax.plot(kc[b, :, 0], kc[b, :, 1], kc[b, :, 2], **THIN)
        colored_path(ax, kc[0], 2.4)
        label_3d(ax, r"KL 1, $u_1$", r"KL 2, $u_2$", r"KL 3, $u_3$")
        ax.view_init(elev=24, azim=-50)
        ax.set_title(f"({let[1]}) {label}: off the chord, KL frame", loc="left",
                     fontsize=13)  # fmt: skip
        ax = fig.add_subplot(2, 4, 4 * row + 3, projection="3d")
        panel_surface(ax, z)
        label_3d(ax, r"Fraction of Path, $s$", r"First-Hump Rank, $j/N$",
                 r"$|W_j|/\max_s|W_j|$")  # fmt: skip
        ax.view_init(elev=32, azim=-55)
        ax.set_title(f"({let[2]}) {label}: particles of flow 0", loc="left",
                     fontsize=13)  # fmt: skip
        ax = fig.add_subplot(2, 4, 4 * row + 4, projection="3d")
        panel_waterfall(ax, z)
        label_3d(ax, r"Fraction of Path, $s$", r"Flow, $b$", r"$|W|/|\Delta r|$")
        ax.view_init(elev=28, azim=-70)
        ax.set_title(f"({let[3]}) {label}: every flow", loc="left", fontsize=13)
    sm = plt.cm.ScalarMappable(cmap=S_CMAP, norm=plt.Normalize(0.0, 1.0))
    cax = fig.add_axes((0.92, 0.3, 0.012, 0.45))
    fig.colorbar(sm, cax=cax, label=r"Flow 0 (a, b, e, f): Fraction of Path, $s$")
    handles = [Line2D([], [], **THIN),
               Line2D([], [], color=BLUE, lw=6, alpha=0.25),
               Line2D([], [], color=RED, lw=0.8),
               Line2D([], [], marker="o", color="black", ls=""),
               Line2D([], [], marker="D", color="black", ls=""),
               Line2D([], [], marker="*", color="black", ls="", ms=8),
               Line2D([], [], color=RED, lw=2.4)]  # fmt: skip
    labels = ["other flows", "start-end plane", "flow 0 to the plane", r"$r(0)$",
              r"$s^*$", r"$r(\infty)$", "(d, h) mean over flows"]  # fmt: skip
    legend_below(fig, handles, labels, 7)
    finish(fig, out / f"{NAMES['prefix']}_N{n}_offplane_3d", dpi)


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
    add_common_arguments(parser)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configure(args.chi, args.prefix, args.dim)
    root = args.root / f"N{args.n}"
    figure_particles(root, args.n, args.out, args.dpi)
    figure_modes(root, args.n, args.out, args.dpi)
    figure_3d(root, args.n, args.out, args.dpi)
    print("Done.", flush=True)
