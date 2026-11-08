#!/usr/bin/env python3

"""
KL eigenfunctions of the 2D flows against the linear model, and why they differ.

Reads offplane_chi<tag>.npz (batten2008_offplane.py) and winding_chi<tag>.npz
(batten2008_winding.py) of each system root (--roots, one --labels entry
each; chi tags 1-10 and 2-5). The linear relaxation model relaxes wave n
independently at rate |k_n|^2 from its start amplitude, along fixed
orthogonal directions, y(t) = sum_n sqrt(2 S0_n)/|k_n| (1 - exp(-|k_n|^2 t)).
Writes to <out>:

- offplane_kl_compare: (a-e, f-j) per start and chi, flow 0's six KL
  eigenfunctions, measured (solid) and model (dashed), stacked against
  sin(k pi s), with the median overlap |<phi_k, phi_k^model>| over flows at
  the right; (k, l) interior nodes against the sine nodes j/k (small: every
  flow; large: mean, filled measured, open model); (m) KL spectra at
  chi = 0.4; (n) measured over model variance fraction, both chi; (o) the
  median overlap against k, both chi.
- offplane_kl_origin: (a) path fraction s against t / tau_1, measured
  (solid) and model (dashed), chi = 0.4, median over flows; (b, c) per flow,
  the overlap deficit 1 - mean_{k=3..6} overlap against the clock lag (time
  to half the path, flow over model) and against the path-length excess
  (flow over model); (d) per system, the deficit against the share of the
  model's displacement carried by short waves (|k| > 0.6 K) times their
  measured slaving 1 - Gamma_k/|k|^2 at Phi = 10, where Gamma_k =
  -d ln|rho_k|/dt is the measured decay rate (|k|^2 if uncoupled).

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
MARKERS = "osD^vP"
TAGS = (("1-10", r"$\chi = 0.1$", "-", "o"), ("2-5", r"$\chi = 0.4$", "--", "s"))
GUIDE = "0.6"
STACK = 2.6  # vertical offset between stacked eigenfunctions
SHORT = 0.6  # short waves: |k| > SHORT K
HIGH_K = slice(2, 6)  # KL modes k = 3..6 for the overlap deficit
SLAVE_LEVEL = 1.0  # log10 Phi at which the short-wave decay is measured

# ============================================================
# HELPERS
# ============================================================


def interior_zeros(s: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Sign changes of f(s) strictly inside (0.02, 0.98), linearly interpolated."""
    inside = (s > 0.02) & (s < 0.98)
    ss, ff = s[inside], f[inside]
    k = np.flatnonzero(np.sign(ff[:-1]) * np.sign(ff[1:]) < 0)
    out: np.ndarray = ss[k] - ff[k] * (ss[k + 1] - ss[k]) / (ff[k + 1] - ff[k])
    return out


def overlaps(o: Dict[str, np.ndarray]) -> np.ndarray:
    """|<phi_k, phi_k^model>| per flow and KL mode (unit vectors on the s grid)."""
    out: np.ndarray = np.abs(np.einsum("bgk,bgk->bk", o["kl_phi"], o["lin_kl_phi"]))
    return out


def model_clock(weight: np.ndarray, rate: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(t, arclength) of the linear relaxation model path, as in batten2008_offplane."""
    assert weight.shape == rate.shape and np.all(rate > 0.0), "one rate per mode"
    lo, hi = np.log10(0.01 / rate.max()), np.log10(40.0 / rate.min())
    t = np.concatenate([[0.0], np.logspace(lo, hi, 6000)])
    y = np.sqrt(weight)[None, :] * -np.expm1(-np.outer(t, rate))
    seg = np.linalg.norm(np.diff(y, axis=0), axis=1)
    return t, np.concatenate([[0.0], np.cumsum(seg)])


def flow_metrics(o: Dict[str, np.ndarray], w: Dict[str, np.ndarray]) -> Dict:
    """Per flow: overlap deficit, clock lag, length excess; s(t) curves on a grid."""
    km, tau1 = o["kmag"], float(w["tau1"])
    grid = np.logspace(-4.5, 1.5, 121)  # t / tau_1
    lag, excess, s_meas, s_model = [], [], [], []
    for b in range(o["chord"].shape[0]):
        t, ell = model_clock(o["lin_power"][b], km**2)
        fl, ft = w["frame_length"][b], w["frame_t"][b]
        ok = np.isfinite(fl) & np.isfinite(ft)
        fl, ft = fl[ok], ft[ok]
        half_model = np.interp(0.5, ell / ell[-1], t)
        lag.append(np.interp(0.5, fl / fl[-1], ft) / half_model)
        excess.append(float(o["length"][b]) / ell[-1])
        s_meas.append(np.interp(grid * tau1, ft, fl / fl[-1]))
        s_model.append(np.interp(grid * tau1, t, ell / ell[-1]))
    return {"deficit": 1.0 - overlaps(o)[:, HIGH_K].mean(axis=1),
            "lag": np.array(lag), "excess": np.array(excess), "grid": grid,
            "s_meas": np.array(s_meas), "s_model": np.array(s_model)}  # fmt: skip


def slaved_share(o: Dict[str, np.ndarray], w: Dict[str, np.ndarray]) -> float:
    """Model displacement share on |k| > SHORT K, times its slaving at Phi = 10."""
    km = o["kmag"]
    short = km > SHORT * km[-1]
    lp = o["lin_power"]
    share = float(np.median(lp[:, short].sum(axis=1) / lp.sum(axis=1)))
    i = list(np.round(w["levels"], 2)).index(SLAVE_LEVEL)
    ok = (w["S"][:, i] > 1e-8) & short[None, :]
    ratio = -w["lnrho_rate"][:, i] / (w["kmag"] ** 2)[None, :]
    return share * (1.0 - float(np.median(ratio[ok])))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (no ties expected)."""
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


# ============================================================
# FIGURES
# ============================================================


def panel_stack(ax: plt.Axes, o: Dict[str, np.ndarray], color: str) -> None:
    """Flow 0's eigenfunctions, measured and model, stacked; overlaps at right."""
    s, g = o["s"], o["s"].shape[0]
    ov = np.median(overlaps(o), axis=0)
    for k in range(o["kl_phi"].shape[2]):
        base = -STACK * k
        ax.plot(s, np.sin((k + 1) * np.pi * s) + base, color=GUIDE, lw=0.8, ls=":")
        ax.axhline(base, color=GUIDE, lw=0.5, zorder=0)
        ax.plot(s, o["kl_phi"][0, :, k] * np.sqrt(g / 2.0) + base, color=color,
                lw=1.8)  # fmt: skip
        ax.plot(s, o["lin_kl_phi"][0, :, k] * np.sqrt(g / 2.0) + base, color=color,
                lw=1.0, ls="--")  # fmt: skip
        ax.text(1.03, base, f"{ov[k]:.2f}", color=color, fontsize=10, va="center",
                transform=ax.get_yaxis_transform())  # fmt: skip
    ax.set_yticks([-STACK * k for k in range(6)], [f"$k={k + 1}$" for k in range(6)])
    ax.set_xlabel(r"Fraction of Path, $s$")


def panel_nodes(ax: plt.Axes, o: Dict[str, np.ndarray], color: str, mk: str) -> None:
    """Interior zeros of each flow's eigenfunctions against j/k; means enlarged."""
    s = o["s"]
    for key, fill in (("kl_phi", color), ("lin_kl_phi", "none")):
        for k in range(1, o[key].shape[2]):
            nodes = np.arange(1, k + 1) / (k + 1)
            rows = [z for z in (interior_zeros(s, f) for f in o[key][:, :, k])
                    if z.shape[0] == k]  # fmt: skip
            if not rows:
                continue
            if key == "kl_phi":
                for z in rows[:30]:
                    ax.plot(nodes, z, ls="", marker=".", ms=2, color=color, alpha=0.25)
            ax.plot(nodes, np.mean(rows, axis=0), ls="", marker=mk, ms=6,
                    color=color, mfc=fill)  # fmt: skip


def figure_compare(data: List, labels: List[str], out: Path, dpi: int) -> None:
    """KL eigenfunctions of every start against the model, with nodes and spectra."""
    fig, axes = plt.subplots(3, 5, figsize=(25.0, 18.0), constrained_layout=True)
    letters = iter("abcdefghijklmno")
    for row, (tag, chi, ls, _) in enumerate(TAGS):
        for i, sys_d in enumerate(data):
            o = sys_d[tag]["o"]
            ax = axes[row, i]
            panel_stack(ax, o, COLORS[i])
            ax.set_title(f"({next(letters)}) {labels[i]}, {chi}", loc="left",
                         fontsize=13)  # fmt: skip
            panel_nodes(axes[2, row], o, COLORS[i], MARKERS[i])
            k = np.arange(1, o["kl_frac"].shape[1] + 1)
            meas = np.median(o["kl_frac"], axis=0)
            model = np.median(o["lin_kl_frac"], axis=0)
            if row == 1:
                axes[2, 2].plot(k, meas, color=COLORS[i], lw=2.0, marker=MARKERS[i],
                                ms=5)  # fmt: skip
                axes[2, 2].plot(k, model, color=COLORS[i], lw=1.0, ls="--")
            axes[2, 3].plot(k, meas / model, color=COLORS[i], lw=2.0, ls=ls,
                            marker=MARKERS[i], ms=4)  # fmt: skip
            ov = np.median(overlaps(o), axis=0)
            axes[2, 4].plot(np.arange(1, ov.shape[0] + 1), ov, color=COLORS[i],
                            lw=2.0, ls=ls, marker=MARKERS[i], ms=5)  # fmt: skip
    for col in range(2):
        ax = axes[2, col]
        ax.plot([0, 1], [0, 1], color=GUIDE, lw=0.8, zorder=0)
        ax.set_title(f"({next(letters)}) nodes of $\\phi_k$, {TAGS[col][1]}",
                     loc="left", fontsize=13)  # fmt: skip
        ax.set_xlabel(r"Sine Node, $j/k$")
        ax.set_ylabel(r"Node of $\phi_k$, $s$")
    ax = axes[2, 2]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"({next(letters)}) KL spectrum, {TAGS[1][1]}", loc="left",
                 fontsize=13)  # fmt: skip
    ax.set_xlabel(r"Mode, $k$")
    ax.set_ylabel(r"Variance Fraction, $\lambda_k/\sum\lambda$")
    ax = axes[2, 3]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(1.0, color=GUIDE, lw=0.8, zorder=0)
    ax.set_title(f"({next(letters)}) measured over model variance", loc="left",
                 fontsize=13)  # fmt: skip
    ax.set_xlabel(r"Mode, $k$")
    ax.set_ylabel(r"$\lambda_k/\lambda_k^{\mathrm{model}}$ (fractions)")
    ax = axes[2, 4]
    ax.set_ylim(0.0, 1.03)
    ax.set_title(f"({next(letters)}) overlap with the model", loc="left", fontsize=13)
    ax.set_xlabel(r"Mode, $k$")
    ax.set_ylabel(r"Median $|\langle\phi_k, \phi_k^{\mathrm{model}}\rangle|$")
    legend(fig, labels, ["measured (flow 0)", "linear relaxation model",
                         r"$\sin k\pi s$"])  # fmt: skip
    finish(fig, out / "offplane_kl_compare", dpi)


def legend(fig: plt.Figure, labels: List[str], extra: List[str]) -> None:
    """Starts, the two chi line styles and the measured/model/sine styles."""
    handles = [Line2D([], [], color=COLORS[i], lw=2.0, marker=MARKERS[i])
               for i in range(len(labels))]  # fmt: skip
    handles += [Line2D([], [], color="0.3", lw=2.0, ls=ls, marker=mk, mfc="none")
                for _, _, ls, mk in TAGS]  # fmt: skip
    styles = (("-", 1.8, "0.3"), ("--", 1.0, "0.3"), (":", 0.8, GUIDE))
    handles += [Line2D([], [], color=c, lw=lw, ls=ls) for ls, lw, c in styles]
    names = labels + [chi for _, chi, _, _ in TAGS] + extra
    fig.legend(handles, names[: len(handles)], loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 0.0), frameon=False)  # fmt: skip


def figure_origin(data: List, labels: List[str], out: Path, dpi: int) -> None:
    """Where the model fails: the clock lags, the path lengthens, short waves slave."""
    fig, axes = plt.subplots(1, 4, figsize=(24.0, 6.2), constrained_layout=True)
    every: Dict[str, List[np.ndarray]] = {"deficit": [], "lag": [], "excess": []}
    sys_x, sys_y = [], []
    for i, sys_d in enumerate(data):
        for tag, _, _ls, mk in TAGS:
            m = sys_d[tag]["m"]
            for key in every:
                every[key].append(m[key])
            dots = dict(ls="", marker=mk, ms=4, color=COLORS[i], mfc="none", alpha=0.6)
            axes[1].plot(m["lag"], m["deficit"], **dots)
            axes[2].plot(m["excess"], m["deficit"], **dots)
            sys_x.append(sys_d[tag]["slaved"])
            sys_y.append(float(np.median(m["deficit"])))
            axes[3].plot(sys_x[-1], sys_y[-1], ls="", marker=mk, ms=10,
                         color=COLORS[i])  # fmt: skip
            if tag == TAGS[1][0]:
                axes[0].plot(m["grid"], np.median(m["s_meas"], axis=0),
                             color=COLORS[i], lw=2.0)  # fmt: skip
                axes[0].plot(m["grid"], np.median(m["s_model"], axis=0),
                             color=COLORS[i], lw=1.2, ls="--")  # fmt: skip
    cat = {k: np.concatenate(v) for k, v in every.items()}
    rho_lag = spearman(cat["lag"], cat["deficit"])
    rho_len = spearman(cat["excess"], cat["deficit"])
    print(f"Spearman deficit-lag {rho_lag:.2f}, deficit-length {rho_len:.2f}",
          flush=True)  # fmt: skip
    ax = axes[0]
    ax.set_xscale("log")
    ax.set_title(r"(a) path covered vs time, $\chi = 0.4$", loc="left", fontsize=13)
    ax.set_xlabel(r"Time, $t/\tau_1$")
    ax.set_ylabel(r"Fraction of Path, $s$")
    texts = (
        (1, "Clock Lag, $t_{1/2}/t_{1/2}^{\\mathrm{model}}$", rho_lag, "b"),
        (2, r"Path Length, $\ell/\ell^{\mathrm{model}}$", rho_len, "c"),
    )
    for j, xlab, rho, letter in texts:
        ax = axes[j]
        ax.set_xlabel(xlab)
        ax.set_ylabel(r"Overlap Deficit, $1 - \langle|\langle\phi_k,"
                      r"\phi_k^{\mathrm{model}}\rangle|\rangle_{k=3..6}$")  # fmt: skip
        ax.set_title(f"({letter}) per flow, Spearman {rho:.2f}", loc="left",
                     fontsize=13)  # fmt: skip
    ax = axes[3]
    rho_sys = spearman(np.array(sys_x), np.array(sys_y))
    ax.set_xlabel(r"Model Share on $|\mathbf{k}| > 0.6K$ $\times$ "
                  r"$(1 - \Gamma_k/|\mathbf{k}|^2)$ at $\Phi = 10$")  # fmt: skip
    ax.set_ylabel("Median Overlap Deficit")
    ax.set_title(f"(d) per system: slaved short waves, Spearman {rho_sys:.2f}",
                 loc="left", fontsize=13)  # fmt: skip
    print(f"Spearman per system {rho_sys:.2f}", flush=True)
    legend(fig, labels, ["measured", "linear relaxation model"])
    finish(fig, out / "offplane_kl_origin", dpi)


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
    data = []
    for root in args.roots:
        sys_d = {}
        for tag, *_ in TAGS:
            o = load(root / f"offplane_chi{tag}.npz")
            w = load(root / f"winding_chi{tag}.npz")
            sys_d[tag] = {"o": o, "m": flow_metrics(o, w), "slaved": slaved_share(o, w)}
            m = sys_d[tag]["m"]
            print(f"{root} chi{tag}: deficit {np.median(m['deficit']):.3f}, lag "
                  f"{np.median(m['lag']):.2f}, length {np.median(m['excess']):.2f}, "
                  f"slaved share {sys_d[tag]['slaved']:.3f}", flush=True)  # fmt: skip
        data.append(sys_d)
    figure_compare(data, args.labels, args.out, args.dpi)
    figure_origin(data, args.labels, args.out, args.dpi)
    print("Done.", flush=True)
