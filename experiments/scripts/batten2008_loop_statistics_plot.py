#!/usr/bin/env python3

"""
Two-point statistics agree; loop sums do not: ground states, surrogates, SCBA.

For 2D Poisson ground states (--root, every chi tag), the three models of the
vibrational spectrum (batten2008_loop_moments.py) share their two-point
statistics by construction: the random-phase surrogates keep every |rho(k)|,
and the SCBA (S(k)-only theory) is built from the ground-state |rho(k)|^2
averaged over states. Beyond two points:

- amplitudes: the SCBA is the pairing theory of a Gaussian field, whose
  intensities I = |rho(k)|^2 / <|rho|^2> are exponential (<I^2> = 2); the
  ground states and surrogates share the measured |rho(k)|;
- loop phases: a closed walk of the coupling W_ab = (Q_a . Q_b) rho(Q_a -
  Q_b) / V contributes prod (Q_i . Q_(i+1)) rho(k_i), k_i = Q_i - Q_(i+1), to
  tr W^n. Surrogates (and the SCBA) average every term whose transfers do not
  cancel in pairs to zero; the ground states need not. Each loop's phase is
  nearly uniform (the coherence per loop is of order 1/2M), so the difference
  is measured exactly, summed: the irreducible share of each wave's return
  amplitude, I_4(q) / P_4(q) = [(W^4)_qq,GS - <(W^4)_qq>_sur] / <(W^4)_qq>_sur
  (triads: I_3(q) / P_4(q)^(3/4)), and of tr W_B^4 for loops built only from
  transfers |k| in a band B.

Cached in <root>/loopstats_chi<tag>.npz. Writes <out>/loop_statistics.png:

- (a) S(k) to |k| = 2K (the range of the transfers): ground states (mean and
  range over states) and the SCBA input; the surrogates equal the ground
  states wave vector by wave vector;
- (b) |S_model - S_GS| (absolute): surrogates (round-off) and the SCBA input
  against single ground states (state-to-state scatter);
- (c) intensity distribution for K < |k| < 2K: ground states (= surrogates)
  against the exponential of a Gaussian field (SCBA);
- (d) the irreducible share I_4(q)/P_4(q) of each wave's fourth-order return
  amplitude against |q|/K, every chi (0 for surrogates and the SCBA);
- (e) the irreducible share of tr W_B^4 for loops of transfers in |k|/K bands,
  every chi;
- (f) the scorecard against chi: S(k) difference, intensity non-Gaussianity
  <I^2> - 2, triad share and quadrilateral share.

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
from batten2008_loop_moments import full_waves, h_matrix  # noqa: E402
from batten2008_mp_plot import CHI_COLOR  # noqa: E402
from batten2008_mp_shoulder_plot import (  # noqa: E402
    lattice_kmag,
    lattice_rho,
    phase_randomized,
)
from batten2008_plot import finish, plt  # noqa: E402  (house rcParams)
from matplotlib.lines import Line2D  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

TAGS = ("1-10", "1-4", "7-20", "2-5", "9-20", "49-100")
K_BINS = np.arange(0.0, 2.0 + 1e-9, 0.02)  # |k|/K bins of S(k)
I_BINS = np.linspace(0.0, 8.0, 81)  # normalized intensity bins
Q_BINS = np.arange(0.0, 1.0 + 1e-9, 0.05)  # |q|/K bins of the per-wave shares
T_BANDS = ((1.0, 1.25), (1.25, 1.5), (1.5, 1.75), (1.75, 2.0))  # transfer |k|/K
N_SUR = 4  # random-phase surrogates per ground state
SEED = 7
GUIDE = "0.6"
MODEL_COLORS = {"gs": "black", "sur": "#c0392b", "scba": "#1f4e9c"}

# ============================================================
# STATISTICS
# ============================================================


def radial(values: np.ndarray, kmag: np.ndarray) -> np.ndarray:
    """Mean of lattice values in |k|/K shells (K_BINS); NaN where empty."""
    idx = np.digitize(kmag.ravel(), K_BINS) - 1
    nb = K_BINS.shape[0] - 1
    ok = (idx >= 0) & (idx < nb)
    s = np.bincount(idx[ok], values.ravel()[ok], minlength=nb)
    c = np.bincount(idx[ok], minlength=nb)
    out = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    assert out.shape == (nb,), "one value per shell"
    return out


def coupling_w(rho: np.ndarray, nfull: np.ndarray, side: float,
               scale: float) -> np.ndarray:  # fmt: skip
    """Off-diagonal coupling W/K^2 in the complex basis."""
    h = h_matrix(rho, nfull, side) / scale
    out: np.ndarray = h - np.diag(np.diag(h))
    assert np.allclose(np.diag(out), 0.0) and scale > 0.0, "off-diagonal"
    return out


def return_amplitudes(w: np.ndarray) -> Dict[str, np.ndarray]:
    """Diagonals of W^3 and W^4 (per-wave return amplitudes) of Hermitian W."""
    w2 = w @ w
    out = {"r3": np.real(np.einsum("ab,ba->a", w2, w)),
           "r4": np.sum(np.abs(w2) ** 2, axis=1)}  # fmt: skip
    assert out["r4"].shape == (w.shape[0],) and np.all(out["r4"] >= 0.0), "diag"
    return out


def band_traces(w: np.ndarray, kt: np.ndarray) -> np.ndarray:
    """tr W_B^4 / 2M for W restricted to transfers |k|/K in each band."""
    out = []
    for lo, hi in T_BANDS:
        wb = np.where((kt >= lo) & (kt < hi), w, 0.0)
        w2 = wb @ wb
        out.append(float(np.sum(np.abs(w2) ** 2)) / w.shape[0])
    res = np.asarray(out)
    assert res.shape == (len(T_BANDS),) and np.all(res >= 0.0), "one per band"
    return res


def scorecard(d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Whole-spectrum irreducible shares (triad, quadrilateral, per band)."""
    p4 = d["r4_sur"].sum(axis=1)
    m2 = d["r4_sur"].shape[1]
    tri = (d["r3_gs"] - d["r3_sur"]).sum(axis=1) / (p4 / m2) ** 0.75 / m2
    out = {"quad": np.array(np.mean((d["r4_gs"].sum(1) - p4) / p4)),
           "tri": np.array(np.mean(tri)),
           "bands": np.mean((d["band_gs"] - d["band_sur"]) / d["band_sur"],
                            axis=0)}  # fmt: skip
    assert out["bands"].shape == (len(T_BANDS),), "one share per band"
    return out


def per_wave(d: Dict[str, np.ndarray], key: str) -> Dict[str, np.ndarray]:
    """Irreducible share of the return amplitude, binned in |q|/K (Q_BINS)."""
    gs, sur = d[f"{key}_gs"], d[f"{key}_sur"]
    norm = d["r4_sur"] ** (0.75 if key == "r3" else 1.0)
    share = (gs - sur) / norm  # (states, 2M)
    idx = np.digitize(d["kq"], Q_BINS) - 1
    nb = Q_BINS.shape[0] - 1
    mean, err = np.full(nb, np.nan), np.full(nb, np.nan)
    for b in range(nb):
        sel = idx == b
        if sel.sum() >= 4:
            per_state = share[:, sel].mean(axis=1)
            mean[b] = per_state.mean()
            err[b] = per_state.std(ddof=1) / np.sqrt(per_state.size)
    out = {"x": 0.5 * (Q_BINS[1:] + Q_BINS[:-1]), "mean": mean, "err": err}
    assert out["mean"].shape == (nb,), "one value per bin"
    return out


def intensities(i_all: List[np.ndarray], mean_lattice: np.ndarray,
                kmag: np.ndarray) -> np.ndarray:  # fmt: skip
    """Normalized intensities |rho|^2 / <|rho|^2>_shell for K < |k| < 2K."""
    shell = radial(mean_lattice, kmag)
    idx = np.clip(np.digitize(kmag, K_BINS) - 1, 0, K_BINS.shape[0] - 2)
    ring = (kmag > 1.0) & (kmag < 2.0)
    out = np.concatenate([(s / shell[idx])[ring] for s in i_all])
    assert out.ndim == 1 and np.all(out >= 0.0), "intensities"
    return out


def state_stats(rho: np.ndarray, n: int, geo: Dict, rng: np.random.Generator) -> Dict:
    """S(k), return amplitudes and band traces of one state and its surrogates."""
    s_gs = np.abs(rho) ** 2 / n
    w = coupling_w(rho, geo["nfull"], geo["side"], geo["scale"])
    ra = return_amplitudes(w)
    sur_r3, sur_r4, sur_band, dev = [], [], [], []
    for _ in range(N_SUR):
        sur = phase_randomized(rho, rng)
        dev.append(radial(np.abs(np.abs(sur) ** 2 / n - s_gs), geo["kmag"]))
        ws = coupling_w(sur, geo["nfull"], geo["side"], geo["scale"])
        rs = return_amplitudes(ws)
        sur_r3.append(rs["r3"])
        sur_r4.append(rs["r4"])
        sur_band.append(band_traces(ws, geo["kt"]))
    out = {"s_lattice": s_gs, "s_gs": radial(s_gs, geo["kmag"]), "r3_gs": ra["r3"],
           "r4_gs": ra["r4"], "band_gs": band_traces(w, geo["kt"]),
           "dev_sur": np.nanmax(dev, axis=0), "r3_sur": np.mean(sur_r3, axis=0),
           "r4_sur": np.mean(sur_r4, axis=0),
           "band_sur": np.mean(sur_band, axis=0)}  # fmt: skip
    assert out["r4_sur"].shape == ra["r4"].shape, "per-wave arrays"
    return out


def compute(root: Path, tag: str, n_states: int) -> Dict[str, np.ndarray]:
    """Two-point, amplitude and loop statistics for one chi tag."""
    meta = json.loads((root / f"flows_chi{tag}.json").read_text())
    side, k_cut, n = float(meta["basis"][0][0]), float(meta["K"]), int(meta["N"])
    nfull = full_waves(side, k_cut)
    nmax = int(np.ceil(k_cut * side / (2.0 * np.pi)))
    unit = 2.0 * np.pi / side / k_cut
    kt = np.linalg.norm(nfull[:, None, :] - nfull[None], axis=2) * unit
    geo = {"nfull": nfull, "side": side, "scale": k_cut**2,
           "kmag": lattice_kmag(nmax, side) / k_cut, "kt": kt}  # fmt: skip
    rng = np.random.default_rng(SEED)
    with np.load(root / f"flows_chi{tag}.npz", mmap_mode="r") as f:
        states = np.asarray(f["s_final"][:n_states]) % 1.0 * side
    rows: Dict[str, List[np.ndarray]] = {}
    for x in states:
        for key, val in state_stats(lattice_rho(x, side, nmax), n, geo, rng).items():
            rows.setdefault(key, []).append(val)
    out = {k: np.asarray(v) for k, v in rows.items() if k != "s_lattice"}
    mean_lattice = np.mean(rows["s_lattice"], axis=0)
    out["s_scba"] = radial(mean_lattice, geo["kmag"])
    out["dev_scba"] = np.abs(out["s_scba"][None, :] - out["s_gs"])
    inten = intensities(rows["s_lattice"], mean_lattice, geo["kmag"])
    out["i_hist"] = np.histogram(inten, bins=I_BINS, density=True)[0]
    out["i_m2"] = np.array(float(np.mean(inten**2)))
    out["kq"] = np.linalg.norm(nfull, axis=1) * unit
    out.update(chi=np.array(float(meta["chi_actual"])), n=np.array(n))
    sc = scorecard(out)
    print(f"{tag}: max |S_sur - S_GS| {np.nanmax(out['dev_sur']):.1e}; <I^2> "
          f"{float(out['i_m2']):.3f}; triad share {float(sc['tri']):+.4f}; quad "
          f"share {float(sc['quad']):+.4f}; band shares "
          f"{np.round(sc['bands'], 3).tolist()}", flush=True)  # fmt: skip
    return out


def load_or_compute(root: Path, tag: str, n_states: int) -> Dict[str, np.ndarray]:
    """Cached in <root>/loopstats_chi<tag>.npz."""
    path = root / f"loopstats_chi{tag}.npz"
    if path.exists():
        with np.load(path) as z:
            cached = {k: np.asarray(z[k]) for k in z.files}
        if cached["s_gs"].shape[0] >= n_states and "r4_gs" in cached:
            return cached
    res = compute(root, tag, n_states)
    np.savez(path, **res)
    assert path.exists(), "cache written"
    return res


# ============================================================
# FIGURE
# ============================================================


def figure(data: Dict[str, Dict], out: Path, dpi: int) -> None:
    """S(k) agreement, amplitude agreement, and where the loop sums differ."""
    fig, axes = plt.subplots(2, 3, figsize=(20.0, 11.5), constrained_layout=True)
    x = 0.5 * (K_BINS[1:] + K_BINS[:-1])
    centres = 0.5 * (I_BINS[1:] + I_BINS[:-1])
    bands = [0.5 * (a + b) for a, b in T_BANDS]
    tags = list(data)
    for tag in tags:
        d, col = data[tag], CHI_COLOR[tag]
        axes[0, 0].fill_between(x, np.nanmin(d["s_gs"], 0), np.nanmax(d["s_gs"], 0),
                                color=col, alpha=0.2, lw=0)  # fmt: skip
        axes[0, 0].plot(x, np.nanmean(d["s_gs"], 0), color=col, lw=1.8)
        axes[0, 0].plot(x[::4], d["s_scba"][::4], ls="", marker="o", ms=4, mfc="none",
                        color=col)  # fmt: skip
        axes[0, 1].plot(x, np.maximum(np.nanmedian(d["dev_scba"], 0), 1e-30),
                        color=col, lw=1.6)  # fmt: skip
        axes[0, 1].plot(x, np.maximum(np.nanmax(d["dev_sur"], 0), 1e-30), color=col,
                        lw=1.0, ls=":")  # fmt: skip
        axes[0, 2].plot(centres, np.maximum(d["i_hist"], 1e-5), color=col, lw=1.6)
        pw = per_wave(d, "r4")
        axes[1, 0].errorbar(pw["x"], pw["mean"], yerr=pw["err"], color=col, lw=1.8,
                            marker="o", ms=4, capsize=2)  # fmt: skip
        axes[1, 1].plot(bands, scorecard(d)["bands"], color=col, lw=1.8, marker="o")
    axes[0, 2].plot(centres, np.exp(-centres), color=MODEL_COLORS["scba"], lw=2.5,
                    ls="--")  # fmt: skip
    chi = np.array([float(data[t]["chi"]) for t in tags])
    cards = [scorecard(data[t]) for t in tags]
    rows = (("$|S_{\\rm sur} - S_{\\rm GS}|$ (max)", "#c0392b", "s",
             [float(np.nanmax(data[t]["dev_sur"])) for t in tags]),
            ("$\\langle I^2\\rangle - 2$", "#2e7d32", "D",
             [float(data[t]["i_m2"]) - 2.0 for t in tags]),
            ("triad share $I_3$", "#b9770e", "^", [float(c["tri"]) for c in cards]),
            ("quadrilateral share $I_4/P_4$", "black", "o",
             [float(c["quad"]) for c in cards]))  # fmt: skip
    for name, col, mk, vals in rows:
        axes[1, 2].plot(chi, vals, color=col, marker=mk, ms=7, lw=2.0, label=name)
    style(axes)
    legend(fig, data, tags)
    finish(fig, out / "loop_statistics", dpi)


def style(axes: np.ndarray) -> None:
    """Labels and titles."""
    ax = axes[0, 0]
    ax.axvline(1.0, color=GUIDE, lw=0.8, zorder=0)
    ax.set_xlim(0.02, 2.0)
    ax.set_ylim(-0.05, 3.0)
    ax.set_xlabel(r"Wavenumber, $|\mathbf{k}|/K$ (forward peak $k = 0$ omitted)")
    ax.set_ylabel(r"Structure Factor, $S(k)$")
    ax.set_title("(a) $S(k)$: ground states = surrogates = SCBA input", loc="left",
                 fontsize=13)  # fmt: skip
    ax = axes[0, 1]
    ax.set_yscale("log")
    ax.set_ylim(1e-20, 10.0)
    ax.axvline(1.0, color=GUIDE, lw=0.8, zorder=0)
    ax.set_xlabel(r"Wavenumber, $|\mathbf{k}|/K$")
    ax.set_ylabel(r"$|S_{\rm model} - S_{\rm GS}|$")
    ax.set_title("(b) solid: SCBA input vs one state; dotted: surrogates",
                 loc="left", fontsize=13)  # fmt: skip
    ax = axes[0, 2]
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 2.0)
    ax.set_xlabel(r"Intensity, $I = |\rho(\mathbf{k})|^2 / \langle|\rho|^2\rangle$, "
                  r"$K < |\mathbf{k}| < 2K$")  # fmt: skip
    ax.set_ylabel("Probability Density")
    ax.set_title("(c) amplitudes: measured (= surrogates) vs Gaussian (SCBA)",
                 loc="left", fontsize=13)  # fmt: skip
    ax = axes[1, 0]
    ax.axhline(0.0, color=GUIDE, lw=0.8, zorder=0)
    ax.set_xlabel(r"Wave, $|\mathbf{q}|/K$")
    ax.set_ylabel(r"Irreducible Share, $I_4(\mathbf{q})/P_4(\mathbf{q})$")
    ax.set_title("(d) four-wave loops, wave by wave (surrogates, SCBA: 0)",
                 loc="left", fontsize=13)  # fmt: skip
    ax = axes[1, 1]
    ax.axhline(0.0, color=GUIDE, lw=0.8, zorder=0)
    ax.set_xlabel(r"Transfer Band, $|\mathbf{k}|/K$ (centre)")
    ax.set_ylabel(r"Irreducible Share of $\mathrm{tr}\,W_B^4$")
    ax.set_title("(e) single-band loops carry little: the loops mix lengths",
                 loc="left", fontsize=13)  # fmt: skip
    ax = axes[1, 2]
    ax.axhline(0.0, color=GUIDE, lw=0.8, zorder=0)
    ax.set_xlabel(r"Constraint Ratio, $\chi$")
    ax.set_ylabel("Difference from the Ground States")
    ax.legend(frameon=False, fontsize=10)
    ax.set_title("(f) scorecard: only the four-wave loops differ", loc="left",
                 fontsize=13)  # fmt: skip


def legend(fig: plt.Figure, data: Dict[str, Dict], tags: List[str]) -> None:
    """Shared legend: models and chi."""
    handles = [Line2D([], [], color="0.3", lw=1.8),
               Line2D([], [], color="0.3", ls="", marker="o", mfc="none"),
               Line2D([], [], color="0.3", lw=1.0, ls=":"),
               Line2D([], [], color=MODEL_COLORS["scba"], lw=2.5, ls="--")]  # fmt: skip
    names = ["ground states (a-e)", "(a) SCBA input $S$", "(b) surrogates",
             "(c) Gaussian field (SCBA)"]  # fmt: skip
    handles += [Line2D([], [], color=CHI_COLOR[t], lw=2.0) for t in tags]
    names += [rf"$\chi = {float(data[t]['chi']):.2f}$" for t in tags]
    fig.legend(handles, names, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.0),
               frameon=False)  # fmt: skip


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    default_root = Path("experiments/trajectories/flows2d/N1024")
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--states", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("experiments/figures"))
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    assert args.states >= 2, "need at least two states for the scatter"
    args.out.mkdir(parents=True, exist_ok=True)
    data = {t: load_or_compute(args.root, t, args.states) for t in TAGS
            if (args.root / f"flows_chi{t}.npz").exists()}  # fmt: skip
    figure(data, args.out, args.dpi)
    print("Done.", flush=True)
