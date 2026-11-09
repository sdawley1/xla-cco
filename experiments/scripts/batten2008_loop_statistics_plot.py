#!/usr/bin/env python3

"""
Two-point statistics agree; loop sums do not: ground states, surrogates, SCBA.

For 2D ground states (--root, every chi tag present; Poisson by default,
--prefix names the other starts), the three models of the
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

Rectangular cells use k = 2 pi (n_x / L_x, n_y / L_y) and V = L_x L_y.
Cached in <root>/loopstats_chi<tag>.npz. Writes
<out>/[<prefix>_]loop_statistics.png:

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
  <I^2> - 2, triad share and quadrilateral share;
- (g) single-wave phases arg rho(k), K < |k| < 2K, largest chi: ground
  states, the random-phase model (|rho| kept) and Gaussian-field (SCBA)
  realizations;
- (h) the |T|-weighted distribution of four-wave loop phases arg T at the
  largest chi (sampled walks with no cancelling pair), the same three models,
  as density / uniform - 1, with the exact first harmonic of the ground
  states, 2 c_1 cos(phi), for scale;
- (i) the exact first harmonic c_1 = sum Re T / sum |T| over irreducible
  loops, times 2M, against chi: triads (c_1 = tr W^3 / tr |W|^3) and
  quadrilaterals ((tr W^4 - P_4) / (tr |W|^4 - P_4), P_4 the pairing sum),
  for all three models. Phase caches: <root>/loopphase_chi<tag>.npz.

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
from batten2008_lattice import (  # noqa: E402
    cell_lengths,
    full_waves,
    h_matrix,
    lattice_kmag,
    lattice_limits,
    lattice_rho,
    phase_randomized,
    scaled,
    surrogate,
)
from batten2008_mp_plot import (  # noqa: E402
    CHI_COLOR,
    default_tags,
    figure_path,
    root_n,
    start_title,
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
PHASE_BINS = np.linspace(-np.pi, np.pi, 37)  # loop and single-wave phase bins
N_WALKS = 500_000  # sampled closed walks per realization (phase histograms)
N_GAUSS_PAIR = 2  # phase randomizations of a Gaussian realization (its pairings)

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


def coupling_w(rho: np.ndarray, nfull: np.ndarray, side: np.ndarray,
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
    side = cell_lengths(meta["basis"])
    k_cut, n = float(meta["K"]), int(meta["N"])
    nfull = full_waves(side, k_cut)
    nmax = lattice_limits(side, k_cut)
    unit = 2.0 * np.pi / side[0] / k_cut  # |k|/K = |scaled(n)| unit
    kt = np.linalg.norm(scaled(nfull[:, None, :] - nfull[None], side), axis=2) * unit
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
    out["kq"] = np.linalg.norm(scaled(nfull, side), axis=1) * unit
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
# PHASES
# ============================================================


def harmonics(w: np.ndarray) -> Dict[str, float]:
    """tr W^3, tr W^4 and tr |W|^3, tr |W|^4 per wave (|W| elementwise)."""
    a = np.abs(w)
    w2, a2 = w @ w, a @ a
    m = w.shape[0]
    out = {"t3": float(np.real(np.einsum("ab,ba->", w2, w))) / m,
           "t4": float(np.sum(np.abs(w2) ** 2)) / m,
           "a3": float(np.einsum("ab,ba->", a2, a)) / m,
           "a4": float(np.sum(a2**2)) / m}  # fmt: skip
    assert out["a4"] >= out["t4"] - 1e-9 and out["a3"] > 0.0, "|T| bounds"
    return out


def first_harmonics(h: Dict[str, float], pairing: float) -> np.ndarray:
    """(triad c_1, quadrilateral c_1) of one realization."""
    out = np.array([h["t3"] / h["a3"], (h["t4"] - pairing) / (h["a4"] - pairing)])
    assert np.all(np.abs(out) <= 1.0 + 1e-9), "a mean cosine"
    return out


def loop_phases(rho: np.ndarray, nfull: np.ndarray, side: np.ndarray,
                rng: np.random.Generator) -> np.ndarray:  # fmt: skip
    """|T|-weighted histogram of four-wave loop phases (no cancelling pair)."""
    off = (np.asarray(rho.shape) - 1) // 2
    a = rng.integers(0, nfull.shape[0], size=(N_WALKS, 4))
    q = nfull[a]
    k = q - np.roll(q, -1, axis=1)
    keep = np.all(np.any(k != 0, axis=2), axis=1)
    for i in range(4):
        for j in range(i + 1, 4):
            keep &= np.any(k[:, i] + k[:, j] != 0, axis=1)
    q, k = q[keep], k[keep]
    qs = scaled(q, side)  # Q_i . Q_j up to the common factor (2 pi / L_x)^2
    dots = np.einsum("wid,wid->wi", qs, np.roll(qs, -1, axis=1))
    tt = np.prod(dots, axis=1) * np.prod(rho[k[..., 0] + off[0], k[..., 1] + off[1]],
                                         axis=1)  # fmt: skip
    h, _ = np.histogram(np.angle(tt), bins=PHASE_BINS, weights=np.abs(tt))
    out: np.ndarray = h / h.sum()
    assert out.shape == (PHASE_BINS.shape[0] - 1,), "one value per bin"
    return out


def single_phases(rho: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Histogram (fractions) of arg rho(k) over the ring K < |k| < 2K."""
    h, _ = np.histogram(np.angle(rho[ring]), bins=PHASE_BINS)
    out: np.ndarray = h / h.sum()
    assert out.shape == (PHASE_BINS.shape[0] - 1,), "one value per bin"
    return out


def model_phase_stats(rhos: Dict[str, List[np.ndarray]], geo: Dict,
                      rng: np.random.Generator) -> Dict[str, np.ndarray]:  # fmt: skip
    """First harmonics and phase histograms of every realization of each model."""
    out: Dict[str, List[np.ndarray]] = {}
    ring = (geo["kmag"] > 1.0) & (geo["kmag"] < 2.0)
    for key, items in rhos.items():
        for rho, pairing in items:
            w = coupling_w(rho, geo["nfull"], geo["side"], geo["scale"])
            out.setdefault(f"c1_{key}", []).append(
                first_harmonics(harmonics(w), pairing)
            )
            out.setdefault(f"loop_{key}", []).append(
                loop_phases(rho, geo["nfull"], geo["side"], rng)
            )
            out.setdefault(f"single_{key}", []).append(single_phases(rho, ring))
    res = {k: np.asarray(v) for k, v in out.items()}
    assert all(v.shape[0] >= 1 for v in res.values()), "realizations"
    return res


def pairing_sum(rho: np.ndarray, geo: Dict, rng: np.random.Generator,
                reps: int) -> float:  # fmt: skip
    """P_4 = <tr W^4 / 2M> over phase randomizations (the pairing sum)."""
    vals = [harmonics(coupling_w(phase_randomized(rho, rng), geo["nfull"],
                                 geo["side"], geo["scale"]))["t4"]
            for _ in range(reps)]  # fmt: skip
    out = float(np.mean(vals))
    assert out > 0.0 and reps >= 1, "a positive sum"
    return out


def phase_compute(root: Path, tag: str, n_states: int) -> Dict[str, np.ndarray]:
    """Phase statistics of ground states, random-phase and Gaussian models."""
    meta = json.loads((root / f"flows_chi{tag}.json").read_text())
    side = cell_lengths(meta["basis"])
    k_cut, n = float(meta["K"]), int(meta["N"])
    nfull = full_waves(side, k_cut)
    nmax = lattice_limits(side, k_cut)
    geo = {"nfull": nfull, "side": side, "scale": k_cut**2,
           "kmag": lattice_kmag(nmax, side) / k_cut}  # fmt: skip
    rng = np.random.default_rng(SEED + 1)
    with np.load(root / f"flows_chi{tag}.npz", mmap_mode="r") as f:
        states = np.asarray(f["s_final"][:n_states]) % 1.0 * side
    gs = [lattice_rho(x, side, nmax) for x in states]
    s_mean = np.mean([np.abs(r) ** 2 / n for r in gs], axis=0)
    rhos: Dict[str, List] = {"gs": [], "sur": [], "scba": []}
    for rho in gs:
        pair = pairing_sum(rho, geo, rng, N_SUR)
        rhos["gs"].append((rho, pair))
        rhos["sur"].append((phase_randomized(rho, rng), pair))
        g = surrogate(s_mean, n, rng)
        rhos["scba"].append((g, pairing_sum(g, geo, rng, N_GAUSS_PAIR)))
    out = model_phase_stats(rhos, geo, rng)
    out.update(chi=np.array(float(meta["chi_actual"])),
               m2=np.array(nfull.shape[0]))  # fmt: skip
    print(f"{tag} phases: c1 x 2M (triad, quad) GS "
          f"{np.round(out['c1_gs'].mean(0) * nfull.shape[0], 3).tolist()}, random "
          f"phase {np.round(out['c1_sur'].mean(0) * nfull.shape[0], 3).tolist()}, "
          f"Gaussian {np.round(out['c1_scba'].mean(0) * nfull.shape[0], 3).tolist()}",
          flush=True)  # fmt: skip
    return out


def load_or_compute_phase(root: Path, tag: str, n_states: int) -> Dict[str, np.ndarray]:
    """Cached in <root>/loopphase_chi<tag>.npz."""
    path = root / f"loopphase_chi{tag}.npz"
    if path.exists():
        with np.load(path) as z:
            cached = {k: np.asarray(z[k]) for k in z.files}
        if cached["c1_gs"].shape[0] >= n_states:
            return cached
    res = phase_compute(root, tag, n_states)
    np.savez(path, **res)
    assert path.exists(), "cache written"
    return res


def phase_row(axes: np.ndarray, pdata: Dict[str, Dict]) -> None:
    """(g) single-wave phases, (h) loop-phase distributions, (i) exact harmonics."""
    tags = list(pdata)
    big = pdata[tags[-1]]
    pc = 0.5 * (PHASE_BINS[1:] + PHASE_BINS[:-1])
    uniform = 1.0 / pc.shape[0]
    for key, ls in (("gs", "-"), ("sur", "-"), ("scba", "--")):
        for ax, name in ((axes[0], "single"), (axes[1], "loop")):
            h = big[f"{name}_{key}"]
            mid = h.mean(0) / uniform - 1.0
            se = h.std(0, ddof=1) / np.sqrt(h.shape[0]) / uniform
            ax.fill_between(pc, mid - 2 * se, mid + 2 * se, color=MODEL_COLORS[key],
                            alpha=0.15, lw=0)  # fmt: skip
            ax.plot(pc, mid, color=MODEL_COLORS[key], ls=ls,
                    lw=2.0 if key == "gs" else 1.4)  # fmt: skip
    c1 = float(big["c1_gs"][:, 1].mean())
    axes[1].plot(pc, 2.0 * c1 * np.cos(pc), color="#2e7d32", lw=2.0, ls=":")
    chi = np.array([float(pdata[t]["chi"]) for t in tags])
    for key in ("gs", "sur", "scba"):
        for col, mk, ls in ((1, "o", "-"), (0, "^", "--")):
            v = [pdata[t][f"c1_{key}"][:, col] * float(pdata[t]["m2"]) for t in tags]
            axes[2].errorbar(chi, [a.mean() for a in v],
                             yerr=[a.std(ddof=1) / np.sqrt(a.size) for a in v],
                             color=MODEL_COLORS[key], marker=mk, ms=7, ls=ls,
                             lw=2.0 if key == "gs" else 1.2, capsize=3,
                             mfc=None if key == "gs" else "none")  # fmt: skip
    for ax in axes[:2]:
        ax.axhline(0.0, color=GUIDE, lw=0.8, zorder=0)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
                      [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])  # fmt: skip
        ax.set_ylabel("Density / Uniform $-$ 1")
    chi_big = float(big["chi"])
    axes[0].set_xlabel(r"Phase, $\arg\rho(\mathbf{k})$, $K < |\mathbf{k}| < 2K$")
    axes[0].set_title(rf"(g) $\chi = {chi_big:.2f}$: single-wave phases are uniform",
                      loc="left", fontsize=13)  # fmt: skip
    axes[1].set_xlabel(r"Four-Wave Loop Phase, $\arg T$ ($|T|$-weighted)")
    axes[1].set_title(rf"(h) $\chi = {chi_big:.2f}$: loop phases, loop by loop",
                      loc="left", fontsize=13)  # fmt: skip
    ax = axes[2]
    ax.axhline(0.0, color=GUIDE, lw=0.8, zorder=0)
    ax.set_xlabel(r"Constraint Ratio, $\chi$")
    ax.set_ylabel(r"First Harmonic $\times\,2M$, $2M\langle\cos\arg T\rangle_{|T|}$")
    ax.set_title("(i) exact loop-phase bias (o: quadrilaterals, ^: triads)",
                 loc="left", fontsize=13)  # fmt: skip


# ============================================================
# FIGURE
# ============================================================


def figure(data: Dict[str, Dict], pdata: Dict[str, Dict], out: Path, dpi: int,
           n: int, prefix: str = "") -> None:  # fmt: skip
    """S(k) and amplitudes agree; loop sums and loop-phase biases differ."""
    fig, axes = plt.subplots(3, 3, figsize=(20.0, 17.0), constrained_layout=True)
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
    phase_row(axes[2], pdata)
    legend(fig, data, tags)
    start_title(fig, prefix, n)
    finish(fig, figure_path(out, prefix, "loop_statistics"), dpi)


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
    ax.set_title("(f) effect on the spectrum: only four-wave loops matter", loc="left",
                 fontsize=13)  # fmt: skip


def legend(fig: plt.Figure, data: Dict[str, Dict], tags: List[str]) -> None:
    """Shared legend: models and chi."""
    handles = [Line2D([], [], color="0.3", lw=1.8),
               Line2D([], [], color="0.3", ls="", marker="o", mfc="none"),
               Line2D([], [], color="0.3", lw=1.0, ls=":"),
               Line2D([], [], color=MODEL_COLORS["scba"], lw=2.5, ls="--")]  # fmt: skip
    names = ["ground states (a-e)", "(a) SCBA input $S$", "(b) surrogates",
             "(c) Gaussian field (SCBA)"]  # fmt: skip
    handles += [Line2D([], [], color=MODEL_COLORS["gs"], lw=2.0),
                Line2D([], [], color=MODEL_COLORS["sur"], lw=1.4),
                Line2D([], [], color=MODEL_COLORS["scba"], lw=1.4, ls="--"),
                Line2D([], [], color="#2e7d32", lw=2.0, ls=":")]  # fmt: skip
    names += ["(g-i) ground states", "(g-i) random-phase model ($|\\rho|$ kept)",
              "(g-i) Gaussian field (SCBA)",
              "(h) exact ground-state harmonic $2c_1\\cos\\phi$"]  # fmt: skip
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
    parser.add_argument("--tags", nargs="+", default=None,
                        help="default: every chi tag present in the root")  # fmt: skip
    parser.add_argument("--states", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("experiments/figures"))
    parser.add_argument("--prefix", default="",
                        help="figure name prefix and start, e.g. hip2d")  # fmt: skip
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    assert args.states >= 2, "need at least two states for the scatter"
    args.out.mkdir(parents=True, exist_ok=True)
    tags = default_tags(args.root, args.tags, TAGS)
    data = {t: load_or_compute(args.root, t, args.states) for t in tags}
    pdata = {t: load_or_compute_phase(args.root, t, args.states) for t in data}
    figure(data, pdata, args.out, args.dpi, root_n(args.root), args.prefix)
    print("Done.", flush=True)
