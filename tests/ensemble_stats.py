#!/usr/bin/env python3

"""
Statistics battery for ensemble parity (pure NumPy/SciPy; no JAX, no data files).

Two ensembles of relaxed configurations, each summarized by an ``EnsembleSample``,
are compared with

1. a two-sided two-proportion z-test on the success fraction;
2. two-sample KS tests (``scipy.stats.ks_2samp``) on min NN distance, mean NN
   distance, first g2-peak height and RMS displacement (all configs);
3. a per-shell z-score of the ensemble-mean S(k): z = (m1 - m2) / sqrt(SEM1^2 +
   SEM2^2), where the per-config value of a shell is the mean of S over its
   k-vectors and SEM is taken over configs.

Acceptance follows the Bonferroni rule of T12: every p-test must exceed its
threshold and every |z| must stay below ``zmax``. The -log10(Phi) distribution
among successes is compared for REPORTING only (ruling R22).

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import dataclasses
from typing import Dict, Tuple

# ----- numerics -----
import numpy as np
from scipy import stats

# ============================================================
# CONSTANTS
# ============================================================

KS_NAMES: Tuple[str, ...] = ("nn_min", "nn_mean", "g2_peak", "rms_disp")
SHELL_RTOL: float = 1e-9  # |k|^2 values within this relative gap share a shell
TINY_PHI: float = 1e-300  # floor before log10 so Phi = 0 stays finite

# ============================================================
# DATA
# ============================================================


@dataclasses.dataclass(frozen=True)
class EnsembleSample:
    """Per-config summaries of one ensemble (B configurations).

    ``sk_shell`` has shape (B, S): the mean of S over the k-vectors of each of
    S shells (same shells for both ensembles being compared).
    """

    energy: np.ndarray
    success: np.ndarray
    nn_min: np.ndarray
    nn_mean: np.ndarray
    g2_peak: np.ndarray
    rms_disp: np.ndarray
    sk_shell: np.ndarray

    def __post_init__(self) -> None:
        """Check that all per-config arrays share the leading size B."""
        b = self.energy.shape[0]
        for f in dataclasses.fields(self):
            assert getattr(self, f.name).shape[0] == b, f"{f.name}: wrong length"
        assert self.sk_shell.ndim == 2, "sk_shell must be (B, S)"

    @property
    def size(self) -> int:
        """Return the number of configurations."""
        return int(self.energy.shape[0])


@dataclasses.dataclass(frozen=True)
class BatteryResult:
    """Outcome of one ensemble comparison (see ``run_battery``)."""

    n_a: int
    n_b: int
    frac_a: float
    frac_b: float
    prop_z: float
    prop_p: float
    prop_pass: bool
    ks_stat: Dict[str, float]
    ks_p: Dict[str, float]
    ks_pass: Dict[str, bool]
    sk_z: np.ndarray
    sk_worst: float
    sk_pass: bool
    logphi_ks_p: float  # report only (R22): excluded from ``passed``
    p_threshold_prop: float
    p_threshold_ks: float
    zmax: float

    @property
    def passed(self) -> bool:
        """Return True iff every acceptance criterion holds."""
        return self.prop_pass and all(self.ks_pass.values()) and self.sk_pass

    @property
    def n_p_tests(self) -> int:
        """Return the number of p-value tests in the Bonferroni family."""
        return 1 + len(self.ks_p)


# ============================================================
# TESTS
# ============================================================


def two_proportion_test(k1: int, n1: int, k2: int, n2: int) -> Tuple[float, float]:
    """Return (z, two-sided p) of the pooled two-proportion z-test.

    A degenerate pooled proportion (0 or 1) has zero variance: both samples are
    then identical in outcome, so z = 0 and p = 1.
    """
    assert n1 > 0 and n2 > 0, "need non-empty samples"
    assert 0 <= k1 <= n1 and 0 <= k2 <= n2, "counts out of range"
    pooled = (k1 + k2) / (n1 + n2)
    var = pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2)
    if var <= 0.0:
        return 0.0, 1.0
    z = (k1 / n1 - k2 / n2) / np.sqrt(var)
    return float(z), float(2.0 * stats.norm.sf(abs(z)))


def shell_groups(k_sq: np.ndarray, rtol: float = SHELL_RTOL) -> np.ndarray:
    """Return the shell index of every k-vector (shells sorted by |k|^2).

    Vectors whose |k|^2 differ by less than ``rtol`` relative share a shell.
    """
    k_sq = np.asarray(k_sq, dtype=np.float64)
    assert k_sq.ndim == 1 and k_sq.size > 0, "k_sq must be a non-empty vector"
    order = np.argsort(k_sq, kind="stable")
    srt = k_sq[order]
    new = np.concatenate([[True], np.diff(srt) > rtol * np.maximum(srt[1:], 1e-300)])
    idx_sorted = np.cumsum(new) - 1
    out = np.empty(k_sq.size, dtype=np.int64)
    out[order] = idx_sorted
    return out


def shell_means(s_k: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Return the per-config mean of ``s_k`` (B, M) over each shell: (B, S)."""
    assert s_k.ndim == 2 and groups.shape == (s_k.shape[1],), "shape mismatch"
    n_shells = int(groups.max()) + 1
    onehot = np.zeros((groups.size, n_shells))
    onehot[np.arange(groups.size), groups] = 1.0
    return np.asarray((s_k @ onehot) / onehot.sum(axis=0))


def sk_z_scores(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the per-shell z = (mean_a - mean_b) / sqrt(SEM_a^2 + SEM_b^2).

    ``a`` and ``b`` are (B, S) per-config shell values; SEM uses ddof = 1.
    Shells with zero combined variance get z = 0 when the means agree and
    +-inf otherwise.
    """
    assert a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[1], "bad shells"
    assert min(a.shape[0], b.shape[0]) >= 2, "need at least 2 configs for a SEM"
    ma, mb = a.mean(axis=0), b.mean(axis=0)
    va = a.var(axis=0, ddof=1) / a.shape[0]
    vb = b.var(axis=0, ddof=1) / b.shape[0]
    denom = np.sqrt(va + vb)
    diff = ma - mb
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(denom > 0.0, diff / denom, np.where(diff == 0.0, 0.0, np.inf))
    return np.asarray(z)


def neg_log10_phi(energy: np.ndarray) -> np.ndarray:
    """Return -log10(Phi), with Phi floored at ``TINY_PHI``."""
    return -np.log10(np.maximum(np.asarray(energy, dtype=np.float64), TINY_PHI))


# ============================================================
# BATTERY
# ============================================================


def bonferroni_thresholds(
    alpha: float, n_tests: int, ks_pmin: float
) -> Tuple[float, float]:
    """Return (proportion threshold, KS threshold) after Bonferroni correction.

    The proportion test needs p > alpha / n_tests. The KS tests need
    p > min(ks_pmin, alpha / n_tests): ``ks_pmin`` is the per-test floor, and
    the Bonferroni level replaces it whenever that is stricter.
    """
    assert 0.0 < alpha < 1.0 and n_tests >= 1 and ks_pmin > 0.0, "bad arguments"
    corrected = alpha / n_tests
    return corrected, min(ks_pmin, corrected)


def run_battery(
    a: EnsembleSample,
    b: EnsembleSample,
    p_prop: float,
    p_ks: float,
    zmax: float,
) -> BatteryResult:
    """Compare ensembles ``a`` and ``b`` with the full battery."""
    assert a.sk_shell.shape[1] == b.sk_shell.shape[1], "shell sets differ"
    ka, kb = int(a.success.sum()), int(b.success.sum())
    z, p = two_proportion_test(ka, a.size, kb, b.size)
    ks_stat: Dict[str, float] = {}
    ks_p: Dict[str, float] = {}
    for name in KS_NAMES:
        res = stats.ks_2samp(getattr(a, name), getattr(b, name))
        ks_stat[name], ks_p[name] = float(res.statistic), float(res.pvalue)
    sk_z = sk_z_scores(a.sk_shell, b.sk_shell)
    worst = float(np.max(np.abs(sk_z)))
    lp_p = _logphi_p(a, b)
    return BatteryResult(
        n_a=a.size,
        n_b=b.size,
        frac_a=ka / a.size,
        frac_b=kb / b.size,
        prop_z=z,
        prop_p=p,
        prop_pass=p > p_prop,
        ks_stat=ks_stat,
        ks_p=ks_p,
        ks_pass={k: v > p_ks for k, v in ks_p.items()},
        sk_z=sk_z,
        sk_worst=worst,
        sk_pass=bool(worst < zmax),
        logphi_ks_p=lp_p,
        p_threshold_prop=p_prop,
        p_threshold_ks=p_ks,
        zmax=zmax,
    )


def _logphi_p(a: EnsembleSample, b: EnsembleSample) -> float:
    """Return the KS p of -log10(Phi) among successes (nan if a side is empty)."""
    xa = neg_log10_phi(a.energy[a.success])
    xb = neg_log10_phi(b.energy[b.success])
    if xa.size == 0 or xb.size == 0:
        return float("nan")
    return float(stats.ks_2samp(xa, xb).pvalue)
