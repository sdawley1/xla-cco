#!/usr/bin/env python3

"""
Unit tests of the ensemble-parity statistics battery on synthetic data.

Identical (same-distribution) samples must pass; shifted ones must fail. This
keeps the battery logic of T12 CI-tested without Della data.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- numerics -----
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# ----- project -----
import ensemble_stats as es
from tolerances import KS_PMIN, SK_ZMAX

# ============================================================
# HELPERS
# ============================================================

B, S = 600, 12
P_PROP, P_KS = es.bonferroni_thresholds(0.01, 50, KS_PMIN)


def sample(
    seed: int, shift: float = 0.0, frac: float = 0.5, b: int = B
) -> es.EnsembleSample:
    """Return a synthetic ensemble; ``shift`` moves every statistic."""
    rng = np.random.default_rng(seed)
    energy = np.where(rng.uniform(size=b) < frac, 1e-25, 1e-5)
    return es.EnsembleSample(
        energy=energy,
        success=energy < 1e-20,
        nn_min=rng.normal(0.1 + shift, 0.02, b),
        nn_mean=rng.normal(0.5 + shift, 0.05, b),
        g2_peak=rng.normal(3.0 + shift, 0.4, b),
        rms_disp=rng.normal(1.0 + shift, 0.1, b),
        sk_shell=rng.normal(1.0 + shift, 0.3, (b, S)),
    )


# ============================================================
# TESTS
# ============================================================


def test_identical_distribution_passes() -> None:
    """Two independent draws from one distribution pass every criterion."""
    res = es.run_battery(sample(1), sample(2), P_PROP, P_KS, SK_ZMAX)
    assert res.passed, res
    assert res.n_p_tests == 5


def test_exactly_identical_samples_pass_with_zero_stat() -> None:
    """Literally identical samples give p = 1, z = 0 and KS statistic 0."""
    a = sample(3)
    res = es.run_battery(a, a, P_PROP, P_KS, SK_ZMAX)
    assert res.passed and res.prop_z == 0.0 and res.sk_worst == 0.0
    assert all(v == 0.0 for v in res.ks_stat.values())


@pytest.mark.parametrize("name", es.KS_NAMES)
def test_shift_in_one_scalar_fails(name: str) -> None:
    """Shifting a single per-config scalar fails exactly its KS test."""
    a, b = sample(4), sample(5)
    b = es.EnsembleSample(
        **{**b.__dict__, name: getattr(b, name) + 0.5 * np.std(getattr(b, name))}
    )
    res = es.run_battery(a, b, P_PROP, P_KS, SK_ZMAX)
    assert not res.ks_pass[name] and not res.passed


def test_sk_shift_in_single_shell_fails() -> None:
    """A shift of 0.5 sigma in one S(k) shell trips only the S(k) criterion."""
    a, b = sample(6), sample(7)
    sk = b.sk_shell.copy()
    sk[:, 3] += 0.5 * 0.3
    b = es.EnsembleSample(**{**b.__dict__, "sk_shell": sk})
    res = es.run_battery(a, b, P_PROP, P_KS, SK_ZMAX)
    assert not res.sk_pass and res.sk_worst > SK_ZMAX
    assert int(np.argmax(np.abs(res.sk_z))) == 3


def test_success_fraction_difference_fails() -> None:
    """60% vs 40% success over 600 configs fails the proportion test."""
    res = es.run_battery(sample(8, frac=0.6), sample(9, frac=0.4), P_PROP, P_KS, 4.0)
    assert not res.prop_pass and not res.passed


def test_logphi_is_report_only() -> None:
    """A different Phi distribution among successes does not affect ``passed``."""
    a, b = sample(10), sample(11)
    e = b.energy.copy()
    e[b.success] = 1e-30  # still successes, different -log10 Phi
    b = es.EnsembleSample(**{**b.__dict__, "energy": e})
    res = es.run_battery(a, b, P_PROP, P_KS, SK_ZMAX)
    assert res.logphi_ks_p < 1e-10 and res.passed


def test_two_proportion_known_value() -> None:
    """Hand value: 30/100 vs 50/100: z = -0.2/sqrt(0.4*0.6*0.02)."""
    z, p = es.two_proportion_test(30, 100, 50, 100)
    assert z == pytest.approx(-0.2 / np.sqrt(0.4 * 0.6 * 0.02))
    assert 1e-3 < p < 1e-2
    assert es.two_proportion_test(0, 50, 0, 70) == (0.0, 1.0)


def test_shell_groups_and_means() -> None:
    """Shell grouping merges ulp-level duplicates; means are per shell."""
    k_sq = np.array([2.0, 1.0, 1.0 + 1e-14, 3.0, 2.0])
    g = es.shell_groups(k_sq)
    assert g.tolist() == [1, 0, 0, 2, 1]
    s = np.array([[10.0, 1.0, 3.0, 7.0, 20.0]])
    assert es.shell_means(s, g).tolist() == [[2.0, 15.0, 7.0]]


def test_bonferroni_thresholds() -> None:
    """alpha = 0.01 over 50 tests gives 2e-4; KS takes the smaller of it and PMIN."""
    p, k = es.bonferroni_thresholds(0.01, 50, 1e-3)
    assert p == pytest.approx(2e-4) and k == pytest.approx(2e-4)
    assert es.bonferroni_thresholds(0.01, 2, 1e-3)[1] == 1e-3


@settings(deadline=None, max_examples=40)
@given(
    seed=st.integers(0, 10_000),
    shift=st.floats(-3.0, 3.0),
    scale=st.floats(0.5, 2.0),
)
def test_sk_z_antisymmetric_and_affine_invariant(
    seed: int, shift: float, scale: float
) -> None:
    """z(a, b) = -z(b, a); a common affine map of both samples leaves z fixed."""
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=(50, 4)), rng.normal(0.1, 1.0, size=(60, 4))
    z = es.sk_z_scores(a, b)
    np.testing.assert_allclose(z, -es.sk_z_scores(b, a), atol=1e-12)
    np.testing.assert_allclose(
        z, es.sk_z_scores(scale * a + shift, scale * b + shift), atol=1e-9
    )


@settings(deadline=None, max_examples=40)
@given(
    k1=st.integers(0, 80),
    n1=st.integers(80, 200),
    k2=st.integers(0, 80),
    n2=st.integers(80, 200),
)
def test_two_proportion_symmetry(k1: int, n1: int, k2: int, n2: int) -> None:
    """Swapping the samples flips z and keeps p; p lies in [0, 1]."""
    z, p = es.two_proportion_test(k1, n1, k2, n2)
    z2, p2 = es.two_proportion_test(k2, n2, k1, n1)
    assert z == pytest.approx(-z2, abs=1e-12) and p == pytest.approx(p2)
    assert 0.0 <= p <= 1.0
