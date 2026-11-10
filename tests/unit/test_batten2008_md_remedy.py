#!/usr/bin/env python3

"""
Tests for the INCONCLUSIVE-remedy evaluation of the MD winding experiment
(experiments/scripts/batten2008_md_remedy.py): the determinism check of a
2x-cycles rerun against its production run, the substitution of the rerun
cell's parts into the registered part list, and the re-evaluated H3 / H7.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import sys
from pathlib import Path
from typing import Any, Dict, List

# ----- numerics -----
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

# ----- project -----
from tests.tolerances import (
    ATOL_MD_ONSET_REL,
    ATOL_MD_PREFIX_DELTA,
    RTOL_MD_PREFIX_DELTA,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "scripts"))
import batten2008_md_remedy as rem  # noqa: E402
import batten2008_md_stats as mds  # noqa: E402

# ============================================================
# HELPERS
# ============================================================


def _raw(n_cycles: int, per_seg: int, seed: int = 0) -> Dict[str, np.ndarray]:
    """Fake MD output: 2 lanes, 3 waves, ``per_seg`` snapshots per 100-step segment.

    Cycle c covers steps (100 c, 100 (c + 1)]; every quantity is a fixed
    function of (step, lane, wave), so runs with different cycle counts or
    snapshot grids agree wherever they overlap.
    """
    lanes, waves = 2, 3
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((lanes, waves))
    steps = np.concatenate([100 * c + np.linspace(100 / per_seg, 100, per_seg)
                            for c in range(n_cycles)]).astype(np.int64)  # fmt: skip
    wind = steps[None, :, None] * base[:, None, :]
    cyc = np.arange(1, n_cycles + 1)[:, None, None] * base[None]
    return {
        "record_steps": steps, "snap_wind_abs": np.abs(wind), "snap_wind_net": wind,
        "snap_phi": steps[None, :] * np.ones((lanes, 1)),
        "cycle_wind_abs": np.abs(cyc), "cycle_n_valid": np.ones_like(cyc, np.int64),
        "cycle_status": np.zeros((n_cycles, lanes), np.int64),
        "trace_time": np.tile(np.arange(10.0 * n_cycles), (lanes, 1)),
        "trace_rho": (np.arange(20.0 * n_cycles) + 1.0).reshape(1, -1, 1, 2)
        * np.ones((lanes, 1, waves, 1)),
        "trace_count": np.full(lanes, 10 * n_cycles, np.int64),
        "temperatures": np.array([1e-6]), "indices": np.arange(lanes),
        "equil_gate": base[:, 0].copy(),
    }  # fmt: skip


def _h3_cell(seed: int, n_seg: int, shift: float = 0.0) -> Dict[str, Any]:
    """Minimal analysis for h3_parts: 10 ground states, 2 harmonic T, n_seg segments."""
    rng = np.random.default_rng(seed)
    w1 = 1.0 + 1e-4 * rng.standard_normal((10, 2, n_seg))
    w2 = w1 * (1.0 + shift + 1e-4 * rng.standard_normal((10, 2, n_seg)))
    return {"system": "flows2d", "tag": "2-5", "temps": np.array([1e-6, 1e-5]),
            "w1_c": w1, "w2_c": w2, "nu_c": np.ones((10, 2)),
            "indices": np.arange(10)}  # fmt: skip


def _lang(gamma: float, nu: np.ndarray) -> Dict[str, Any]:
    """Minimal BAOAB analysis for h7_parts (5 ground states, T = 1e-6, 1e-4)."""
    temps = np.array([1e-6, 1e-4])
    return {"system": "flows2d", "tag": "1-10", "temps": temps,
            "gamma": np.asarray(gamma), "nu_c": nu,
            "indices": np.arange(5)}  # fmt: skip


# ============================================================
# DETERMINISM CHECK
# ============================================================


@pytest.mark.critical
def test_prefix_agreement_of_a_longer_rerun() -> None:
    """A 4-cycle rerun on a coarser snapshot grid agrees bitwise where it overlaps."""
    prod, rerun = _raw(2, 10), _raw(4, 5)
    out = rem.prefix_agreement(prod, rerun)
    assert out["bitwise"] and out["max_abs"] == 0.0
    assert out["n_cycles"] == 2
    assert out["n_snapshots_common"] == 10  # every 2nd production snapshot
    assert {"cycle_wind_abs", "snap_wind_net", "trace_rho", "equil_gate"} <= set(
        out["keys"]
    )
    assert all(v["n"] > 0 for v in out["keys"].values())


@pytest.mark.critical
@given(
    key=st.sampled_from(["cycle_wind_abs", "snap_wind_net", "equil_gate", "trace_rho"]),
    delta=st.floats(1e-12, 1e3, allow_nan=False),
)
@example(key="cycle_wind_abs", delta=1e-12)
def test_prefix_agreement_flags_any_difference(key: str, delta: float) -> None:
    """One perturbed overlapping value is reported, with its size, under its key."""
    prod, rerun = _raw(2, 10), _raw(4, 5)
    prod[key] = prod[key].astype(np.float64)
    flat = prod[key].reshape(-1)
    flat[-1] += delta  # the last element lies in the overlap for every key
    out = rem.prefix_agreement(prod, rerun)
    assert not out["bitwise"] and not out["keys"][key]["bitwise"]
    assert out["keys"][key]["max_abs"] == pytest.approx(
        delta, rel=RTOL_MD_PREFIX_DELTA, abs=ATOL_MD_PREFIX_DELTA
    )
    assert all(v["bitwise"] for k, v in out["keys"].items() if k != key)


@pytest.mark.medium
def test_prefix_agreement_needs_a_longer_rerun() -> None:
    """A rerun with fewer cycles than production is rejected."""
    with pytest.raises(ValueError, match="cycles"):
        rem.prefix_agreement(_raw(4, 5), _raw(2, 10))


# ============================================================
# SUBSTITUTION AND RE-EVALUATION
# ============================================================


@pytest.mark.high
@given(
    n_base=st.integers(1, 12),
    picks=st.lists(st.integers(0, 15), max_size=8, unique=True),
)
@example(n_base=3, picks=[1, 14])
def test_replace_parts_property(n_base: int, picks: List[int]) -> None:
    """Rerun parts replace production parts of the same (part, cell); others stay."""

    def part(i: int, value: float) -> mds.Part:
        return mds.make_part(f"segment {i}", "c", value, 0.0, 0.0, True)

    base = [part(i, 1.0) for i in range(n_base)]
    new = [part(i, 2.0) for i in picks]
    out = rem.replace_parts(base, new)
    keys = [(p["part"], p["cell"]) for p in out]
    assert len(keys) == len(set(keys)) == len(set(range(n_base)) | set(picks))
    vals = {p["part"]: p["value"] for p in out}
    assert all(vals[f"segment {i}"] == 2.0 for i in picks)
    assert all(vals[f"segment {i}"] == 1.0 for i in range(n_base) if i not in picks)


@pytest.mark.critical
def test_remedy_h3_cell_replaces_production_cell() -> None:
    """H3: the rerun cell's 4 segments replace its 2 production segments only."""
    prod_cell = _h3_cell(0, 2, shift=-0.0045)  # a sub-percent miss, |z| >> 3
    other = dict(_h3_cell(1, 2), system="url2d")
    rerun_cell = _h3_cell(2, 4)
    res = rem.remedy([prod_cell, other], [], [rerun_cell], [])
    h3 = res["hypotheses"]["H3"]
    assert h3["production"]["verdict"] == mds.INCONCLUSIVE
    assert h3["production_cell"]["verdict"] == mds.INCONCLUSIVE
    assert h3["rerun_cell"]["verdict"] == mds.PASS
    assert sum(h3["rerun_cell"]["counts"].values()) == 2 * 4
    assert sum(h3["production_cell"]["counts"].values()) == 2 * 2
    assert h3["after_remedy"]["verdict"] == mds.PASS
    assert sum(h3["after_remedy"]["counts"].values()) == 2 * 4 + 2 * 2
    assert res["hypotheses"]["H7"]["rerun_cell"]["verdict"] == mds.NOT_EVALUATED


@pytest.mark.critical
def test_remedy_h7_pairs_rerun_langevin_with_production_nve() -> None:
    """H7: BAOAB / NVE uses the production NVE cell; gamma / gamma both reruns."""
    rng = np.random.default_rng(4)
    nve = {"system": "flows2d", "tag": "1-10", "temps": np.array([1e-6, 1e-4, 1e-3]),
           "nu_c": 1.0 + 1e-3 * rng.standard_normal((10, 3)),
           "w1_c": np.ones((10, 3, 2)), "w2_c": np.ones((10, 3, 2)),
           "indices": np.arange(10)}  # fmt: skip
    nu = 1.0 + 1e-3 * rng.standard_normal((5, 2))
    noise = 1.0 + 1e-4 * rng.standard_normal((5, 2))
    lang = [_lang(0.003, nu), _lang(0.03, nu * noise)]
    res = rem.remedy([nve], [], [], lang)
    h7 = res["hypotheses"]["H7"]
    names = sorted({p["part"] for p in h7["rerun_cell"]["parts"]})
    assert names == ["BAOAB / NVE", "gamma"]
    assert sum(h7["rerun_cell"]["counts"].values()) == 2 * 2 + 2
    assert h7["rerun_cell"]["verdict"] == mds.PASS
    md = rem.markdown(res)
    assert "INCONCLUSIVE remedy" in md and "| H7 |" in md


@pytest.mark.high
@given(rate=st.floats(0.05, 2.0), eps=st.floats(1e-15, 1e-12))
@example(rate=0.5, eps=1e-13)
def test_divergence_onset_of_exponential_growth(rate: float, eps: float) -> None:
    """A difference eps exp(rate t) is found at rounding level and timed per level."""
    prod, rerun = _raw(2, 10), _raw(4, 5)
    t = prod["trace_time"][0]
    growth = 1.0 + np.minimum(eps * np.exp(rate * t), 1.0)
    rerun["trace_rho"] = rerun["trace_rho"].copy()
    rerun["trace_rho"][:, : t.size] *= growth[None, :, None, None]
    out = rem.divergence_onset(prod, rerun)
    assert out["rel"][0] == pytest.approx(eps, abs=ATOL_MD_ONSET_REL)
    for level, t_hit in out["t_reach"].items():
        expect = t[np.flatnonzero(growth - 1.0 > float(level))]
        assert t_hit == (float(expect[0]) if expect.size else None), level


@pytest.mark.high
def test_cycle_winding_ratio_of_the_shared_cycles() -> None:
    """Unsigned constrained winding of rerun cycles 0..C-1 over production's."""
    prod, rerun = _raw(2, 10), _raw(4, 5)
    prod["waves_constrained"] = np.array([True, True, False])
    rerun["waves_constrained"] = prod["waves_constrained"]
    rerun["cycle_wind_abs"] = rerun["cycle_wind_abs"].copy()
    rerun["cycle_wind_abs"][:2, 1] *= 1.02  # lane 1, both shared cycles
    rerun["cycle_wind_abs"][2:] *= 5.0  # later cycles are not compared
    out = rem.cycle_winding_ratio(prod, rerun)
    assert out["n"] == 2 * 2
    assert out["min"] == pytest.approx(1.0) and out["max"] == pytest.approx(1.02)
    assert out["median"] == pytest.approx(1.01)


@pytest.mark.medium
def test_markdown_reports_determinism_onset_and_winding_ratio() -> None:
    """The remedy markdown names the differing arrays, the onset and the ratio."""
    prod, rerun = _raw(2, 10), _raw(4, 5)
    for z in (prod, rerun):
        z["waves_constrained"] = np.array([True, True, False])
    rerun["trace_rho"] = rerun["trace_rho"] * (1.0 + 1e-13)
    chk = rem.prefix_agreement(prod, rerun)
    onset = rem.divergence_onset(prod, rerun)
    chk["onset"] = {"t_first": onset["t"][0], "rel_first": onset["rel"][0],
                    "t_last": onset["t"][-1], "rel_last": onset["rel"][-1],
                    "t_reach": onset["t_reach"]}  # fmt: skip
    chk["winding_ratio"] = rem.cycle_winding_ratio(prod, rerun)
    res = rem.remedy([_h3_cell(0, 2)], [], [_h3_cell(2, 4)], [])
    res["determinism"] = {"x.npz": chk}
    md = rem.markdown(res)
    assert "bitwise False" in md and "differ: trace_rho" in md
    assert "1e-09 at t = never" in md and "median 1.00000" in md
    assert "| segment 0 | flows2d chi=2-5 T=1e-06 | 1.0" in md  # production value
