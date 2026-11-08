#!/usr/bin/env python3

"""
Unit gate and protocol checks of the C++ MD reference data (plan T4).

Reads ``tests/data/md_cxx_reference.json`` and the stored configurations
``tests/data/md_cxx_samples.npz`` written by
``tools/cpp_reference/md_reference.py collect`` and recomputes Phi with
``xla_cco`` in the C++ box. Also tests the pure helpers of that script (the
Andersen schedule transcription, the log parser, the K choice).

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import importlib.util
import json
import math
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List

# ----- numerics -----
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

# ----- project -----
from tolerances import MD_CXX_UNITS_TOL, RTOL_CXX_LOG_6SIG, RTOL_MD_CXX_PHI
from xla_cco.box import Box
from xla_cco.energy import make_stealthy_energy
from xla_cco.kvectors import enumerate_kvectors, k_outer_from_ka

# ============================================================
# CONSTANTS AND FIXTURES
# ============================================================

ROOT: Path = Path(__file__).resolve().parents[2]
JSON_PATH: Path = ROOT / "tests" / "data" / "md_cxx_reference.json"
NPZ_PATH: Path = ROOT / "tests" / "data" / "md_cxx_samples.npz"
SCRIPT: Path = ROOT / "tools" / "cpp_reference" / "md_reference.py"
REAL_LINE: str = (
    "at time=1,\t2:0/1,energy=0.000414431,kinetic_energy=0.00207993,dt=0.01,"
    "dimensionless_pressure=2.16065"
)


def _load_script() -> ModuleType:
    """Import ``tools/cpp_reference/md_reference.py`` (not a package)."""
    spec = importlib.util.spec_from_file_location("md_reference", SCRIPT)
    assert spec is not None and spec.loader is not None, f"cannot load {SCRIPT}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MDR: ModuleType = _load_script()


@pytest.fixture(scope="module")
def ref() -> Dict[str, Any]:
    """The reference JSON."""
    data: Dict[str, Any] = json.loads(JSON_PATH.read_text())
    assert data["runs"], "no runs in the reference JSON"
    return data


@pytest.fixture(scope="module")
def samples(ref: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """The stored C++ configurations with their JSON energies attached."""
    with np.load(NPZ_PATH) as z:
        out = {k: np.asarray(z[k]) for k in z.files}
    assert out["s"].ndim == 3, "s must be (B, N, d)"
    runs = {r["seed"]: r for r in ref["runs"]}
    pairs = list(zip(out["seed"].tolist(), out["sample"].tolist(), strict=True))
    assert len(pairs) == out["s"].shape[0], "seed/sample columns do not match s"
    for key in ("phi_cpp", "phi_log"):
        out[key] = np.array([runs[sd][key][i] for sd, i in pairs])
    return out


def _xla_phi(ref: Dict[str, Any], basis: np.ndarray, s: np.ndarray) -> np.ndarray:
    """xla_cco Phi of (B, N, d) fractional configs in the reference box and K."""
    assert s.ndim == 3 and s.shape[2] == basis.shape[0], "s must be (B, N, d)"
    assert ref["K"] > 0.0, "K must be positive"
    box = Box(basis=jnp.asarray(basis))
    energy = make_stealthy_energy(box, enumerate_kvectors(basis, 0.0, ref["K"]))
    return np.array([float(energy(jnp.asarray(c))) for c in s])


def _naive_resets(n_steps: int, prob: float) -> List[int]:
    """Independent oracle: the C++ loop written with an unbounded while."""
    assert n_steps >= 0 and prob > 0.0, "need n_steps >= 0 and prob > 0"
    out: List[int] = []
    acc = 0.0
    for step in range(1, n_steps + 1):
        acc += prob
        while acc > 0:
            out.append(step)
            acc -= 1.0
    return out


# ============================================================
# TESTS: UNIT GATE
# ============================================================


@pytest.mark.high
def test_unit_gate_stored_samples(
    ref: Dict[str, Any], samples: Dict[str, np.ndarray]
) -> None:
    """xla Phi of stored C++ samples equals the C++ energy to relative 1e-8."""
    phi = _xla_phi(ref, samples["basis"], samples["s"])
    rel = np.abs(phi - samples["phi_cpp"]) / samples["phi_cpp"]
    assert samples["s"].shape[1:] == (ref["N"], ref["d"])
    assert np.all(rel <= RTOL_MD_CXX_PHI), f"max rel diff {rel.max():.3e}"
    assert np.array_equal(samples["basis"], np.array(ref["basis"]))


@pytest.mark.high
def test_unit_gate_all_samples_in_json(ref: Dict[str, Any]) -> None:
    """Every recorded sample passed the gate; arrays are complete and positive."""
    worst = 0.0
    for run in ref["runs"]:
        cpp, xla = np.array(run["phi_cpp"]), np.array(run["phi_xla"])
        n = run["n_samples"]
        assert n >= 1 and all(len(run[k]) == n for k in ("phi_log", "ke_log", "dt_log"))
        assert np.all(cpp > 0.0) and cpp.size == xla.size == n
        worst = max(worst, float(np.max(np.abs(xla - cpp) / cpp)))
    assert worst <= RTOL_MD_CXX_PHI, f"max rel diff {worst:.3e}"
    assert ref["unit_gate"]["max_rel_diff_xla_vs_cpp"] <= RTOL_MD_CXX_PHI


@pytest.mark.high
def test_regression_D052_log_is_six_digit_rounding_of_cpp_energy(
    ref: Dict[str, Any], samples: Dict[str, np.ndarray]
) -> None:
    """D-052: the log is the %g (6-digit) rounding of the full C++ energy."""
    phi = _xla_phi(ref, samples["basis"], samples["s"])
    rel = np.abs(phi - samples["phi_log"]) / samples["phi_log"]
    assert np.all(rel <= RTOL_CXX_LOG_6SIG), f"max rel diff {rel.max():.3e}"
    for run in ref["runs"]:
        pairs = zip(run["phi_cpp"], run["phi_log"], strict=True)
        assert all(f"{c:g}" == f"{g:g}" for c, g in pairs), f"seed {run['seed']}"


# ============================================================
# TESTS: K-SET, BOX AND UNITS
# ============================================================


@pytest.mark.high
def test_kset_and_box_match_cpp(ref: Dict[str, Any]) -> None:
    """K from Ka bit-equals C++ k2; the k-set and the D-004 box are reproduced."""
    d, n = ref["d"], ref["N"]
    basis = np.array(ref["basis"])
    assert np.array_equal(basis, MDR.cpp_box_side(d) * np.eye(d))
    assert ref["L"] == math.pow(1000.0, 1.0 / d), "D-004 box side"
    assert float.hex(k_outer_from_ka(ref["Ka"], ref["phi_fake"], d)) == ref["K_hex"]
    kset = enumerate_kvectors(basis, 0.0, ref["K"])
    assert kset.n.shape[0] == ref["M"] == len(ref["n_vectors"])
    assert sorted(map(tuple, kset.n.tolist())) == sorted(map(tuple, ref["n_vectors"]))
    assert ref["chi"] == ref["M"] / (d * (n - 1))
    assert abs(ref["M"] - ref["M_target"]) <= 1, "M far from round(chi d (N-1))"


@pytest.mark.high
def test_units_equipartition(ref: Dict[str, Any]) -> None:
    """Pooled <Phi>/(M T) and <KE>/(N d T/2) are 1 (mass 1, kT = T, no COM removal)."""
    phi = np.concatenate([r["phi_cpp"] for r in ref["runs"]])
    ke = np.concatenate([r["ke_log"] for r in ref["runs"]])
    t, m, n, d = ref["T"], ref["M"], ref["N"], ref["d"]
    phi_ratio = float(phi.mean()) / (m * t)
    ke_ratio = float(ke.mean()) / (n * d * t / 2.0)
    assert abs(phi_ratio - 1.0) < MD_CXX_UNITS_TOL, f"<Phi>/(MT) = {phi_ratio}"
    assert abs(ke_ratio - 1.0) < MD_CXX_UNITS_TOL, f"<KE>/(NdT/2) = {ke_ratio}"


@pytest.mark.high
def test_protocol_record(ref: Dict[str, Any]) -> None:
    """The JSON carries the C++ dt and Andersen schedule V2 must replicate."""
    proto = ref["protocol"]
    assert proto["dt"] == 0.01 and proto["step_per_sample"] == 100_000
    for run in ref["runs"]:
        assert set(run["dt_log"]) == {proto["dt"]}, f"seed {run['seed']} dt"
    sched = MDR.andersen_reset_steps(proto["steps_per_half"], proto["andersen"]["p"])
    assert proto["andersen"]["resets_per_call"] == sched.size
    assert proto["andersen"]["first_reset_steps"] == sched[:4].tolist()
    assert proto["andersen"]["mode"] == "count" and proto["andersen"]["p"] == 0.01


# ============================================================
# TESTS: HELPERS OF md_reference.py
# ============================================================


@pytest.mark.high
def test_regression_D050_andersen_count_schedule() -> None:
    """D-050: one 50000-step call resets at 1, 100, 200, ..., 50000 (501 resets)."""
    sched = MDR.andersen_reset_steps(50_000, 0.01)
    assert sched.size == 501
    assert sched[:4].tolist() == [1, 100, 200, 300] and sched[-1] == 50_000
    assert MDR.andersen_reset_steps(1, 0.01).tolist() == [1]
    assert MDR.andersen_reset_steps(0, 0.01).size == 0
    assert MDR.andersen_reset_steps(6, 0.5).tolist() == [1, 3, 5]


@pytest.mark.high
@given(n_steps=st.integers(0, 3000), prob=st.floats(1e-3, 1.0))
@example(n_steps=50_000, prob=0.01)  # D-050 repro
def test_andersen_schedule_properties(n_steps: int, prob: float) -> None:
    """Oracle loop equality, first reset at step 1, count within 1 of ceil(n p)."""
    sched = MDR.andersen_reset_steps(n_steps, prob)
    assert sched.tolist() == _naive_resets(n_steps, prob)
    exact = math.ceil(n_steps * Fraction(prob))
    assert abs(sched.size - exact) <= 1
    if n_steps >= 1:
        assert sched[0] == 1 and np.all(np.diff(sched) > 0) and sched[-1] <= n_steps


@pytest.mark.high
def test_parse_sample_lines_real_line() -> None:
    """A verbatim C++ sample line parses to its fields."""
    out = MDR.parse_sample_lines("noise\n" + REAL_LINE + "\n")
    assert out["phi"].tolist() == [0.000414431] and out["ke"].tolist() == [0.00207993]
    assert out["dt"].tolist() == [0.01] and out["pressure"].tolist() == [2.16065]
    assert MDR.parse_sample_lines("no samples here")["phi"].size == 0


_pos = st.floats(1e-12, 1e6, allow_nan=False, allow_infinity=False)


@pytest.mark.high
@given(phi=_pos, ke=_pos, dt=_pos, pres=st.floats(-1e6, 1e6))
def test_parse_sample_lines_round_trip(
    phi: float, ke: float, dt: float, pres: float
) -> None:
    """Lines printed with the C++ %g default parse back to the printed values."""
    line = (
        f"at time=7,\t2:0/3,energy={phi:g},kinetic_energy={ke:g},dt={dt:g},"
        f"dimensionless_pressure={pres:g}"
    )
    out = MDR.parse_sample_lines(line)
    assert out["phi"][0] == float(f"{phi:g}") and out["ke"][0] == float(f"{ke:g}")
    assert out["dt"][0] == float(f"{dt:g}") and out["pressure"][0] == float(f"{pres:g}")


@pytest.mark.high
@given(m_target=st.integers(1, 120), side=st.floats(5.0, 40.0))
@example(m_target=51, side=math.sqrt(1000.0))  # the T4 set-up: M = 50
@example(m_target=36, side=6.0)  # D-053 repro: K split the n^2 = 25 shell
@example(m_target=76, side=6.25)  # D-053 repro: K^2 landed on a shell
def test_choose_k_outer_properties(m_target: int, side: float) -> None:
    """K holds exactly M waves, sits mid-gap, and M is the nearest shell count.

    Oracle: in a square box the shells are the integers n^2 = n . n exactly.
    """
    basis = side * np.eye(2)
    m, k_out = MDR.choose_k_outer(basis, m_target)
    n_all = enumerate_kvectors(basis, 0.0, 2.0 * k_out + 1.0).n
    nsq = np.sort((n_all * n_all).sum(axis=1))
    shells = np.unique(nsq)
    counts = np.searchsorted(nsq, shells, side="right")
    q = k_out**2 * (side / (2.0 * math.pi)) ** 2  # K^2 in units of n^2
    assert enumerate_kvectors(basis, 0.0, k_out).n.shape[0] == m
    assert np.min(np.abs(shells - q)) > 0.25, "K^2 is not mid-gap"
    assert m in counts.tolist()
    assert abs(m - m_target) == int(np.min(np.abs(counts - m_target)))


@pytest.mark.high
def test_regression_D053_shell_split_by_ulp() -> None:
    """D-053: an ulp-split shell must not put K inside or on the n^2 = 25 shell."""
    for side, m_target in ((6.0, 36), (6.25, 76)):
        m, k_out = MDR.choose_k_outer(side * np.eye(2), m_target)
        q = k_out**2 * (side / (2.0 * math.pi)) ** 2
        assert enumerate_kvectors(side * np.eye(2), 0.0, k_out).n.shape[0] == m
        assert abs(q - round(q)) > 0.25, f"K^2 = {q} n^2-units is on a shell"
    assert MDR.choose_k_outer(6.0 * np.eye(2), 36)[0] == 34  # shells 34, 40


@pytest.mark.high
def test_build_stdin_tokens() -> None:
    """The stdin stream has main.cpp's 17 tokens with ``MD step T`` last."""
    toks = MDR.build_stdin(0.25, 200, 20).split()
    assert len(toks) == 17 and toks[0] == "2" and toks[2] == "0.25"
    assert toks[9:12] == ["200", "20", "random"]
    assert toks[-3:] == ["MD", "step", repr(1e-5)]
