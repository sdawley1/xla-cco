#!/usr/bin/env python3

"""
Statistical parity of the JAX MD with the C++ ``CollectiveCoordinateMD`` (V2).

C++ side: ``tests/data/md_cxx_reference.json`` (plan T4), 3 seeds x 200
samples of Phi (full precision) and KE (6 digits) from the unmodified binary;
N = 256, d = 2, M = 50 (chi = 0.098), box side sqrt(1000) (density 0.256),
T = 1e-5, dt = 0.01, count-mode Andersen p = 0.01, COM never removed.

JAX side: ``tests/data/md_jax_replica.json``, written by
``tools/cpp_reference/md_jax_replica.py`` (SLURM array
``slurm/md_jax_replica_cpu.slurm``; the full protocol costs ~1e8 lane-steps,
hours on CPU, so the test reads its committed output). 40 independent lanes at
T and 40 at 1.1 T (the power control), each from its own L-BFGS-relaxed
random start, through the full C++ protocol (20 x 2 x [MB redraw, NVE 5e4],
then 20 x [Andersen 5e4, NVE 5e4, record]), with the C++ Andersen schedule
reproduced exactly (``ANDERSEN_ACC_START``; checked below). A live test
reruns the protocol's sample stage on CPU from the stored C++ configurations
(the decision table's "JAX replica on a C++ configuration").

Observables: x = Phi / (M T) and y = KE / (N d T / 2), both 1 in the harmonic
canonical limit (KE counts all N d components: no COM removal). Statistics:

- KS: two-sample, after keeping every MD_PARITY_KS_THIN-th sample of each
  series (C++ seed or JAX lane; lag-1 autocorrelation ~ 0.2);
- means: z = (m_a - m_b) / sqrt(se_a^2 + se_b^2), JAX SE from lane means (lanes
  are independent), C++ SE from batch means of MD_PARITY_BLOCK samples;
- controls: C++ seed vs seed must pass both; JAX at 1.1 T (normalized by the
  nominal T) must fail the KS test with p < MD_PARITY_CONTROL_P.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import dataclasses
import importlib.util
import itertools
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Tuple

# ----- numerics -----
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

# ----- project -----
from tolerances import (
    MD_PARITY_BLOCK,
    MD_PARITY_CONTROL_P,
    MD_PARITY_KS_PVALUE,
    MD_PARITY_KS_THIN,
    MD_PARITY_Z,
)
from xla_cco.dynamics.integrators import andersen_reset

# ============================================================
# CONSTANTS AND MODULES
# ============================================================

ROOT: Path = Path(__file__).resolve().parents[2]
CXX_JSON: Path = ROOT / "tests" / "data" / "md_cxx_reference.json"
CXX_NPZ: Path = ROOT / "tests" / "data" / "md_cxx_samples.npz"
JAX_JSON: Path = ROOT / "tests" / "data" / "md_jax_replica.json"
TOOLS: Path = ROOT / "tools" / "cpp_reference"
OBS: Tuple[str, ...] = ("phi", "ke")
LIVE_SEED: int = 4242


def _load(name: str) -> ModuleType:
    """Import ``tools/cpp_reference/<name>.py`` (not a package)."""
    path = TOOLS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.__file__ is not None, "module without a file"
    return mod


REP: ModuleType = _load("md_jax_replica")
MDR: ModuleType = _load("md_reference")

# ============================================================
# STATISTICS
# ============================================================


def _norm(ref: Dict[str, Any], phi: np.ndarray, ke: np.ndarray) -> Dict[str, Any]:
    """{'phi': Phi / (M T), 'ke': KE / (N d T / 2)} at the nominal C++ T."""
    t, n, d = float(ref["T"]), int(ref["N"]), int(ref["d"])
    assert ref["protocol"]["ke_dof"] == n * d, "KE counts all N d components"
    out = {"phi": phi / (ref["M"] * t), "ke": ke / (0.5 * n * d * t)}
    assert all(np.all(np.isfinite(v)) for v in out.values()), "non-finite data"
    return out


def _thin(series: np.ndarray) -> np.ndarray:
    """Every MD_PARITY_KS_THIN-th sample of each row of (s, n) series, flat."""
    assert series.ndim == 2 and series.size >= 1, "need (s, n) series"
    out = series[:, ::MD_PARITY_KS_THIN].ravel()
    assert out.size >= series.shape[0], "at least one sample per series"
    return out


def _blocked(series: np.ndarray, block: int) -> Tuple[float, float]:
    """(mean, SE) from consecutive means of ``block`` samples of each row."""
    s, n = series.shape
    nb = n // block
    assert block >= 1 and s * nb >= 2, f"need >= 2 groups, got {s} x {nb}"
    means = series[:, : nb * block].reshape(s, nb, block).mean(axis=2).ravel()
    se = float(means.std(ddof=1)) / math.sqrt(means.size)
    assert se > 0.0, "zero spread of block means"
    return float(series.mean()), se


def _compare(
    a: np.ndarray, a_block: int, b: np.ndarray, b_block: int
) -> Dict[str, float]:
    """KS p (thinned) and z of the means of two sets of (s, n) series.

    ``*_block`` is the samples per independent group: MD_PARITY_BLOCK for
    C++ series, the full lane length for JAX lanes (lane means).
    """
    ma, sa = _blocked(a, a_block)
    mb, sb = _blocked(b, b_block)
    ks = stats.ks_2samp(_thin(a), _thin(b))
    out = {"p": float(ks.pvalue), "D": float(ks.statistic), "mean_a": ma,
           "se_a": sa, "mean_b": mb, "se_b": sb,
           "z": (ma - mb) / math.hypot(sa, sb)}  # fmt: skip
    assert 0.0 <= out["p"] <= 1.0 and math.isfinite(out["z"]), out
    return out


def _report(label: str, res: Dict[str, float]) -> None:
    """One line: means +- SE, z and the KS p (pytest -s)."""
    assert "p" in res and "z" in res, "need a comparison"
    assert label, "need a label"
    print(f"  {label:<34s} {res['mean_a']:.4f}+-{res['se_a']:.4f} vs "
          f"{res['mean_b']:.4f}+-{res['se_b']:.4f}  z = {res['z']:+.2f}  "
          f"KS D = {res['D']:.3f} p = {res['p']:.3g}", flush=True)  # fmt: skip


# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture(scope="module")
def ref() -> Dict[str, Any]:
    """The C++ reference JSON (plan T4)."""
    data: Dict[str, Any] = json.loads(CXX_JSON.read_text())
    assert len(data["runs"]) >= 2, "need >= 2 C++ seeds for the seed control"
    assert data["protocol"]["remove_com"] is False, "C++ keeps the COM momentum"
    return data


@pytest.fixture(scope="module")
def cxx(ref: Dict[str, Any]) -> Dict[int, Dict[str, np.ndarray]]:
    """Per C++ seed, normalized (1, n) series of Phi/(M T) and KE/(N d T/2)."""
    out = {}
    for r in ref["runs"]:
        x = _norm(ref, np.asarray(r["phi_cpp"]), np.asarray(r["ke_log"]))
        out[int(r["seed"])] = {k: v[None, :] for k, v in x.items()}
    assert all(v["phi"].shape == v["ke"].shape for v in out.values()), "shapes"
    assert sum(v["phi"].size for v in out.values()) >= 600, "600 C++ samples"
    return out


@pytest.fixture(scope="module")
def pooled(cxx: Dict[int, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """C++ series of all seeds stacked (s, n)."""
    out = {k: np.concatenate([v[k] for v in cxx.values()]) for k in OBS}
    assert out["phi"].shape[0] == len(cxx), "one series per seed"
    assert out["phi"].shape == out["ke"].shape, "aligned observables"
    return out


@pytest.fixture(scope="module")
def replica(ref: Dict[str, Any]) -> Dict[str, Any]:
    """Stored JAX replica: normalized (lanes, samples) arrays per t_factor."""
    data: Dict[str, Any] = json.loads(JAX_JSON.read_text())
    fac = np.asarray(data["t_factor"])
    phi, ke = np.asarray(data["phi"]), np.asarray(data["ke"])
    x = _norm(ref, phi, ke)
    hot = np.isclose(fac, REP.CONTROL_FACTOR)
    out = {
        "meta": data,
        "main": {k: v[fac == 1.0] for k, v in x.items()},
        "hot": {k: v[hot] for k, v in x.items()},
    }
    assert phi.shape == ke.shape == (fac.size, data["protocol"]["n_samples"])
    assert out["main"]["phi"].shape[0] >= 2 and out["hot"]["phi"].shape[0] >= 2
    return out


# ============================================================
# PROTOCOL
# ============================================================


@pytest.mark.high
def test_andersen_phase_reproduces_cxx_schedule(ref: Dict[str, Any]) -> None:
    """Library count mode started at ANDERSEN_ACC_START resets at exactly the
    C++ steps of one AndersonEvolve call (1, 100, ..., 50000: 501, D-050);
    the carried-over accumulator from 0 would give 500 resets from step 100.
    """
    n_steps, p = int(ref["protocol"]["andersen_steps"]), ref["protocol"]["andersen_p"]
    v = jnp.zeros((4, 2))

    def fire_steps(acc0: float) -> np.ndarray:
        def body(c: Tuple[jax.Array, jax.Array], _: Any) -> Tuple[Any, jax.Array]:
            key, acc = c
            key, _, acc, fired = andersen_reset(key, v, acc, jnp.asarray(1.0), p,
                                                "count")  # fmt: skip
            return (key, acc), fired

        c0 = (jax.random.PRNGKey(0), jnp.asarray(acc0))
        _, fired = jax.lax.scan(body, c0, None, length=n_steps)
        return np.flatnonzero(np.asarray(fired)) + 1

    got = fire_steps(REP.ANDERSEN_ACC_START)
    want = MDR.andersen_reset_steps(n_steps, p)
    sched = ref["protocol"]["andersen"]
    assert np.array_equal(got, want), (got[:4], want[:4], got.size, want.size)
    assert got.size == sched["resets_per_call"] == 501
    assert got[:4].tolist() == sched["first_reset_steps"]
    carried = fire_steps(0.0)
    assert carried.size == 500 and carried[0] == 100, carried[:4]


def _check_settings(ref: Dict[str, Any], meta: Dict[str, Any], proto: Any) -> None:
    """MDSettings and lane temperatures: the recorded ones when the stored file
    has them (files written after 26af1b2), else the current settings_pair of
    the recorded protocol, must be the C++ rule (AUDIT-V2 F2, D-081)."""
    rec = meta.get("settings", REP.settings_record(proto))
    proto_ref = ref["protocol"]
    for name, scheme in (("nve", "verlet"), ("andersen", "andersen")):
        st = rec[name]
        assert st["scheme"] == scheme and st["remove_com"] is False, st
        assert st["dt"] == proto_ref["dt"] and st["record_steps"] == [], st
    ander = rec["andersen"]
    assert ander["andersen_mode"] == "count", ander
    assert ander["andersen_p"] == proto_ref["andersen_p"], ander
    fac = np.asarray(meta["t_factor"])
    assert set(fac.tolist()) == {1.0, REP.CONTROL_FACTOR}, "unexpected T factors"
    if "temperature" in meta:
        np.testing.assert_allclose(meta["temperature"], fac * ref["T"], rtol=1e-15)


@pytest.mark.high
def test_replica_phase_restarts_andersen_counter(ref: Dict[str, Any]) -> None:
    """REP.phase sets reset_acc = ANDERSEN_ACC_START before an Andersen phase
    (AUDIT-V2 B1): from reset_acc = 0.5 one Andersen step redraws exactly one
    velocity row relative to an NVE step from the same state (the C++ step-1
    reset) and leaves reset_acc = p (1 + p - 1); without the restart the
    accumulator would reach 0.5 + p and nothing would fire."""
    with np.load(CXX_NPZ) as z:
        r0 = np.asarray(z["s"][:2]) @ np.asarray(z["basis"])
    system = REP.build_system(ref)
    proto = REP.protocol_from_reference(ref, 1)
    st_nve, st_and = REP.settings_pair(proto)
    v0 = math.sqrt(system.temperature) * np.random.default_rng(7).standard_normal(
        r0.shape
    )
    temps = np.full(r0.shape[0], system.temperature)
    lanes = REP.init_lanes(system, r0, v0, temps, [11, 12], st_nve)
    lanes = dataclasses.replace(lanes, reset_acc=jnp.full_like(lanes.reset_acc, 0.5))
    ander = REP.phase(system, st_and, lanes, 1)
    nve = REP.phase(system, st_nve, lanes, 1)
    changed = np.any(np.asarray(ander.v) != np.asarray(nve.v), axis=-1)
    assert changed.sum(axis=1).tolist() == [1, 1], changed.sum(axis=1)
    np.testing.assert_allclose(np.asarray(ander.reset_acc), proto.andersen_p,
                               rtol=0, atol=1e-15)  # fmt: skip
    assert np.array_equal(np.asarray(ander.r), np.asarray(nve.r)), "same drift"


@pytest.mark.high
def test_stored_replica_ran_the_cxx_protocol(
    ref: Dict[str, Any], replica: Dict[str, Any]
) -> None:
    """The stored replica's protocol equals the C++ one and it holds at least
    as many samples as C++ at T, from independent lanes."""
    meta = replica["meta"]
    want = REP.protocol_from_reference(ref, meta["protocol"]["n_samples"])
    assert meta["protocol"] == want.__dict__, (meta["protocol"], want)
    assert meta["andersen_acc_start"] == REP.ANDERSEN_ACC_START
    assert meta["control_factor"] == REP.CONTROL_FACTOR
    assert want.equi_segments == 2 * ref["Nc_equi"] and want.nve_steps == 50_000
    assert len(set(meta["seeds"])) == len(meta["seeds"]), "lane seeds repeat"
    _check_settings(ref, meta, want)
    n_cxx = sum(r["n_samples"] for r in ref["runs"])
    assert replica["main"]["phi"].size >= n_cxx, "fewer JAX samples than C++"
    assert replica["hot"]["phi"].size >= n_cxx // 2, "control too small"
    print(f"\n  replica: {replica['main']['phi'].shape} at T, "
          f"{replica['hot']['phi'].shape} at {REP.CONTROL_FACTOR} T, "
          f"cost {meta['cost']['cpu_hours']:.1f} CPU-h", flush=True)  # fmt: skip


# ============================================================
# PARITY
# ============================================================


def _jax_vs(
    replica: Dict[str, Any], which: str, other: Dict[str, np.ndarray], block: int
) -> Dict[str, Dict[str, float]]:
    """Comparisons of the stored JAX lanes ``which`` against C++ series."""
    lanes = replica[which]
    n = lanes["phi"].shape[1]
    out = {k: _compare(lanes[k], n, other[k], block) for k in OBS}
    assert set(out) == set(OBS) and n >= 2, "both observables"
    return out


@pytest.mark.slow
@pytest.mark.high
def test_jax_matches_pooled_cxx(
    replica: Dict[str, Any], pooled: Dict[str, np.ndarray],
    cxx: Dict[int, Dict[str, np.ndarray]],
) -> None:  # fmt: skip
    """KS p > MD_PARITY_KS_PVALUE and |z| < MD_PARITY_Z for Phi/(M T) and
    KE/(N d T/2), JAX at T vs pooled C++ (per-seed numbers printed)."""
    res = _jax_vs(replica, "main", pooled, MD_PARITY_BLOCK)
    print("", flush=True)
    for seed, ser in cxx.items():
        for k, r in _jax_vs(replica, "main", ser, MD_PARITY_BLOCK).items():
            _report(f"{k} JAX vs C++ seed {seed}", r)
    for k in OBS:
        _report(f"{k} JAX vs C++ pooled", res[k])
        mj, sj = res[k]["mean_a"], res[k]["se_a"]
        mc, sc = res[k]["mean_b"], res[k]["se_b"]
        print(f"  {k}: vs theory 1: JAX t = {(mj - 1) / sj:+.2f}, "
              f"C++ t = {(mc - 1) / sc:+.2f}", flush=True)  # fmt: skip
    for k, r in res.items():
        assert r["p"] > MD_PARITY_KS_PVALUE, (k, r)
        assert abs(r["z"]) < MD_PARITY_Z, (k, r)


@pytest.mark.slow
@pytest.mark.high
def test_controls_seed_pairs_pass_and_hot_jax_fails(
    replica: Dict[str, Any], pooled: Dict[str, np.ndarray],
    cxx: Dict[int, Dict[str, np.ndarray]],
) -> None:  # fmt: skip
    """Controls: every pair of C++ seeds passes the parity tests; the JAX
    replica run at 1.1 T fails the KS test against pooled C++ with
    p < MD_PARITY_CONTROL_P for both observables (and |z| > MD_PARITY_Z)."""
    print("", flush=True)
    pairs = list(itertools.combinations(sorted(cxx), 2))
    for a, b in pairs:
        for k in OBS:
            r = _compare(cxx[a][k], MD_PARITY_BLOCK, cxx[b][k], MD_PARITY_BLOCK)
            _report(f"{k} C++ seed {a} vs seed {b}", r)
            assert r["p"] > MD_PARITY_KS_PVALUE and abs(r["z"]) < MD_PARITY_Z, r
    hot = _jax_vs(replica, "hot", pooled, MD_PARITY_BLOCK)
    for k, r in hot.items():
        _report(f"{k} JAX {REP.CONTROL_FACTOR} T vs C++", r)
        assert r["p"] < MD_PARITY_CONTROL_P, (k, r)
        assert abs(r["z"]) > MD_PARITY_Z, (k, r)
    assert len(pairs) == math.comb(len(cxx), 2) >= 1, "seed pairs tested"


# ============================================================
# LIVE REPLICA FROM C++ CONFIGURATIONS
# ============================================================


@pytest.mark.slow
@pytest.mark.high
def test_live_replica_from_cxx_configurations(
    ref: Dict[str, Any], pooled: Dict[str, np.ndarray]
) -> None:
    """Live CPU run of the replica's sample stage (one Andersen 5e4 + NVE 5e4
    cycle) from each stored C++ configuration with fresh MB velocities, i.e.
    a canonical start: its Phi and KE must pass the parity tests against
    pooled C++ (one sample per lane, so lanes are the independent groups)."""
    with np.load(CXX_NPZ) as z:
        s, basis = np.asarray(z["s"]), np.asarray(z["basis"])
    system = REP.build_system(ref)
    proto = dataclasses.replace(REP.protocol_from_reference(ref, 1), equi_segments=0)
    r0 = s @ basis
    rng = np.random.default_rng(LIVE_SEED)
    v0 = math.sqrt(system.temperature) * rng.standard_normal(r0.shape)
    seeds = [LIVE_SEED + i for i in range(r0.shape[0])]
    temps = np.full(r0.shape[0], system.temperature)
    lanes = REP.init_lanes(system, r0, v0, temps, seeds, REP.settings_pair(proto)[0])
    keys = jnp.stack([jax.random.PRNGKey(sd) for sd in seeds])  # unused: no redraw
    res = REP.run_protocol(system, proto, lanes, keys)
    x = _norm(ref, res["phi"].ravel(), res["ke"].ravel())
    print("", flush=True)
    for k in OBS:
        r = _compare(x[k][:, None], 1, pooled[k], MD_PARITY_BLOCK)
        _report(f"{k} live JAX ({r0.shape[0]} lanes) vs C++", r)
        assert r["p"] > MD_PARITY_KS_PVALUE and abs(r["z"]) < MD_PARITY_Z, (k, r)
    assert np.array_equal(np.asarray(basis), np.asarray(ref["basis"]))
