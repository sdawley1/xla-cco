#!/usr/bin/env python3

"""
Integration tests of the MD driver CLI (``xla_cco.cli.md``).

The protocol, outputs and resume logic run against ``md_states.FakeDriver``
(monkeypatched into ``xla_cco.dynamics.run`` and ``thermal``); the
end-to-end test runs the real driver on relaxed ground states.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ----- numerics -----
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from md_states import (
    FakeDriver,
    fake_begin_segment,
    fake_harmonic_sample,
    fake_init_state,
    fake_make_waves,
    fake_maxwell_boltzmann,
)
from systems import small_system

# ----- project -----
from xla_cco.cli import md
from xla_cco.cli import relax as relax_cli
from xla_cco.dynamics import run as md_run
from xla_cco.dynamics import thermal
from xla_cco.minimize.flow import FlowSettings, gradient_flow

# ============================================================
# CONSTANTS
# ============================================================

N, D, SIDE = 16, 2, 4.0
K = 2.0 * math.pi / SIDE * 1.5  # |n| < 1.5: n = (1,0), (0,1), (1,1), (1,-1)
M = 4
TEMPS = ["1e-4", "1e-3"]
# default _argv protocol: equilibration 6 = 3 + 3, Andersen 4, NVE 10, 2 cycles
FULL = [("andersen", 3), ("andersen", 3), ("andersen", 4), ("verlet", 10),
        ("andersen", 4), ("verlet", 10)]  # fmt: skip
EXTENDED = FULL[:2] + FULL[:2] + FULL[2:]
NPZ_KEYS = {
    *(f"snap_{k}" for k in md.SNAP_FIELDS),
    *md.RECORD_FIELDS,
    *(f"cycle_{k}" for k in md.RECORD_FIELDS),
    "r_final", "v_final", "temperatures", "indices", "lane_index",
    "lane_temperature", "basis", "waves_n_vectors", "waves_kvec",
    "waves_constrained", "record_steps", "settings", "equil_gate_first",
    "equil_gate", "equil_gate_pass", "equil_extended",
}  # fmt: skip
JSON_KEYS = {
    "N", "d", "M", "K", "k_track", "chi", "n_waves", "n_constrained", "input",
    "input_sha256", "indices", "temperatures", "n_lanes", "args", "settings",
    "phases", "n_record_steps", "status_counts", "n_running", "gate_tol",
    "equil_gate", "equil_gate_pass", "equil_extended", "phase_seconds",
    "wall_seconds", "npz_keys", "device", "jax_version", "resumed",
}  # fmt: skip
# every fingerprint field: (arguments of the first run, changed arguments)
RHO = ["--record-rho-every", "2", "--max-rho-records", "3"]
BAOAB = ["--scheme", "baoab", "--gamma", "0.1"]
CHANGES: Dict[str, Tuple[List[str], List[str]]] = {
    "indices": ([], ["--indices", "0", "1"]),
    "temperature": ([], ["--temperature", "1e-4", "2e-3"]),
    "dt": ([], ["--dt", "0.02"]),
    "n_equil": ([], ["--n-equil", "8"]),
    "n_andersen": ([], ["--n-andersen", "5"]),
    "n_nve": ([], ["--n-nve", "11"]),
    "cycles": ([], ["--cycles", "3"]),
    "andersen_p": ([], ["--andersen-p", "0.02"]),
    "andersen_mode": ([], ["--andersen-mode", "bernoulli"]),
    "scheme": ([], BAOAB),
    "gamma": (BAOAB, ["--gamma", "0.2"]),
    "wind_every": ([], ["--wind-every", "2"]),
    "n_snapshots": ([], ["--n-snapshots", "9"]),
    "chunk_steps": ([], ["--chunk-steps", "6"]),
    "seed": ([], ["--seed", "1"]),
    "k_outer": ([], ["--k-outer", repr(1.01 * K)]),  # same M = 4
    "k_track": ([], ["--k-track", repr(2.5 * K)]),
    "record_rho_every": (RHO, ["--record-rho-every", "3"]),
    "max_rho_records": (RHO, ["--max-rho-records", "4"]),
    "phase_floor": ([], ["--phase-floor", "1e-12"]),
    "drift_tol": ([], ["--drift-tol", "1e-4"]),
    "k_chunk": ([], ["--k-chunk", "2"]),
    "input_sha256": ([], []),  # the input file's content changes instead
}

# ============================================================
# HELPERS
# ============================================================


def _unit(step: np.ndarray) -> np.ndarray:
    """Phi = M T at every step: the equilibration gate passes exactly."""
    return np.ones(np.shape(step))


def _driver(**kw: Any) -> FakeDriver:
    """A fake driver whose Phi passes the gate unless told otherwise."""
    kw.setdefault("phi_factor", _unit)
    return FakeDriver(m=M, **kw)


def _write_input(
    root: Path, b: int = 3, meta: Optional[Dict[str, Any]] = None, seed: int = 3,
    **over: Any,
) -> Path:  # fmt: skip
    """flows_chi1-10.npz (+ .json) with B random states in a 4 x 4 cell."""
    rng = np.random.default_rng(seed)
    arrays = {"s_final": rng.uniform(size=(b, N, D)), "basis": SIDE * np.eye(D)}
    arrays.update(over)
    path = root / "flows_chi1-10.npz"
    np.savez(path, **{k: v for k, v in arrays.items() if v is not None})
    if meta is not False:
        text = json.dumps({"K": K, "M": M} if meta is None else meta)
        path.with_suffix(".json").write_text(text)
    return path


def _argv(root: Path, *extra: str, save: str = "out") -> List[str]:
    base = [
        "--input", str(root / "flows_chi1-10.npz"), "--indices", "0", "2",
        "--temperature", *TEMPS, "--dt", "0.01", "--n-equil", "6",
        "--n-andersen", "4", "--n-nve", "10", "--cycles", "2",
        "--n-snapshots", "8", "--chunk-steps", "5", "--device", "cpu",
        "--save", str(root / save),
    ]  # fmt: skip
    return base + list(extra)


class _Counter:
    """Wraps a fake; logs its arguments, results and (optionally) rng states."""

    def __init__(self, fn: Any) -> None:
        self.fn = fn
        self.args: List[Tuple[Any, ...]] = []
        self.results: List[Any] = []
        self.rng_states: List[str] = []

    @property
    def n(self) -> int:
        return len(self.args)

    def __call__(self, *a: Any, **k: Any) -> Any:
        if a and isinstance(a[0], np.random.Generator):
            self.rng_states.append(json.dumps(a[0].bit_generator.state))
        self.args.append(a)
        out = self.fn(*a, **k)
        self.results.append(np.asarray(out))
        return out


def _install(mp: pytest.MonkeyPatch, driver: FakeDriver) -> Dict[str, _Counter]:
    """Monkeypatch the fake driver into run / thermal; return call loggers."""
    counters = {
        "harmonic": _Counter(fake_harmonic_sample),
        "begin": _Counter(fake_begin_segment),
        "mb": _Counter(fake_maxwell_boltzmann),
        "init": _Counter(fake_init_state),
    }
    mp.setattr(md_run, "make_waves", fake_make_waves)
    mp.setattr(md_run, "init_state", counters["init"])
    mp.setattr(md_run, "begin_segment", counters["begin"])
    mp.setattr(md_run, "simulate", driver.simulate)
    mp.setattr(thermal, "harmonic_sample", counters["harmonic"])
    mp.setattr(thermal, "maxwell_boltzmann", counters["mb"])
    return counters


def _load(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def _phases(**kw: Any) -> List[md.Phase]:
    ns = md.build_parser().parse_args(
        ["--input", "x", "--temperature", "1", "--dt", "1", "--save", "y"]
        + [f"--{k.replace('_', '-')}={v}" for k, v in kw.items()]
    )
    return md.protocol_phases(ns)


def _key_bytes(key: Any) -> bytes:
    return np.asarray(key).tobytes()


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> Any:
    driver = _driver()
    return driver, _install(monkeypatch, driver)


# ============================================================
# HELP, USAGE AND DEVICE
# ============================================================


@pytest.mark.low
def test_md_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert md.main(["--help"]) == 0
    out = capsys.readouterr().out
    for flag in ("--k-track", "--checkpoint", "--scheme", "--n-snapshots"):
        assert flag in out
    proc = subprocess.run(
        [sys.executable, "-m", "xla_cco.cli.md", "--help"],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert proc.returncode == 0 and "--temperature" in proc.stdout


@pytest.mark.high
@pytest.mark.parametrize(
    "argv",
    [[], ["--dt"], ["--input", "x", "--temperature", "1", "--dt", "abc",
                    "--save", "y"],
     ["--input", "x", "--temperature", "1", "--dt", "1", "--save", "y",
      "--scheme", "verlet"]],
)  # fmt: skip
def test_regression_D041_usage_error_exits_1(
    argv: List[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """D-041: argparse usage errors returned 2 instead of 1."""
    assert md.main(argv) == 1
    assert "usage" in capsys.readouterr().err


@pytest.mark.low
@pytest.mark.parametrize("input_exists", [True, False])
def test_md_cli_gpu_guard_mocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake: Any,
    capsys: pytest.CaptureFixture[str], input_exists: bool,
) -> None:  # fmt: skip
    # the device is selected before the input is read: a missing input file
    # must still report the GPU problem, not the file
    driver, counters = fake
    if input_exists:
        _write_input(tmp_path)

    def no_gpu() -> None:
        raise RuntimeError("no GPU visible: the job probably lacks --gres=gpu")

    monkeypatch.setattr(relax_cli, "require_gpu", no_gpu)
    argv = _argv(tmp_path)
    argv[argv.index("--device") + 1] = "gpu"
    assert md.main(argv) == 1
    err = capsys.readouterr().err
    assert "ERROR" in err and "--gres=gpu" in err and "No such file" not in err
    assert driver.calls == [] and counters["harmonic"].n == 0
    assert not (tmp_path / "out.npz").exists()


@pytest.mark.low
def test_md_cli_gpu_guard_subprocess(tmp_path: Path) -> None:
    # CPU-only JAX: the real guard must name the cause and exit 1
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    env["JAX_PLATFORMS"] = "cpu"
    argv = _argv(tmp_path)
    argv[argv.index("--device") + 1] = "gpu"
    proc = subprocess.run(
        [sys.executable, "-m", "xla_cco.cli.md", *argv],
        capture_output=True, text=True, env=env, check=False,
    )  # fmt: skip
    assert proc.returncode == 1
    assert "ERROR" in proc.stderr and "JAX_PLATFORMS=cpu" in proc.stderr


# ============================================================
# VALIDATION
# ============================================================


@pytest.mark.medium
@pytest.mark.parametrize(
    "extra, flag",
    [
        (["--dt", "0"], "--dt"), (["--dt", "nan"], "--dt"),
        (["--temperature", "0"], "--temperature"),
        (["--temperature", "1e-3", "1e-3"], "--temperature"),
        (["--n-nve", "0"], "--n-nve"), (["--cycles", "0"], "--cycles"),
        (["--n-equil", "-1"], "--n-equil"), (["--andersen-p", "0"], "--andersen-p"),
        (["--andersen-p", "1.5"], "--andersen-p"), (["--gamma", "0.1"], "--gamma"),
        (["--scheme", "baoab"], "--gamma"), (["--n-snapshots", "257"], "--n-snap"),
        (["--n-snapshots", "0"], "--n-snap"), (["--wind-every", "0"], "--wind"),
        (["--record-rho-every", "5"], "--record-rho-every"),
        (["--indices", "0", "0"], "--indices"), (["--indices", "-1"], "--indices"),
        (["--k-track", "-1"], "--k-track"), (["--chunk-steps", "0"], "--chunk"),
        (["--n-nve", str(2**31)], "steps exceed"),
    ],
)  # fmt: skip
def test_md_cli_rejects_bad_arguments(
    tmp_path: Path, fake: Any, capsys: pytest.CaptureFixture[str],
    extra: List[str], flag: str,
) -> None:  # fmt: skip
    driver, counters = fake
    _write_input(tmp_path)
    assert md.main(_argv(tmp_path, *extra)) == 1
    err = capsys.readouterr().err
    assert "ERROR" in err and flag in err
    assert driver.calls == [] and counters["harmonic"].n == 0


@pytest.mark.medium
@pytest.mark.parametrize(
    "case, match",
    [
        ("float32", "float64"), ("no_basis", "'basis'"), ("basis_shape", "basis"),
        ("density", "density 1"), ("nan", "non-finite"), ("index", "out of range"),
        ("no_json", "--k-outer"), ("m_mismatch", "M = 5"),
        ("k_track", "--k-track"), ("missing", "No such file"),
    ],
)  # fmt: skip
def test_md_cli_rejects_bad_input(
    tmp_path: Path, fake: Any, capsys: pytest.CaptureFixture[str], case: str,
    match: str,
) -> None:  # fmt: skip
    driver, counters = fake
    s = np.random.default_rng(0).uniform(size=(3, N, D))
    s_nan = s.copy()
    s_nan[1, 2, 0] = np.nan
    over: Dict[str, Any] = {
        "float32": {"s_final": s.astype(np.float32)},
        "no_basis": {"basis": None}, "basis_shape": {"basis": SIDE * np.eye(3)},
        "density": {"basis": 5.0 * np.eye(D)}, "nan": {"s_final": s_nan},
    }.get(case, {})  # fmt: skip
    meta: Any = {"no_json": False, "m_mismatch": {"K": K, "M": 5}}.get(case)
    _write_input(tmp_path, meta=meta, **over)
    extra = {"index": ["--indices", "3"], "k_track": ["--k-track", "1.0"]}
    argv = _argv(tmp_path, *extra.get(case, []))
    if case == "missing":
        argv[1] = str(tmp_path / "absent.npz")
    assert md.main(argv) == 1
    err = capsys.readouterr().err
    assert "ERROR" in err and match in err
    assert driver.calls == [] and counters["harmonic"].n == 0


# ============================================================
# SCHEDULE AND SETTINGS
# ============================================================


@pytest.mark.high
def test_record_schedule_example() -> None:
    phases = _phases(n_equil=10, n_andersen=5, n_nve=100, cycles=2)
    # 1, 10, 100 after each NVE start (15 and 120)
    assert md.record_schedule(phases, 6) == (16, 25, 115, 121, 130, 220)
    # a segment shorter than its share records every step
    short = _phases(n_equil=0, n_andersen=0, n_nve=3, cycles=1)
    assert md.record_schedule(short, 200) == (1, 2, 3)


@pytest.mark.high
def test_regression_D045_default_snapshot_count() -> None:
    """D-045: the default 200 snapshots deduplicated to 184 record steps."""
    default = _phases(n_equil=200_000, n_andersen=100_000, n_nve=1_000_000)
    steps = md.record_schedule(default, 200)
    assert len(steps) == 200 and len(set(steps)) == 200
    assert len(md.record_schedule(_phases(n_nve=50, cycles=1), 40)) == 40
    assert len(md.record_schedule(_phases(n_nve=7, cycles=2), 20)) == 14


@pytest.mark.high
@settings(max_examples=200)
@given(length=st.integers(1, 10**7), n=st.integers(0, md.MAX_SNAPSHOTS))
@example(length=3, n=200)
@example(length=1_000_000, n=100)
@example(length=257, n=256)
def test_segment_offsets_properties(length: int, n: int) -> None:
    out = md.segment_offsets(length, n)
    count = min(n, length)
    assert out.shape == (count,) and out.dtype == np.int64
    if count == 0:
        return
    assert np.all(np.diff(out) > 0)  # distinct and sorted
    assert out[-1] == length and out[0] == (1 if count > 1 else length)
    if count >= 3:
        # log spacing: the middle point lies far below the arithmetic middle
        assert out[count // 2] <= max(count, (length + 1) // 2)


@pytest.mark.high
@settings(max_examples=150)
@given(
    n_equil=st.integers(0, 50), n_andersen=st.integers(0, 20),
    n_nve=st.integers(1, 5000), cycles=st.integers(1, 4),
    n_snap=st.integers(1, md.MAX_SNAPSHOTS), baoab=st.booleans(),
)  # fmt: skip
@example(n_equil=0, n_andersen=0, n_nve=3, cycles=1, n_snap=200, baoab=False)
@example(n_equil=2, n_andersen=1, n_nve=2, cycles=4, n_snap=256, baoab=False)
@example(n_equil=4, n_andersen=3, n_nve=5000, cycles=2, n_snap=200, baoab=False)
def test_record_schedule_properties(
    n_equil: int, n_andersen: int, n_nve: int, cycles: int, n_snap: int,
    baoab: bool,
) -> None:  # fmt: skip
    scheme = "baoab" if baoab else "andersen-nve"
    phases = _phases(n_equil=n_equil, n_andersen=n_andersen, n_nve=n_nve,
                     cycles=cycles, scheme=scheme)  # fmt: skip
    steps = md.record_schedule(phases, n_snap)
    arr = np.asarray(steps)
    assert all(isinstance(x, int) for x in steps)
    assert 1 <= arr.size <= n_snap <= md.MAX_SNAPSHOTS
    assert np.all(np.diff(arr) > 0)  # sorted and unique
    starts = np.cumsum([0] + [ph.n_steps for ph in phases])
    segs = [(starts[i], starts[i + 1]) for i, ph in enumerate(phases) if ph.cycle >= 0]
    inside = np.zeros(arr.size, dtype=bool)
    expected = 0
    for c, (lo, hi) in enumerate(segs):
        inside |= (arr > lo) & (arr <= hi)
        share = n_snap // len(segs) + (1 if c < n_snap % len(segs) else 0)
        expected += min(share, hi - lo)
        if share >= 1:
            assert hi in steps  # every recorded segment ends with a snapshot
        if share >= 2:
            assert lo + 1 in steps  # and starts one step in (log spacing)
    assert inside.all() and arr.size == expected  # exactly the requested count


@pytest.mark.high
@pytest.mark.parametrize("scheme", ["andersen-nve", "baoab"])
def test_settings_mapping(tmp_path: Path, scheme: str) -> None:
    extra = ["--scheme", scheme, "--andersen-p", "0.02", "--wind-every", "3",
             "--record-rho-every", "2", "--max-rho-records", "7",
             "--phase-floor", "1e-12", "--drift-tol", "1e-4",
             "--andersen-mode", "bernoulli"]  # fmt: skip
    if scheme == "baoab":
        extra += ["--gamma", "0.03"]
    args = md.build_parser().parse_args(_argv(tmp_path, *extra))
    md.validate_args(args)
    steps = md.record_schedule(md.protocol_phases(args), args.n_snapshots)
    got = md.make_settings(args, steps)
    want = {"andersen-nve": {"andersen", "verlet"}, "baoab": {"baoab"}}[scheme]
    assert set(got) == want
    for name, s in got.items():
        assert s.scheme == name and s.record_steps == steps
        assert (s.dt, s.andersen_p, s.andersen_mode) == (0.01, 0.02, "bernoulli")
        assert s.gamma == (0.03 if scheme == "baoab" else 0.0)
        assert (s.wind_every, s.chunk_steps, s.remove_com) == (3, 5, True)
        assert (s.record_rho_every, s.max_rho_records) == (2, 7)
        assert (s.phase_floor, s.drift_tol) == (1e-12, 1e-4)


# ============================================================
# PROTOCOL AND OUTPUTS (fake driver)
# ============================================================


@pytest.mark.high
@pytest.mark.parametrize(
    "extra, calls, n_begin",
    [
        ([], FULL, 3),
        (BAOAB, [("baoab", 3), ("baoab", 3), ("baoab", 10), ("baoab", 10)], 3),
        (["--n-equil", "0", "--n-andersen", "0", "--cycles", "3"],
         [("verlet", 10)] * 3, 3),
        (["--n-equil", "1"], [("andersen", 1)] + FULL[2:], 3),
    ],
)  # fmt: skip
def test_md_cli_protocol(
    tmp_path: Path, fake: Any, extra: List[str], calls: List[Any], n_begin: int
) -> None:
    driver, counters = fake
    _write_input(tmp_path)
    assert md.main(_argv(tmp_path, *extra)) == 0
    assert driver.calls == calls
    assert counters["begin"].n == n_begin and counters["harmonic"].n == 4
    # every begin_segment resets both the sums and the winding
    assert all(a[2:] == (True, True) for a in counters["begin"].args)


@pytest.mark.critical
def test_md_cli_winding_reset_each_segment(tmp_path: Path, fake: Any) -> None:
    # the fake adds 1 to n_valid per step, so a segment that starts with
    # reset winding ends with n_valid == n_nve
    _write_input(tmp_path)
    assert md.main(_argv(tmp_path, "--cycles", "3")) == 0
    out = _load(tmp_path / "out.npz")
    assert out["cycle_n_valid"].shape[:2] == (3, 4)
    np.testing.assert_array_equal(out["cycle_n_valid"], 10)
    np.testing.assert_array_equal(out["cycle_n_sum"], np.full((3, 4), 10))


@pytest.mark.critical
def test_md_cli_lane_streams_are_independent(tmp_path: Path, fake: Any) -> None:
    _write_input(tmp_path)
    _, counters = fake
    assert md.main(_argv(tmp_path)) == 0
    n_lanes = 4
    states = counters["harmonic"].rng_states
    mb_keys = [_key_bytes(a[0]) for a in counters["mb"].args]
    lane_keys = [_key_bytes(a[6]) for a in counters["init"].args]
    assert len(set(states)) == len(states) == n_lanes  # one seed per lane
    assert len(set(mb_keys)) == len(mb_keys) == n_lanes
    assert len(set(lane_keys)) == len(lane_keys) == n_lanes
    assert not set(mb_keys) & set(lane_keys)  # lane key != velocity key


@pytest.mark.critical
def test_md_cli_identical_lanes_draw_different_starts(
    tmp_path: Path, fake: Any
) -> None:
    # two copies of one ground state at one temperature: different r0 and v0
    s = np.random.default_rng(5).uniform(size=(1, N, D))
    _write_input(tmp_path, s_final=np.concatenate([s, s]))
    _, counters = fake
    argv = _argv(tmp_path, "--indices", "0", "1", "--temperature", "1e-3")
    assert md.main(argv) == 0
    r0, v0 = counters["harmonic"].results, counters["mb"].results
    assert len(r0) == len(v0) == 2
    assert not np.array_equal(r0[0], r0[1]) and not np.array_equal(v0[0], v0[1])


@pytest.mark.medium
def test_md_cli_outputs(tmp_path: Path, fake: Any) -> None:
    _write_input(tmp_path)
    assert md.main(_argv(tmp_path)) == 0
    out = _load(tmp_path / "out.npz")
    meta = json.loads((tmp_path / "out.json").read_text())
    assert set(out) == NPZ_KEYS and set(meta) >= JSON_KEYS
    assert meta["npz_keys"] == sorted(out) and meta["resumed"] is False
    n_lanes, w = 4, out["waves_n_vectors"].shape[0]
    np.testing.assert_array_equal(out["lane_index"], [0, 0, 2, 2])
    np.testing.assert_array_equal(out["lane_temperature"], [1e-4, 1e-3] * 2)
    np.testing.assert_array_equal(out["indices"], [0, 2])
    assert out["cycle_wind_abs"].shape == (2, n_lanes, w)
    assert out["wind_abs"].shape == (n_lanes, w) and out["status"].shape == (n_lanes,)
    assert int(out["waves_constrained"].sum()) == M == meta["M"]
    steps = md.record_schedule(_phases(n_equil=6, n_andersen=4, n_nve=10), 8)
    np.testing.assert_array_equal(out["record_steps"], steps)
    # the fake writes a snapshot at every record step it passes
    np.testing.assert_array_equal(out["snap_step"], np.tile(steps, (n_lanes, 1)))
    assert json.loads(str(out["settings"]))["record_steps"] == list(steps)
    assert meta["status_counts"] == {"STATUS_RUNNING": n_lanes}
    assert meta["K"] == K and len(meta["input_sha256"]) == 64
    assert out["equil_gate_pass"].all() and not bool(out["equil_extended"])
    np.testing.assert_array_equal(out["cycle_wind_abs"][-1], out["wind_abs"])


# ============================================================
# EQUILIBRATION GATE (fake driver)
# ============================================================


@pytest.mark.high
def test_regression_D044_gate_excludes_transient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-044: the gate averaged over the whole equilibration (transient)."""
    # Phi = 10 M T in the first half of equilibration, M T afterwards
    driver = _driver(phi_factor=lambda s: np.where(s <= 3, 10.0, 1.0))
    _install(monkeypatch, driver)
    _write_input(tmp_path)
    assert md.main(_argv(tmp_path)) == 0
    out = _load(tmp_path / "out.npz")
    assert driver.calls == FULL and not bool(out["equil_extended"])
    assert out["equil_gate_pass"].all()


@pytest.mark.high
@pytest.mark.parametrize("settles", [True, False])
def test_gate_failure_extends_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settles: bool
) -> None:
    # Phi = 1.2 M T through the first equilibration; afterwards M T (settles)
    # or still 1.2 M T (never passes): extended once, never aborted
    driver = _driver(phi_factor=lambda s: np.where((s <= 6) | (not settles), 1.2, 1.0))
    _install(monkeypatch, driver)
    _write_input(tmp_path)
    assert md.main(_argv(tmp_path)) == 0
    out = _load(tmp_path / "out.npz")
    meta = json.loads((tmp_path / "out.json").read_text())
    assert driver.calls == EXTENDED and bool(out["equil_extended"])
    assert not md.gate_pass(out["equil_gate_first"]).any()
    assert out["equil_gate_pass"].all() == settles
    assert not out["equil_gate_pass"].any() or settles
    assert len(meta["equil_gate"]) == 2 and meta["equil_extended"] is True
    # production moved by n_equil = 6 steps; snapshots follow
    steps = np.asarray(md.record_schedule(_phases(n_equil=6, n_andersen=4,
                                                  n_nve=10), 8)) + 6  # fmt: skip
    np.testing.assert_array_equal(out["record_steps"], steps)
    np.testing.assert_array_equal(out["snap_step"], np.tile(steps, (4, 1)))


# ============================================================
# CHECKPOINT AND RESUME (fake driver)
# ============================================================


def _reference(root: Path, mp: pytest.MonkeyPatch, *extra: str, **kw: Any) -> Any:
    """Outputs of an uninterrupted checkpointed run in ``root``."""
    root.mkdir()
    _write_input(root)
    _install(mp, _driver(**kw))
    assert md.main(_argv(root, *extra, "--checkpoint", str(root / "c.npz"))) == 0
    return _load(root / "out.npz")


def _crash_and_resume(
    root: Path, mp: pytest.MonkeyPatch, crash_at: int, inphase: bool,
    *extra: str, **kw: Any,
) -> Tuple[FakeDriver, FakeDriver]:  # fmt: skip
    """Crash at simulate call ``crash_at``, then resume; returns both drivers."""
    root.mkdir()
    _write_input(root)
    argv = _argv(root, *extra, "--checkpoint", str(root / "c.npz"))
    crashing = _driver(crash_at=crash_at, inphase=inphase, **kw)
    _install(mp, crashing)
    assert md.main(argv) == 1
    assert (root / "c.npz").exists()  # written before the first phase
    assert md.inphase_path(root / "c.npz").exists() == inphase
    resumed = _driver(**kw)
    _install(mp, resumed)
    assert md.main(argv) == 0
    assert not md.inphase_path(root / "c.npz").exists()
    return crashing, resumed


def _assert_same_outputs(ref: Dict[str, np.ndarray], root: Path) -> None:
    got = _load(root / "out.npz")
    assert set(ref) == set(got)
    for k in ref:
        assert ref[k].tobytes() == got[k].tobytes(), k


@pytest.mark.critical
@pytest.mark.parametrize(
    "crash_at, inphase",
    [(0, False), (0, True), (1, True), (2, False), (3, True), (5, True)],
)
def test_md_cli_resume_after_crash_is_bitwise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crash_at: int, inphase: bool
) -> None:
    ref = _reference(tmp_path / "ref", monkeypatch)
    crashing, resumed = _crash_and_resume(
        tmp_path / "run", monkeypatch, crash_at, inphase
    )
    first = FULL[crash_at]
    if inphase:
        first = (first[0], first[1] - first[1] // 2)
    assert resumed.calls == [first] + FULL[crash_at + 1 :]
    # simulate is never handed a path that already holds a file
    assert not any(crashing.existed) and not any(resumed.existed)
    _assert_same_outputs(ref, tmp_path / "run")
    meta = json.loads((tmp_path / "run" / "out.json").read_text())
    assert meta["resumed"] is True


@pytest.mark.critical
@pytest.mark.parametrize("crash_at, inphase", [(2, False), (3, True), (4, False)])
def test_md_cli_resume_after_extension_is_bitwise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crash_at: int, inphase: bool
) -> None:
    # the gate fails once: the extended schedule must survive a restart
    def factor(s: np.ndarray) -> np.ndarray:
        return np.where(s <= 6, 1.2, 1.0)

    ref = _reference(tmp_path / "ref", monkeypatch, phi_factor=factor)
    _, resumed = _crash_and_resume(
        tmp_path / "run", monkeypatch, crash_at, inphase, phi_factor=factor
    )
    first = EXTENDED[crash_at]
    if inphase:
        first = (first[0], first[1] - first[1] // 2)
    assert resumed.calls == [first] + EXTENDED[crash_at + 1 :]
    _assert_same_outputs(ref, tmp_path / "run")


@pytest.mark.high
def test_md_cli_resume_twice_inside_one_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # two in-phase crashes in the same phase add up their progress
    ref = _reference(tmp_path / "ref", monkeypatch)
    root = tmp_path / "run"
    root.mkdir()
    _write_input(root)
    argv = _argv(root, "--checkpoint", str(root / "c.npz"))
    for driver in (_driver(crash_at=3, inphase=True),
                   _driver(crash_at=0, inphase=True)):  # fmt: skip
        _install(monkeypatch, driver)
        assert md.main(argv) == 1
    resumed = _driver()
    _install(monkeypatch, resumed)
    assert md.main(argv) == 0
    # phase 3 (NVE, 10 steps): 5 done, then 5 // 2 = 2 more, 3 left
    assert resumed.calls == [("verlet", 3)] + FULL[4:]
    _assert_same_outputs(ref, root)


@pytest.mark.medium
def test_md_cli_ignores_stale_inphase_checkpoint(tmp_path: Path, fake: Any) -> None:
    driver, _ = fake
    _write_input(tmp_path)
    stale = md.inphase_path(tmp_path / "c.npz")
    stale.write_bytes(b"left over by an earlier run")
    assert md.main(_argv(tmp_path, "--checkpoint", str(tmp_path / "c.npz"))) == 0
    assert not stale.exists() and driver.calls == FULL
    assert not any(driver.existed)


@pytest.mark.high
def test_regression_D046_simulate_never_gets_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-046: a stale in-phase file stayed at the path handed to simulate."""
    # a simulate that writes its checkpoint at the end of every call: the CLI
    # removes it once the phase checkpoint is written, so the next call (and
    # a later resume) never finds a file there
    ref = _reference(tmp_path / "ref", monkeypatch)
    root = tmp_path / "run"
    root.mkdir()
    _write_input(root)
    driver = _driver(always_write=True)
    _install(monkeypatch, driver)
    assert md.main(_argv(root, "--checkpoint", str(root / "c.npz"))) == 0
    assert driver.calls == FULL and not any(driver.existed)
    assert not md.inphase_path(root / "c.npz").exists()
    _assert_same_outputs(ref, root)


@pytest.mark.medium
def test_md_cli_inphase_checkpoint_behind_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # an in-phase file older than the phase checkpoint must not roll back
    _write_input(tmp_path)
    ckpt = tmp_path / "c.npz"
    argv = _argv(tmp_path, "--checkpoint", str(ckpt))
    _install(monkeypatch, _driver(crash_at=1))
    assert md.main(argv) == 1  # checkpoint after phase 0
    shutil.copy(ckpt, tmp_path / "old.npz")
    _install(monkeypatch, _driver(crash_at=1))
    assert md.main(argv) == 1  # phase 1 done, crash in phase 2
    shutil.copy(tmp_path / "old.npz", md.inphase_path(ckpt))
    resumed = _driver()
    _install(monkeypatch, resumed)
    assert md.main(argv) == 0
    assert resumed.calls == FULL[2:] and not md.inphase_path(ckpt).exists()


@pytest.mark.medium
@pytest.mark.parametrize("field", sorted(CHANGES))
def test_md_cli_resume_rejects_any_changed_setting(
    tmp_path: Path, fake: Any, capsys: pytest.CaptureFixture[str], field: str
) -> None:
    base, change = CHANGES[field]
    _write_input(tmp_path)
    ckpt = ["--checkpoint", str(tmp_path / "c.npz")]
    assert md.main(_argv(tmp_path, *base, *ckpt)) == 0
    capsys.readouterr()
    if field == "input_sha256":
        _write_input(tmp_path, seed=4)
    assert md.main(_argv(tmp_path, *base, *change, *ckpt)) == 1
    assert "other settings" in capsys.readouterr().err


@pytest.mark.medium
def test_fingerprint_cases_cover_every_field() -> None:
    assert set(CHANGES) == set(md.FINGERPRINT_ARGS) | {"input_sha256"}
    assert "k_chunk" in md.FINGERPRINT_ARGS and "dt" in md.FINGERPRINT_ARGS


@pytest.mark.medium
@pytest.mark.parametrize("damage", ["truncate", "garbage", "no_next_phase"])
def test_md_cli_damaged_checkpoint_exits_1(
    tmp_path: Path, fake: Any, capsys: pytest.CaptureFixture[str], damage: str
) -> None:
    _write_input(tmp_path)
    ckpt = tmp_path / "c.npz"
    argv = _argv(tmp_path, "--checkpoint", str(ckpt))
    assert md.main(argv) == 0
    capsys.readouterr()
    raw = ckpt.read_bytes()
    if damage == "truncate":
        ckpt.write_bytes(raw[: len(raw) // 2])
    elif damage == "garbage":
        ckpt.write_bytes(b"\x00garbage" * 50)
    else:
        arrays = _load(ckpt)
        meta = json.loads(str(arrays["__meta__"]))
        del meta["next_phase"]
        arrays["__meta__"] = np.asarray(json.dumps(meta))
        np.savez(ckpt, **arrays)
    assert md.main(argv) == 1
    err = capsys.readouterr().err
    match = "next_phase" if damage == "no_next_phase" else "truncated or corrupt"
    assert "ERROR" in err and match in err


# ============================================================
# END TO END (real driver)
# ============================================================


def _ground_states(b: int = 3, seed: int = 3) -> np.ndarray:
    """(b, N, d) fractional ground states (gradient flow from _write_input's starts).

    The real driver aborts NVE lanes whose energy drifts by more than
    drift_tol * dof T / 2; far from a minimum (random states) the Verlet
    energy error exceeds that at T = 1e-4, so the end-to-end run starts from
    ground states, as production does.
    """
    box, _, energy = small_system(N, D, K)
    starts = np.random.default_rng(seed).uniform(size=(b, N, D))
    fs = FlowSettings(energy_tol=1e-20, max_steps=100_000)
    out = np.stack([np.asarray(gradient_flow(energy, s, box.basis, fs).s)
                    for s in jnp.asarray(starts)])  # fmt: skip
    assert out.shape == (b, N, D) and np.all(np.isfinite(out)), "bad flow"
    return out


@pytest.mark.high
@pytest.mark.slow
def test_md_cli_end_to_end_small(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref_dir, run_dir = tmp_path / "ref", tmp_path / "run"
    s_final = _ground_states()
    for root in (ref_dir, run_dir):
        root.mkdir()
        _write_input(root, s_final=s_final)
    ref_args = md.build_parser().parse_args(_argv(ref_dir))
    summary = md.run(ref_args)
    ref = _load(ref_dir / "out.npz")
    assert set(ref) >= NPZ_KEYS and summary["n_lanes"] == 4
    assert summary["status_counts"] == {"STATUS_RUNNING": 4}
    assert np.all(ref["snap_count"] == len(ref["record_steps"]))
    assert np.all(ref["wind_abs"] >= np.abs(ref["wind_net"]))
    assert np.all(np.isfinite(ref["equil_gate"]))
    # crash in the fifth simulate call (a later Andersen phase), then resume
    real = md_run.simulate
    calls: List[int] = []

    def crash_once(*a: Any, **k: Any) -> Any:
        calls.append(1)
        if len(calls) == 5:
            raise RuntimeError("simulated crash")
        return real(*a, **k)

    argv = _argv(run_dir, "--checkpoint", str(run_dir / "c.npz"))
    monkeypatch.setattr(md_run, "simulate", crash_once)
    assert md.main(argv) == 1
    assert md.main(argv) == 0
    got = _load(run_dir / "out.npz")
    for k in ref:
        assert ref[k].tobytes() == got[k].tobytes(), k


# ============================================================
# REGRESSIONS
# ============================================================


@pytest.mark.critical
def test_regression_D042_phase0_inphase_progress_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-042: a crash inside phase 0 discarded its in-phase progress."""
    ref = _reference(tmp_path / "ref", monkeypatch)
    _, resumed = _crash_and_resume(tmp_path / "run", monkeypatch, 0, True)
    assert resumed.calls == [("andersen", 2)] + FULL[1:]  # 1 of 3 steps kept
    _assert_same_outputs(ref, tmp_path / "run")


@pytest.mark.high
@settings(max_examples=30)
@given(k_chunk=st.integers(1, 64), payload=st.binary(min_size=0, max_size=64))
@example(k_chunk=2, payload=b"")
def test_regression_D043_fingerprint_has_k_chunk_and_input_hash(
    k_chunk: int, payload: bytes
) -> None:
    """D-043: k_chunk and the input's content were not fingerprinted."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_input(root)
        args = md.build_parser().parse_args(_argv(root))
        info = {"M": M, "K": K}
        base = md.fingerprint(args, info, md.input_digest(args.input))
        args.k_chunk = k_chunk
        assert md.fingerprint(args, info, md.input_digest(args.input)) != base
        args.k_chunk = None
        with open(args.input, "ab") as fh:
            fh.write(b"\x00" + payload)  # same path, other bytes
        assert md.fingerprint(args, info, md.input_digest(args.input)) != base
        assert "input" not in base  # the path itself is not part of it
