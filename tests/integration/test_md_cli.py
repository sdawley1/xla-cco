#!/usr/bin/env python3

"""
Integration tests of the MD driver CLI (``xla_cco.cli.md``).

Until ``xla_cco.dynamics.run`` and ``thermal`` are implemented, the protocol,
outputs and resume logic run against ``md_states.FakeDriver`` (monkeypatched
into those modules); the end-to-end test on the real driver is a strict
xfail that flips to a failure once they land.

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
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ----- numerics -----
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

# ----- project -----
from xla_cco.cli import md
from xla_cco.cli import relax as relax_cli
from xla_cco.dynamics import run as md_run
from xla_cco.dynamics import thermal

# ============================================================
# CONSTANTS
# ============================================================

N, D, SIDE = 16, 2, 4.0
K = 2.0 * math.pi / SIDE * 1.5  # |n| < 1.5: n = (1,0), (0,1), (1,1), (1,-1)
M = 4
TEMPS = ["1e-4", "1e-3"]
NPZ_KEYS = {
    *(f"snap_{k}" for k in md.SNAP_FIELDS),
    *md.RECORD_FIELDS,
    *(f"cycle_{k}" for k in md.RECORD_FIELDS),
    "r_final", "v_final", "equil_phi_mean", "temperatures", "indices",
    "lane_index", "lane_temperature", "basis", "waves_n_vectors", "waves_kvec",
    "waves_constrained", "record_steps", "settings",
}  # fmt: skip
JSON_KEYS = {
    "N", "d", "M", "K", "k_track", "chi", "n_waves", "n_constrained", "input",
    "indices", "temperatures", "n_lanes", "args", "settings", "phases",
    "n_record_steps", "status_counts", "n_running", "equil_phi_over_MT",
    "phase_seconds", "wall_seconds", "npz_keys", "device", "jax_version",
    "resumed",
}  # fmt: skip

# ============================================================
# HELPERS
# ============================================================


def _write_input(
    root: Path, b: int = 3, meta: Optional[Dict[str, Any]] = None, **over: Any
) -> Path:
    """flows_chi1-10.npz (+ .json) with B random states in a 4 x 4 cell."""
    rng = np.random.default_rng(3)
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
    """Wraps a fake and counts its calls."""

    def __init__(self, fn: Any) -> None:
        self.fn, self.n = fn, 0

    def __call__(self, *a: Any, **k: Any) -> Any:
        self.n += 1
        return self.fn(*a, **k)


def _install(mp: pytest.MonkeyPatch, driver: FakeDriver) -> Dict[str, _Counter]:
    """Monkeypatch the fake driver into run / thermal; return call counters."""
    counters = {
        "harmonic": _Counter(fake_harmonic_sample),
        "begin": _Counter(fake_begin_segment),
    }
    mp.setattr(md_run, "make_waves", fake_make_waves)
    mp.setattr(md_run, "init_state", fake_init_state)
    mp.setattr(md_run, "begin_segment", counters["begin"])
    mp.setattr(md_run, "simulate", driver.simulate)
    mp.setattr(thermal, "harmonic_sample", counters["harmonic"])
    mp.setattr(thermal, "maxwell_boltzmann", fake_maxwell_boltzmann)
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


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> Any:
    driver = FakeDriver()
    return driver, _install(monkeypatch, driver)


# ============================================================
# HELP AND DEVICE
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


@pytest.mark.low
def test_md_cli_gpu_guard_mocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    driver, counters = fake
    _write_input(tmp_path)

    def no_gpu() -> None:
        raise RuntimeError("no GPU visible: the job probably lacks --gres=gpu")

    monkeypatch.setattr(relax_cli, "require_gpu", no_gpu)
    argv = _argv(tmp_path)
    argv[argv.index("--device") + 1] = "gpu"
    assert md.main(argv) == 1
    err = capsys.readouterr().err
    assert "ERROR" in err and "--gres=gpu" in err
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
    # geomspace(1, 100, 3) = 1, 10, 100 after each NVE start (15 and 120)
    assert md.record_schedule(phases, 6) == (16, 25, 115, 121, 130, 220)
    # a short segment rounds many log points onto the same steps
    short = _phases(n_equil=0, n_andersen=0, n_nve=3, cycles=1)
    assert md.record_schedule(short, 200) == (1, 2, 3)


@pytest.mark.high
@settings(max_examples=150)
@given(
    n_equil=st.integers(0, 50), n_andersen=st.integers(0, 20),
    n_nve=st.integers(1, 5000), cycles=st.integers(1, 4),
    n_snap=st.integers(1, md.MAX_SNAPSHOTS), baoab=st.booleans(),
)  # fmt: skip
@example(n_equil=0, n_andersen=0, n_nve=3, cycles=1, n_snap=200, baoab=False)
@example(n_equil=2, n_andersen=1, n_nve=2, cycles=4, n_snap=256, baoab=False)
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
    for c, (lo, hi) in enumerate(segs):
        inside |= (arr > lo) & (arr <= hi)
        share = n_snap // len(segs) + (1 if c < n_snap % len(segs) else 0)
        if share >= 1:
            assert hi in steps  # every recorded segment ends with a snapshot
        if share >= 2:
            assert lo + 1 in steps  # and starts one step in (log spacing)
    assert inside.all()


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
        ([], [("andersen", 6), ("andersen", 4), ("verlet", 10), ("andersen", 4),
              ("verlet", 10)], 2),
        (["--scheme", "baoab", "--gamma", "0.1"],
         [("baoab", 6), ("baoab", 10), ("baoab", 10)], 2),
        (["--n-equil", "0", "--n-andersen", "0", "--cycles", "3"],
         [("verlet", 10)] * 3, 3),
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
    assert len(meta["equil_phi_over_MT"]) == n_lanes and meta["K"] == K
    np.testing.assert_array_equal(out["cycle_wind_abs"][-1], out["wind_abs"])


# ============================================================
# CHECKPOINT AND RESUME (fake driver)
# ============================================================


@pytest.mark.critical
@pytest.mark.parametrize(
    "crash_at, inphase", [(0, False), (1, True), (2, False), (2, True), (4, True)]
)
def test_md_cli_resume_after_crash_is_bitwise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crash_at: int, inphase: bool
) -> None:
    ref_dir, run_dir = tmp_path / "ref", tmp_path / "run"
    for root in (ref_dir, run_dir):
        root.mkdir()
        _write_input(root)
    _install(monkeypatch, FakeDriver())
    assert md.main(_argv(ref_dir, "--checkpoint", str(ref_dir / "c.npz"))) == 0
    argv = _argv(run_dir, "--checkpoint", str(run_dir / "c.npz"))
    crashing = FakeDriver(crash_at=crash_at, inphase=inphase)
    _install(monkeypatch, crashing)
    assert md.main(argv) == 1
    assert (run_dir / "c.npz").exists() == (crash_at > 0)
    resumed = FakeDriver()
    _install(monkeypatch, resumed)
    assert md.main(argv) == 0
    full = [("andersen", 6), ("andersen", 4), ("verlet", 10), ("andersen", 4),
            ("verlet", 10)]  # fmt: skip
    first = full[crash_at]
    if inphase:
        first = (first[0], first[1] - first[1] // 2)
    assert resumed.calls == [first] + full[crash_at + 1 :]
    ref, got = _load(ref_dir / "out.npz"), _load(run_dir / "out.npz")
    assert set(ref) == set(got)
    for k in ref:
        assert ref[k].tobytes() == got[k].tobytes(), k
    meta = json.loads((run_dir / "out.json").read_text())
    assert meta["resumed"] is (crash_at > 0)


@pytest.mark.medium
def test_md_cli_resume_rejects_other_settings(
    tmp_path: Path, fake: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_input(tmp_path)
    ckpt = ["--checkpoint", str(tmp_path / "c.npz")]
    assert md.main(_argv(tmp_path, *ckpt)) == 0
    assert md.main(_argv(tmp_path, *ckpt, "--seed", "1")) == 1
    assert "other settings" in capsys.readouterr().err


@pytest.mark.medium
def test_md_cli_ignores_stale_inphase_checkpoint(tmp_path: Path, fake: Any) -> None:
    driver, _ = fake
    _write_input(tmp_path)
    stale = md.inphase_path(tmp_path / "c.npz")
    stale.write_bytes(b"left over by an earlier run")
    assert md.main(_argv(tmp_path, "--checkpoint", str(tmp_path / "c.npz"))) == 0
    assert not stale.exists() and len(driver.calls) == 5


@pytest.mark.medium
def test_md_cli_inphase_checkpoint_behind_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # an in-phase file older than the phase checkpoint must not roll back
    _write_input(tmp_path)
    argv = _argv(tmp_path, "--checkpoint", str(tmp_path / "c.npz"))
    _install(monkeypatch, FakeDriver(crash_at=1, inphase=True))
    assert md.main(argv) == 1
    inphase = md.inphase_path(tmp_path / "c.npz")
    _install(monkeypatch, FakeDriver(crash_at=3))
    assert md.main(argv) == 1  # phases 1-3 done; the in-phase file is now behind
    resumed = FakeDriver()
    _install(monkeypatch, resumed)
    assert inphase.exists() and md.main(argv) == 0
    assert resumed.calls == [("verlet", 10)]


# ============================================================
# END TO END (real driver)
# ============================================================


@pytest.mark.high
@pytest.mark.slow
@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="run.py / thermal.py not implemented yet (T1, T5)",
)
def test_md_cli_end_to_end_small(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref_dir, run_dir = tmp_path / "ref", tmp_path / "run"
    for root in (ref_dir, run_dir):
        root.mkdir()
        _write_input(root)
    ref_args = md.build_parser().parse_args(_argv(ref_dir))
    summary = md.run(ref_args)
    ref = _load(ref_dir / "out.npz")
    assert set(ref) >= NPZ_KEYS and summary["n_lanes"] == 4
    assert summary["status_counts"] == {"STATUS_RUNNING": 4}
    assert np.all(ref["snap_count"] == len(ref["record_steps"]))
    assert np.all(ref["wind_abs"] >= np.abs(ref["wind_net"]))
    assert np.all(np.isfinite(ref["equil_phi_mean"]))
    # crash in the second NVE segment's Andersen phase, then resume
    real = md_run.simulate
    calls: List[int] = []

    def crash_once(*a: Any, **k: Any) -> Any:
        calls.append(1)
        if len(calls) == 4:
            raise RuntimeError("simulated crash")
        return real(*a, **k)

    argv = _argv(run_dir, "--checkpoint", str(run_dir / "c.npz"))
    monkeypatch.setattr(md_run, "simulate", crash_once)
    assert md.main(argv) == 1
    assert md.main(argv) == 0
    got = _load(run_dir / "out.npz")
    for k in ref:
        assert ref[k].tobytes() == got[k].tobytes(), k
