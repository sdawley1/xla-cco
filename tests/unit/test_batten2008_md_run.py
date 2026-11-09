#!/usr/bin/env python3

"""
Tests for the MD production run script (experiments/scripts/batten2008_md_run.py):
input assembly per cell (E1, E1b, E4) against the plan and bench.json, the
SLURM cost helpers, and an end-to-end run on a tiny system whose output the
E2 analysis consumes.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import argparse
import hashlib
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ----- numerics -----
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from systems import stealthy_ground_state

# ----- project -----
from xla_cco.cli import md

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "scripts"))
import batten2008_md_analysis as ana  # noqa: E402
import batten2008_md_run as mdr  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

BENCH = mdr.DEFAULT_BENCH
TINY_N, TINY_D, TINY_SIDE = 16, 2, 4.0
TINY_K = 2.0 * math.pi / TINY_SIDE * 1.5  # M = 4 waves (as test_md_cli)
TINY_GS_SEEDS = (3, 4)

# ============================================================
# HELPERS
# ============================================================


def _args(
    step: str, system: str = "flows2d", chi: str = "1-10",
    gamma: Optional[float] = None, **over: Any,
) -> argparse.Namespace:  # fmt: skip
    """Parsed run-script arguments of one cell (bench = the committed bench.json)."""
    argv = ["--step", step, "--system", system, "--chi", chi]
    if gamma is not None:
        argv += ["--gamma", repr(gamma)]
    args = mdr.build_parser().parse_args(argv)
    for k, v in over.items():
        setattr(args, k, v)
    return args


def _assemble(args: argparse.Namespace, k_outer: float = 2.0) -> List[str]:
    """CLI argv of a cell (validated, chosen + protocol attached)."""
    mdr.validate(args)
    args.chosen = mdr.load_chosen(args.bench, args.chi)
    args.proto = mdr.protocol(args)
    prefix = Path("/data") / args.system / "N1" / mdr.output_stem(
        args.step, args.chi, args.gamma)  # fmt: skip
    return mdr.cli_argv(args, prefix, k_outer)


def _cli(argv: List[str]) -> argparse.Namespace:
    """The CLI's parse of argv (and its own validation)."""
    ns = md.build_parser().parse_args(argv)
    md.validate_args(ns)
    return ns


def _trace_rows_needed(ns: argparse.Namespace) -> int:
    """Rows the CLI's trace takes over the extended protocol (every step % e)."""
    total = sum(ph.n_steps for ph in md.protocol_phases(ns, extended=True))
    return total // ns.record_rho_every


def _bench() -> Dict[str, Any]:
    """The committed V3 bench.json."""
    out: Dict[str, Any] = json.loads(BENCH.read_text())
    return out


def _plan_cells() -> List[Any]:
    """Every (step, system, chi, gamma) cell of the plan (28)."""
    cells = [(s, y, c, None) for s, y, c in itertools.product(
        ("run", "spectra"), mdr.SYSTEMS, mdr.CHI_TAGS)]  # fmt: skip
    cells += [("langevin", y, c, g) for y, c, g in itertools.product(
        mdr.LANGEVIN_SYSTEMS, mdr.CHI_TAGS, mdr.LANGEVIN_GAMMAS)]  # fmt: skip
    assert len(cells) == 28, "E1 + E1b (2 x 5 x 2) + E4 (2 x 2 x 2)"
    return cells


def _git(repo: Path, *argv: str) -> None:
    """Run git in ``repo`` with a throwaway identity (raises on failure)."""
    ident = ["-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", "-C", str(repo), *ident, *argv], check=True,
                   capture_output=True, timeout=60)  # fmt: skip


# ============================================================
# INPUT ASSEMBLY
# ============================================================


@pytest.mark.critical
@pytest.mark.parametrize("chi", mdr.CHI_TAGS)
def test_run_argv_follows_plan_and_bench(chi: str) -> None:
    """E1: 10 ground states x 4 T, V3 dt / cadence / drift_tol, A4 trace."""
    chosen = _bench()["chosen"][chi]
    ns = _cli(_assemble(_args("run", chi=chi)))
    assert ns.indices == list(range(10))
    assert ns.temperature == [1e-6, 1e-5, 1e-4, 1e-3]
    assert (ns.dt, ns.wind_every, ns.drift_tol) == (
        chosen["dt"], chosen["wind_every"], chosen["drift_tol"])  # fmt: skip
    assert (ns.n_equil, ns.n_andersen, ns.n_nve, ns.cycles) == (
        200_000, 100_000, 1_000_000, 2)  # fmt: skip
    assert (ns.andersen_mode, ns.andersen_p, ns.scheme) == (
        "count", 0.01, "andersen-nve")  # fmt: skip
    assert (ns.n_snapshots, ns.k_track, ns.k_chunk) == (200, None, None)
    assert ns.device == "gpu"
    assert (ns.record_rho_every, ns.max_rho_records) == (
        chosen["record_rho_every"], chosen["max_rho_records"])  # fmt: skip
    assert ns.n_nve // ns.record_rho_every >= mdr.TRACE_SAMPLES  # A4 per segment
    assert ns.max_rho_records == _trace_rows_needed(ns)
    assert ns.record_rho_every % ns.wind_every == 0  # F5: fresh unconstrained rows
    assert ns.seed == mdr.cell_seed("run", "flows2d", chi, None)  # F1
    assert ns.checkpoint == Path(str(ns.save) + ".ckpt.npz")
    assert ns.save.name == f"md_chi{chi}" and ns.input.name == f"flows_chi{chi}.npz"


@pytest.mark.high
@pytest.mark.parametrize("chi", mdr.CHI_TAGS)
def test_spectra_argv_follows_plan(chi: str) -> None:
    """E1b: 2 ground states, T = 1e-6, constrained only, t = 200, trace every 5."""
    plan = _bench()["plan"]["E1b_per_job"][chi]
    ns = _cli(_assemble(_args("spectra", chi=chi), k_outer=2.25))
    assert (ns.indices, ns.temperature, ns.cycles) == ([0, 1], [1e-6], 1)
    assert ns.k_track == 2.25 and ns.n_nve == round(200.0 / ns.dt)
    assert (ns.n_nve, ns.record_rho_every, ns.max_rho_records) == (
        plan["n_nve"], 5, plan["max_rho_records"])  # fmt: skip
    assert ns.max_rho_records == _trace_rows_needed(ns)
    assert ns.record_rho_every % ns.wind_every == 0  # F5
    assert ns.seed == mdr.cell_seed("spectra", "flows2d", chi, None)  # F1
    assert ns.save.name == f"md_spectra_chi{chi}"


@pytest.mark.high
@pytest.mark.parametrize("gamma", mdr.LANGEVIN_GAMMAS)
def test_langevin_argv_follows_plan(gamma: float) -> None:
    """E4: BAOAB at gamma, 5 ground states x T in {1e-6, 1e-4}, E1 cadence."""
    chosen = _bench()["chosen"]["2-5"]
    ns = _cli(_assemble(_args("langevin", system="hip2d", chi="2-5", gamma=gamma)))
    assert (ns.scheme, ns.gamma) == ("baoab", gamma)
    assert (ns.indices, ns.temperature) == ([0, 1, 2, 3, 4], [1e-6, 1e-4])
    assert (ns.dt, ns.wind_every) == (chosen["dt"], chosen["wind_every"])
    assert ns.record_rho_every == chosen["record_rho_every"]
    assert ns.max_rho_records == _trace_rows_needed(ns)
    assert ns.record_rho_every % ns.wind_every == 0  # F5
    assert ns.seed == mdr.cell_seed("langevin", "hip2d", "2-5", gamma)  # F1
    assert gamma < chosen["omega_min_all_starts"]  # F6 (the run script checks it)
    assert ns.save.name == f"md_langevin_g{gamma:g}_chi2-5"


@pytest.mark.high
def test_off_plan_cells_rejected() -> None:
    """E4 only on flows2d / hip2d at the two gammas; --gamma only for E4."""
    bad = [
        _args("langevin", system="gau2d", gamma=0.03),
        _args("langevin", system="hip2d", gamma=0.1),
        _args("langevin", system="hip2d"),
        _args("run", gamma=0.03),
    ]
    for args in bad:
        with pytest.raises(ValueError):
            mdr.validate(args)
    mdr.validate(_args("langevin", system="flows2d", gamma=0.003))


@pytest.mark.high
def test_overrides_recorded_as_plan_deviations() -> None:
    """A changed protocol option is applied and listed; the plan is not."""
    args = _args("run", n_nve=2500, temperature=[1e-4])  # every 5 = wind_every
    ns = _cli(_assemble(args))
    dev = args.proto["deviations"]
    assert ns.n_nve == 2500 and ns.temperature == [1e-4]
    assert dev == {
        "n_nve": {"plan": 1_000_000, "used": 2500},
        "temperature": {"plan": list(mdr.TEMPS_RUN), "used": [1e-4]},
    }
    assert _assemble(_args("run")) and _args("run").n_gs is None


@pytest.mark.high
def test_seeds_distinct_per_cell() -> None:
    """Every (step, system, chi, gamma) cell of the plan has its own seed."""
    cells = _plan_cells()
    seeds = [mdr.cell_seed(*cell) for cell in cells]
    assert len(set(seeds)) == len(cells) == 28
    assert min(seeds) >= 1


@pytest.mark.critical
def test_cli_seed_is_cell_seed_for_every_cell() -> None:
    """AUDIT-E1 F1: the --seed cli.md receives is cell_seed, distinct per cell.

    Checks the assembled argv (not cell_seed alone), so a constant --seed in
    cli_argv (audit mutation M2b) fails here.
    """
    got = {}
    for step, system, chi, gamma in _plan_cells():
        ns = _cli(_assemble(_args(step, system=system, chi=chi, gamma=gamma)))
        assert ns.seed == mdr.cell_seed(step, system, chi, gamma), (step, system)
        got[(step, system, chi, gamma)] = ns.seed
    assert len(set(got.values())) == len(got) == 28


@pytest.mark.high
@pytest.mark.parametrize(("chi", "n_nve"), [("1-10", 600), ("1-10", 1500),
                                             ("2-5", 1500)])  # fmt: skip
def test_trace_cadence_not_multiple_of_wind_every_rejected(
    chi: str, n_nve: int
) -> None:
    """AUDIT-E1 F5: an override whose trace cadence is not a multiple of
    wind_every (stale unconstrained rows in the A4 trace) is a ValueError.

    1-10 has wind_every 5: n_nve 600 -> every 1, 1500 -> 3 (both rejected);
    2-5 has wind_every 2: 1500 -> every 3 (rejected); 1000 -> 2 is accepted.
    """
    with pytest.raises(ValueError, match="multiple of wind_every"):
        _assemble(_args("run", chi=chi, n_nve=n_nve))
    ns = _cli(_assemble(_args("run", chi="2-5", n_nve=1000)))
    assert ns.record_rho_every == 2 and ns.wind_every == 2
    assert mdr.trace_every("run", 1_000_000, 5) == 2000  # plan: n_nve // 500
    assert mdr.trace_every("spectra", 5000, 1) == mdr.SPECTRA_EVERY  # wound each step


@pytest.mark.high
def test_langevin_gamma_below_omega_min(tmp_path: Path) -> None:
    """AUDIT-E1 F6: E4 needs gamma < omega_min (V3) whenever bench.json has it.

    A bench whose omega_min_all_starts lies between the two gammas rejects
    gamma = 0.03 and accepts 0.003; a bench without the key (written before
    V3 recorded omega_min, e.g. the tiny end-to-end bench) is not checked,
    and omega_min is None in the returned value (recorded in provenance).
    """
    chosen = dict(_bench()["chosen"]["2-5"])
    assert mdr.check_gamma(0.03, chosen) == chosen["omega_min_all_starts"]
    low = dict(chosen, omega_min_all_starts=0.02)
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps({"chosen": {"2-5": low}}))
    with pytest.raises(ValueError, match="omega_min"):
        _assemble(_args("langevin", system="hip2d", chi="2-5", gamma=0.03, bench=bench))
    assert _assemble(_args("langevin", system="hip2d", chi="2-5", gamma=0.003,
                           bench=bench))  # fmt: skip
    del low["omega_min_all_starts"]
    assert mdr.check_gamma(0.03, low) is None
    with pytest.raises(ValueError, match="omega_min"):
        mdr.check_gamma(0.03, dict(chosen, omega_min_all_starts=0.03))  # gamma = w


@pytest.mark.high
@given(step=st.sampled_from(("run", "spectra", "langevin")),
       tag=st.sampled_from(mdr.CHI_TAGS),
       gamma=st.sampled_from(mdr.LANGEVIN_GAMMAS))  # fmt: skip
def test_output_stem_matches_analysis_naming(step: str, tag: str, gamma: float) -> None:
    """Outputs map to md_analysis[_<kind>]_chi<tag> as E2 / E3 classify them."""
    stem = mdr.output_stem(step, tag, gamma if step == "langevin" else None)
    name = ana.output_name(Path(stem + ".npz"))
    kind = {"run": "md_analysis_chi", "spectra": "md_analysis_spectra",
            "langevin": "md_analysis_langevin"}[step]  # fmt: skip
    assert name.startswith(kind) and name.endswith(f"_chi{tag}.npz")
    assert name.split("_chi")[-1] == f"{tag}.npz"


@pytest.mark.critical
@given(n_equil=st.integers(0, 50), n_andersen=st.integers(0, 50),
       n_nve=st.integers(1, 200), cycles=st.integers(1, 4),
       every=st.integers(1, 30), baoab=st.booleans())  # fmt: skip
@example(n_equil=200_000, n_andersen=100_000, n_nve=1_000_000, cycles=2,
         every=2000, baoab=False)  # fmt: skip
def test_trace_capacity_covers_extended_protocol(
    n_equil: int, n_andersen: int, n_nve: int, cycles: int, every: int, baoab: bool
) -> None:
    """trace_records = rows the CLI's trace takes over the extended protocol."""
    ns = argparse.Namespace(n_equil=n_equil, n_andersen=n_andersen, n_nve=n_nve,
                            cycles=cycles, scheme="baoab" if baoab else "andersen-nve",
                            record_rho_every=every)  # fmt: skip
    cap = mdr.trace_records(n_equil, n_andersen, n_nve, cycles, every, baoab)
    assert cap == max(1, _trace_rows_needed(ns))
    total = sum(ph.n_steps for ph in md.protocol_phases(ns, extended=True))
    assert cap * every <= total or cap == 1


@pytest.mark.high
def test_git_changes_reports_dirty_worktree(tmp_path: Path) -> None:
    """AUDIT-E1 F3: tracked uncommitted changes listed; clean -> []; no repo -> None."""
    assert mdr.git_changes(tmp_path) is None  # not a repository
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(tmp_path, "add", "a.py")
    _git(tmp_path, "commit", "-q", "-m", "init")
    assert mdr.git_changes(tmp_path) == []
    (tmp_path / "untracked.log").write_text("log\n")
    assert mdr.git_changes(tmp_path) == []  # untracked files are not code changes
    (tmp_path / "a.py").write_text("x = 2\n")
    assert mdr.git_changes(tmp_path) == ["a.py"]
    head = mdr.git_head(tmp_path)
    assert len(head) == 40 and mdr.git_head(tmp_path / "nowhere") == ""


@pytest.mark.medium
def test_check_input_reads_k_and_checks_n(tmp_path: Path) -> None:
    """K from flows_chi<tag>.json; a wrong N or a missing file is an error."""
    cell = tmp_path / "flows2d" / "N1024"
    cell.mkdir(parents=True)
    with pytest.raises(ValueError, match="missing input metadata"):
        mdr.check_input(cell, "flows2d", "1-10")
    (cell / "flows_chi1-10.json").write_text(json.dumps({"N": 1024, "K": 2.5}))
    with pytest.raises(ValueError, match="missing input"):
        mdr.check_input(cell, "flows2d", "1-10")
    np.savez(cell / "flows_chi1-10.npz", basis=np.eye(2))
    assert mdr.check_input(cell, "flows2d", "1-10") == 2.5
    (cell / "flows_chi1-10.json").write_text(json.dumps({"N": 1000, "K": 2.5}))
    with pytest.raises(ValueError, match="want 1024"):
        mdr.check_input(cell, "flows2d", "1-10")


# ============================================================
# JOBS AND COST
# ============================================================


@pytest.mark.high
@given(rows=st.lists(st.tuples(st.integers(10_000_000, 99_999_999),
                               st.integers(0, 10**6), st.integers(0, 8)),
                     max_size=6))  # fmt: skip
@example(rows=[(17143149, 1454, 1)])
def test_parse_sacct_gpu_hours(rows: List[Any]) -> None:
    """GPU-h = ElapsedRaw x gres/gpu / 3600 per row; malformed lines skipped."""
    lines = [f"{j}|mdw_x|{e}|billing=4,cpu=4,gres/gpu={g},mem=64G,node=1|COMPLETED"
             for j, e, g in rows]  # fmt: skip
    text = "\n".join(lines + ["garbage", "1|x|n/a|cpu=1|FAILED"])
    got = mdr.parse_sacct(text)
    assert [r["job_id"] for r in got] == [str(j) for j, _, _ in rows]
    for r, (_, e, g) in zip(got, rows, strict=True):
        assert (r["elapsed_s"], r["n_gpu"]) == (e, g)
        assert r["gpu_hours"] == e * g / 3600.0


@pytest.mark.medium
def test_gpus_of_examples() -> None:
    """gres/gpu=N counted; gres/gpu:<type> and missing entries are not."""
    assert mdr.gpus_of("billing=4,cpu=4,gres/gpu=1,mem=64G,node=1") == 1
    assert mdr.gpus_of("cpu=4,gres/gpu:a100=2") == 0
    assert mdr.gpus_of("cpu=4,mem=8G") == 0 and mdr.gpus_of("") == 0


@pytest.mark.medium
def test_job_list_appends_once(tmp_path: Path) -> None:
    """A resumed job is not listed twice; outside SLURM nothing is recorded."""
    path = tmp_path / "md_chi1-10.jobs.json"
    assert mdr.job_record({}) is None and mdr.append_job(path, None) == []
    rec = mdr.job_record({"SLURM_JOB_ID": "123", "SLURM_JOB_NAME": "mdw_pilot"})
    assert rec is not None and rec["job_name"] == "mdw_pilot"
    assert len(mdr.append_job(path, rec)) == 1
    assert len(mdr.append_job(path, rec)) == 1
    rec2 = dict(rec, job_id="124")
    assert [j["job_id"] for j in mdr.append_job(path, rec2)] == ["123", "124"]
    assert mdr.sacct_cost([]) == {"available": False, "reason": "no SLURM job recorded"}


# ============================================================
# END TO END (tiny system, CPU)
# ============================================================


def _tiny_cell(root: Path) -> Path:
    """flows2d/N16 with 2 ground states, its JSON and a bench.json; -> bench."""
    cell = root / "flows2d" / f"N{TINY_N}"
    cell.mkdir(parents=True)
    states = [stealthy_ground_state(TINY_N, TINY_D, TINY_K, s) for s in TINY_GS_SEEDS]
    s_final = np.stack([s for s, _, _ in states])
    np.savez(cell / "flows_chi1-10.npz", s_final=s_final, basis=states[0][1])
    meta = {"N": TINY_N, "d": TINY_D, "K": TINY_K, "M": 4}
    (cell / "flows_chi1-10.json").write_text(json.dumps(meta))
    bench = root / "bench.json"
    chosen = {"dt": 0.01, "wind_every": 2, "drift_tol": 1e-3, "k_chunk": "none"}
    bench.write_text(json.dumps({"chosen": {"1-10": chosen}}))
    return bench


@pytest.mark.high
@pytest.mark.slow
def test_run_cell_end_to_end_tiny(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tiny E1 cell writes md_chi1-10.{npz,json} that the E2 analysis reads.

    The JSON carries the run-script provenance (argv, deviations, bench
    hash, git state) and a cost entry; rerunning resumes the final
    checkpoint and rewrites identical arrays without integrating again
    (AUDIT-E1 F2: ``resumed`` is True and the phase timings are those of
    the first run, so a silent restart from scratch fails).
    """
    bench = _tiny_cell(tmp_path)
    monkeypatch.setitem(mdr.SYSTEMS, "flows2d", f"N{TINY_N}")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(mdr, "git_changes", lambda repo=mdr.REPO: ["src/x.py"])
    over = {"n_gs": 2, "temperature": [1e-4, 1e-3], "n_equil": 40, "n_andersen": 20,
            "n_nve": 1000, "n_snapshots": 30}  # fmt: skip
    args = _args("run", data_root=tmp_path, bench=bench, device="cpu", **over)
    assert mdr.run_cell(args) == 0
    out = tmp_path / "flows2d" / f"N{TINY_N}" / "md_chi1-10"
    meta = json.loads(Path(str(out) + ".json").read_text())
    rs = meta["run_script"]
    assert rs["step"] == "run" and rs["chosen"]["dt"] == 0.01
    assert set(rs["plan_deviations"]) == set(over)
    assert rs["bench_sha256"] == hashlib.sha256(bench.read_bytes()).hexdigest()  # F3
    assert rs["cli_argv"][rs["cli_argv"].index("--seed") + 1] == str(
        mdr.cell_seed("run", "flows2d", "1-10", None))  # fmt: skip
    assert len(rs["git_head"]) in (0, 40) and rs["omega_min"] is None
    assert rs["git_dirty"] is True and rs["git_changed"] == ["src/x.py"]  # F3
    assert meta["resumed"] is False and len(meta["phase_seconds"]) >= 6
    assert meta["cost"]["available"] is False and meta["jobs"] == []
    assert meta["status_counts"] == {"STATUS_RUNNING": 4}
    with np.load(str(out) + ".npz") as z:
        first = {k: z[k].copy() for k in z.files}
    assert set(ana.REQUIRED_KEYS) <= set(first)
    assert int(first["trace_count"].min()) == first["trace_rho"].shape[1]
    res = ana.analyze_file(Path(str(out) + ".npz"), out.parent / "flows_chi1-10.npz")
    assert res["nu_c"].shape == (2, 2) and np.all(np.isfinite(res["nu_c"]))
    assert np.all(np.isfinite(res["inv_rho"]))  # A4 trace: 500 rows per segment
    assert mdr.run_cell(args) == 0  # resume at the end: outputs rewritten as-is
    with np.load(str(out) + ".npz") as z:
        for k in first:
            assert z[k].tobytes() == first[k].tobytes(), k
    again = json.loads(Path(str(out) + ".json").read_text())
    assert again["resumed"] is True  # F2: resumed, not restarted
    assert again["phase_seconds"] == meta["phase_seconds"]  # no phase ran again
