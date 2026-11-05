#!/usr/bin/env python3

"""
Generate (or --check) the C++ golden data under ``tests/data/cpp/``.

Drives the harness ``build/dump_reference`` (built by ``make`` in this
directory) and the unmodified L-BFGS binary ``soft_core_stealthy2.out`` via
``subprocess``; writes seeded input ConfigPacks, the ``.npz`` goldens and
``MANIFEST.json``. Independent of ``xla_cco``: it carries its own ConfigPack
reader/writer and k-space formulas. Run with ``python -I``.

    python -I tools/cpp_reference/make_goldens.py            # regenerate
    python -I tools/cpp_reference/make_goldens.py --check    # verify

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import argparse
import datetime
import hashlib
import importlib.util
import itertools
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ----- numerics -----
import numpy as np

# ============================================================
# CONSTANTS
# ============================================================

REPO: Path = Path(__file__).resolve().parents[2]
TOOL_DIR: Path = REPO / "tools" / "cpp_reference"
HARNESS: Path = TOOL_DIR / "build" / "dump_reference"
DATA_DIR: Path = REPO / "tests" / "data" / "cpp"
INPUT_DIR: Path = DATA_DIR / "inputs"
CCO_DIR: Path = Path("/home/sd2402/ccoptimization/cco")
LBFGS_BIN: Path = CCO_DIR / "EXC" / "soft_core_stealthy2.out"
RUNS: Path = Path("/scratch/gpfs/TORQUATO/sd2402/cco_runs")
RUN_DIR: Path = RUNS / "p3d-N100-phi0.05-chi0.05-val0.0"
IC_PACK: Path = RUN_DIR / "ic" / "chunk_0000"
SUCCESS_PACK: Path = RUN_DIR / "task_0000" / "ccopt-t0000_Success"
DEFAULT_WORK: Path = Path("/scratch/gpfs/TORQUATO/sd2402/cco_golden_work")

PHI: float = 0.05
SIGMA: str = "0.4570781497340833"
CHIS: Tuple[float, ...] = (0.05, 0.25, 0.45)
CASES: Tuple[Tuple[int, int], ...] = ((1, 50), (2, 64), (3, 100))
SEED_BASE: int = 20261100  # pack seed = SEED_BASE + d
N_CONFIGS: int = 5
LATTICE_NOISE: float = 1e-3
LATTICE_SHAPE: Dict[int, Tuple[int, ...]] = {1: (50,), 2: (8, 8), 3: (5, 5, 4)}
SHELL_CANDIDATES: Tuple[int, ...] = (5, 8, 10, 4, 1)  # cubic-shell n^2, in order
P_MAX: int = 256  # trajectory runs per config (noise-horizon search)
P_STORE: int = 200  # trajectory iterates stored (R9)
NOISE_EPS: float = 1e-15
SK_CONFIGS: int = 50
TOL_F32: float = float(np.float32(1e-20))  # C++ std::stof(argv[2]) value
LBFGS_ARGV: Tuple[str, ...] = ("1", "1e-20", "500000", "0")
MAX_BYTES: int = 5 * 1024 * 1024

# ============================================================
# CONFIGPACK I/O (independent minimal reader/writer)
# ============================================================


def write_pack(prefix: Path, s: np.ndarray) -> None:
    """Write (B,N,d) fractional configs as a unit-cell ConfigPack."""
    b, n, d = s.shape
    rec = np.dtype([("x", "<f8", (d,)), ("nm", "S4")])
    blobs = []
    for cfg in s:
        body = np.zeros(n, dtype=rec)
        body["x"], body["nm"] = cfg, b"a"
        head = (
            np.array(d, "<u2").tobytes()
            + np.array(1.0, "<f8").tobytes()
            + np.eye(d, dtype="<f8").tobytes()
            + np.array(n, "<i4").tobytes()
        )
        blobs.append(head + body.tobytes())
    offs = np.cumsum([0] + [len(x) for x in blobs[:-1]]).astype("<i8")
    Path(str(prefix) + ".ConfigPack").write_bytes(b"".join(blobs))
    Path(str(prefix) + ".ConfigPackIndex").write_bytes(
        np.array(b, "<i8").tobytes() + offs.tobytes()
    )


def read_pack(prefix: Path, count: Optional[int] = None) -> np.ndarray:
    """Read the first ``count`` configs of a ConfigPack as (B,N,d) relative."""
    idx = np.fromfile(str(prefix) + ".ConfigPackIndex", "<i8")
    buf = np.fromfile(str(prefix) + ".ConfigPack", np.uint8)
    out = []
    for off in idx[1:][: (idx[0] if count is None else count)]:
        d = int(buf[off : off + 2].view("<u2")[0])
        n = int(buf[off + 10 + 8 * d * d : off + 14 + 8 * d * d].view("<i4")[0])
        rec = np.dtype([("x", "<f8", (d,)), ("nm", "u1", (4,))])
        start = int(off) + 14 + 8 * d * d
        out.append(np.frombuffer(buf[start : start + n * rec.itemsize], rec)["x"])
    assert len(out) > 0, f"no configs read from {prefix}"
    return np.stack(out)


# ============================================================
# K-SPACE FORMULAS (independent of xla_cco)
# ============================================================


def ball_volume(d: int) -> float:
    """Volume of the unit d-ball."""
    assert d in (1, 2, 3), "dimension 1-3"
    return math.pi ** (d / 2) / math.gamma(d / 2 + 1)


def ka_from_chi(chi: float, phi: float, d: int) -> float:
    """Ka = (2^(d+1) d chi Gamma(d/2+1)^2 phi)^(1/d)."""
    assert 0.0 < chi < 1.0 and 0.0 < phi < 1.0, "chi, phi in (0,1)"
    return (2 ** (d + 1) * d * chi * math.gamma(d / 2 + 1) ** 2 * phi) ** (1 / d)


def particle_radius(phi: float, d: int) -> float:
    """a = (phi / V_d(1))^(1/d)."""
    assert d in (1, 2, 3), "dimension 1-3"
    return (phi / ball_volume(d)) ** (1.0 / d)


# ============================================================
# SUBPROCESS DRIVERS
# ============================================================


def _env() -> Dict[str, str]:
    """Environment with single-threaded OpenMP (deterministic sums)."""
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    return env


def _display(cmd: List[str]) -> str:
    """Shell-quoted command line with the repo root made relative."""
    return shlex.join(c.replace(str(REPO) + "/", "") for c in cmd)


def run_harness(
    sub: str, args: Dict[str, str], work: Path, tag: str
) -> Tuple[Path, str]:
    """Run ``dump_reference <sub> --k v ...``; return (output file, command)."""
    out = work / f"{tag}.txt"
    cmd = [str(HARNESS), sub] + [x for k, v in args.items() for x in (f"--{k}", v)]
    cmd += ["--out", str(out)]
    with open(work / f"{tag}.stdout", "w") as log:
        rc = subprocess.run(
            cmd, cwd=work, env=_env(), stdout=log, stderr=subprocess.STDOUT, check=False
        ).returncode
    assert rc == 0, f"{_display(cmd)} exited {rc}; see {work / (tag + '.stdout')}"
    return out, _display(cmd)


def _parse_out(path: Path) -> Dict[str, Any]:
    """Parse the tagged text written by the harness."""
    o: Dict[str, Any] = {
        "kvec": [],
        "mod2": [],
        "recip": [],
        "cfg": [],
        "E": [],
        "F": [],
        "R": [],
        "T": [],
    }
    for ln in path.read_text().splitlines():
        t = ln.split()
        tag = t[0]
        if tag in ("KINNER", "KOUTER"):
            o[tag] = (float.fromhex(t[1]), float(t[2]))
        elif tag == "RECIP":
            o["recip"].append([float(x) for x in t[1:]])
        elif tag == "KVEC":
            o["kvec"].append([float(x) for x in t[1:-1]])
            o["mod2"].append(float.fromhex(t[-1]))
        elif tag == "M":
            o["M"] = int(t[1])
        elif tag == "CHI":
            o["CHI"] = float(t[1])
        elif tag == "CONFIG":
            o["cfg"].append([])
            o["F"].append([])
        elif tag == "S":
            o["cfg"][-1].append([float(x) for x in t[1:]])
        elif tag == "E":
            o["E"].append(float(t[1]))
        elif tag == "F":
            o["F"][-1].append([float(x) for x in t[1:]])
        elif tag == "R":
            o["R"].append((int(t[1]), float(t[2]), [float(x) for x in t[3:]]))
        elif tag == "T":
            o["T"].append([float(x) for x in t[1:]])
    return o


# ============================================================
# NPZ + PARAMS HELPERS
# ============================================================


def _save(name: str, params: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> Path:
    """Write ``tests/data/cpp/<name>.npz`` with a JSON ``params`` string."""
    path = DATA_DIR / f"{name}.npz"
    np.savez_compressed(
        path, params=np.array(json.dumps(params, sort_keys=True)), **arrays
    )
    return path


def _window_params(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Raw k_inner / k_outer actually used by C++, hex and decimal."""
    (ki_x, ki), (ko_x, ko) = parsed["KINNER"], parsed["KOUTER"]
    return {
        "k_inner_hex": ki_x.hex(),
        "k_inner": ki,
        "k_outer_hex": ko_x.hex(),
        "k_outer": ko,
    }


def _integer_n(parsed: Dict[str, Any]) -> np.ndarray:
    """Map the C++ k-vectors back to integer reciprocal coordinates."""
    k, rec = np.array(parsed["kvec"]), np.array(parsed["recip"])
    n = k @ np.linalg.inv(rec)
    err = float(np.abs(n - np.rint(n)).max())
    assert err < 1e-9, f"integer reconstruction error {err}"
    return np.rint(n).astype(np.int64)


# ============================================================
# INPUT PACKS
# ============================================================


def make_inputs(d: int, n: int) -> np.ndarray:
    """Seeded (5,N,d) configs: 4 Poisson + 1 lattice with 1e-3 noise (last)."""
    rng = np.random.default_rng(SEED_BASE + d)
    poisson = rng.random((N_CONFIGS - 1, n, d))
    axes = [(np.arange(m) + 0.5) / m for m in LATTICE_SHAPE[d]]
    lat = np.array(list(itertools.product(*axes)), dtype=np.float64)
    assert lat.shape == (n, d), f"lattice {lat.shape} != {(n, d)}"
    noisy = np.mod(lat + rng.uniform(-LATTICE_NOISE, LATTICE_NOISE, lat.shape), 1.0)
    return np.concatenate([poisson, noisy[None]], axis=0)


def write_inputs() -> Dict[Tuple[int, int], Path]:
    """Write the Poisson/near-ordered packs and 1e-15-perturbed copies."""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    out: Dict[Tuple[int, int], Path] = {}
    for d, n in CASES:
        s = make_inputs(d, n)
        prefix = INPUT_DIR / f"poisson_d{d}_n{n}"
        write_pack(prefix, s)
        write_pack(Path(str(prefix) + "_pert"), np.mod(s + NOISE_EPS, 1.0))
        assert np.array_equal(read_pack(prefix), s), "input pack round-trip"
        out[(d, n)] = prefix
    return out


# ============================================================
# KERNEL GOLDENS
# ============================================================


def _kernel_golden(
    name: str,
    d: int,
    n: int,
    pack: Path,
    window: Dict[str, str],
    meta: Dict[str, Any],
    work: Path,
    count: int = N_CONFIGS,
) -> List[str]:
    """Run the harness ``kernel`` subcommand and save ``name``.npz."""
    args = {"pack": str(pack), "d": str(d), "n": str(n), "count": str(count), **window}
    path, cmd = run_harness("kernel", args, work, name)
    p = _parse_out(path)
    s = np.array(p["cfg"])
    assert np.array_equal(s, read_pack(pack, count)), f"{name}: C++ s != pack"
    assert p["CHI"] == p["M"] / (d * (n - 1)), f"{name}: chi mismatch"
    params = {
        "d": d,
        "N": n,
        "phi": PHI,
        "M": p["M"],
        "chi": p["CHI"],
        "source_pack": str(pack).replace(str(REPO) + "/", ""),
        "harness_cmd": cmd,
        "rescale": "N^(1/d), main.cpp:211",
        **_window_params(p),
        **meta,
    }
    _save(
        name,
        params,
        {
            "n": _integer_n(p),
            "m": np.array(p["M"], np.int64),
            "chi": np.array(p["CHI"]),
            "s": s,
            "energy": np.array(p["E"]),
            "forces": np.array(p["F"]),
        },
    )
    print(f"  {name}: M={p['M']} chi={p['CHI']:.6f} B={s.shape[0]}", flush=True)
    return [cmd]


def _chi_meta(chi: float, d: int) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Params metadata and harness window arguments for a target chi."""
    ka, a = ka_from_chi(chi, PHI, d), particle_radius(PHI, d)
    meta = {
        "chi_target": chi,
        "Ka": ka,
        "Ka_hex": ka.hex(),
        "a": a,
        "k_outer_python": (ka / a),
        "k_outer_python_hex": (ka / a).hex(),
        "seed": SEED_BASE + d,
    }
    return meta, {"ka": ka.hex(), "phi": repr(PHI)}


def build_kernels(
    packs: Dict[Tuple[int, int], Path], work: Path
) -> Dict[str, List[str]]:
    """kernel_d{1,2,3}_chi*.npz plus the C++-ic d=3 file."""
    cmds: Dict[str, List[str]] = {}
    for (d, n), pack in packs.items():
        for chi in CHIS:
            meta, window = _chi_meta(chi, d)
            name = f"kernel_d{d}_chi{chi}"
            cmds[name] = _kernel_golden(name, d, n, pack, window, meta, work)
            ko = json.loads(str(np.load(DATA_DIR / f"{name}.npz")["params"]))["k_outer"]
            assert abs(ko / meta["k_outer_python"] - 1) < 1e-13, f"{name}: K2 mismatch"
    meta, window = _chi_meta(0.05, 3)
    meta["seed"] = None
    cmds["kernel_d3_cpp_ic"] = _kernel_golden(
        "kernel_d3_cpp_ic", 3, 100, IC_PACK, window, meta, work
    )
    z = np.load(DATA_DIR / "kernel_d3_cpp_ic.npz")
    assert int(z["m"]) == 13 and abs(float(z["chi"]) - 0.043771) < 5e-7, "ic M/chi"
    return cmds


# ============================================================
# BOUNDARY GOLDENS
# ============================================================


def _shell_threshold(
    mod2: np.ndarray, b2: float, n2: int
) -> Optional[Tuple[float, float]]:
    """K2 with K2*K2 == max shell |k|^2 exactly, and the next double above.

    Returns None when no double squares exactly to the shell value (the
    squaring map is not onto for every target).
    """
    target = float(mod2[np.rint(mod2 / b2) == n2].max())
    k = math.sqrt(target)
    for _ in range(8):
        k = math.nextafter(k, 0.0)
    on = None
    for _ in range(17):
        if k * k == target:
            on = k
        k = math.nextafter(k, math.inf)
    if on is None:
        return None
    above = math.nextafter(on, math.inf)
    while above * above <= target:
        above = math.nextafter(above, math.inf)
    return on, above


def build_boundary(pack: Path, work: Path) -> Dict[str, List[str]]:
    """kernel_boundary_{on_shell,above_shell}.npz (d=2, N=64, L=8)."""
    b2 = (2 * math.pi / 8.0) ** 2
    probe, cmd0 = run_harness(
        "kernel",
        {
            "pack": str(pack),
            "d": "2",
            "n": "64",
            "k-inner": (0.0).hex(),
            "k-outer": (3.5 * math.sqrt(b2)).hex(),
        },
        work,
        "boundary_probe",
    )
    mod2 = np.array(_parse_out(probe)["mod2"])
    for n2 in SHELL_CANDIDATES:  # first shell whose |k|^2 is a perfect double square
        found = _shell_threshold(mod2, b2, n2)
        if found is not None:
            break
    assert found is not None, "no usable shell"
    on, above = found
    shell = mod2[np.rint(mod2 / b2) == n2]
    print(
        f"  boundary shell n^2={n2}: {len(shell)} half-space vectors, "
        f"distinct |k|^2 = {[float(x).hex() for x in sorted(set(shell))]}",
        flush=True,
    )
    cmds: Dict[str, List[str]] = {}
    for name, k2, incl in (
        ("kernel_boundary_on_shell", on, False),
        ("kernel_boundary_above_shell", above, True),
    ):
        meta = {
            "case": name,
            "shell_excluded": not incl,
            "shell_n2": n2,
            "shell_mod2_hex": [float(x).hex() for x in sorted(set(shell))],
            "k_outer_on_shell_hex": on.hex(),
            "k_outer_above_hex": above.hex(),
            "probe_cmd": cmd0,
            "seed": SEED_BASE + 2,
            "Ka": None,
        }
        window = {"k-inner": (0.0).hex(), "k-outer": k2.hex()}
        cmds[name] = _kernel_golden(name, 2, 64, pack, window, meta, work)
    m_on, m_ab = (int(np.load(DATA_DIR / f"{k}.npz")["m"]) for k in cmds)
    assert m_ab - m_on == len(shell), f"shell members {m_ab - m_on} != {len(shell)}"
    return cmds


# ============================================================
# TRAJECTORY GOLDENS
# ============================================================


def _traj_run(
    pack: Path, d: int, n: int, window: Dict[str, str], work: Path, tag: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, Dict[str, Any]]:
    """Run ``traj``; return s0 (B,N,d), s (P,B,N,d), E (P,B), cmd, parsed."""
    args = {
        "pack": str(pack),
        "d": str(d),
        "n": str(n),
        "count": str(N_CONFIGS),
        "pmax": str(P_MAX),
        **window,
    }
    path, cmd = run_harness("traj", args, work, tag)
    p = _parse_out(path)
    s0 = np.array(p["cfg"])
    rows = p["R"]
    assert len(rows) == N_CONFIGS * P_MAX, f"{tag}: {len(rows)} trajectory rows"
    s = (
        np.array([r[2] for r in rows])
        .reshape(N_CONFIGS, P_MAX, n, d)
        .transpose(1, 0, 2, 3)
    )
    e = np.array([r[1] for r in rows]).reshape(N_CONFIGS, P_MAX).T
    return s0, s, e, cmd, p


def _accepted(s0: np.ndarray, s: np.ndarray) -> np.ndarray:
    """(P,B) bool: iteration m accepted iff s_m != s_{m-1} (exact)."""
    prev = np.concatenate([s0[None], s[:-1]], axis=0)
    return np.any(prev != s, axis=(2, 3))


def _horizon(acc: np.ndarray, acc_pert: np.ndarray) -> np.ndarray:
    """First m (1-based) at which the two accept sequences differ, else P."""
    diff = acc != acc_pert
    first = np.argmax(diff, axis=0) + 1
    return np.where(diff.any(axis=0), first, acc.shape[0]).astype(np.int64)


def build_traj(packs: Dict[Tuple[int, int], Path], work: Path) -> Dict[str, List[str]]:
    """traj_d2_n64_chi0.25.npz and traj_d3_n100_chi0.05.npz."""
    cmds: Dict[str, List[str]] = {}
    for d, n, chi in ((2, 64, 0.25), (3, 100, 0.05)):
        pack = packs[(d, n)]
        meta, window = _chi_meta(chi, d)
        name = f"traj_d{d}_n{n}_chi{chi}"
        s0, s, e, c1, p = _traj_run(pack, d, n, window, work, name)
        pert = Path(str(pack) + "_pert")
        s0p, sp, _, c2, _ = _traj_run(pert, d, n, window, work, name + "_pert")
        assert np.array_equal(s0p, np.mod(s0 + NOISE_EPS, 1.0)), "perturbed s0"
        acc, accp = _accepted(s0, s), _accepted(s0p, sp)
        hor = _horizon(acc, accp)
        params = {
            "d": d,
            "N": n,
            "phi": PHI,
            "chi": p["CHI"],
            "M": p["M"],
            "p_max": P_MAX,
            "stored": P_STORE,
            "noise_eps": NOISE_EPS,
            "horizon_rule": "first m where accepted sequences differ, else p_max",
            "accepted_rule": "m accepted iff s_m != s_{m-1}",
            "call": "RelaxStructure_LocalGradientDescent(copy, pot, 0, 0, 0, 2*m)",
            "source_pack": str(pack).replace(str(REPO) + "/", ""),
            "harness_cmds": [c1, c2],
            **_window_params(p),
            **meta,
            "perturbed_max_abs_ds_at_p_max": np.abs(sp[-1] - s[-1])
            .max(axis=(1, 2))
            .tolist(),
        }
        _save(
            name,
            params,
            {
                "s0": s0,
                "iters": np.arange(1, P_STORE + 1, dtype=np.int64),
                "s": s[:P_STORE],
                "energy": e[:P_STORE],
                "horizon": hor,
                "accepted": acc[:P_STORE],
                "accepted_full": acc,
                "accepted_perturbed_full": accp,
            },
        )
        print(
            f"  {name}: horizon={hor.tolist()} "
            f"accepted@200={acc[:P_STORE].sum(0).tolist()}",
            flush=True,
        )
        cmds[name] = [c1, c2]
    return cmds


# ============================================================
# STRUCTURE-FACTOR GOLDEN
# ============================================================


def build_sk(work: Path) -> Dict[str, List[str]]:
    """sk_d3_chi0.05.npz from the first 50 relaxed task_0000 configs."""
    args = {"pack": str(SUCCESS_PACK), "count": str(SK_CONFIGS)}
    path, cmd = run_harness("sk", args, work, "sk_d3_chi0.05")
    p = _parse_out(path)
    s = np.array(p["cfg"])
    assert np.array_equal(s, read_pack(SUCCESS_PACK, SK_CONFIGS)), "sk: C++ s != pack"
    table = np.array(p["T"])
    assert table.ndim == 2 and table.shape[1] == 4, "sk table shape"
    params = {
        "d": 3,
        "N": 100,
        "phi": PHI,
        "chi_target": 0.05,
        "n_configs": SK_CONFIGS,
        "call": "IsotropicStructureFactor(getC, 50, 5, 400, Sk)  (KPrecision=0.01)",
        "columns": "mean|k|, mean S, std|k|, std-error S",
        "source_pack": str(SUCCESS_PACK),
        "harness_cmd": cmd,
    }
    _save("sk_d3_chi0.05", params, {"s": s, "table": table})
    print(f"  sk_d3_chi0.05: table {table.shape}", flush=True)
    return {"sk_d3_chi0.05": [cmd]}


# ============================================================
# L-BFGS GOLDENS
# ============================================================


def _lbfgs_stdin(ka: float, pack: Path) -> str:
    """The stdin token stream of the cco binary (ground/input/step)."""
    toks = [
        "3",
        "0",
        repr(ka),
        "0",
        "0",
        SIGMA,
        repr(PHI),
        "1",
        "100",
        str(N_CONFIGS),
        "0",
        "input",
        "LBFGS",
        str(pack),
        "ccopt",
        "ground",
        "step",
    ]
    return " ".join(toks) + "\n"


def _count_pack(prefix: Path) -> int:
    """Number of configs in a ConfigPack index (0 if absent)."""
    f = Path(str(prefix) + ".ConfigPackIndex")
    return int(np.fromfile(f, "<i8")[0]) if f.is_file() else 0


def build_lbfgs(pack: Path, work: Path) -> Dict[str, List[str]]:
    """Run the cco binary on the d=3 Poisson pack for chi = 0.05, 0.25."""
    cmds: Dict[str, List[str]] = {}
    for chi in (0.05, 0.25):
        meta, window = _chi_meta(chi, 3)
        ka = ka_from_chi(chi, PHI, 3)
        wd = work / f"lbfgs_chi{chi}"
        if wd.exists():  # fresh run dir: the binary appends to existing packs
            shutil.rmtree(wd)
        wd.mkdir(parents=True)
        stdin = _lbfgs_stdin(ka, pack)
        cmd = [str(LBFGS_BIN), *LBFGS_ARGV]
        t0 = time.time()
        with open(wd / "run.stdout", "w") as log:
            rc = subprocess.run(
                cmd,
                input=stdin,
                text=True,
                cwd=wd,
                env=_env(),
                stdout=log,
                stderr=subprocess.STDOUT,
            ).returncode
        assert rc == 0, f"L-BFGS binary exited {rc}"
        dt = time.time() - t0
        args = {
            "pack": str(wd / "ccopt"),
            "d": "3",
            "n": "100",
            "rescale": "0",
            "count": str(N_CONFIGS),
            **window,
        }
        path, hcmd = run_harness("kernel", args, wd, "energy_recompute")
        p = _parse_out(path)
        e = np.array(p["E"])
        success = e < TOL_F32
        n_succ = _count_pack(wd / "ccopt_Success")
        assert n_succ == int(success.sum()), f"Success pack {n_succ} != {success.sum()}"
        name = f"lbfgs_d3_chi{chi}"
        params = {
            "d": 3,
            "N": 100,
            "phi": PHI,
            "M": p["M"],
            "chi": p["CHI"],
            "binary": str(LBFGS_BIN),
            "argv": list(LBFGS_ARGV),
            "stdin": stdin.strip(),
            "tol_float32": TOL_F32,
            "success_rule": "energy_final < float32(1e-20)",
            "source_pack": str(pack).replace(str(REPO) + "/", ""),
            "wall_seconds": dt,
            "energy_cmd": hcmd,
            **_window_params(p),
            **meta,
        }
        _save(
            name,
            params,
            {
                "s0": read_pack(pack, N_CONFIGS),
                "energy_final": e,
                "success": success,
                "s_final": np.array(p["cfg"]),
            },
        )
        print(
            f"  {name}: E={e.tolist()} success={success.tolist()} ({dt:.1f}s)",
            flush=True,
        )
        cmds[name] = [f"cd <scratch>/lbfgs_chi{chi} && {_display(cmd)} < stdin", hcmd]
    return cmds


# ============================================================
# MANIFEST
# ============================================================


def sha256(path: Path) -> str:
    """SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _ls_listing() -> str:
    """``ls -l`` of the C++ dirs (full ISO times); hashed into the manifest."""
    out = ""
    for sub in (".", "CCOptimization", "EXC"):
        out += subprocess.run(
            ["ls", "-l", "--time-style=full-iso", str(CCO_DIR / sub)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    return out


def _data_files() -> List[Path]:
    """All files covered by the manifest, sorted."""
    return sorted(
        p for p in DATA_DIR.rglob("*") if p.is_file() and p.name != "MANIFEST.json"
    )


def write_manifest(cmds: Dict[str, List[str]], timings: Dict[str, float]) -> None:
    """Write MANIFEST.json with provenance and SHA-256 of every file."""
    flags = subprocess.run(
        ["make", "-s", "-C", str(TOOL_DIR), "flags"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    gxx = subprocess.run(["g++", "--version"], capture_output=True, text=True).stdout
    listing = _ls_listing()
    srcs = sorted(CCO_DIR.glob("*.cpp")) + sorted(CCO_DIR.glob("*.h"))
    man = {
        "generated_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "compile_command": flags,
        "compiler": gxx.splitlines()[0],
        "omp_num_threads": 1,
        "harness_binary": {
            "path": str(HARNESS),
            "sha256": sha256(HARNESS),
            "mtime": time.ctime(HARNESS.stat().st_mtime),
        },
        "lbfgs_binary": {
            "path": str(LBFGS_BIN),
            "sha256": sha256(LBFGS_BIN),
            "mtime_epoch": LBFGS_BIN.stat().st_mtime,
            "mtime": time.ctime(LBFGS_BIN.stat().st_mtime),
        },
        "cpp_dir_ls_l": listing,
        "cpp_dir_ls_l_sha256": hashlib.sha256(listing.encode()).hexdigest(),
        "cpp_sources_sha256": {p.name: sha256(p) for p in srcs},
        "main_cpp_sha256": sha256(CCO_DIR / "CCOptimization" / "main.cpp"),
        "commands": cmds,
        "timings_seconds": timings,
        "files": {str(p.relative_to(DATA_DIR)): sha256(p) for p in _data_files()},
    }
    (DATA_DIR / "MANIFEST.json").write_text(
        json.dumps(man, indent=1, sort_keys=True) + "\n"
    )


# ============================================================
# CHECK
# ============================================================


def _load_golden_module() -> Any:
    """Import tests/golden.py by file path."""
    spec = importlib.util.spec_from_file_location(
        "golden_schema", REPO / "tests" / "golden.py"
    )
    assert spec is not None and spec.loader is not None, "cannot import tests/golden.py"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check() -> int:
    """Re-hash every file against MANIFEST.json and validate each .npz."""
    man = json.loads((DATA_DIR / "MANIFEST.json").read_text())
    golden = _load_golden_module()
    bad: List[str] = []
    on_disk = {str(p.relative_to(DATA_DIR)) for p in _data_files()}
    bad += [f"unlisted file {x}" for x in sorted(on_disk - set(man["files"]))]
    for rel, digest in man["files"].items():
        path = DATA_DIR / rel
        if not path.is_file():
            bad.append(f"missing {rel}")
        elif sha256(path) != digest:
            bad.append(f"hash mismatch {rel}")
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as z:
                data = {k: z[k] for k in z.files}
            try:
                golden.validate_golden(path.stem, data)
            except AssertionError as exc:
                bad.append(f"schema {rel}: {exc}")
    total = sum(p.stat().st_size for p in _data_files())
    if total > MAX_BYTES:
        bad.append(f"total size {total} > {MAX_BYTES}")
    for msg in bad:
        print("FAIL:", msg, flush=True)
    print(
        ("OK" if not bad else "FAILED"),
        f"({len(man['files'])} files, {total} bytes)",
        flush=True,
    )
    return 1 if bad else 0


# ============================================================
# MAIN
# ============================================================


def generate(work: Path) -> int:
    """Regenerate every golden file and the manifest."""
    assert HARNESS.is_file(), f"build the harness first: make -C {TOOL_DIR}"
    assert LBFGS_BIN.is_file(), f"missing {LBFGS_BIN}"
    work.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cmds: Dict[str, List[str]] = {}
    timings: Dict[str, float] = {}
    packs: Dict[Tuple[int, int], Path] = {}
    stages = (
        ("inputs", lambda: write_inputs()),
        ("kernel", lambda: build_kernels(packs, work)),
        ("boundary", lambda: build_boundary(packs[(2, 64)], work)),
        ("traj", lambda: build_traj(packs, work)),
        ("sk", lambda: build_sk(work)),
        ("lbfgs", lambda: build_lbfgs(packs[(3, 100)], work)),
    )
    for name, fn in stages:
        t0 = time.time()
        print(f"[{name}]", flush=True)
        res = fn()
        if name == "inputs":
            packs.update(res)
        else:
            cmds.update(res)
        timings[name] = round(time.time() - t0, 2)
        print(f"[{name}] {timings[name]} s", flush=True)
    write_manifest(cmds, timings)
    return check()


if __name__ == "__main__":
    # =========================================================
    # ARGUMENTS
    # =========================================================
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify manifest + schema")
    ap.add_argument(
        "--workdir",
        type=Path,
        default=DEFAULT_WORK,
        help="scratch dir for harness/binary output (outside the repo)",
    )
    ns = ap.parse_args()
    # =========================================================
    # RUN
    # =========================================================
    sys.exit(check() if ns.check else generate(ns.workdir))
