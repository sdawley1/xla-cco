# C++ reference goldens

Golden data for the JAX port, produced by the unmodified C++ CCO code in
`/home/sd2402/ccoptimization/cco/` (read-only; only include paths and `-c`
compilation, nothing is written there).

## Rebuild

```bash
export UV_CACHE_DIR=/scratch/gpfs/TORQUATO/sd2402/.uv-cache
make -C tools/cpp_reference                      # build/dump_reference
.venv/bin/python -I tools/cpp_reference/make_goldens.py            # regenerate
.venv/bin/python -I tools/cpp_reference/make_goldens.py --check    # hashes + schema
# or, on Della:  sbatch slurm/make_cpp_reference.slurm   (CPU, ~2 min)
```

The Makefile compiles `dump_reference.cpp` plus every cco source except
`CCOptimization/main.cpp` with `g++ -Wall -std=c++11 -O3 -fopenmp -DDELLA
-I<cco>` and links `-lgsl -lgslcblas -lnlopt -lm` (the cco makefile flags).
Objects go to `tools/cpp_reference/build/` (git-ignored). Generation runs with
`OMP_NUM_THREADS=1` and `ParallelNumThread = 1` so every sum is sequential.
Scratch output (harness text, L-BFGS run directories, `DetailedLog.txt`) goes to
`--workdir` (default `/scratch/gpfs/TORQUATO/sd2402/cco_golden_work`), never the
repository or the cco tree.

`tests/data/cpp/MANIFEST.json` records compile command and compiler, SHA-256 of
the harness and the L-BFGS binary, the binary mtime, the `ls -l` listing and hash
of the cco directories, SHA-256 of every cco source, every command line, stage
timings, and SHA-256 of every file in `tests/data/cpp/`.

## Harness (`dump_reference`)

| subcommand | C++ calls |
|---|---|
| `kernel` | `GetKs(Config, k2, k2, 1)` + strict filter `k1^2 < Modulus2 < k2^2` (main.cpp:242-300); `ShiftedCCPotential::AddConstraint/Energy/AllForce` |
| `traj` | `RelaxStructure_LocalGradientDescent(copy, pot, 0, 0, 0, 2*m)` for m = 1..P, each from the initial config |
| `sk` | `IsotropicStructureFactor(getC, n, 5, 400, Sk)` (default `KPrecision = 0.01`) |

Window: `--ka KA --phi PHI [--k1a K1A]` (k = Ka / a with a from `HyperSphere_Volume`,
as main.cpp) or raw `--k-inner/--k-outer` (C99 hex floats, parsed by `strtod`).
Input packs are unit-cell ConfigPacks scaled by `N^(1/d)` (main.cpp:211) unless
`--rescale 0`. Floats are printed with `%.17g`; k-vector `Modulus2` and the used
window with `%a`. The potential is `ShiftedCCPotential` directly; with `val = 0`
`RepulsiveCCPotential` adds exactly `0.0`, so energies and forces are identical.

## Inputs (`tests/data/cpp/inputs/`)

Seeded NumPy packs `poisson_d{d}_n{N}` for (1, 50), (2, 64), (3, 100): seed
`20261100 + d`, 5 configs = 4 Poisson (indices 0-3) + 1 near-ordered (index 4:
lattice `(i+0.5)/m` with 50 / 8x8 / 5x5x4 points, plus uniform +/-1e-3 fractional
noise, wrapped). `*_pert` are the same with `+1e-15` added and wrapped (noise
horizon). The C++ ChunkPack inputs are the first 5 configs of
`cco_runs/p3d-N100-phi0.05-chi0.05-val0.0/ic/chunk_0000`.

## Goldens (`tests/data/cpp/*.npz`, keys per `tests/golden.py`; `params` is JSON)

All `params` record d, N, phi, M, chi, source pack, seed, the raw `k_inner` /
`k_outer` used (decimal and `float.hex()`), and the exact command lines.

* `kernel_d{1,2,3}_chi{0.05,0.25,0.45}`: phi = 0.05, Ka = `ka_from_chi`.
  `n` (M,d) int64 (C++ order, recovered from C++ k by the reciprocal basis, error
  < 1e-9), `m`, `chi = M/(d(N-1))`, `s` (5,N,d), `energy`, `forces` (Cartesian,
  `AllForce`).
* `kernel_d3_cpp_ic`: chi 0.05 on the C++ ic configs (M = 13, chi = 0.043771,
  matching engine.log).
* `kernel_boundary_on_shell` / `kernel_boundary_above_shell`: d = 2, N = 64,
  L = 8; K2 with `K2*K2 ==` the C++ `Modulus2` of the n^2 = 8 shell (shell
  excluded), and `nextafter(K2, +inf)` (shell included). `params` lists the shell
  `Modulus2` values.
* `traj_d2_n64_chi0.25`, `traj_d3_n100_chi0.05`: `s0`, `iters` = 1..200, `s`
  (200,B,N,d) wrapped fractional as `GetConfig` produces, `energy` (200,B),
  `horizon` (B,) and extras `accepted` (200,B) (iteration m accepted iff
  `s_m != s_{m-1}`), `accepted_full`, `accepted_perturbed_full` (256,B). `horizon` =
  first m in 1..256 where the accept sequences of the base and `+1e-15` runs differ,
  else 256. Runs to 256 are not stored beyond m = 200.
* `sk_d3_chi0.05`: first 50 relaxed configs of `task_0000/ccopt-t0000_Success`;
  `table` columns = mean |k|, mean S, std |k|, std error S.
* `lbfgs_d3_chi{0.05,0.25}`: unmodified `EXC/soft_core_stealthy2.out`
  (argv `1 1e-20 500000 0`, stdin per `params.stdin`) on the d = 3 Poisson pack.
  `s0`, `energy_final` (harness recompute on the output pack), `success = E <
  float32(1e-20)`, extra `s_final`.
