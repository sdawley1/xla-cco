# C++ reference tools

Tools that run the group's C++ CCO code in `/home/sd2402/ccoptimization/cco/`
(read-only: include paths, `-c` compilation and running the existing binary;
nothing is ever written there).

## Harness (`dump_reference`)

`dump_reference.cpp` and the `Makefile` are restored from the removed golden
harness (commit `0013948^`). They compile the cco sources, except
`CCOptimization/main.cpp`, with the cco makefile flags
(`g++ -Wall -std=c++11 -O3 -fopenmp -DDELLA -I<cco>`, linking
`-lgsl -lgslcblas -lnlopt -lm`). The objects go to `build/`, which is
git-ignored.

```bash
make -C tools/cpp_reference          # -> tools/cpp_reference/build/dump_reference
```

The MD reference uses only `kernel`. It builds the k-set as main.cpp:242-300
does (`GetKs(Config, k2, k2, 1)` with a strict `k1^2 < |k|^2 < k2^2`), then
prints `KOUTER` (as `%a`), `M`, every `KVEC`, and for each configuration the
`ShiftedCCPotential` energy `E` (as `%.17g`) and the forces. It runs with
`ParallelNumThread = 1`. Use `--rescale 0` for packs that already hold
Cartesian boxes, such as the MD output.

## MD reference data (plan T4)

`md_reference.py` runs the **unmodified** binary
`/home/sd2402/ccoptimization/cco/EXC/soft_core_stealthy2.out` in MD mode
(`CollectiveCoordinateMD`, main.cpp:725-890). It then writes
`tests/data/md_cxx_reference.json` and `tests/data/md_cxx_samples.npz`, which
`tests/unit/test_md_cxx_reference.py` checks.

### Command

One run per seed, with argv `<hours> 1e-20 1000 <seed>` and this stdin:

```
2 0.0 0.14399483155201312 0.0 0.0 0.5 0.05 1 256 200 20 random LBFGS cxxmd MD step 1e-05
```

The tokens are `d K1a Ka S0 val sigma phi_fake nthreads N Nc Nc_equi`, then
`random`, the algorithm token (parsed but unused in MD), the savename, `MD`,
`potential_type`, and T. T is the required token after `potential_type`
(main.cpp:231-236).

| Parameter | Value | Note |
|---|---|---|
| d, N | 2, 256 | |
| box side L | 1000^(1/2) = 31.6228 | random mode builds the box from the stale `num = 1000` (D-004), so the density is N/V = 0.256 |
| M | 50 | target round(0.1 · 2 · 255) = 51 is not achievable: half-space shell counts are even (…, 48, 50, 54, …) |
| chi | 50/510 = 0.0980 | |
| K | 1.1413972932085459 | the mid-gap between the n² = 32 and n² = 34 shells, K = Ka/a |
| Ka, phi_fake | 0.14399483155201312, 0.05 | a = (phi_fake/π)^(1/2); phi_fake only fixes a |
| S0, val, sigma | 0, 0, 0.5 | pure stealthy step potential; the soft core is off, so sigma is unused |
| T | 1e-5 | kT, with k_B = 1 and m = 1 |
| Nc, Nc_equi | 200, 20 | |
| nthreads | 1 | serial sums, so the harness and the MD energies are bit-identical |
| seeds | 1, 2, 3 | argv[4] |

These values are hard-coded in main.cpp and cannot be set from stdin:
- dt = 0.01 (`MDTimeStep`, line 378). The random mode passes
  `MDAutoTimeStep = false`, so dt never changes.
- `StepPerSample` = 100000 (line 379).
- Andersen p = 0.01 (line 852).
- A pre-relaxation `RelaxStructure_NLOPT` with 1000 evaluations (line 741).

A serial run takes about 64 s per 100000 steps on a login node and 77 s on
della-i13 compute nodes. A seed (220 × 100000 steps) took 4.8 h (array job
17107313, 2026-11-08), so Nc did not need to be reduced.

### Protocol (recorded in the JSON under `protocol`)

1. Place N points uniformly at random. Relax with L-BFGS (1000 evaluations).
   Set the velocities to 0. The first half-kick uses a zero acceleration
   (D-051).
2. Equilibrate: repeat Nc_equi times, 2 × [`SetRandomSpeed(T)`, which draws
   every component as N(0, T) with no COM removal; then NVE velocity Verlet
   for 50000 steps].
3. Sample: repeat Nc times [`AndersonEvolve(50000, T, p = 0.01)`; NVE
   `Evolve(50000)`; then log Φ, KE, dt and pressure, and append the positions
   to `cxxmd.ConfigPack`].
4. Andersen count mode (D-050). `ResetCount = 0` at the start of every call.
   After each full Verlet step, `ResetCount += p`. While `ResetCount > 0`,
   draw one particle `floor(U N)` with replacement, give it fresh N(0, T)
   components, and do `ResetCount -= 1`. The result is 501 resets per
   50000-step call, at steps 1, 100, 200, …, 50000.
5. The COM momentum is never removed. KE counts all N·d components.

### Units mapping (C++ to xla_cco)

- **Box:** build `Box(basis = L I)` with L from the JSON. The xla K equals the
  C++ k2 = `k_outer_from_ka(Ka, phi_fake, d)`, and the test checks the two
  are bit-equal.
- **Φ:** the C++ `Energy()` is `sum_{half-space k} |rho_k|^2 / V + Shift`,
  with Shift = 0. This is exactly `StealthyEnergy`; there is no per-particle
  normalization. In the JSON, `phi_cpp` is the harness `%.17g` recompute,
  `phi_xla` is the xla value, and `phi_log` is the 6-digit log value (D-052).
- **KE:** `sum m v^2 / 2` with m = 1, so `<KE> = N d T / 2`. Velocities are
  not saved, so KE exists only at the log's 6 digits.
- **T:** used directly as kT.

### Unit gate

For every sampled configuration, the xla `StealthyEnergy` must equal the C++
`ShiftedCCPotential` energy of the same ConfigPack configuration to a relative
1e-8. The log alone cannot serve as the reference, because it has only 6
significant digits (D-052). The `collect` step asserts the gate on every
sample. It also checks that `%g` of the C++ energy equals the logged string,
that K is bit-equal, and that the C++ and xla k-sets are the same.
`tests/unit/test_md_cxx_reference.py` recomputes Φ for the stored first 5
configurations of each seed, and checks units through `<Φ>/(MT)` and
`<KE>/(NdT/2)`.

### Regenerate

```bash
make -C tools/cpp_reference
mkdir -p /scratch/gpfs/TORQUATO/sd2402/xla_cco_runs/md_cxx
env -u JAX_PLATFORMS sbatch slurm/md_cxx_reference_cpu.slurm      # 3 array tasks
# after the jobs finish, on a login node:
PYTHONPATH=$PWD/src JAX_PLATFORMS=cpu UV_PROJECT_ENVIRONMENT=<venv> \
    uv run --no-sync python tools/cpp_reference/md_reference.py collect --seeds 1 2 3
```

`run` refuses a seed directory that already holds a run; the binary would
restart from an `.MDDump` or append to the pack. Run logs go to
`/scratch/gpfs/TORQUATO/sd2402/xla_cco_runs/md_cxx/seed<k>/`, and SLURM logs
to `.../md_cxx/slurm-mdcxx-<job>_<k>.out`.

## JAX replica of the MD protocol (plan V2)

`md_jax_replica.py` replays the protocol above with `xla_cco.dynamics` in the
same box and K, for the parity test `tests/parity/test_md_cxx_parity.py`.

- Each lane gets its own uniform random start, relaxed by L-BFGS (1000
  evaluations, one pass), with zero velocities. Equilibration is
  20 × 2 × [all components N(0, T), no COM removal; NVE 5e4 steps], and
  sampling is 20 × [count Andersen 5e4; NVE 5e4; record Φ, KE].
- Andersen schedule: each Andersen phase sets `reset_acc = 1.0` (the C++
  counter plus one). The library rule then fires at exactly the C++ steps
  1, 100, …, 50000 (D-050, D-081). D-051 is not replicated.
- Layout: 10 array tasks × 8 vmapped lanes (4 at T, 4 at 1.1 T, the power
  control), 20 samples per lane: 800 samples at T and 800 at 1.1 T.

```bash
mkdir -p experiments/trajectories
env -u JAX_PLATFORMS sbatch slurm/md_jax_replica_cpu.slurm          # 10 array tasks
# after the jobs finish, on a login node:
PYTHONPATH=$PWD/src JAX_PLATFORMS=cpu UV_PROJECT_ENVIRONMENT=<venv> \
    uv run --no-sync python tools/cpp_reference/md_jax_replica.py collect
```

Task files go to `/scratch/gpfs/TORQUATO/sd2402/xla_cco_runs/md_jax_replica/`,
SLURM logs to `experiments/trajectories/slurm-mdjaxrep-<job>_<k>.out`, and the
merged result (with its cost record) to `tests/data/md_jax_replica.json`.
