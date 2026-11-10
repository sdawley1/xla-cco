# V3: GPU benchmark, time step, winding cadence and production cost

Samuel J. Dawley, 11/2026. Code: `benchmarks/md_bench.py` (modes `throughput`, `grid`, `calib`, `omega`, `project`); SLURM: `experiments/slurm/batten2008_md_gpu.slurm` (STEP=bench). The machine-readable results, including every measurement, are in `experiments/md_winding/bench.json`. The V3 jobs logged to `/scratch/gpfs/TORQUATO/sd2402/xla_cco_runs/v3/` (overriding `--output`); copies of all 8 logs are in `experiments/md_winding/logs/` (git-ignored `slurm-*.out`), which is now the script's default `--output`. Raw outputs are in `/scratch/gpfs/TORQUATO/sd2402/xla_cco_runs/v3/`:
- `throughput.jsonl`, `omega.json`;
- `calib2/calib_<tag>.json` and `_wind.npz` (the ladders and raw winding);
- `calib3/calib_<tag>.json` (the final calibration).

## Chosen settings (production)

| | χ = 0.1 (`1-10`) | χ = 0.4 (`2-5`) |
|---|---|---|
| dt | **0.04** | **0.02** |
| ω_max · dt | 0.10 | 0.12 |
| max \|E − e0\| / (dof T/2) over 1e6 NVE steps | 8.9e-5 | 2.0e-4 |
| `--drift-tol` (5 × max, rounded up to 1-2-5) | **5e-4** | **2e-3** |
| `--wind-every` (unconstrained waves) | **5** | **2** |
| `--k-chunk` | **none** (unchunked) | **none** |
| lanes per E1 job | **40** (10 ground states × 4 T) | **40** |
| `--record-rho-every` / `--max-rho-records` (A4) | 2000 / 1300 | 2000 / 1300 |
| ω_min (softest constrained mode, all starts) | 0.182 | 0.173 |

**Notes on the settings:**
- **Trace (Amendment A4):** 500 samples per NVE segment (n_nve / 2000). The trace cadence is global, so the 1300 rows cover the whole run, including an extended equilibration (2.6e6 steps).
- **Default drift_tol:** the CLI default of 1e-3 would leave only a 5× margin at χ = 0.4. Pass the values above.
- **Langevin (E4):** both γ = 0.003 and γ = 0.03 are below ω_min ≥ 0.173, by factors of 58 and 5.8.

**Projected cost:** no reduction is needed.

| Line | Projected GPU-h | Budget |
|---|---|---|
| Production: E1 + E1b | 18.0 | 60 |
| Langevin: E4 | 5.8 | 12 |

## Hardware and method

- **Throughput hardware.** Throughput was timed on an A100-SXM4-80GB (gputest). A sanity run of the heaviest case on an A100-PCIE-40GB was 17 % slower (210 against 253 steps/s), so projections multiply times by 1.25.
- **Calibration hardware.**
  - The final calibration runs (17119585/6, 17121008/10) used A100 80GB PCIe.
  - The first χ = 0.4 attempt landed on a MIG 3g.40gb slice. MIG slices are billed as one GPU but are much slower, so the SLURM script now requests `--constraint=nomig`.
- **Throughput runs.**
  - Each group (system, χ, lanes, waves, k-chunk, scheme) runs in a fresh process, so `peak_bytes_in_use` is that group's own. Rows after the first wind_every in a group carry the group's running peak, which makes them upper bounds.
  - Each wind_every gets one compile call (1 step), then 3000 timed steps in 1000-step chunks.
  - The production recording buffers are always allocated: 200 snapshots and the A4 trace of 1300 rows × all tracked waves.
- **Lanes.** Lanes are ground state major. 10 lanes = 5 ground states × T ∈ {1e-6, 1e-4}; 40 lanes = 10 × 4 T.
- **Calibration.** Canonical lanes are drawn with `cli.md.init_lanes` (harmonic start + MB velocities), then given 2e4 steps of count-mode Andersen with p = 1 at dt = 0.01. Each system (flows2d, hip2d) has 4 ground states × 4 T. Every trial restarts NVE (`init_state`, COM velocity removed) from these states.

## 1. Throughput

Steps/s of the whole batch (lane-steps/s = steps/s × lanes), mean of flows2d and hip2d (they agree within 4 %). `2K` means all waves with |k| < 2K; at χ = 0.4 that is W = 3196–3278, of which M = 802–820 are constrained.

| χ | lanes | waves | scheme | wind_every | k-chunk none | 64 | 256 | peak GiB (none) |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 10 | constrained | verlet | 1 | 4249 | 3246 | 4206 | 0.3 |
| 0.1 | 10 | 2K | verlet | 1 | 1597 | 1147 | 1454 | 0.9 |
| 0.1 | 10 | 2K | verlet | 5 | 2878 | 2313 | 2871 | ≤ 1.3 |
| 0.1 | 10 | 2K | baoab | 1 | 2539 | – | – | 0.9 |
| 0.1 | 40 | constrained | verlet | 1 | 1731 | 1597 | 1745 | 1.2 |
| 0.1 | 40 | 2K | verlet | 1 | 775 | 557 | 526 | 3.2 |
| 0.1 | 40 | 2K | verlet | 2 | 1044 | 931 | 790 | ≤ 4.2 |
| 0.1 | 40 | 2K | verlet | 5 | 1337 | 1209 | 1147 | ≤ 4.3 |
| 0.1 | 40 | 2K | verlet | 10 | 1475 | 1345 | 1347 | ≤ 4.3 |
| 0.1 | 40 | 2K | andersen | 1 | 937 | – | – | 3.2 |
| 0.4 | 10 | constrained | verlet | 1 | 1970 | 1251 | 1667 | 1.0 |
| 0.4 | 10 | 2K | verlet | 1 | 820 | 469 | 663 | 2.9 |
| 0.4 | 10 | 2K | verlet | 5 | 1471 | 916 | 1225 | ≤ 4.0 |
| 0.4 | 10 | 2K | baoab | 1 | 1035 | – | – | 2.9 |
| 0.4 | 40 | constrained | verlet | 1 | 657 | 564 | 511 | 3.5 |
| 0.4 | 40 | 2K | verlet | 1 | 255 | 229 | 150 | 11.2 |
| 0.4 | 40 | 2K | verlet | 2 | 365 | 324 | 229 | ≤ 14.7 |
| 0.4 | 40 | 2K | verlet | 5 | 491 | 429 | 337 | ≤ 14.7 |
| 0.4 | 40 | 2K | verlet | 10 | 554 | 477 | 401 | ≤ 14.7 |
| 0.4 | 40 | 2K | andersen | 1 | 305 | – | – | 11.2 |

### Compile time
The first call takes 2–4.5 s, including one step; later cadences of the same group take ≈ 1 s.

### k-chunk
**Unchunked (`--k-chunk` omitted) is fastest in every configuration that tracks the 2K waves** (production), by 0.05–39 % (median 13 %) over the best chunked variant, and fits easily, so it is the choice. The one exception is constrained-only at χ = 0.1 with 40 lanes, where k-chunk 256 is 0.8 % faster (1745 against 1731 steps/s, within run-to-run noise).

### Memory
- At χ = 0.4, 40 lanes, 2K waves, 200 snapshots and the 1300-row A4 trace, the wind_every = 1 row peaks at 11.2 GiB. The trace alone is 40 × 1300 × 3278 × 16 B = 2.7 GB.
- The chosen χ = 0.4 setting (wind_every = 2) is bounded by ≤ 14.7 GiB (`bench.json` `peak_bytes_40` = 15.8e9 B; that row carries the process peak of the earlier wind_every = 1 run). The PCIe-40GB sanity run at wind_every = 5 also peaked at 15.8 GB.
- Either way it fits a 40 GB A100 with room to spare, so **40 lanes per job** (no fallback to 20).

### Unconstrained waves
Tracking the waves with K < |k| < 2K costs a factor of 2.2–2.7 at wind_every = 1. Cadence recovers much of this: at χ = 0.4 and 40 lanes, wind_every = 2 gives 1.43× and wind_every = 5 gives 1.93×.

### Observation: Verlet is slower than Andersen and BAOAB
The "verlet" program runs 16–17 % slower than "andersen" at 40 lanes, and 21–37 % slower than "baoab" at 10 lanes, although it does strictly less work.
- **Likely cause:** XLA fusion. Only the non-Verlet path wraps the constrained ρ in `com_correct`.
- **Effect on outputs:** none (performance only).
- **Effect on projection:** the projection never credits the thermostat phases with the faster rate (ratio capped at 1).
- **Follow-up:** worth a profiler look, since it could cut E1 cost by up to ~20 %.

## 2. Time step per χ

**Criteria** (plan V3), at every T ∈ {1e-6, 1e-5, 1e-4, 1e-3}:
- **(a) Drift.** |Ē_last − Ē_first| / ⟨KE⟩ < 1e-4, from window means over the first and last 10 % of steps.
- **(b) Fluctuation.** std(E) / ⟨KE⟩ < 1e-2, from 256 uniformly spaced samples.
- **(c) Winding.** Unsigned winding against dt/2, from the same state over the same time, changes by < 1 % (details below).

### Ladder: (a), (b) and max |E − e0|
The ladder used 2e5 NVE steps for every dt, with 2 ground states × 4 T × 2 systems = 16 lanes per χ. The maximum is over lanes. The secular slope is the fitted drift per decile window.

| χ | dt | (a) drift | secular slope / window | (b) std(E)/⟨KE⟩ | max \|E − e0\| / (dof T/2) |
|---|---|---|---|---|---|
| 0.1 | 0.04 | 8.2e-6 | 7.9e-7 | 1.5e-5 | 8.9e-5 |
| 0.1 | 0.02 | 1.7e-6 | 1.8e-7 | 3.6e-6 | 2.0e-5 |
| 0.1 | 0.01 | 2.4e-7 | 3.0e-8 | 8.6e-7 | 4.8e-6 |
| 0.1 | 0.005 | 7.1e-8 | 1.2e-8 | 2.3e-7 | 1.2e-6 |
| 0.4 | 0.04 | 4.4e-5 | 7.5e-6 | 1.2e-4 | 9.3e-4 |
| 0.4 | 0.02 | 1.9e-5 | 2.8e-6 | 3.0e-5 | 2.1e-4 |
| 0.4 | 0.01 | 3.7e-6 | 4.6e-7 | 7.6e-6 | 4.5e-5 |
| 0.4 | 0.005 | 7.7e-7 | 7.2e-8 | 1.8e-6 | 1.1e-5 |

**Reading the ladder:**
- **Every dt passes (a) and (b),** by a factor of ≥ 2 for (a) and ≥ 80 for (b).
- **The energy error is bounded, not secular.** max |E − e0| falls by a factor of 4.0–4.7 per halving of dt. The (a) value does not grow when the run is 5× longer (next bullet), so what it measures is the window-mean scatter of the oscillation, not a trend.
- **The 1e6-step confirmation at the chosen dt agrees with the 2e5-step ladder.**
  - χ = 0.1, dt = 0.04: drift 5.9e-6, std 1.4e-5, max dev 8.9e-5.
  - χ = 0.4, dt = 0.02: drift 1.5e-5, std 3.0e-5, max dev 2.0e-4.
  - In both, max |E − e0| equals the 2e5-step value: the Verlet oscillation saturates, and nothing accumulates.
  - This justifies the 2e5-step ladder; the chosen dt was nevertheless checked at the full 1e6 steps.

### Criterion (c): winding against dt/2
**Setup.**
- All 32 lanes per χ (4 ground states × 4 T × 2 systems), with every wave |k| < 2K tracked.
- NVE over t = 400 at dt and at dt/2, from the same state.

**Statistic.**
- Per temperature and per |k|/K bin (20 bins over (0, 2), as in E2), pooled over systems and lanes:
  - Δ_b = Σ W_dt / Σ W_dt/2 − 1;
  - SE_b = std(W_dt − R·W_dt/2) / (√n · mean W_dt/2), by the delta method.
- A bin passes if |Δ_b| < 1 % when resolved (SE_b ≤ 0.5 %). If noise-dominated (SE_b > 0.5 %), it passes if |Δ_b| − 2 SE_b < 1 %.
- The all-wave ratio of summed winding must also stay within 1 %.

**Results.** Each cell gives max |Δ_b| / max excess in %; ✗ marks a fail.

| χ | dt | T = 1e-6 | 1e-5 | 1e-4 | 1e-3 | verdict |
|---|---|---|---|---|---|---|
| 0.1 | 0.04 | 0.03 / 0.03 | 0.03 / 0.03 | 0.05 / 0.05 | 0.72 / 0.44 | **pass** |
| 0.1 | 0.02 | 0.01 / 0.01 | 0.01 / 0.01 | 0.02 / 0.02 | 0.62 / 0.62 | pass |
| 0.1 | 0.01 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.61 / 0.37 | pass |
| 0.1 | 0.005 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.63 / 0.63 | pass |
| 0.4 | 0.04 | 0.14 / 0.14 | 0.18 / 0.18 | 1.04 / 0.25 | 2.25 / 1.07 ✗ | fail |
| 0.4 | 0.02 | 0.03 / 0.03 | 0.05 / 0.05 | 1.00 / 0.34 | 0.72 / 0.72 | **pass** |
| 0.4 | 0.01 | 0.01 / 0.01 | 0.01 / 0.01 | 1.18 / 0.59 | 0.88 / 0.64 | pass |
| 0.4 | 0.005 | 0.00 / 0.00 | 0.00 / 0.00 | 0.81 / 0.81 | 0.71 / 0.71 | pass |

**Coherent regime.** Where trajectories stay coherent (all T ≤ 1e-4 at χ = 0.1; T ≤ 1e-5 at χ = 0.4), the dt effect is resolved to < 0.01 % precision. It falls roughly as dt² and is at most 0.18 % (χ = 0.4, dt = 0.04).

**Decorrelated regime.** At T = 1e-3, and at T = 1e-4 for χ = 0.4, the dt and dt/2 runs decorrelate within t = 400.
- **Noise floor:** per-bin SE reaches 0.4–0.9 %, so |Δ_b| ≈ 0.6–1.2 % at every dt, down to 0.005.
- **χ = 0.4, dt = 0.04 fails the pre-set rule on one bin**: T = 1e-3, bin 10 (|k| ≈ K), Δ = −2.25 ± 0.59 % (z = −3.8 with the delta-method SE).
  - With a lane-cluster jackknife SE (0.82 %, from the audit) z is −2.7, and the bin would pass (excess 0.61 %).
  - The sign is opposite to the expected Verlet bias (the effective frequency rises, so winding should increase). In the same bin, dt = 0.02, 0.01 and 0.005 scatter by +0.58, +0.48 and −0.67 %.
  - Over 120 near-null decorrelated bins the largest |z| is already 3.68, so this is consistent with a noise excursion.
  - The rejection is therefore conservative. We keep dt = 0.02 at χ = 0.4 as the conservative choice, since the cost fits the budget (orchestrator decision).
  - The trade-off: in the coherent regime the dt effect at dt = 0.04 is only 0.14–0.18 %, and dt = 0.02 halves the physical NVE time (2e4 instead of 4e4 per segment) at the same cost.
- **Power limit of the rule:** in noise-dominated bins the burden of proof is reversed. A pass means "no bin shift above about 1 % + 2 SE detected", not "below 1 % demonstrated".
  - From the audit's injected-shift tests: a uniform ≥ 1 % shift fails at T = 1e-4 and 1e-3 (the all-wave check catches it), as does a ≥ 1 % shift confined to the unconstrained waves.
  - A single-bin shift is caught at ≥ 1 % in quiet bins, but in the noisiest bins (10, 19) only at 2–3 %. So at T = 1e-3 single-bin shifts of up to 2–3 % can pass.
  - The harmonic temperatures are covered: the dt effect there is resolved in the coherent regime at ≤ 0.18 % (dt = 0.04, χ = 0.4) and ≤ 0.05 % (chosen dts).

**History of criterion (c).** The criterion changed twice after looking at data; both changes are recorded here and in `V3_jobs.md`.
1. **Per-system binned means** (8 lanes, t = 200; jobs 17117177/8).
   - This chose dt = 0.02 at χ = 0.1.
   - At χ = 0.4 no dt passed. The bins with few waves are noisy once trajectories decorrelate.
2. **Pooled matched pairs with an SE-aware threshold** (16 lanes per system, t = 400; jobs 17119585/6).
   - This used the *mean of per-wave ratios*.
   - **At χ = 0.1 it chose dt = 0.01** (job 17119585, confirmed at 1e6 steps). dt = 0.04 failed at excess 1.04 % and dt = 0.02 at 1.17 %, both at T = 1e-3.
   - So the χ = 0.1 choice went 0.02 (criterion 1), then 0.01 (criterion 2), then 0.04 (criterion 3, final).
   - At χ = 0.4 it again failed every dt, with Δ_b ≈ +1–4 % in all unconstrained bins at T ≥ 1e-4, for every dt.
   - That offset is a Jensen bias (D-070): E[a/b] − 1 ≈ CV(b)² for noisy, slowly winding unconstrained waves.
     - It is positive in both directions: E[a/b] − 1 and E[b/a] − 1 are both ≈ +0.7–0.9 % at dt = 0.005.
     - It shrinks only partly with dt, because smaller-dt pairs decorrelate later. At χ = 0.4, T = 1e-4 it goes 1.78 → 1.56 → 1.21 → 0.72 % down the ladder.
     - It persists at the near-null pair 0.005 against 0.0025, where the true dt effect is < 0.01 %.
   - The ratio-of-means estimator above is unbiased to first order and removes it. The same raw winding gives max |Δ_b| ≤ 2.25 %, and only χ = 0.4, dt = 0.04 fails.
   - Both ladders were re-evaluated from the saved winding (`--ladder-from`), with no new GPU time.
3. **For the record:** the first χ = 0.1 run (t = 200, only partly decorrelated) measured the largest per-bin T = 1e-3 change at dt = 0.04 as 0.84–1.09 % (per system, no SE), against 0.16–0.23 % at 0.02.
   - The dt² reading is weakly supported. The outlier bins at dt = 0.04 (flows bin 11; hip bins 12–13) are not where the dt = 0.02 maxima are, and the all-wave change is only 0.002–0.03 %.
   - It looks more like partial-decorrelation noise growing with dt than a band-limited systematic.
   - So dt = 0.04 at χ = 0.1 is close to the 1 % line at T = 1e-3, the anharmonic temperature the plan reports only descriptively.
   - At the harmonic temperatures (T ≤ 1e-4) the change is ≤ 0.05 %.
   - If E2 needs T = 1e-3 rates at < 1 % systematic, rerun that temperature at dt = 0.02. The cost is the same, since the protocol is fixed in steps.

### drift_tol
`--drift-tol` = 5 × the largest max |E − e0| / (dof T/2) seen over 1e6 NVE steps at the chosen dt, rounded up to {1, 2, 5} × 10^k: **5e-4 at χ = 0.1** and **2e-3 at χ = 0.4**.

Lane-to-lane spread is small:
- χ = 0.1: 4.7e-5 to 8.9e-5;
- χ = 0.4: 1.2e-4 to 2.0e-4.

The bound is the saturated Verlet oscillation, with no secular part, so a 5× margin does not trip on it. It still aborts a real secular drift or a blow-up long before ⟨KE⟩ is affected.

## 3. wind_every for the unconstrained waves

**Setup.**
- At the chosen dt, one NVE trajectory (T = 1e-6 and 1e-3; 2 ground states × 2 systems) records every step's ρ of all waves, over t = 200.
- Subsampling every e-th step reproduces the library's cadence exactly, on the same trajectory. With e = 1 the offline winding equals the device winding to 6e-16 relative.

**Rule.** Choose the largest e in {1, 2, 5, 10}, with every smaller e also passing, such that **every** unconstrained wave's W_abs is within 1 % of e = 1.

| χ (dt) | e = 2: max / median per wave | e = 5 | e = 10 | chosen |
|---|---|---|---|---|
| 0.1 (0.04) | 0.061 % / 0.010 % | 0.53 % / 0.086 % | 2.1 % / 0.36 % | **5** |
| 0.4 (0.02) | 0.20 % / 0.030 % | 1.1 % / 0.24 % | 3.7 % / 0.98 % | **2** |

Undercounting grows with the time between updates, e·dt. It reaches 1 % near e·dt ≈ 0.1 at χ = 0.4 and ≈ 0.3 at χ = 0.1, which follows the larger ω_max at χ = 0.4. Constrained waves are always counted every step.

## 4. Softest constrained mode

ω² is an eigenvalue of K = (2/V) J Jᵀ, with J = ∂(Re ρ, Im ρ)/∂r over the M constrained waves; these are the 2M nonzero eigenvalues of the Cartesian Hessian. The figures below are over the first 10 ground states of each (start, χ).
- **Cross-check:** on flows2d χ = 0.1, ground state 0, they match the Hessian to 1e-13.
- **Interpretation:** ω_min sits just below the smallest |k| = 2π/L ≈ 0.196, and is nearly the same in every ground state.

| start | ω_min χ = 0.1 (min / median) | ω_max χ = 0.1 | ω_min χ = 0.4 (min / median) | ω_max χ = 0.4 |
|---|---|---|---|---|
| flows2d | 0.1939 / 0.1943 | 2.568 | 0.1773 / 0.1793 | 5.799 |
| gau2d | 0.1977 / 0.1980 | 2.468 | 0.1763 / 0.1819 | 5.824 |
| url2d | 0.1821 / 0.1822 | 2.489 | 0.1729 / 0.1738 | 5.714 |
| rsa2d | 0.1986 / 0.1987 | 2.318 | 0.1851 / 0.1875 | 5.719 |
| hip2d | 0.1928 / 0.1935 | 2.566 | 0.1800 / 0.1827 | 5.806 |

**Langevin (E4):** both planned values satisfy γ < ω_min.
- γ = 0.03 is 5.8× below the smallest ω_min, 0.173.
- γ = 0.003 is 58× below.

## 5. Production cost projection

**Model.**
- **Per job:** (steps / rate) × 1.05 for checkpoints and I/O × 1.25 hardware margin, plus init (40 × ≈ 0.8 s harmonic samples) plus 120 s for compile and startup.
- **Rates:** the measured rate at the chosen wind_every and k-chunk (flows2d/hip2d mean); for E4, BAOAB at 10 lanes.
- **Worst case:** equilibration is extended once (2·2e5 steps).

| Job | per job (h) | jobs | line total (GPU-h) |
|---|---|---|---|
| E1 χ = 0.1: 40 lanes, 2.6e6 steps/lane at 1337 steps/s | 0.75 | 5 | 3.75 |
| E1 χ = 0.4: 40 lanes, 2.6e6 steps/lane at 365 steps/s | 2.64 | 5 | 13.21 |
| E1b χ = 0.1: 2 lanes, constrained, n_nve = 5000 | 0.08 | 5 | 0.39 |
| E1b χ = 0.4: 2 lanes, constrained, n_nve = 10000 | 0.13 | 5 | 0.64 |
| **E1 + E1b (production)** | | 20 | **17.99 ≤ 60** |
| E4 χ = 0.1: 10 lanes BAOAB, 2.4e6 steps (worst case), per γ | 0.34 | 4 | 1.36 |
| E4 χ = 0.4: same | 1.10 | 4 | 4.41 |
| **E4 (Langevin)** | | 8 | **5.77 ≤ 12** |

**Reduction applied: none.** The order (n_nve → 5e5, then 6 ground states, then drop T = 1e-5) is implemented in `production_plan` but is not triggered.

**Suggested SLURM `--time` per job** (projection plus margin):

| Job | --time |
|---|---|
| E1 χ = 0.1 | 1:30 |
| E1 χ = 0.4 | 4:00 |
| E1b | 0:20 |
| E4 χ = 0.1 | 0:45 |
| E4 χ = 0.4 | 1:45 |

The CLI checkpoints every 30 min and resumes.

**Suggested CLI arguments.** Inputs are `flows_chi<tag>.npz`, with K and M from the sibling JSON.

- **E1:**
  - `--indices 0 1 2 3 4 5 6 7 8 9 --temperature 1e-6 1e-5 1e-4 1e-3`
  - `--dt {0.04|0.02} --wind-every {5|2} --drift-tol {5e-4|2e-3}`
  - `--record-rho-every 2000 --max-rho-records 1300 --n-snapshots 200`
  - `--checkpoint <path> --device gpu`
  - Defaults kept: `--n-equil 2e5 --n-andersen 1e5 --n-nve 1e6 --cycles 2`, k-track 2K, no k-chunk.
- **E1b:**
  - `--indices 0 1 --temperature 1e-6 --k-track <K>` (constrained only).
  - `--cycles 1 --n-nve {5000|10000}` (t = 200).
  - `--record-rho-every 5 --max-rho-records {101000|102000}`.
  - The trace cadence is global, so the trace rows must also cover the 2e5 (+ 2e5) equilibration and 1e5 Andersen steps: about 2.7 GB at χ = 0.4.
- **E4:**
  - `--scheme baoab --gamma {0.003|0.03} --indices 0 1 2 3 4 --temperature 1e-6 1e-4`, same dt, wind_every and trace cadence.
  - `--drift-tol` does not apply (BAOAB).

## 6. GPU budget used by V3

**2.64 GPU-h of the 4 GPU-h cap** (8 jobs; rows in `V3_jobs.md`). 0.55 GPU-h of that was the first χ = 0.4 calibration, which landed on a MIG slice and produced no usable verdict.

## Reproduce

```
R=/scratch/gpfs/TORQUATO/sd2402/xla_cco_runs/v3
S=experiments/slurm/batten2008_md_gpu.slurm
env -u JAX_PLATFORMS sbatch -J mdw_bench \
  --export=ALL,STEP=bench,STEP_ARGS="--mode grid --out $R/throughput.jsonl" $S
env -u JAX_PLATFORMS sbatch -J mdw_bench \
  --export=ALL,STEP=bench,STEP_ARGS="--mode calib --chi 1-10 --out $R/calib2/calib_1-10.json" $S
# re-evaluate a saved ladder, then confirm and choose wind_every
env -u JAX_PLATFORMS sbatch -J mdw_bench --export=ALL,STEP=bench,STEP_ARGS="--mode calib \
  --chi 1-10 --ladder-from $R/calib2/calib_1-10.json --out $R/calib3/calib_1-10.json" $S
JAX_PLATFORMS=cpu uv run --no-sync python benchmarks/md_bench.py --mode omega --device cpu \
  --out $R/omega.json
JAX_PLATFORMS=cpu uv run --no-sync python benchmarks/md_bench.py --mode project --device cpu \
  --out experiments/md_winding/bench.json --throughput $R/throughput.jsonl \
  --calib-dir $R/calib3 --omega $R/omega.json --gpu-hours-used 2.642
```
