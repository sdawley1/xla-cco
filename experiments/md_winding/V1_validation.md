# V1: MD statistical validation against exact harmonic results

Test file: `tests/integration/test_md_equipartition.py` (markers `slow`, `high`). Job script: `experiments/slurm/test_md_equipartition.slurm`.

**Verdict: PASS.** All 5 tests passed on a SLURM CPU node (AMD EPYC 9654, 16 cores) in 17.5 min wall (simulation fixture 1046 s; an earlier run of the same code on another node took 14.4 min with bit-identical numbers).
- Commit: 01ff531
- Log: `/scratch/gpfs/TORQUATO/sd2402/xla_cco_runs/v1/slurm-mdw_v1_equipartition-17118223.out`

## Setup
- **System:** N = 64, d = 2, K = 3.2, so M = 24 constrained waves and χ = 24/126 ≈ 0.19.
  - Ground state: gradient flow, Φ = 8e-22.
  - Density 1, so V = N.
  - 2M = 48 stiff modes, with λ ∈ [0.52, 12.4] and mean λ = 4.93.
- **Lanes:** 512 independent lanes per T, with T ∈ {1e-8, 1e-6}. Both temperatures run in one vmapped batch of 1024 lanes.
- **Start:** `harmonic_sample` positions plus full Maxwell–Boltzmann velocities.
- **Equilibration:** count-mode Andersen with p = 0.01, dt = 0.02, for 4000 steps.
- **Branches.** All three start from the same equilibrated state through `begin_segment`:
  - NVE (velocity Verlet, COM removed), dt = 0.02, 5000 steps;
  - Andersen, dt = 0.02, 10 000 steps;
  - Andersen, dt = 0.01, 10 000 steps.
- **Errors:** lane-level only. Lanes are independent, so the test uses Student t with 511 degrees of freedom.
  - Bonferroni correction over the m comparisons of each test, at family rate MD_FAMILY_ALPHA = 1e-3.
  - Within-lane blocking is not used: harmonic NVE keeps each lane's mode energies, which blocking would miss.
- **COM momentum:** the start keeps it, a deliberate departure from "COM removed". It is part of the Andersen ensemble.
  - A COM-free start leaves that degree of freedom cold for about N/(2p) = 3200 steps, which biases the Andersen ⟨KE⟩ by up to 1/N.
  - The NVE branch removes the COM in `begin_segment`. Its KE target is therefore d(N−1)T/2, and Andersen's is dNT/2.

## Means (lane mean ± SE; t against 1)
The family for these comparisons has m = 8 (Φ and KE × 2 T × 2 branches), giving t_crit = 3.87.

| T | branch | ⟨Φ⟩/(MT) | t | ⟨KE⟩/(dof·T/2) | t |
|---|---|---|---|---|---|
| 1e-8 | NVE | 1.0038 ± 0.0063 | +0.61 | 1.0019 ± 0.0046 | +0.42 |
| 1e-8 | Andersen | 1.0038 ± 0.0054 | +0.70 | 0.9992 ± 0.0038 | −0.20 |
| 1e-6 | NVE | 0.9937 ± 0.0064 | −0.98 | 0.9957 ± 0.0050 | −0.87 |
| 1e-6 | Andersen | 0.9933 ± 0.0054 | −1.25 | 0.9910 ± 0.0038 | −2.33 |

## ⟨S(k)⟩/T for each constrained k
The family has m = 96 (24 k × 2 T × 2 branches), giving t_crit = 4.45.

| T | branch | range of lane means over k | max abs(t) (wave) |
|---|---|---|---|
| 1e-8 | NVE | 0.983 – 1.055 | 2.57 (k#22) |
| 1e-8 | Andersen | 0.976 – 1.052 | 3.10 (k#22) |
| 1e-6 | NVE | 0.969 – 1.022 | 2.70 (k#14) |
| 1e-6 | Andersen | 0.977 – 1.015 | 2.54 (k#13) |

## Distribution: KS test of Φ/T against Gamma(24, 1)
Each T uses 512 independent samples: the final Φ of the NVE branch from lanes 0–255 and of the Andersen branch from lanes 256–511.

| T | p | p with T reported as 1.1 T |
|---|---|---|
| 1e-8 | 0.787 | 6.7e-20 |
| 1e-6 | 0.067 | 5.4e-26 |

## Power and control
- **Definitions:**
  - n_eff = Var₁/SE², where Var₁ is the per-sample variance of Φ/(MT), measured from the running sums of Φ and Φ² (theory: 1/M = 0.0417).
  - z(5%) = 0.05/SE.
- **Requirements:** z(5%) ≥ 6 and n_eff ≥ 600.
- **Control:** the same data, reported at 1.1 T, must fail the ⟨Φ⟩/(MT) = 1 test (abs(t) > 3.87).

| T | branch | Var₁ | n_eff | z(5%) | t (true T) | t (1.1 T) |
|---|---|---|---|---|---|---|
| 1e-8 | NVE | 0.0410 | 1049 | 8.00 | +0.61 | −15.4 |
| 1e-8 | Andersen | 0.0421 | 1470 | 9.34 | +0.70 | −18.0 |
| 1e-6 | NVE | 0.0415 | 1003 | 7.77 | −0.98 | −16.5 |
| 1e-6 | Andersen | 0.0417 | 1455 | 9.34 | −1.25 | −19.9 |

## dt independence (Andersen, paired lanes, dt = 0.02 against 0.01)
The family has m = 50 (Φ and 24 S(k), × 2 T), giving t_crit = 4.31.
- **Predicted shadow-measure gap:** 3.7e-4. Verlet's bias in ⟨Φ⟩/(MT) is (1/2M)Σλdt²/(4−λdt²), which is 4.9e-4 at dt = 0.02. That is below 0.1 SE.

| T | Δ⟨Φ⟩/(MT) (0.02 − 0.01) | t |
|---|---|---|
| 1e-8 | +0.0030 ± 0.0044 | +0.68 |
| 1e-6 | +0.0017 ± 0.0043 | +0.40 |

The largest abs(t) over all 50 comparisons is 1.55.

## Defects
None found. No harmonic-T assertion failed, so no root-cause investigation was needed.
