# Phase winding in low-temperature MD around stealthy ground states: report

Plan stages E2 (analysis), E3 (figures) and E5 (write-up), on the complete production data (2026-11-09). The pre-registered criteria are in `HYPOTHESES.md` (with Amendments A1–A4); this report applies them unchanged. Sections 4 and 5 are post hoc and exploratory; they never alter a verdict.

**Sources of numbers.** Every number below comes from one of these files (all under `/scratch/gpfs/TORQUATO/sd2402/xla-cco/experiments/`, git-ignored):

- **[V]** `trajectories/md_results/verdicts.{json,md}`: the H1–H8 verdict parts, the descriptive T = 1e-3 parts, the H5 reports and the exploratory pooled bins (`batten2008_md_stats.py`);
- **[D]** `trajectories/md_results/diagnostics.{json,md}`: the post hoc diagnostics (`batten2008_md_diagnostics.py`);
- **[A]** `trajectories/md_results/<start>/N<N>/md_analysis*_chi<tag>.npz`: the per-cell analyses (`batten2008_md_analysis.py`);
- **[J]** `trajectories/<start>/N<N>/md*_chi<tag>.json`: the run metadata, the equilibration gate, and the final sacct cost;
- **[B]** `md_winding/BUDGET.md`: the GPU ledger.

## 1. Summary

- **Q1 (does arg ρ_k keep winding at fixed T?):** yes. In every cell a constrained wave winds at the Rice rate |k|/2π turns per unit time. The normalized rate ν (rate over |k|/2π, averaged over the constrained waves) has per-start medians 0.9970–1.0025 at harmonic T [A]. That is 9.49–9.69 × 10³ unsigned turns per NVE segment, at every T [D]. The rate over [t/2, t] equals the rate over [t/4, t/2] to within 0.44% in all 60 harmonic segments [V]. Winding never becomes stationary.
- **Q2 (does the rate depend on the start?):** no for the unsigned rate. The mean bias-corrected Kruskal–Wallis effect size is η²_H = −0.010 over the 6 harmonic cells, and the largest pairwise difference of start medians is ≤ 0.53% (H5 PASS) [V]. The net-winding diffusion constant D_θ is start-independent at χ = 0.4. At χ = 0.1 and T ≤ 1e-5 it is 15–30% higher for URL than for the other starts [V].
- **Registered verdicts:** H5 and H6b PASS; H3 and H7 INCONCLUSIVE; H1, H2, H4, H6 and H8 FAIL (Sec. 3). The FAILs fall into three groups:
  - H1, H2, and through H2 part 1 of H8, fail only in |k|/K bins holding 2–8 waves (median 2), where 10 ground states resolve a 10% miss only marginally (Sec. 4) [D].
  - H4 fails because net winding is superdiffusive at χ = 0.1 (α = 1.18–1.28) under Andersen + NVE. The diagnostics show this comes from the near-integrable NVE dynamics, since weak Langevin friction removes it (α = 1.01–1.02 at γ = 0.03) [D].
  - H6 fails because drift along the ground-state manifold is diffusive (MSD ∝ √T t), not ballistic. Each H6 part misses its prediction by a definite factor: the unsigned rate is √π times the fixed-ρ prediction, and β = 1/4 instead of 1/2 [V, D].

## 2. What was run

Five starts × 2 χ × 10 ground states (flows 0–9 of `flows_chi<tag>.npz` `s_final`) × T ∈ {1e-6, 1e-5, 1e-4, 1e-3} make 400 E1 lanes. The protocol per lane:

1. harmonic start;
2. count-mode Andersen (p = 0.01) for 2 × 10⁵ steps, with the ⟨Φ⟩/(MT) gate and one extension;
3. 2 cycles of [Andersen 10⁵ steps, COM removal, NVE 10⁶ steps], all waves with |k| < 2K wound.

The cells [J, A]:

| χ tag | χ | N | M | K | dt | wind_every (unconstrained) | NVE segment length | tracked waves |
|---|---|---|---|---|---|---|---|---|
| 1-10 | 0.100–0.103 | 999–1024 | 200–210 | 2.242 | 0.04 | 5 | t = 4 × 10⁴ | 798–821 |
| 2-5 | 0.400–0.401 | 999–1024 | 798–821 | 4.484 | 0.02 | 2 | t = 2 × 10⁴ | 3192–3278 |

- **E1b:** ground states 0–1 at T = 1e-6, one NVE segment of t = 200 with ρ_k traced every 5 steps (20 lanes).
- **E4:** BAOAB at γ ∈ {0.003, 0.03} for flows2d and hip2d, ground states 0–4, T ∈ {1e-6, 1e-4} (80 lanes).

**Integrity:**
- All 28 outputs exist (10 E1, 10 E1b, 8 E4). Every SLURM job ended COMPLETED [J].
- All 500 lanes ended with status RUNNING (`cycle_status` = 0 in every cycle); **no lane hit STATUS_DRIFT**, so the drift rule of the decision table never applied [J].
- The pilot (flows2d, χ = 0.1, job 17148922) was produced by the production command and is part of the data set.

**Equilibration gate.** After the one extension, |⟨Φ⟩/(MT) − 1| > 5% held in the following lanes, which continued into production as the CLI does [D, J]:
- at χ = 0.1: 8, 13, 13, 13 and 10 of 40 lanes (flows2d, gau2d, url2d, rsa2d, hip2d);
- at χ = 0.4: 2, 0, 0, 0 and 0 of 40;
- in E1b: 2 of 20 lanes.

This is the canonical scatter of a finite harmonic system (`NOTES.md` §1). The sd of ⟨Φ⟩/(MT) is 1/√(2M) = 0.049 at M = 210, so a 5% miss has probability 0.31, i.e. about 12 of 40 lanes. In the production segments, the snapshot Φ/T at harmonic T has mean/M 0.994–1.008 and sd/√M 0.93–1.04 against Gamma(M, 1) (1 and 1) [D].

## 3. Pre-registered verdicts (HYPOTHESES.md, A1–A4)

The unit is a ground state (n = 10; n = 5 for E4). "Value" is the median over ground states; z uses the ground-state standard error. Harmonic T means T ≤ 1e-4. Parts are listed as P / I / F (PASS / INCONCLUSIVE / FAIL) [V].

| ID | Hypothesis | Verdict | Parts P / I / F | Key numbers |
|---|---|---|---|---|
| H1 | Equipartition | **FAIL** | 532 / 40 / 34 | ⟨Φ⟩/(MT): 28 of 30 cells in [0.98, 1.02], range 0.971–1.034. ⟨S⟩/T per bin: 504 of 576 parts PASS, median 1.001, range 0.735–1.371 (worst: gau2d χ = 0.4, T = 1e-6, bin 0, value 1.371, z 1.73) |
| H2 | Rice rate, constrained | **FAIL** | 737 / 24 / 7 | rate / Rice per bin: 545 of 576 PASS, median 1.0006, range 0.856–1.140 (worst: gau2d χ = 0.1, T = 1e-6, bin 2, value 0.856, z −3.75). Rate ratio T = 1e-6 : 1e-4: 192 of 192 PASS, range 0.984–1.008 |
| H3 | Winding persists (Q1) | **INCONCLUSIVE** | 57 / 3 / 0 | window ratio [t/2, t] / [t/4, t/2], range 0.9956–1.0042 over the 60 segments. The 3 misses are all at T = 1e-5, from \|z\| > 3 at deviations ≤ 0.44%: flows2d χ = 0.4 segment 0 (0.9956, z −8.0), url2d χ = 0.4 segment 0 (0.9964, z −3.95), flows2d χ = 0.1 segment 1 (0.9972, z −3.71) |
| H4 | Net winding is a random walk | **FAIL** | 45 / 0 / 15 | mean net rate consistent with 0 in 30 of 30 cells (max \|z\| 2.72). MSD exponent α: χ = 0.4 0.990–1.008 (15 PASS); χ = 0.1 1.176–1.277 (15 FAIL, z 27–49) |
| H5 | Start independence (Q2) | **PASS** | 7 / 0 / 0 | mean η²_H = −0.0102 over 6 cells (per cell −0.066 to 0.151). Largest pairwise median difference of ν per cell 0.07–0.53% |
| H6 | Unconstrained waves | **FAIL** | 5 / 0 / 615 | rate / \|k\|√(T/(πS))/2π: 1.734–1.799 in all 600 bins (median 1.768). β = 0.247–0.254 in all 10 cells (band [0.4, 0.6]). t√T collapse spread: χ = 0.1 0.143–0.169 (5 PASS); χ = 0.4 0.936–1.031 (5 FAIL) |
| H6b | Conditional Rice, unconstrained (A3, A4) | **PASS** | 600 / 0 / 0 | rate / (\|k\|√(NT/π)⟨1/\|ρ_k\|⟩_t/2π), range 0.9845–1.0201 |
| H7 | Thermostat independence | **INCONCLUSIVE** | 23 / 1 / 0 | BAOAB / NVE ν: 16 of 16 PASS, range 0.9985–1.0061. γ = 0.003 / 0.03: 8 parts, range 0.9998–1.0010; 7 PASS and 1 INCONCLUSIVE (flows2d χ = 0.1 T = 1e-4: 1.0007, z 3.41) |
| H8 | χ dependence | **FAIL** | 15 / 0 / 2 | part 1 "H2 passes at both χ" fails at both χ, because H2 fails. Part 2: D_θ(χ = 0.4)/D_θ(χ = 0.1) = 0.148–0.266, Mann–Whitney p = 1.8 × 10⁻⁴ in 15 of 15 (start, T) cells |

**Decision table.**
- The H1, H2, H4, H6 and H8 FAILs each have at least one part that misses by ≥ 10%. By the table they are reported as physics; Sec. 4 shows which physics.
- H3 and H7 are INCONCLUSIVE: some parts miss with |z| > 3 at deviations < 10% (here ≤ 0.44% and 0.07%). The table's remedy is to rerun one cell at 2× cycles (`NOTES.md` §1) from the 16 GPU-h reserve and re-evaluate once. **That rerun has not been done**, because this stage ran no GPU jobs; the decision is left to the orchestrator.

**T = 1e-3 (descriptive, no verdict).** Parts within / outside the criteria [V]:

| Hypothesis | Within / outside | Values |
|---|---|---|
| H1 | 195 / 7 | ⟨Φ⟩/(MT) 0.976–1.022 |
| H2 | 192 / 0 | rate / Rice 0.970–1.034 |
| H3 | 20 / 0 | window ratio 0.9994–1.0007 |
| H4 | 15 / 5 | α 1.181–1.193 at χ = 0.1; 1.000–1.007 at χ = 0.4 |
| H6b | 200 / 0 | |
| H6 | 5 / 215 | β 0.250–0.260 (fitted over all four T, A1.4 descriptive) |

ν at T = 1e-3 has per-start medians 1.0002–1.0009 [A]. At this anharmonic edge, the winding statistics are indistinguishable from the harmonic ones.

## 4. Why the FAILs fail (post hoc, descriptive; not pre-registered)

### H1, H2 (and H8 part 1): few-wave bins

The bins behind the FAIL parts hold a median of 2 constrained waves (H1: 2–8; H2: 2–4). The PASS parts sit in bins with a median of 16–20 waves [D]. The per-wave scatter across ground states, measured on the pilot, is σ_w = 0.358 for S/T and 0.166 for rate / Rice [V, `exploratory.sigma_w`]. With 10 ground states, the standard error of a 2-wave bin is therefore ≈ 0.08 for S/T and ≈ 0.04 for rate / Rice, so a 10% miss is ≈ 1.2 SE for S/T and ≈ 2.5 SE for rate / Rice.

In most bins the FAIL parts are consistent with sampling scatter from the canonical mode-energy fluctuation that near-integrable NVE preserves (`NOTES.md` §1): the median of ⟨Φ⟩/(MT) over the 30 harmonic cells is 1.001, and the medians of all 576 per-bin values are 1.001 for S/T and 1.0006 for the rate [V]. One deficit is not explained by this scatter. The post hoc pooled H1 group url2d χ = 0.1, T = 1e-5, |k|/K ∈ (0.30, 0.45] (bins 6–8, 25 waves) has a median ⟨S⟩/T of 0.915 with SE 0.008 (z = −9.9), an 8.5% deficit that the ground-state sample resolves [V, `exploratory.rows`; Sec. 5]. We have no explanation for it.

### H4: superdiffusive net winding is an NVE effect

Net winding at χ = 0.1 has α = 1.18–1.28 under Andersen + NVE. On the same ground states, BAOAB gives [D]:

| Friction | α | D_θ |
|---|---|---|
| Andersen + NVE | 1.188–1.266 | 0.735–1.145 |
| BAOAB, γ = 0.003 | 1.083–1.095 | 0.338–0.388 |
| BAOAB, γ = 0.03 | 1.011–1.021 | 0.201–0.220 |

At χ = 0.4, α = 0.992–1.005 and D_θ = 0.177–0.191 under every thermostat [D]. Within one NVE segment, the 2M harmonic modes keep their energies, so ρ_k(t) is quasi-periodic. Its phase then carries persistent rotation over the 4 × 10⁴ time units of a segment. Friction dephases the modes and restores a random walk. With 2M = 420 modes against Nd = 2048 degrees of freedom (χ = 0.1), the NVE phase decorrelates more slowly than with 2M = 1640 (χ = 0.4). This last point is an interpretation; the measured facts are the α and D_θ values above.

### H6: diffusive drift along the ground-state manifold

**Part 1 (the rate).** The ratio of the unsigned rate to the registered prediction |k|√(T/(πS))/2π, divided by √π, is 0.978–1.015 in every cell and T (cell medians 0.993–1.002) [D]. The unconstrained waves therefore wind at |k|√(T/S)/2π, the Rice rate of a circular Gaussian ρ_k. This is what the conditional Rice formula of A3 gives once drift randomizes ρ_k (with ⟨|ρ|²⟩ = NS, E[1/|ρ|] = √(π/(NS)) and E|θ̇| = |k|√(T/S)). H6b confirms the conditional formula directly (0.985–1.020).

**Part 2 (β).** At T ≥ 1e-4, the particle MSD grows as t^1.00–1.04 over the last decade of a segment, i.e. it is diffusive. At the segment end, MSD ∝ T^0.504–0.519 [D]. The end-of-segment net phase of the unconstrained waves follows (2πW_net)² ∝ ⟨|k|²⟩·MSD (F8, right). The net-winding rate therefore scales as (√T t)^{1/2}/t, which gives β = 1/4; the measured β is 0.247–0.254. The registered β ∈ [0.4, 0.6] assumed ballistic drift.

**Part 3 (collapse).** The t√T collapse holds at χ = 0.1 (spread 0.14–0.17). At χ = 0.4 it fails (0.94–1.03) because the vibrational plateau of the 2M modes, which scales as T and not as t√T, dominates the MSD at t√T ≈ 10⁻³–10⁻¹ (F8, left).

### E1b spectra (F10)

97.6–99.9% of the ρ_k power of the constrained waves lies at ω ≤ 1.05 max √eig K, for every start and both χ [D]. At χ = 0.4 the power has a dip at ω ≈ 3.4, where the √eig K density has its gap (location read off F10; no stored number).

## 5. Exploratory pooled low-|k| bins (post hoc, NOT pre-registered)

This analysis was added after the pilot (`NOTES.md` §2) and kept at the user's request. It does not change any verdict in Sec. 3. Adjacent constrained bins are merged, from low |k| upward, until each holds ≥ W_min waves:
- W_min = 15 for ⟨S⟩/T and 4 for rate / Rice;
- W_min = ⌈σ_w²/(n · 0.03²)⌉ with the pilot σ_w and n = 10.

Results [V, `exploratory`]:

- **All pooled bins (merged and single):** 395 of 408 H1 values and 526 of 543 H2 values lie in the registered bands.
- **Merged groups only:**
  - H1: 94 of 105 in [0.95, 1.05]; medians 0.904–1.124. Nine of the 11 misses lie at χ = 0.1 with |k|/K ≤ 0.45, the other two at χ = 0.4 with |k|/K ≤ 0.15 (gau2d and rsa2d, T = 1e-6). Three have |z| > 2.5: url2d χ = 0.1 T = 1e-5, bins 6–8 (0.915, z −9.9); flows2d χ = 0.1 T = 1e-5, bins 1–5 (0.914, z −3.75); gau2d χ = 0.4 T = 1e-6, bins 0–2 (1.077, z 2.85).
  - H2: 27 of 33 in [0.95, 1.05]; medians 0.893–1.063. All 6 misses are the χ = 0.1 group |k|/K ∈ (0.05, 0.15] (4 waves), with |z| ≤ 3.95.
- **Reading:** away from the lowest |k| (the few softest modes), pooled ⟨S⟩/T and rate / Rice agree with equipartition and the Rice rate to within the 5% bands. The lowest-|k| groups scatter by up to 12%.

Figures: `md_X1_pooled_posthoc_chi1-10.png`, `md_X1_pooled_posthoc_chi2-5.png`.

## 6. Answers

**Q1: does arg ρ_k keep winding at fixed T, or become stationary?**
It keeps winding. A constrained wave accumulates unsigned phase at |k|/2π turns per unit time, independent of T, χ and the start (ν medians 0.997–1.0025 at harmonic T, 1.0002–1.0009 at T = 1e-3) [A]. That is 9.49–9.69 × 10³ turns per segment in every cell [D]. The windowed rates agree to ≤ 0.44% (H3 is INCONCLUSIVE only through |z| on sub-percent deviations) [V]. The winding is the Rice rate of a Gaussian field whose phase velocity is set by the Maxwell–Boltzmann velocities, ⟨|ρ̇_k|²⟩ = N|k|²T. Since ⟨S_k⟩ = T, the T dependence cancels.

**Q2: does the rate depend on which start the ground state came from?**
No for the unsigned rate (H5 PASS: mean η²_H = −0.010; start medians differ by ≤ 0.53%) [V]. The net-winding diffusion constant D_θ is also start-independent at χ = 0.4 (no Holm-significant pair at any T). At χ = 0.1, URL ground states have a higher D_θ at T = 1e-6 (0.849 against 0.691–0.740) and at T = 1e-5 (1.283 against 0.986–1.065), with Holm p = 0.003–0.020 and |Cliff's δ| = 0.80–0.96 against 4 and 3 of the other starts; no pair differs at T = 1e-4 [V, `reports.H5_harmonic`]. Spearman correlations of D_θ with moments of the K spectrum reach |ρ| ≤ 0.48 in every cell [V, `reports.H5_harmonic.*.spearman_d_theta`]. At χ = 0.4 they do not recur across temperatures (e.g. k_cv: 0.48 at T = 1e-6; 0.06 and 0.09 at 1e-5 and 1e-4). At χ = 0.1 two moments recur with the same sign at T = 1e-6 and 1e-5 and vanish at 1e-4: k_q10_rel (ρ = 0.30, p = 0.032; ρ = 0.42, p = 0.0023; ρ = 0.04 at 1e-4) and k_min_rel (ρ = −0.47, p = 5.5 × 10⁻⁴; ρ = −0.29, p = 0.044; ρ = −0.001 at 1e-4). These are the two temperatures at which URL has the higher D_θ, so the correlations likely track the URL offset (a start effect) and not a spectral dependence within a start; we have not tested this (e.g. by excluding URL).

**Temperature.**
- The constrained unsigned rate does not depend on T: the 1e-6 : 1e-4 rate ratio is 0.984–1.008 in all 192 bins [V].
- The unconstrained unsigned rate scales as √T (|k|√(T/S)/2π; F2) [D].
- The unconstrained net-winding rate scales as T^{1/4} (β = 0.247–0.254) [V].
- The particle drift MSD at the segment end scales as T^{1/2} (exponent 0.504–0.519) [D].
- The constrained net-winding D_θ under NVE is non-monotonic in T at χ = 0.1: start medians 0.69–0.85 at 1e-6, 1.11–1.15 at 1e-4, 0.64–0.71 at 1e-3. At χ = 0.4 it is flat, 0.18–0.19 [V, `reports`; F6].

**|k|: constrained (|k| < K) against unconstrained (K < |k| < 2K).**
- Constrained waves wind at |k|/2π: S_k = T, so the rate does not depend on T.
- Unconstrained waves wind at |k|√(T/S_k)/2π. With S_k = O(1), this is 1e-3 (T = 1e-6) to 3e-2 (T = 1e-3) of |k|/2π (derived from √(T/S_k); consistent with F2, not a stored number).
- Both are the same circular-Gaussian Rice law evaluated at the respective S_k [D].

**χ.**
- The normalized unsigned rate ν is χ-independent (ν ≈ 1 at both χ) [A]. H8 part 1 nevertheless fails, because it is defined through H2, whose few-wave bins fail at both χ [V].
- D_θ differs between χ by a factor 3.8–6.8 (ratio 0.148–0.266; p = 1.8 × 10⁻⁴ in every cell) [V]. Under BAOAB at γ = 0.03, D_θ at χ = 0.1 (0.201–0.220) is only 14–20% above the χ = 0.4 values for the same start and T (0.177–0.183) [D]. Most of the χ dependence of D_θ under NVE is therefore the NVE persistence of Sec. 4, not a static property of the ground states.

**Signed against unsigned.**
- Net winding has zero mean (30 of 30 cells, |z| ≤ 2.72) [V].
- Its RMS after one segment is 2.4–3.5% of the unsigned winding at χ = 0.1 (231–340 turns out of 9.5–9.7 × 10³) and 0.89–0.92% at χ = 0.4 (84–88 turns) [D].
- It is diffusive at χ = 0.4 (α = 0.99–1.01) and superdiffusive under NVE at χ = 0.1 (α = 1.18–1.28) [V].

**Drift along the ground-state manifold.**
- The configuration wanders far along the flat manifold. Over one segment the particle MSD reaches 351–368 (T = 1e-6) to 1.20–1.25 × 10⁴ (T = 1e-3) at χ = 0.1, and 9.7–10.0 to 319–325 at χ = 0.4 [D]. Here ρ = 1, so the mean spacing is 1.
- The drift is diffusive at long times (MSD ∝ t^1.00–1.04 at T ≥ 1e-4) with a diffusion constant ∝ √T [D].
- It randomizes the unconstrained ρ_k, which is why their unsigned rate takes the circular-Gaussian value and their net winding obeys (2πW_net)² ∝ ⟨|k|²⟩ MSD (F8).

**Thermostat.**
- The unsigned constrained rate does not depend on the thermostat: BAOAB / NVE ν = 0.9985–1.0061 and the γ = 0.003 / 0.03 ratio is 0.9998–1.0010 (H7, 23 of 24 parts PASS; the one INCONCLUSIVE deviates by 0.07%) [V].
- Net-winding statistics at χ = 0.1 do depend on the thermostat (α and D_θ, Sec. 4). At χ = 0.4 they do not.

## 7. GPU-hours

19.77 of the 100 GPU-h cap were used [B]:

| Line | GPU-h |
|---|---|
| Verification and benchmarks | 3.46 |
| Pilot | 0.56 |
| Production (E1 + E1b) | 12.93 |
| Langevin (E4) | 2.81 |
| Reserve | 0 (of 16) |

The final sacct costs in the 28 output JSONs (refreshed with `--step cost`) sum to 16.30 GPU-h, which equals pilot + production + Langevin [J]. E2, E3 and E5 used CPU only: SLURM jobs 17171421 (16 cores, 22 min) and 17172995 (1 min).

## 8. Deviations from the plan

1. **Equilibration gate.** 59 of 400 E1 lanes (and 2 of 20 E1b lanes) still missed the 5% ⟨Φ⟩/(MT) gate after the one extension. They were kept, as the CLI does, because the misses match the canonical 1/√(2M) scatter (Sec. 2). The decision table has no rule for this case.
2. **INCONCLUSIVE remedy not run.** The 2×-cycles rerun for H3 and H7 needs GPU time, and this stage ran no GPU jobs. The reserve (16 GPU-h) is untouched.
3. **Location of the analysis outputs.** They are in `trajectories/md_results/<start>/N<N>/`, not beside the inputs. D-089 (a shared `--out-dir` overwrote the starts) was found and fixed during this stage. The production analysis (job 17171421) used per-start output directories, so its 28 files were never affected; the fixed script produces the same layout.
4. **Post hoc additions.** The exploratory pooled bins (Sec. 5, `NOTES.md` §2) and the diagnostics (Sec. 4, `batten2008_md_diagnostics.py`) are post hoc and descriptive. Neither enters a verdict.
5. **Provenance.** `run_script.git_dirty` is `null` in all 28 output JSONs: `git status` was not available inside the jobs. `git_head` is recorded: 6c48bfd6 for the pilot, fe54c5d0 for all other cells.

## 9. Figure index

All figures are PNG, in `/scratch/gpfs/TORQUATO/sd2402/xla-cco/experiments/figures/`:

| File | Content |
|---|---|
| `md_F1_kymograph_chi1-10.png`, `md_F1_kymograph_chi2-5.png` | unsigned W_n(t) against log t and \|k\|/K, ground state 0, first segment; rows T, columns starts. Unconstrained waves are wound every `wind_every` steps (5 and 2), so their first snapshots read 0 |
| `md_F2_rates.png` | ν per \|k\|/K bin and T, with the Rice (dashed) and conditional Rice (dotted) lines |
| `md_F3_stationarity.png` | window ratio − 1 per ground state and segment, with the 3% H3 band |
| `md_F4_net_msd.png` | net-winding MSD at T = 1e-6 and α per ground state (H4 band) |
| `md_F5_starts.png` | ν and D_θ by start, with η²_H, median difference and max \|Cliff's δ\| |
| `md_F6_temperature.png` | ν, unconstrained net rate (with T^{1/2}) and D_θ against T |
| `md_F7_equipartition.png` | ⟨S⟩/T per bin, ⟨Φ⟩/(MT) by start, Φ/T histogram against Gamma(M, 1) |
| `md_F8_drift.png` | particle MSD against t√T; (2πW_net)² against ⟨\|k\|²⟩ MSD |
| `md_F9_thermostat.png` | BAOAB against Andersen + NVE ν per ground state |
| `md_F10_spectra.png` | ρ_k power spectra (E1b) against the √eig K density |
| `md_X1_pooled_posthoc_chi1-10.png`, `md_X1_pooled_posthoc_chi2-5.png` | post hoc pooled bins (Sec. 5) |

## 10. Reproduction

From the repository root. The SLURM job runs on a CPU node and takes about 23 min.

```bash
mkdir -p experiments/trajectories
for s in flows2d gau2d url2d rsa2d hip2d; do for c in 1-10 2-5; do   # final sacct cost into each JSON
  uv run --no-sync python experiments/scripts/batten2008_md_run.py --step cost --system $s --chi $c
done; done
env -u JAX_PLATFORMS sbatch --export=ALL,PAR=4 experiments/slurm/batten2008_md_analysis_cpu.slurm
```

The job writes `trajectories/md_results/` (per-cell analyses, `verdicts.{json,md}`, `diagnostics.{json,md}`) and the figures. `SKIP_ANALYSIS=1` reruns only the statistics and figures.
