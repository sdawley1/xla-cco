# Pre-registered hypotheses: phase winding in low-temperature MD around stealthy ground states

Committed before any production MD job was submitted (plan E0, 2026-11-08). The verdicts in `REPORT.md` must use exactly these criteria. A criterion may change only through an amendment section appended at the end of this file, with date and reason, written before the affected analysis is run.

## Setup

- **Systems.** Ground states from the 2D gradient flows: Poisson (`flows2d/N1024`), Gaussian (`gau2d/N1000`), URL (`url2d/N1024`), saturated RSA (`rsa2d/N999`) and HIP (`hip2d/N1000`), each at χ = 0.1 (tag `1-10`) and χ = 0.4 (tag `2-5`). There are 10 ground states per (start, χ): flows 0–9 of `s_final`.
- **Dynamics.** Mass 1, k_B = 1, Φ = (1/V) Σ_{half-space k, |k|<K} |ρ_k|², density 1.
- **Temperatures.** k_BT ∈ {1e-6, 1e-5, 1e-4, 1e-3}. "Harmonic T" means T ≤ 1e-4.
- **Protocol.**
  1. Harmonic start and Maxwell–Boltzmann velocities.
  2. Andersen thermalization (count mode, p = 0.01).
  3. Repeated segments: Andersen, then zero centre-of-mass (COM) momentum, then NVE velocity Verlet with winding accumulated every step for the constrained waves (|k| < K).
- **Unconstrained waves** (K < |k| < 2K) use the cadence fixed by V3.
- **Langevin subset.** BAOAB on flows2d and hip2d with γ ∈ {0.003, 0.03}.
- **Statistics.** The statistical unit is a ground state (n = 10 per cell). Rates are in turns per unit time. Unsigned winding is W_abs = (1/2π) Σ|Δθ| and net winding is W_net = (1/2π) Σ Δθ.

## Predictions (theory in the plan)

- **Equipartition.** In the harmonic limit, ⟨u uᵀ⟩ = (VT/2) I over the 2M constrained coordinates. So ⟨Φ⟩ = M T and ⟨S(k)⟩ = T for every constrained k.
- **Rice formula.** For a circular complex Gaussian ρ_k with ρ̇_k independent of ρ_k at equal times, E|θ̇| = σ_ρ̇/σ_ρ. Maxwell–Boltzmann velocities give exactly ⟨|ρ̇_k|²⟩ = N|k|²T. Hence:
  - **Constrained waves:** E|θ̇_k| = |k|·√(T/S_k), i.e. |k|/(2π) turns per unit time when S_k = T.
  - **Unconstrained waves (S = O(1)):** E|θ̇_k| ≈ |k|·√(T/(π S_k)).

## Hypotheses and pass criteria

Every z-test below uses autocorrelation-corrected errors (Flyvbjerg–Petersen blocking or ground-state-level standard errors). "Within x%" refers to the median over ground states.

| ID | Statement | PASS if (all parts) |
|---|---|---|
| H1 | Equipartition at harmonic T | ⟨Φ⟩/(MT) ∈ [0.98, 1.02]; ⟨S_k⟩/T ∈ [0.95, 1.05] in each of the 20 constrained \|k\|/K bins |
| H2 | Rice rate of the constrained waves | the unsigned rate / (\|k\|√(T/⟨S_k⟩)/2π) ∈ [0.95, 1.05] in each constrained bin at each harmonic T; the rate ratio T = 1e-6 : T = 1e-4 ∈ [0.95, 1.05] per bin |
| H3 | Winding persists, not stationary (Q1) | the unsigned rate over [t/4, t/2] and over [t/2, t] of each NVE segment agree within 3% with \|z\| < 3, in every (start, χ, harmonic T) cell |
| H4 | Net winding is a random walk | the mean net rate is consistent with 0 (\|z\| < 3); the MSD exponent of W_net(t) for t ≫ τ_corr is α ∈ [0.9, 1.1] |
| H5 | Start independence of the unsigned constrained rate (Q2) | for each (χ, harmonic T): Kruskal–Wallis effect size ε² < 0.06 across the five starts, and the largest pairwise difference of medians < 3%. Reported, not pass/fail: D_θ (the net-winding diffusion constant) by start, with Holm-corrected Mann–Whitney tests, Cliff's δ, and Spearman ρ against moments of the spectrum of K = (2/V) J Jᵀ |
| H6 | Unconstrained waves | the unsigned rate is within 10% of \|k\|√(T/(πS_k))/2π; the net-winding rate scales as T^β with β ∈ [0.4, 0.6]; the particle MSD curves of the four T collapse against t√T (within 20%) |
| H7 | Thermostat independence | BAOAB and Andersen+NVE unsigned rates agree within 3% (\|z\| < 3) at γ = 0.003 and 0.03; no dependence on γ beyond that |
| H8 | χ dependence | the normalized unsigned rate is χ-independent (H2 passes at both χ); D_θ differs between χ (Mann–Whitney p < 0.01) |

**T = 1e-3** (anharmonic edge): all quantities are reported descriptively, with no verdict.

**Verdicts:**
- **PASS:** all parts are met.
- **FAIL:** any part misses by ≥ 10% relative.
- **INCONCLUSIVE:** a part misses with \|z\| > 3 but by < 10%. One cell is then rerun at 2× length from the reserve budget, and the verdict is re-evaluated once.

## Amendment A1 (2026-11-08, before any production MD job)

These changes were made after a synthetic dry run of the analysis pipeline and before any production data existed (D-047).

1. **H5 effect size.** The plain Kruskal–Wallis ε² = H(n+1)/(n²−1) has a null expectation of about (k−1)/(n−1). For k = 5 starts and n = 50 ground states that is 0.082, above the 0.06 threshold, so a start-independent system would fail H5 with probability ≈ 0.99. H5 therefore uses the bias-corrected η²_H = (H − k + 1)/(n − k), whose null expectation is 0. The threshold (< 0.06) and the median-difference part (< 3%) are unchanged.
2. **Mapping misses to verdicts.**
   - *Interval and tolerance parts* (ratios in [a, b]; "within x%"): a part that misses by less than 10% relative is INCONCLUSIVE whatever its \|z\|; a miss of 10% or more is FAIL. Deviations are measured from the predicted value; for H5 they are the largest pairwise median difference.
   - *Significance-test parts* (H4 mean net rate \|z\| < 3; H8 part 2, Mann–Whitney p < 0.01): a miss is FAIL. There is no INCONCLUSIVE for these.
3. **Normalized unsigned rate.** H5 and H7 use, per ground state, ν = the mean over constrained waves of (unsigned rate)/(\|k\|/2π).
4. **H6.** The t√T collapse and the exponent β use the harmonic temperatures only. Values including T = 1e-3 are reported descriptively.

## Amendment A2 (2026-11-08, before any production MD job)

Under A1, a start-independent set still passes the η²_H part of H5 in all 6 harmonic cells with probability only ≈ 0.85⁶ ≈ 0.37. Each cell passes when H < 6.7, i.e. P(χ²₄ < 6.7) = 0.847; this was measured on synthetic null sets (E2 dry run). H5 part 1 is therefore evaluated on the mean of η²_H over the 6 harmonic (χ, T) cells: PASS if mean η²_H < 0.06. Under the null this mean has expectation 0 and standard error ≈ √(2(k−1))/(n−k)/√6 ≈ 0.026, so the null pass rate is ≈ 0.99. Per-cell η²_H values are reported descriptively.

H5 part 2 is unchanged: in every harmonic cell, the largest pairwise difference of the per-start medians of ν must be below 3%.

## Amendment A3 (2026-11-08, before any production MD job)

H6 part 1 assumes that an unconstrained ρ_k stays near a fixed ρ0. In the N = 64 CPU dry run, drift along the flat ground-state manifold made |ρ_k|² vary within a segment (coefficient of variation 0.38–0.81). The measured unsigned rate was then 1.34–1.77 times |k|√(T/(πS_k))/2π (AUDIT-E2 §5).

H6 stays exactly as registered; a FAIL of its part 1 is reported as evidence of manifold drift. A secondary hypothesis is added. It uses the conditional Rice formula, which holds for any slowly varying amplitude. Given ρ, θ̇ is Gaussian with variance N|k|²T/(2|ρ|²), so E|θ̇| = |k|·√(NT/π)·E[1/|ρ_k|].

| ID | Statement | PASS if |
|---|---|---|
| H6b | Conditional Rice rate of the unconstrained waves | at each harmonic T, the per-bin median over ground states of the unsigned rate divided by \|k\|√(NT/π)·⟨1/\|ρ_k\|⟩_t/(2π) lies in [0.90, 1.10] for every unconstrained \|k\|/K bin |

Here ⟨1/|ρ_k|⟩_t is estimated from the production-segment snapshots, which are log-spaced. That estimator is noted in the report. The A1 verdict mapping applies.
