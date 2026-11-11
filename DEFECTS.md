# Defects log

Every defect gets an entry here and a `test_regression_D###_<slug>`.
Upstream (C++ reference at `/home/sd2402/ccoptimization/cco/`) defects are
`wontfix-upstream`: the reference is read-only; we document and, where
needed for parity, reproduce the behaviour.

Statuses:
- `open`: found and logged, not yet fixed.
- `fixed`: fixed in this repository; the entry names the regression test
  that failed before the fix.
- `wontfix-upstream`: a defect of the read-only C++ reference; documented
  and, where parity needs it, reproduced.
- `documented`: a deliberate, accepted deviation from the plan, not a bug.
  Nothing is fixed, so there is no `test_regression_` test; the entry names
  the existing tests that pin the chosen behaviour.

| ID | Status | Location | Summary |
|----|--------|----------|---------|
| D-001 | wontfix-upstream | `RepulsiveCCPotential.h:103-118` | Soft-core `AllForce` double-counts the CC force. |
| D-002 | wontfix-upstream | `main.cpp:119-136` | Tolerance parsed with `std::stof`; `1e-20` becomes the float32 value 9.99999968e-21. Parity runs use `float(np.float32(1e-20))`. |
| D-003 | wontfix-upstream | `StructureOptimization.cpp:724` | `MaxStep -= 2` on a `size_t` underflows for odd budgets. The port is immune: the budget is `max_evals // 2` iterations. Regression: `test_regression_D003_odd_budget_no_underflow`. |
| D-004 | wontfix-upstream | `main.cpp:171` | Random-mode box is built from the stale default `num = 1000`. |
| D-005 | wontfix-upstream | `StructureOptimization.cpp:841` | NLopt L-BFGS stopval is `-MaxEnergy` with no ftol/xtol, so it runs to maxeval or roundoff. |
| D-006 | fixed | `kvectors.py` (`unit_ball_volume`) | `unit_ball_volume` not bit-equal to the C++ hard-coded constants (d <= 4); k window 1 ulp off. |
| D-007 | fixed | `tests/conftest.py` | Hypothesis wall-clock deadline flakes on loaded nodes. |
| D-008 | fixed | `tests/property/test_energy_props.py` | Central-difference gradient test tolerance blind to energy magnitude and phase rounding. |
| D-009 | fixed | `io/tables.py` (`write_function_table`) | Rejected NaN, which C++ `WriteFunction` writes. |
| D-010 | fixed | `energy.py` (`StealthyEnergy`) | Undersized `n_max` and non-integer `n_vectors` failed silently on the table path. |
| D-011 | fixed | `tests/property/test_energy_props.py` | Gradient-sum test tolerance blind to pre-cancellation magnitude. |
| D-012 | fixed | `minimize/descent.py` (`_status`) | Exactly zero gradient: the C++ 0/0 accept test accepts forever; with `grow = 2` the step overflows and positions become NaN. |
| D-013 | fixed | `analysis/transport.py` (`torus_sq_cost`) | Fixed 3^d image scan is not exact for unreduced (strongly sheared) cells, though documented as exact for sheared cells. |
| D-014 | fixed | `experiments/scripts/analyze_wasserstein.py` (`certified`) | Converged Sinkhorn plans with a colliding rounding were trusted; their greedy-repaired matching is often suboptimal. |
| D-015 | fixed | `cli/relax.py` (`load_input`) | The CLI rejected its own output packs (basis scaled by N^(1/d)) as input. |
| D-016 | fixed | `_device.py` (`classify_failure`) | GPU guard blamed `--gres=gpu` when `JAX_PLATFORMS=cpu` hid the GPU, and its fix hint synced CUDA JAX into the CPU env. |
| D-017 | fixed | `tests/property/test_energy_tables_props.py` | Table-path gradient tolerance blind to pre-cancellation magnitude (sibling of D-011). |
| D-018 | fixed | `experiments/scripts/batten2008_mp_plot.py` (`stats`, `ks_eff`) | Coupling statistics asserted a positive spectrum; HIP starts have round-off zero modes in C and Var λ > c. |
| D-019 | fixed | `experiments/scripts/batten2008_mp_shoulder_plot.py` (`min_density`) | Dip window empty for the narrow saturated-RSA spectrum at χ = 0.1; asserted instead of giving NaN. |
| D-020 | fixed | `tests/property/test_md_props.py` | MD reversibility property tolerance blind to \|r\|. |
| D-040 | fixed | `dynamics/checkpoint.py` (`load_state`) | Corrupt MD checkpoint escaped as BadZipFile instead of a ValueError. |
| D-041 | fixed | `cli/md.py` (`main`) | MD CLI usage errors exited 2 instead of 1. |
| D-042 | fixed | `cli/md.py` (`start_state`) | A crash in phase 0 discarded its in-phase progress. |
| D-043 | fixed | `cli/md.py` (`fingerprint`) | Resume fingerprint missed k_chunk and the input content. |
| D-044 | fixed | `cli/md.py` (`run_phases`) | Equilibration gate averaged over the transient and was never acted on. |
| D-045 | fixed | `cli/md.py` (`record_schedule`) | Default 200 snapshots deduplicated to 184. |
| D-046 | fixed | `cli/md.py` (`_run_phase`) | Stale in-phase checkpoint left at the path handed to simulate. |
| D-050 | wontfix-upstream | `MD_System.cpp:199-250` | Andersen count mode fires on step 1 and float accumulation of p = 0.01 shifts the schedule: 501 resets per 50000-step call, not 500. |
| D-051 | wontfix-upstream | `MD_System.cpp:19-30, 128-136` | The MD system starts with zero acceleration, so the first Verlet half-kick after construction omits the force. |
| D-052 | wontfix-upstream | `main.cpp:864-877` | MD sample log prints energy, KE, dt and pressure with the 6-significant-digit stream default. |
| D-053 | fixed | `tools/cpp_reference/md_reference.py` (`shell_table`) | Shells grouped with exact float equality, so an ulp split of one shell put K inside it. |
| D-060 | fixed | `dynamics/run.py` (`begin_segment`) | Frozen `begin_segment` cannot recompute rho_prev (no waves); a scheme switch mixed COM-corrected and raw rho in the first winding step. |
| D-061 | fixed | `tests/integration/test_md_cli.py` (end to end) | The real-driver end-to-end test ran NVE from random (non-ground) states, which the drift guard correctly aborts. |
| D-062 | fixed | `cli/md.py` (`_run_phase`) | Andersen phases after an NVE segment started without begin_segment: their first winding step compared raw with COM-corrected rho. |
| D-070 | fixed | `benchmarks/md_bench.py` (`_bin_stats`) | V3 winding criterion (c) used the mean of per-wave ratios, biased by ~CV^2 for decorrelated waves; no dt passed at chi = 0.4. |
| D-071 | fixed | `benchmarks/md_bench.py` (`round_up_125`) | log10 rounding put x just below 10^k in the next decade (assertion error). |
| D-072 | fixed | `tests/unit/test_md_bench.py` (offline vs device winding) | Flaky high-tier test: subnormal S = rho^2/N counted by NumPy, flushed to 0 by XLA on CPU. |
| D-080 | documented | `tests/integration/test_md_equipartition.py` (V1) | Plan deviations of V1: lane-level t statistics instead of Flyvbjerg-Petersen blocking, and the Andersen KE target d N T / 2 (not d (N - 1) T / 2). |
| D-081 | documented | `tests/parity/test_md_cxx_parity.py` (V2) | Plan deviations of V2: the full JAX replica runs offline (SLURM CPU array) and the test reads its committed output; the C++ Andersen schedule is reproduced through `reset_acc`, D-051 is not replicated. |
| D-082 | fixed | `dynamics/run.py` (`begin_segment`) | begin_segment reset lanes already stopped (STATUS_DRIFT): their max_drift, sums and winding were zeroed, so the CLI reported 0 drift and 0 winding for a drifted lane. |
| D-083 | fixed | `dynamics/run.py` (`init_state`, `simulate`) | The constrained waves were never checked against the energy's n_vectors; mismatched waves silently used padding rows (n = 0, rho = N) as constrained densities. |
| D-084 | fixed | `dynamics/run.py` (`_step`) | Drift limit drift_tol * dof T / 2 is inf * 0 = NaN at T = 0 with drift_tol = inf, so every "verlet" lane aborted at step 1. |
| D-085 | fixed | `cli/relax.py` (`select_device`) | `--device cpu` failed once a GPU backend was initialized in the same process: `jax_platforms` is ignored after init and `jax.devices()[0]` stayed the GPU (76 in-process CLI test failures on the GPU suite). |
| D-086 | fixed | `tests/unit/test_energy.py` (`test_value_grad_density_bitwise_trig`) | Bitwise equality of separately compiled programs is a CPU property; on the GPU the jitted variants differ by ~1 ulp. |
| D-087 | fixed | `tests/property/test_energy_props.py` (D-011 regression) | The D-011 demonstration (the old tolerance fails on the pinned input) depends on CPU rounding; on the GPU the old tolerance passes. |
| D-091 | fixed | `theory/hardsphere.py` (`solve_py`) | PY cache key rounds eta to 12 decimal places, a relative perturbation up to ~1e-9 at small eta that shows up as noise in g(chi)/chi^2 and its derivatives. |
| D-092 | fixed | `theory/geometry.py` (`_intersection_fraction_beta`) | alpha_d via I_{1 - x^2/4}((d+1)/2, 1/2) loses absolute accuracy ~eps/x at small x (1 - x^2/4 rounds); used in production for d >= 4. |

## D-006 — `unit_ball_volume` is not bit-equal to the C++ hard-coded constants
- **Found:** 2026-11-05, by the T4 auditor's preview parity
- **Status:** fixed
- **Where:** src/xla_cco/kvectors.py (`unit_ball_volume`, hence `k_outer_from_ka`)
- **Symptom:** for d=1 at chi 0.25 and 0.45, `k_outer_from_ka` differs from the C++ `k_outer_hex` by 1 ulp (...d18 vs ...d19)
- **Repro:** `k_outer_from_ka(0.03926990816987242, 0.05, 1).hex()` gives `0x1.921fb54442d18p+0`, C++ gives `0x1.921fb54442d19p+0`
- **Impact:** the recomputed k window can sit 1 ulp off the C++ one; parity tests replay the stored hex window, so no stored result is affected
- **Regression test:** tests/unit/test_kvectors.py::test_regression_D006_k_outer_matches_cpp_bitwise (removed with the C++ parity suite) (and tests/property/test_kvectors_props.py::test_unit_ball_volume_closed_form_and_cpp_constants)
- **Fix:** `HyperSphere_C` (etc.cpp:23-37) returns constants for d<=4 (2, pi, 4.1887902047863909846, 4.9348022005446793094); `unit_ball_volume` now uses the same constants; the closed-form d=3 volume differs from the C++ literal by 1 ulp (0x1.0c152382d7365p+2 vs 0x1.0c152382d7366p+2), so the literal is needed for bit-exactness even though k2 happened to be unaffected for the goldens; d>=5 uses `pi^(d/2)/Gamma(d/2+1)` (`math.gamma` vs `gsl_sf_gamma` may differ by an ulp; untested, no goldens)

## D-007 — Hypothesis wall-clock deadline flakes on loaded nodes
- **Found:** 2026-11-05, while integrating T6 (CI)
- **Status:** fixed
- **Where:** tests/conftest.py (`dev` and `ci` Hypothesis profiles)
- **Symptom:** property tests intermittently fail with `DeadlineExceeded` although the code is correct
- **Repro:** tests/property/test_configpack_props.py::test_wrong_length_names_raise hit DeadlineExceeded at 332 ms against the 200 ms default deadline on the Della login node; it passed on re-run
- **Impact:** spurious red runs locally and in CI; no effect on computed results
- **Regression test:** tests/unit/test_conftest_profiles.py::test_regression_D007_profiles_have_no_deadline
- **Fix:** `deadline=None` set in the dev, ci and thorough profiles in tests/conftest.py (example counts unchanged)

## D-008 — Central-difference gradient test has an energy-blind tolerance
- **Found:** 2026-11-05, by the controller on main (dev profile, non-derandomized)
- **Status:** fixed
- **Where:** tests/property/test_energy_props.py::test_gradient_matches_central_difference (tolerance `ATOL_FD * max(1, |fd|max)`)
- **Symptom:** property test fails with ACTUAL 1.13e-13 vs DESIRED (FD) 1.36e-6 against atol 1e-6, although the analytic gradient is correct
- **Repro:** 16 near-coincident particles (s = 0.3 + 1e-12 u, u uniform, `default_rng(2)`), n integers in 1..3, w = 2, V = 0.5: |E| ~ 1e4, true |grad| ~ 2e-7, FD error 1.69e-6 > 1e-6 (pinned in `_coincident_case`)
- **Impact:** test-only; spurious red runs. No src defect: the FD reference carries roundoff ~ eps |E| / h that grows with |E|
- **Regression test:** tests/property/test_energy_props.py::test_regression_D008_fd_tolerance_scales_with_energy
- **Fix:** tolerance is now an error model, `FD_ROUNDOFF_C eps max|E(s +- h)| / h + FD_TRUNC_C h^2 (2 pi n_max)^3 sum(w) N^2 / V` (constants in tests/tolerances.py); plus the phase-rounding term (theta = 2 pi n.s carries eps theta; `theta_max * slope`, as in `_fd_atol`) in the roundoff part; `ATOL_FD` unchanged

## D-009 — `write_function_table` rejected NaN, which C++ `WriteFunction` writes
- **Found:** 2026-11-06, during T9 (CLI S(k) table with a negative-variance error bin)
- **Status:** fixed
- **Where:** src/xla_cco/io/tables.py (`write_function_table`)
- **Symptom:** `ValueError: non-finite value in table` when a S(k) standard-error column holds NaN (C++ writes it unclamped)
- **Repro:** `write_function_table(path, [np.array([np.nan])])`
- **Impact:** the CLI had to substitute 0.0 for NaN, which differs from C++ output
- **Regression test:** tests/unit/test_tables.py::test_regression_D009_function_table_writes_nan_like_cpp
- **Fix:** NaN is written as glibc printf `%1.10e` does (`nan` or `-nan` by sign bit; verified with ctypes snprintf); only +/-inf still raises


## D-010 — Undersized `n_max` and non-integer `n_vectors` failed silently on the table path
- **Found:** 2026-11-06, by the independent audit of T11
- **Status:** fixed
- **Where:** src/xla_cco/energy.py (`StealthyEnergy`, phase-table fast path of `value_and_grad`)
- **Symptom:** an `n_max` below max|n| gave NaN energy and gradient with no error at construction; a non-integer `n_vectors` was silently truncated by `astype(int32)` and gave a finite wrong energy
- **Repro:** `dataclasses.replace(en, n_max=en.n_max - 1).value_and_grad(s)` returned NaN; `StealthyEnergy(jnp.asarray([[0.5, 1.0]]), w, v, None, 1).value_and_grad(s)` returned a wrong finite value
- **Impact:** hand-built energies could produce NaN or wrong values far from the cause
- **Regression test:** tests/unit/test_energy_tables.py::test_regression_D010_undersized_n_max_raises_at_construction (and test_non_integer_n_vectors_raise_at_construction)
- **Fix:** `StealthyEnergy.__check_init__` raises `ValueError` for negative or undersized `n_max` and for non-integer-valued `n_vectors`; it is skipped when `n_vectors` is a tracer, where the NaN fill of out-of-range table gathers still guards (test_traced_n_vectors_skip_checks_and_nan_guards_the_table)

## D-011 — Gradient-sum test tolerance is blind to pre-cancellation magnitude
- **Found:** 2026-11-06, intermittently under the dev Hypothesis profile
- **Status:** fixed
- **Where:** tests/property/test_energy_props.py::test_gradient_sums_to_zero (atol `1e-11 * max(1, |g|max N)`)
- **Symptom:** sum_i dPhi/ds_i = -1.095e-11 (exact value 0) exceeds atol 1e-11
- **Repro:** d=1, 16 coincident particles at s=0.40625, 13 k-rows, V=1; reproduced with `default_rng(34)` rows n in [-4,4]\{0}, w in [0.5,2): sum 1.82e-11 vs old atol 1e-11 (pinned in the regression test)
- **Impact:** test-only; spurious red runs. For coincident particles |rho| = N, so each gradient is a large sum that cancels to ~0; the rounding error follows the pre-cancellation size, not |g|
- **Regression test:** tests/property/test_energy_props.py::test_regression_D011_grad_sum_tolerance_scales_with_gradient_magnitude
- **Fix:** atol = GRAD_SUM_C eps N (4 pi n_max sum(w) N / V) + GRAD_SUM_FLOOR (constants in tests/tolerances.py)

## D-012 — Descent overflows to NaN at an exactly zero gradient
- **Found:** 2026-11-06, repository audit (gstack /review)
- **Status:** fixed
- **Where:** src/xla_cco/minimize/descent.py (`chunk_one`, `_status`)
- **Symptom:** N = 1 at s = (0, 0) with `grow = 2.0`, `max_evals = 2200` returns NaN positions and NaN energy
- **Repro:** the single-particle stealthy energy is constant, so the gradient is exactly zero at s = 0; df2 = step * |g| = 0, the reject test is 0/0 = NaN and accepts (C++ semantics), and the step doubles until it overflows; s - 0 * inf = NaN
- **Impact:** the library could return NaN without an error status. Production runs stop on the energy tolerance before a stationary point, so no stored trajectory is affected
- **Regression test:** tests/unit/test_descent.py::test_regression_D012_zero_gradient_stops_finite (also test_descent_flat_energy_stays_finite)
- **Fix:** an exactly zero gradient is a stationary point; `_status` returns STATUS_GRAD_TOL (precedence: energy tol > zero gradient > step floor > budget), so descent stops with the configuration unchanged

## D-013 — Minimum-image cost wrong for unreduced cells
- **Found:** 2026-11-06, repository audit (gstack /review)
- **Status:** fixed
- **Where:** src/xla_cco/analysis/transport.py (`torus_sq_cost`)
- **Symptom:** with basis [[1, 0], [5, 1]], 36 % of cost entries were wrong, by up to 2.04 in squared distance
- **Repro:** rounding the fractional difference and scanning only the 3^d neighbouring images misses the nearest translate when the cell is far from reduced
- **Impact:** wrong W_2 for strongly sheared cells. The experiments use square boxes (single-image path), so no stored result is affected
- **Regression test:** tests/unit/test_transport.py::test_regression_D013_cost_exact_for_unreduced_cells (also test_cost_rejects_too_few_images, test_cost_needs_explicit_images_under_jit)
- **Fix:** `required_images(basis)` bounds the per-axis image radius (|n_k| <= 1/2 + R |B^-1 e_k|, R = max |ds B| over the half-cube), which is exact for any cell; `images=None` (default) uses it, an explicit radius below it raises for a concrete basis, and the scan runs as a `lax.scan` over the translates

## D-014 — Converged-but-colliding Sinkhorn frames trusted
- **Found:** 2026-11-06, repository audit (gstack /review)
- **Status:** fixed
- **Where:** experiments/scripts/analyze_wasserstein.py (`certified`)
- **Symptom:** frames whose plan had converged (marginal error < tol) but whose row-argmax rounding collided were certified; the greedy repair there exceeded the exact W_2 on 343 of 1860 compared chi = 0.4 POS frames (worst 2.8e-3 relative) and 28 of 1699 at chi = 0.1
- **Repro:** genuine near-ties (two close particles whose partner swap costs less than ~eps) split the plan's mass, so the argmax collides however long Sinkhorn runs
- **Impact:** the full-resolution Sinkhorn curves carried errors up to 2.8e-3 at about 1.5 % of POS frames; the exact 201-frame curves were unaffected
- **Regression test:** experiments/tests/test_wasserstein_analysis.py::test_regression_D014_converged_collision_not_certified (also the apply_fallback tests)
- **Fix:** only a bijective rounding certifies a frame (every compared bijective frame equalled the exact optimum); `wasserstein_fallback.py` solves the exact assignment at all other frames and `apply_fallback` substitutes those values, failing loudly if any are missing

## D-015 — CLI cannot read its own output
- **Found:** 2026-11-06, repository audit (gstack /devex-review)
- **Status:** fixed
- **Where:** src/xla_cco/cli/relax.py (`load_input` vs `pack_configs`)
- **Symptom:** relaxing `S.ConfigPack` again fails with "input basis has |det| = N, not unit volume"
- **Repro:** outputs are written C++-style with basis x N^(1/d) (|det| = N), while the input check accepted only |det| = 1
- **Impact:** relaxed packs could not be fed back (continuation, re-relaxation) without manual rescaling
- **Regression test:** tests/integration/test_cli.py::test_regression_D015_cli_reads_its_own_output (also test_cli_input_basis_mode_is_enforced)
- **Fix:** `--input-basis {auto,unit,scaled}` (default auto) accepts |det| = 1 or |det| = N and normalizes scaled bases to unit volume; any other volume still raises

## D-016 — GPU guard misdiagnosis
- **Found:** 2026-11-06, repository audit (gstack /devex-review)
- **Status:** fixed
- **Where:** src/xla_cco/_device.py (`classify_failure`)
- **Symptom:** with `JAX_PLATFORMS=cpu` on a GPU node, the error said "probably no --gres=gpu"; the wrong-environment hint `uv sync --dev --extra cuda --locked` installed CUDA JAX into the CPU `.venv`
- **Repro:** `JAX_PLATFORMS=cpu python -m xla_cco.cli.relax ... --device gpu` on della-gpu
- **Impact:** misleading failure message; following the hint polluted the CPU environment
- **Regression test:** tests/unit/test_gpu_assert.py::test_regression_D016_jax_platforms_hiding_gpu_is_named (also test_jax_platforms_allowing_gpu_falls_through, test_cli_gpu_guard)
- **Fix:** `JAX_PLATFORMS` without cuda/gpu is reported first ("unset it, or pass --device cpu"); the hint now gives the full `UV_PROJECT_ENVIRONMENT=.venv-cuda uv sync ...` command

## D-017 — Table-path gradient tolerance blind to pre-cancellation magnitude
- **Found:** 2026-11-07, GitHub Actions (PR #13, derandomized `ci` profile), on both Python versions
- **Status:** fixed
- **Where:** tests/property/test_energy_tables_props.py (`test_tables_match_oracle_and_baseline`, `test_tables_gradient_matches_autodiff`)
- **Symptom:** phase-table vs trig gradient differ by 1.05e-12 where the test allowed 1e-12 * max(1, max|grad|) = 1e-12
- **Repro:** 16 coincident particles at s = (0.8125, 0.8125), 10 k-rows n = (1, 1) with weights summing to 12.26, 6 zero padding rows, V = 1, chunk = 1; the exact gradient is 0, so both values are pure round-off of terms of size (4 pi n_max / V) sum(w) N
- **Impact:** test-only; spurious red CI. The energy code is correct
- **Regression test:** tests/property/test_energy_tables_props.py::test_regression_D017_coincident_particles_gradient_tolerance
- **Fix:** atol = max(1e-12 * max(1, max|grad|), GRAD_SUM_C eps (4 pi n_max / V) sum(w) N + GRAD_SUM_FLOOR), the D-011 bound per particle, in both gradient comparisons

## D-018 — coupling_mp_compare asserts a positive spectrum at HIP starts
- **Found:** 2026-11-09, per-start coupling run (hip2d/N1000, SLURM 17145072)
- **Status:** fixed
- **Where:** experiments/scripts/batten2008_mp_plot.py (`stats`)
- **Symptom:** `AssertionError: positive spectrum` (`stats`), then `AssertionError: MP normalization` (`ks_distance` with c_eff >= 1), while drawing coupling_mp_compare for HIP starts
- **Repro:** HIP starts at chi = 0.4 (tag 2-5): the coupling C of the start configurations has 42-134 eigenvalues at round-off (|lambda| <= 6e-16, some negative), so <1/lambda>(1 - c) is a huge number of either sign, and Var lambda = 2.96 c = 2.37 > 1, so MP(c_eff) does not exist
- **Impact:** no per-start figure for HIP; the stored Poisson figures are unaffected (their smallest start eigenvalue, 1e-11 at chi = 0.49, is a genuine void mode)
- **Regression test:** experiments/tests/test_batten2008_coupling_plots.py::test_regression_D018_stats_with_zero_modes and ::test_regression_D018_ks_eff_undefined_for_variance_above_one
- **Fix:** <1/lambda> is NaN when lambda_min / lambda_max <= 1e-13 (round-off zeros), and the assertion checks only that it is not <= 0; `ks_eff` returns NaN (no point, no MP(c_eff) curve) unless 0 < c_eff < 1

## D-019 — coupling_mp_shoulder dip window outside a narrow spectrum
- **Found:** 2026-11-09, per-start coupling run (rsa2d/N999, SLURM 17145071)
- **Status:** fixed
- **Where:** experiments/scripts/batten2008_mp_shoulder_plot.py (`min_density`)
- **Symptom:** `AssertionError: window inside the histogram` in panel (d)
- **Repro:** saturated-RSA ground states at chi = 0.1 have lambda(C) within 0.85-1.15 (5-95 %), so the 0.3-0.8 window clipped to the quantiles is empty
- **Impact:** no per-start figure for saturated RSA; Poisson unaffected
- **Regression test:** experiments/tests/test_batten2008_coupling_plots.py::test_regression_D019_min_density_window_outside_support
- **Fix:** an empty clipped window returns NaN (no point drawn) instead of asserting

## D-020 — MD reversibility property used an absolute tolerance blind to |r|
- **Found:** 2026-11-08, T1 fix round (Hypothesis `dev` profile, random seed)
- **Status:** fixed
- **Where:** tests/property/test_md_props.py (`test_verlet_reversible_random_states`), tests/tolerances.py (`ATOL_MD_REVERSE`)
- **Symptom:** forward/backward Verlet returned to r0 within 1.08e-12, where the test allowed 1e-12; the derivation assumed |r| <= 10 but the 33-particle 1D cell has |r| up to 33
- **Repro:** `shape=(33, 1), seed=3192540, dt=0.05` (40 steps each way, stealthy force of the cubic cell)
- **Impact:** test-only; spurious failure. The integrator is correct (error 538 eps |r| over 80 steps at the repro, chaotic amplification of roundoff)
- **Regression test:** tests/property/test_md_props.py::test_regression_D020_reverse_tolerance_scales_with_r (repro also pinned by `@example`)
- **Fix:** bound MD_REVERSE_ULPS_PER_STEP * 2 n_steps * eps * max(1, max|r0|) with C = 16 (measured worst 538 eps |r| at the repro, 17 eps |r| at dt <= 0.02); the property draws dt from {0.01, 0.02}

## D-040 — Corrupt MD checkpoint escapes as BadZipFile
- **Found:** 2026-11-08, audit of T3 (AUDIT-T3.md finding 2)
- **Status:** fixed
- **Where:** src/xla_cco/dynamics/checkpoint.py (`load_state`); src/xla_cco/cli/md.py (`resume`)
- **Symptom:** a truncated or garbage checkpoint raised `zipfile.BadZipFile` (and a checkpoint without `next_phase` a KeyError), outside the documented Raises and outside the CLI's except clause: a traceback instead of `ERROR:` and exit 1
- **Repro:** `save_state(p, s, {}); p.write_bytes(p.read_bytes()[:n // 2]); load_state(p, s)` → BadZipFile
- **Impact:** CLI crash with a traceback on a damaged checkpoint; no wrong results
- **Regression test:** tests/unit/test_md_checkpoint.py::test_regression_D040_truncated_checkpoint_raises_value_error (also test_damaged_checkpoint_raises_value_error, tests/integration/test_md_cli.py::test_md_cli_damaged_checkpoint_exits_1)
- **Fix:** zip/npy/JSON/KeyError failures become `CheckpointError` (a ValueError); `read_meta` added; the CLI checks every metadata field it uses

## D-041 — MD CLI usage errors exit 2
- **Found:** 2026-11-08, audit of T3 (finding 3)
- **Status:** fixed
- **Where:** src/xla_cco/cli/md.py (`main`)
- **Symptom:** argparse usage errors returned 2; the contract is exit 1 on any error
- **Repro:** `md.main(["--dt"])` → 2
- **Impact:** job scripts testing for exit 1 miss usage errors
- **Regression test:** tests/integration/test_md_cli.py::test_regression_D041_usage_error_exits_1
- **Fix:** SystemExit codes other than 0/None map to 1 (--help still 0)

## D-042 — Crash in phase 0 discards its in-phase progress
- **Found:** 2026-11-08, audit of T3 (finding 5)
- **Status:** fixed
- **Where:** src/xla_cco/cli/md.py (`start_state`)
- **Symptom:** with no phase checkpoint yet, the in-phase file of phase 0 was deleted as stale on restart, losing up to n_equil / 2 steps
- **Repro:** crash at the first simulate call after an in-phase save, restart → equilibration restarts from step 0
- **Impact:** lost compute only (the rerun is identical)
- **Regression test:** tests/integration/test_md_cli.py::test_regression_D042_phase0_inphase_progress_kept
- **Fix:** the lanes are checkpointed (next_phase 0) before the first phase; a newer in-phase file is folded into the checkpoint on resume

## D-043 — Resume fingerprint misses k_chunk and the input content
- **Found:** 2026-11-08, audit of T3 (finding 6)
- **Status:** fixed
- **Where:** src/xla_cco/cli/md.py (`fingerprint`)
- **Symptom:** a checkpoint was resumed after --k-chunk changed or the input file was rewritten at the same path
- **Repro:** run with --checkpoint, rewrite the input npz with other ground states, rerun → resumed silently
- **Impact:** a resumed run could mix two inputs (or summation orders) without notice
- **Regression test:** tests/integration/test_md_cli.py::test_regression_D043_fingerprint_has_k_chunk_and_input_hash (plus the per-field test_md_cli_resume_rejects_any_changed_setting)
- **Fix:** k_chunk and the SHA-256 of the input bytes are fingerprinted (the path is not)

## D-044 — Equilibration gate averaged over the transient and never acted on
- **Found:** 2026-11-08, audit of T3 ("Against E1")
- **Status:** fixed
- **Where:** src/xla_cco/cli/md.py (`run_phases`, `protocol_phases`)
- **Symptom:** <Phi>/(M T) included the start-up transient and was only reported
- **Repro:** Phi = 10 M T in the first half of equilibration, M T after → gate 5.5
- **Impact:** a gate value that does not measure equilibrium; unequilibrated lanes went to production unflagged
- **Regression test:** tests/integration/test_md_cli.py::test_regression_D044_gate_excludes_transient (also test_gate_failure_extends_once)
- **Fix:** equilibration runs in halves, the second starting with begin_segment; a failed gate (|g - 1| > 5 %) extends equilibration once; per-lane gate values and pass flags are written

## D-045 — Default 200 snapshots deduplicate to 184
- **Found:** 2026-11-08, audit of T3 ("Against E1")
- **Status:** fixed
- **Where:** src/xla_cco/cli/md.py (`record_schedule`)
- **Symptom:** rounding geomspace(1, L, n) produced repeated early steps, so fewer snapshots than requested
- **Repro:** defaults (n_nve = 1e6, 2 cycles, 200 snapshots) → 184 record steps
- **Impact:** fewer early-time snapshots than documented
- **Regression test:** tests/integration/test_md_cli.py::test_regression_D045_default_snapshot_count (also test_segment_offsets_properties)
- **Fix:** `segment_offsets` builds min(n, L) distinct log-spaced integers per segment

## D-046 — Stale in-phase checkpoint left at the path handed to simulate
- **Found:** 2026-11-08, audit of T3 (finding 4)
- **Status:** fixed
- **Where:** src/xla_cco/cli/md.py (`_run_phase`, `_save_phase`)
- **Symptom:** an old `.inphase.npz` stayed at the path passed to `run.simulate`; a simulate that resumed from it would re-apply steps
- **Repro:** a simulate that writes its checkpoint at the end of each call → the file exists when the next phase starts
- **Impact:** none with a write-only simulate; double-applied steps otherwise
- **Regression test:** tests/integration/test_md_cli.py::test_regression_D046_simulate_never_gets_existing_file
- **Fix:** the in-phase file is removed after every phase checkpoint and before every simulate call

## D-050 — Andersen count mode: reset on step 1 and a 501-reset schedule
- **Found:** 2026-11-08, while building the C++ MD reference (plan T4)
- **Status:** wontfix-upstream
- **Where:** /home/sd2402/ccoptimization/cco/MD_System.cpp:199-250 (`ParticleMolecularDynamics::AndersonEvolve`)
- **Symptom:** `ResetCount` starts at 0.0 in every call and the reset loop runs `for (; ResetCount > 0; ResetCount -= 1.0)` after `ResetCount += p`, so the first step of every call resets a velocity; the float sum of 0.01 then reaches > 0 at step 100 (not 101), and later every 100 steps. One `AndersonEvolve(50000, ..., 0.01)` call does 501 resets at steps 1, 100, 200, ..., 50000, not floor(p n) = 500 (the plan's T1 count-mode description)
- **Repro:** transcription `tools/cpp_reference/md_reference.py::andersen_reset_steps(50000, 0.01)` gives 501 steps starting `[1, 100, 200, 300]`
- **Impact:** a JAX count-mode replica must reset at these exact steps (recorded in `tests/data/md_cxx_reference.json` `protocol.andersen`) to replicate the C++ protocol for V2; statistics change by O(1/500)
- **Regression test:** tests/unit/test_md_cxx_reference.py::test_regression_D050_andersen_count_schedule
- **Fix:** none (C++ is read-only); reproduced in the transcription and recorded in the reference JSON

## D-051 — First Verlet half-kick after MD construction uses a zero acceleration
- **Found:** 2026-11-08, while building the C++ MD reference (plan T4)
- **Status:** wontfix-upstream
- **Where:** /home/sd2402/ccoptimization/cco/MD_System.cpp:19-30 (constructor zero-fills `Acceleration`), :128-136 (`Evolve` kicks with the stored acceleration)
- **Symptom:** `CollectiveCoordinateMD` builds `ParticleMolecularDynamics` from the relaxed config without computing forces, so step 1 of stage 1 kicks with a = 0 instead of F(r0)/m
- **Repro:** read the constructor and `Evolve`; no `_GetAcc` call precedes the first step
- **Impact:** none measurable here: the start is an L-BFGS-relaxed config with |F| near 0, and 2.2e7 later steps follow. Recorded in the reference JSON protocol so V2 does not chase it
- **Regression test:** tests/unit/test_md_cxx_reference.py::test_regression_D051_first_kick_zero_acceleration_recorded (guards the protocol record; the quirk itself is statistically invisible: one kick from a relaxed start with |F| ~ 0 before 2.2e7 steps)
- **Fix:** none (C++ is read-only)

## D-052 — MD sample log has 6 significant digits
- **Found:** 2026-11-08, while building the C++ MD reference (plan T4)
- **Status:** wontfix-upstream
- **Where:** /home/sd2402/ccoptimization/cco/CCOptimization/main.cpp:864-877 (`std::cout`/`logfile` default precision)
- **Symptom:** `energy=0.000414431`: the logged Phi, KE, dt and pressure are rounded to 6 significant digits, so a relative 1e-8 unit gate against the log is impossible
- **Repro:** any MD run; compare the logged energy with the `%.17g` harness value `0.00041443080610021411`
- **Impact:** the T4 unit gate compares xla Phi with full-precision C++ energies recomputed from the sampled ConfigPack by the unmodified `ShiftedCCPotential` (`tools/cpp_reference/build/dump_reference kernel`); the log is checked only to its rounding (relative 5e-6, and `%g` string equality). KE is available only at 6 digits (velocities are not saved)
- **Regression test:** tests/unit/test_md_cxx_reference.py::test_regression_D052_log_is_six_digit_rounding_of_cpp_energy
- **Fix:** none (C++ is read-only); full precision comes from the harness

## D-053 — `shell_table` split symmetry-equivalent shells at the ulp level
- **Found:** 2026-11-08, by Hypothesis (`test_choose_k_outer_properties`) before the first commit of T4
- **Status:** fixed
- **Where:** tools/cpp_reference/md_reference.py (`shell_table`, `choose_k_outer`)
- **Symptom:** `np.unique(k_sq)` treated the float |k|^2 of n = (5, 0) and (3, 4) as two shells; the mid-gap K then fell inside the n^2 = 25 shell (34 waves instead of 36) or on it
- **Repro:** `choose_k_outer(6.0 * np.eye(2), 36)` claimed M = 36 (an ulp sub-shell of n^2 = 25) but its K holds 34 waves (achievable counts are 34 and 40); `choose_k_outer(6.25 * np.eye(2), 76)` put K^2 on a shell
- **Impact:** none on the committed data (L = sqrt(1000), M = 50 was unaffected: the cutoff K was checked by enumeration and the C++ harness), only on other boxes
- **Regression test:** tests/unit/test_md_cxx_reference.py::test_regression_D053_shell_split_by_ulp, plus the `@example` pins in `test_choose_k_outer_properties`
- **Fix:** group sorted |k|^2 into shells with relative tolerance `SHELL_RTOL = 1e-9` and place K between the largest value of one shell and the smallest of the next

## D-060 — `begin_segment` cannot recompute rho_prev; scheme switch mixes COM rules
- **Found:** 2026-11-08, while implementing T5 (run.py)
- **Status:** fixed
- **Where:** src/xla_cco/dynamics/run.py (`begin_segment`, `md_chunk`); frozen signature in the plan's interface
- **Symptom:** the contract asks begin_segment to recompute rho_prev "from the current positions with the COM rule of the scheme", but its frozen signature `(state, settings, reset_sums, reset_winding)` has no waves or energy. Keeping the old rho_prev, Andersen -> begin_segment -> NVE compares the Andersen phase's COM-corrected rho with raw NVE rho in the first step (an error of k . R_cm per wave)
- **Repro:** Andersen (p = 0) lane with a COM velocity, 6 steps; begin_segment(verlet settings); 1 NVE step -> wind_net differs from angle(rho_raw(r1) conj rho_raw(r0)) by O(1) rad
- **Impact:** one spurious winding step per production segment (no outputs existed before the fix)
- **Regression test:** tests/unit/test_md_run.py::test_regression_D060_begin_segment_recomputes_rho_prev_on_scheme_switch (property: test_begin_segment_property)
- **Fix:** with reset_winding, begin_segment marks rho_prev pending (NaN); md_chunk recomputes pending rows of running lanes from the current positions with its scheme's COM rule before the first step. Interface concern reported to the orchestrator (an optional `waves`/`energy` argument would make it eager)

## D-061 — CLI end-to-end test ran NVE from random states
- **Found:** 2026-11-08, while implementing T5 (removing the strict xfail)
- **Status:** fixed
- **Where:** tests/integration/test_md_cli.py (`test_md_cli_end_to_end_small`)
- **Symptom:** with the real driver every lane ended in STATUS_DRIFT: far from a minimum (Phi ~ 4 at T = 1e-4) the Verlet energy error exceeds drift_tol * dof T / 2 ~ 1.5e-6
- **Repro:** remove the xfail and run the test on the unchanged input -> status_counts == {"STATUS_DRIFT": 4}
- **Impact:** test only (the drift guard behaved as specified)
- **Regression test:** tests/integration/test_md_cli.py::test_regression_D061_md_cli_end_to_end_small (renamed from test_md_cli_end_to_end_small; passes on ground states)
- **Fix:** the test relaxes its three random starts with gradient_flow (Phi < 1e-20) and feeds those ground states, as production does

## D-062 — Andersen phases after NVE start without begin_segment
- **Found:** 2026-11-08, while implementing T5 (reading cli/md.py against run.py)
- **Status:** fixed
- **Where:** src/xla_cco/cli/md.py (`_run_phase`; phases built by `protocol_phases` with begin=False)
- **Symptom:** after an NVE segment rho_prev holds raw rho; the next Andersen phase (no begin_segment) compares it with COM-corrected rho in its first step, adding |k . R_cm| (wrapped) to its winding
- **Repro:** CLI run with cycles >= 2 on ground states: the first step of andersen1 added 3.13 rad (max over waves) more than |angle(rho~(r1) conj rho~(r0))|
- **Impact:** none on written outputs (Andersen-phase winding and sums are reset by the next NVE begin_segment; snapshots fall in production segments only), but the in-state Andersen winding was wrong
- **Regression test:** tests/integration/test_md_cli.py::test_regression_D062_andersen_after_nve_restarts_winding (test_md_cli_protocol now checks the restart flags)
- **Fix:** every phase that does not begin a segment starts (done == 0) with `_rewind`: begin_segment(reset_sums=False, reset_winding=True) with remove_com=False, so rho_prev is recomputed under the phase's COM rule; velocities, sums and step are untouched, resume semantics unchanged

## D-070 — V3 criterion (c) estimator biased upward for decorrelated waves
- **Found:** 2026-11-08, while implementing V3 (job 17119586, chi = 0.4 calibration)
- **Status:** fixed
- **Where:** benchmarks/md_bench.py (`_bin_stats`, used by `compare_pooled`)
- **Symptom:** every dt in {0.04, ..., 0.005} failed criterion (c) at chi = 0.4. The per-bin change was Delta_b = +1-4 % in all unconstrained bins at T >= 1e-4, positive, and shrank only partly with dt (1.78 -> 0.72 % at T = 1e-4 down the ladder, as smaller-dt pairs decorrelate later); it persisted at the near-null pair 0.005 vs 0.0025.
- **Repro:** matched iid lognormal pairs (CV 0.3, n = 4000, no dt effect): mean(a / b) - 1 = +0.09 (> 100 SE), while mean(a) / mean(b) - 1 is within its delta-method SE.
- **Cause:** a Jensen bias. E[a / b] - 1 ~ CV(b)^2 for waves whose dt and dt/2 runs decorrelate (slow, noisy unconstrained winding).
- **Impact:** one failed calibration job (0.51 GPU-h). No production setting was taken from it.
- **Regression test:** tests/unit/test_md_bench.py::test_regression_D070_ratio_of_means_unbiased_under_noise
- **Fix:** Delta_b is now the ratio of bin means, with SE by the delta method. Both saved ladders were re-evaluated from their raw winding (`--ladder-from`).

## D-071 — `round_up_125` assertion for x just below a power of ten
- **Found:** 2026-11-08, by the Hypothesis property test while implementing V3
- **Status:** fixed
- **Where:** benchmarks/md_bench.py (`round_up_125`)
- **Symptom:** AssertionError "decade of x" for x = 999999999999.998.
- **Cause:** math.log10(x) rounds to 12.0, so base = 1e12 > x.
- **Impact:** none on outputs (drift_tol inputs were 4e-4 and 1e-3).
- **Regression test:** tests/unit/test_md_bench.py::test_regression_D071_round_up_125_property (@example pins the repro)
- **Fix:** step the base down one decade when it exceeds x.

## D-072 — Flaky offline-winding test on subnormal |rho|^2
- **Found:** 2026-11-08, by the V3 auditor (3 of 15 runs failed under the default `dev` Hypothesis profile)
- **Status:** fixed
- **Where:** tests/unit/test_md_bench.py (`test_offline_winding_every_step_matches_device_rule`, now `test_regression_D072_...`)
- **Symptom:** rho = [[[[-1.04e-158, -1.04e-158]]]] with floor 0: NumPy gives pi, the device rule gives 0.
- **Cause:** S = |rho|^2/N ~ 5e-317 is subnormal. NumPy keeps it (> 0, the step is counted); XLA on CPU flushes it to zero (the step is masked). The tolerance derivation assumed O(1) arguments, but the strategy drew any float in [-3, 3].
- **Impact:** test only. Production uses phase_floor = 1e-16 and |rho| >= ~0.03.
- **Regression test:** the same test, renamed test_regression_D072_offline_winding_every_step_matches_device_rule, with @example pinning the repro.
- **Fix:** the test zeroes rho components below MD_BENCH_TINY_RHO = 1e-100 (tests/tolerances.py), so S is never subnormal. Checked with 20 consecutive passes under the default profile.

## D-080 — V1 uses lane-level errors and a d N Andersen KE target
- **Found:** 2026-11-08, audit of V1 (AUDIT-V1 F4)
- **Status:** documented (plan deviation, accepted)
- **Where:** tests/integration/test_md_equipartition.py; experiments/md_winding/V1_validation.md
- **Symptom:** the plan asks for Flyvbjerg-Petersen blocked errors and <KE> / (d (N - 1) T / 2) = 1; V1 instead uses Student t over 512 independent lanes per T, and for the Andersen branch the target d N T / 2. Both were recorded only in the test docstring and V1_validation.md
- **Repro:** read `_lane_t` and `_summary(..., dof = D * N)` for the Andersen branches
- **Impact:** none on the verdict. Lane means are exactly independent (each lane has its own start and key), so their t statistic needs no correlation model, whereas within-lane blocking would miss the near-integrable NVE harmonic dynamics that keep each lane's mode energies for the whole run (blocking would underestimate the SE). Andersen redraws whole velocity rows without COM removal, so the COM momentum is thermalized and the canonical KE has d N quadratic degrees of freedom; the NVE branch removes the COM at begin_segment and keeps d (N - 1). The difference (1/N = 1.6 %) is below V1's resolution (AUDIT-V1 F2), so the choice is a matter of correctness, not of passing
- **Regression test:** tests/integration/test_md_equipartition.py::test_energy_means_match_harmonic_values (asserts both targets)
- **Fix:** none needed; recorded here (V2 uses the same d N target, `protocol.ke_dof = 512`, for the C++ protocol, which never removes the COM)

## D-081 — V2 parity reads an offline replica; Andersen schedule via reset_acc
- **Found:** 2026-11-08, while implementing V2
- **Status:** documented (plan deviation, accepted)
- **Where:** tests/parity/test_md_cxx_parity.py, tools/cpp_reference/md_jax_replica.py, slurm/md_jax_replica_cpu.slurm, tests/data/md_jax_replica.json
- **Symptom:** (1) the plan's V2 test "runs on CPU", but the full C++ protocol (2e6 equilibration steps + 1e5 steps per sample, N = 256, M = 50) costs ~1e8 lane-steps for >= 600 samples plus the 1.1 T control, i.e. ~110-300 us per lane-step on CPU and hours of wall time (measured: SLURM array 17122788, 10 tasks x 8 lanes x 2 CPUs, 2.2-3.1 h each, 26.3 task-hours = 53 core-hours); (2) the library count-mode Andersen (fires when acc + p >= 1, accumulator carried across calls) gives 500 resets per 5e4-step call starting at step 100, the C++ rule (counter reset to 0 per call, fires while > 0) 501 starting at step 1 (D-050); (3) the C++ zero-acceleration first kick (D-051)
- **Repro:** `andersen_reset` iterated 5e4 times from reset_acc = 0 vs `md_reference.andersen_reset_steps(50000, 0.01)`
- **Impact:** (1) the parity statistics come from committed data produced by the replica script at a recorded commit; the test file also reruns the replica's sample stage live on CPU (15 C++ configurations, one Andersen + NVE cycle) and checks the stored protocol record against the C++ JSON. (2) none: setting reset_acc = 1.0 (the C++ counter plus one) at the start of each Andersen phase through the public state reproduces the C++ reset steps exactly, so no library knob was added. (3) not replicated: one kick from a relaxed start (|F| ~ 0) 2.1e6 steps before the first sample. (4) the stored data (generated at 26af1b2) predate the `settings` and per-lane `temperature` fields that `md_jax_replica.py` now records (AUDIT-V2 F2); they were not regenerated: for them the test checks `t_factor` and that the current `settings_pair` of the recorded protocol is the C++ rule, and it asserts the recorded fields whenever a regenerated file carries them
- **Regression test:** tests/parity/test_md_cxx_parity.py::test_andersen_phase_reproduces_cxx_schedule, ::test_replica_phase_restarts_andersen_counter (AUDIT-V2 B1), ::test_stored_replica_ran_the_cxx_protocol
- **Fix:** none needed; regenerate the data with `env -u JAX_PLATFORMS sbatch slurm/md_jax_replica_cpu.slurm` then `md_jax_replica.py collect` after any change to the dynamics

## D-082 — `begin_segment` resets lanes that are already stopped
- **Found:** 2026-11-08, audit of the MD PR (AUDIT-V4 S1)
- **Status:** fixed
- **Where:** src/xla_cco/dynamics/run.py (`begin_segment`), reached through src/xla_cco/cli/md.py (`_begin`, `_rewind`)
- **Symptom:** begin_segment is vmapped over every lane whatever its status. On a lane with STATUS_DRIFT it zeroed max_drift, the sums and the winding, reset e0, set rho_prev to NaN (never repaired, the lane is not running) and removed the COM momentum from v, contradicting the md_chunk promise that a stopped lane "keeps every leaf from then on"
- **Repro:** two lanes, drift_tol = 1, lane 1 with e0 + 1e6 so it trips at step 1; simulate 3 steps; `jax.vmap(begin_segment(., settings, True, True))` -> lane 1 still STATUS_DRIFT but max_drift 0, n_sum 0, winding 0, rho_prev NaN
- **Impact:** a lane that drifts in an early CLI cycle reports max_drift = 0, wind_abs = 0 and n_sum = 0 in the final npz (only `status` and `cycle_*[0]` keep the evidence); averages over lanes that do not mask on status silently include zeros. No production run had a drifted lane
- **Regression test:** tests/unit/test_md_run.py::test_regression_D082_begin_segment_keeps_stopped_lanes (Hypothesis over the three flags and three non-running statuses, repro pinned by @example)
- **Fix:** begin_segment selects the new leaves only where status == STATUS_RUNNING (`_freeze`), so a stopped lane is returned bitwise unchanged

## D-083 — Waves never checked against the energy
- **Found:** 2026-11-08, audit of the MD PR (AUDIT-V4 S2)
- **Status:** fixed
- **Where:** src/xla_cco/dynamics/run.py (`_lane_rho` assumes `energy.n_vectors[:nc] == waves.n_vectors[:nc]`; `init_state`, `simulate`)
- **Symptom:** the constrained rho is read from the force pass (`out[2][:nc]`), so waves built with a larger k_constraint than the energy (or another basis) take the energy's zero-padding rows (n = 0, rho = N) or other waves as their constrained densities, with no error
- **Repro:** energy from `enumerate_kvectors(b, 0, K)`, waves from `make_waves(b, 1.3 K, 2 K)` (same total w) -> init_state and simulate run, with S = N on the padding rows
- **Impact:** none in production: the CLI builds both from one k-set and `build_system` raises when they differ. Library callers (E-series scripts) were unprotected
- **Regression test:** tests/unit/test_md_run.py::test_regression_D083_waves_must_match_energy
- **Fix:** host-side `_check_waves` (np.array_equal of the first nc rows, skipped for traced arrays) in init_state and simulate; ValueError on a mismatch

## D-084 — Drift limit is NaN at T = 0 with drift_tol = inf
- **Found:** 2026-11-08, audit of the MD PR (AUDIT-V4 N2)
- **Status:** fixed
- **Where:** src/xla_cco/dynamics/run.py (`_step`, the "verlet" drift guard)
- **Symptom:** limit = drift_tol * d (N - 1) T / 2 is inf * 0 = NaN for drift_tol = inf (the documented "no abort" setting used by select_dt and the benchmark) and T = 0; `drift <= NaN` is False, so every lane aborted at step 1
- **Repro:** `_settings(drift_tol=inf)`, lanes at temperature 0, simulate 10 verlet steps -> status STATUS_DRIFT, step 1
- **Impact:** unreachable from the CLI (T > 0 enforced). tests/unit/test_md_run.py::test_com_rule_under_rigid_translation[verlet] runs at T = 0 and so checked one step instead of 50; it passes over all 50 after the fix
- **Regression test:** tests/unit/test_md_run.py::test_regression_D084_infinite_drift_tol_at_zero_temperature
- **Fix:** the guard is skipped statically when drift_tol is infinite

## D-085 — `select_device("cpu")` fails after a GPU backend started
- **Found:** 2026-11-08, GPU test suite (slurm/test_gpu.slurm, job 17137898, JAX_PLATFORMS=cuda,cpu, A100)
- **Status:** fixed
- **Where:** src/xla_cco/cli/relax.py (`select_device`), used by the relax and md CLIs
- **Symptom:** `select_device("cpu")` only did `jax.config.update("jax_platforms", "cpu")`, which JAX ignores once the backend is initialized; `jax.devices()[0]` was then the GPU and the call raised "requested --device cpu but JAX uses gpu". In one pytest process every in-process CLI test after the first GPU computation failed (13 in test_cli.py, 63 in test_md_cli.py; the FileNotFoundError of test_md_cli_inphase_checkpoint_behind_is_ignored is the same failure: the CLI exited 1 before writing c.npz). The check also looked at `jax.devices()[0]`, not at the device computations actually use
- **Repro:** in a process with a GPU backend: `jax.devices(); select_device("cpu")` -> RuntimeError. On a CPU-only host: monkeypatch `jax.devices()` to list a GPU first
- **Impact:** test-only on the GPU suite; production CLI runs start a fresh process and select the device before any computation. Fails loudly, never silently
- **Regression test:** tests/integration/test_cli.py::test_regression_D085_select_device_cpu_after_backend_init (plus test_select_device_gpu_never_falls_back)
- **Fix:** select_device sets `jax_default_device` to the first device of the requested platform (cpu, or a GPU after require_gpu) and checks the platform of a freshly placed array; tests/conftest.py restores `jax_default_device` after each test so a CLI test does not move the rest of the GPU suite to the CPU

## D-086 — Bitwise `value_grad_density` test is CPU-only by nature
- **Found:** 2026-11-08, GPU test suite (job 17137898)
- **Status:** fixed
- **Where:** tests/unit/test_energy.py::test_value_grad_density_bitwise_trig (jit = True, 6 parameter sets)
- **Symptom:** gradients of `filter_jit(value_grad_density)` and `filter_jit(value_and_grad)` differ by <= 2.1e-14 absolute (7.5e-16 relative) on the A100
- **Repro:** `JAX_PLATFORMS=cuda,cpu pytest tests/unit/test_energy.py -k bitwise_trig` on a GPU node
- **Impact:** test-only. The test asserts that the two entry points share one computation; XLA's CPU backend compiles them to the same arithmetic, but separately compiled GPU programs may fuse and reduce in different orders (GPU kernels are not bitwise reproducible across programs). Device agreement is covered by tests/integration/test_gpu_device.py with tolerances
- **Regression test:** tests/unit/test_energy.py::test_value_grad_density_bitwise_trig (now asserts its outputs live on the CPU)
- **Fix:** the test runs under `jax.default_device(cpu)`, keeping the bitwise check where it is meaningful

## D-087 — D-011 regression demonstration depends on CPU rounding
- **Found:** 2026-11-08, GPU test suite (job 17137898)
- **Status:** fixed
- **Where:** tests/property/test_energy_props.py::test_regression_D011_grad_sum_tolerance_scales_with_gradient_magnitude
- **Symptom:** "old tolerance no longer reproduces D-011": the test asserts that the pre-fix tolerance fails on the pinned input, i.e. that the rounding error of the gradient sum exceeds 1e-11; on the GPU the sum rounds differently and stays below it
- **Repro:** run the test with a GPU default device
- **Impact:** test-only. The pinned repro is a statement about CPU (float64, XLA CPU reduction order) rounding; the fixed tolerance (D-011) bounds the error on any device
- **Regression test:** the same test, now pinned to the CPU device (asserts the gradient lives on the CPU)
- **Fix:** the test runs under `jax.default_device(cpu)`

## D-091 — PY cache key rounds eta to 12 decimal places
- **Found:** 2026-11-11, while building the PHS theory layer (feat/phs-theory), before its first commit
- **Status:** fixed
- **Where:** src/xla_cco/theory/hardsphere.py (`solve_py`, `ETA_CACHE_DIGITS`)
- **Symptom:** `solve_py` solved at `round(eta, 12)`: an absolute rounding of up to 5e-13, i.e. a relative perturbation of up to ~6e-10 at eta ~ 1e-3. g(chi) ~ eta^2, so the Chebyshev interpolant of g(chi)/chi^2 (built from solves at small chi) carried ~1e-9 relative noise, which differentiation amplifies to ~1e-6 in h'(chi) (measured: h' of the d = 3 interpolant changed by 2e-6 between 16 and 32 nodes, independent of the PY tolerance)
- **Repro:** `solve_py(8.0004e-4 + 3e-13, 1).eta` -> `0.00080004`, relative error 3.7e-10
- **Impact:** none released (found before the first commit of the module); numbers would have carried the ~1e-9 relative noise
- **Regression test:** tests/unit/test_theory_hardsphere.py::test_regression_D091_eta_cache_key_keeps_relative_precision (failed on all three pinned eta before the fix); tests/property/test_theory_props.py::test_eta_cache_key_is_relative (generalization, D-091 repro pinned with `@example`)
- **Fix:** the key is eta rounded to 15 significant figures (`_eta_cache_key`, relative change <= 5e-15). Afterwards the 20-node interpolant agrees with direct PY solves to < 1e-13 relative (was ~1e-9) and h' is stable to ~3e-9 between 16 and 32 nodes (was ~2e-6)

## D-092 — Incomplete-beta alpha_d loses accuracy at small separation
- **Found:** 2026-11-11, Hypothesis (tests/property/test_theory_props.py::test_alpha_closed_form_matches_incomplete_beta), before the first commit of the module
- **Status:** fixed
- **Where:** src/xla_cco/theory/geometry.py (`_intersection_fraction_beta`; production path of `intersection_fraction` for d >= 4)
- **Symptom:** alpha_d(x) = I_z((d+1)/2, 1/2) with z = 1 - x^2/4 formed in floating point: the rounding of z (~1e-16) is amplified by dI/dz ~ 1/x near z = 1, so the absolute error grows like eps/x at small x
- **Repro:** `_intersection_fraction_beta(1e-5, 1)` differs from the exact 1 - x/2 by 2.1e-13 (shrunk Hypothesis input x = 1e-5, d = 1)
- **Impact:** none released; d <= 3 use the closed forms, d >= 4 (not used by the PY/PHS layer) would carry ~1e-16/x absolute error
- **Regression test:** tests/unit/test_theory_geometry.py::test_regression_D092_beta_form_accurate_at_small_separation (failed at x = 1e-5, 1e-7, 1e-9 before the fix); property generalization with the repro pinned: tests/property/test_theory_props.py::test_alpha_closed_form_matches_incomplete_beta
- **Fix:** for x^2/4 < 1/2 use the complement 1 - I_{x^2/4}(1/2, (d+1)/2), where the argument x^2/4 is formed without cancellation
