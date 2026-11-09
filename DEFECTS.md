# Defects log

Every defect gets an entry here and a `test_regression_D###_<slug>`.
Upstream (C++ reference at `/home/sd2402/ccoptimization/cco/`) defects are
`wontfix-upstream`: the reference is read-only; we document and, where
needed for parity, reproduce the behaviour.

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
- **Regression test:** tests/unit/test_wasserstein_analysis.py::test_regression_D014_converged_collision_not_certified (also the apply_fallback tests)
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
- **Regression test:** tests/unit/test_batten2008_coupling_plots.py::test_regression_D018_stats_with_zero_modes and ::test_regression_D018_ks_eff_undefined_for_variance_above_one
- **Fix:** <1/lambda> is NaN when lambda_min / lambda_max <= 1e-13 (round-off zeros), and the assertion checks only that it is not <= 0; `ks_eff` returns NaN (no point, no MP(c_eff) curve) unless 0 < c_eff < 1

## D-019 — coupling_mp_shoulder dip window outside a narrow spectrum
- **Found:** 2026-11-09, per-start coupling run (rsa2d/N999, SLURM 17145071)
- **Status:** fixed
- **Where:** experiments/scripts/batten2008_mp_shoulder_plot.py (`min_density`)
- **Symptom:** `AssertionError: window inside the histogram` in panel (d)
- **Repro:** saturated-RSA ground states at chi = 0.1 have lambda(C) within 0.85-1.15 (5-95 %), so the 0.3-0.8 window clipped to the quantiles is empty
- **Impact:** no per-start figure for saturated RSA; Poisson unaffected
- **Regression test:** tests/unit/test_batten2008_coupling_plots.py::test_regression_D019_min_density_window_outside_support
- **Fix:** an empty clipped window returns NaN (no point drawn) instead of asserting
