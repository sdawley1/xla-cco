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

## D-006 — `unit_ball_volume` is not bit-equal to the C++ hard-coded constants
- **Found:** 2026-11-05, by the T4 auditor's preview parity
- **Status:** fixed
- **Where:** src/xla_cco/kvectors.py (`unit_ball_volume`, hence `k_outer_from_ka`)
- **Symptom:** for d=1 at chi 0.25 and 0.45, `k_outer_from_ka` differs from the C++ `k_outer_hex` by 1 ulp (...d18 vs ...d19)
- **Repro:** `k_outer_from_ka(0.03926990816987242, 0.05, 1).hex()` gives `0x1.921fb54442d18p+0`, C++ gives `0x1.921fb54442d19p+0`
- **Impact:** the recomputed k window can sit 1 ulp off the C++ one; parity tests replay the stored hex window, so no stored result is affected
- **Regression test:** tests/unit/test_kvectors.py::test_regression_D006_k_outer_matches_cpp_bitwise (and tests/property/test_kvectors_props.py::test_unit_ball_volume_closed_form_and_cpp_constants)
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
- **Fix:** tolerance is now an error model, `FD_ROUNDOFF_C eps max|E(s +- h)| / h + FD_TRUNC_C h^2 (2 pi n_max)^3 sum(w) N^2 / V` (constants in tests/tolerances.py); `ATOL_FD` unchanged

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
