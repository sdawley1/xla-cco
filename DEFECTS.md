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
