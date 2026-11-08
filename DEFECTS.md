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
| D-020 | fixed | `tests/property/test_md_props.py` | MD reversibility property tolerance blind to \|r\|. |
| D-040 | fixed | `dynamics/checkpoint.py` (`load_state`) | Corrupt MD checkpoint escaped as BadZipFile instead of a ValueError. |
| D-041 | fixed | `cli/md.py` (`main`) | MD CLI usage errors exited 2 instead of 1. |
| D-042 | fixed | `cli/md.py` (`start_state`) | A crash in phase 0 discarded its in-phase progress. |
| D-043 | fixed | `cli/md.py` (`fingerprint`) | Resume fingerprint missed k_chunk and the input content. |
| D-044 | fixed | `cli/md.py` (`run_phases`) | Equilibration gate averaged over the transient and was never acted on. |
| D-045 | fixed | `cli/md.py` (`record_schedule`) | Default 200 snapshots deduplicated to 184. |
| D-046 | fixed | `cli/md.py` (`_run_phase`) | Stale in-phase checkpoint left at the path handed to simulate. |
| D-060 | fixed | `dynamics/run.py` (`begin_segment`) | Frozen `begin_segment` cannot recompute rho_prev (no waves); a scheme switch mixed COM-corrected and raw rho in the first winding step. |
| D-061 | fixed | `tests/integration/test_md_cli.py` (end to end) | The real-driver end-to-end test ran NVE from random (non-ground) states, which the drift guard correctly aborts. |
| D-062 | fixed | `cli/md.py` (`_run_phase`) | Andersen phases after an NVE segment started without begin_segment: their first winding step compared raw with COM-corrected rho. |

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

