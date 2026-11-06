# Parity results

## Level 3: ensemble parity vs C++ NLopt L-BFGS (T12)

Test: `tests/parity/test_ensemble_parity.py` (marks `della`, `slow`); run with
`sbatch slurm/parity_ensemble.slurm`. Statistics: `tests/ensemble_stats.py`.

Evidence run: SLURM job **16988911** (gputest, 1 GPU, COMPLETED exit 0,
13 passed, 307 s test session). Evaluated commit `974582ccedc51df3ad6d512ccf1d1eb7501f9d01`; the src tree
is `78ddc262a32553ef6fbfb5bb646b482005368936` (`git rev-parse HEAD:src`, clean working tree), src SHA-256
`e5b1077d9fe1e3b3abc2ade10a9bb7688dbe1ad074e44ebe2bc253c8f4b27502`. This is the rebased code that ships (includes the T11 phase
tables, used by the factory for M >= 75, i.e. chi = 0.35 and 0.45). JAX
0.12.0.1; 400 configs per task (tasks 0000-0002 = 1200 per chi);
success = Phi < 9.99999968e-21.

Cache: JAX results are cached under `/scratch/gpfs/TORQUATO/sd2402/xla_cco_runs/ensemble/`.
The key is the SHA-256 of every `src/xla_cco` Python file plus the settings (chi,
task, K2 scale, config count, tol, L-BFGS settings, JAX version). The device
(CPU or GPU) is NOT part of the key; a cache filled on one device is reused on
the other.

Change versus the earlier job 16986040 (pre-T11 src, identical settings): every
relaxation was recomputed; no statistic moved materially (max change over all KS
statistics and worst S(k) |z| is 1.3e-9; all p-values, success fractions and
status counts equal at the printed precision).

### Acceptance rules and honest false-alarm rates

- KS + success family: family-wise alpha = 0.01 by Bonferroni over 50 p-tests
  (5 chi x 5 tests x {JAX vs C++, C++ false-alarm control}); per-test threshold
  alpha/50 = 2e-4. KS threshold = min(KS_PMIN, alpha/n_tests) = 2e-4 (ruling R27).
  Success fraction, JAX vs C++: ONE-SIDED (H1: JAX < C++), because the port must
  not reach ground states less often; the two-sided p is reported too. The
  false-alarm control (C++ vs C++) keeps the two-sided success test.
- S(k) rule (pre-registered, ruling R29): every shell bin in (K2, 3 K2] must have
  |z| < SK_ZMAX = 4. It is NOT part of the Bonferroni family. Per bin the
  two-sided false-alarm probability is 2 Phi(-4) = 6.3e-5; with the actual
  number of bins (25, 53, 73, 91, 107 = 349 per comparison) and an independence
  approximation, 1 - (1 - 6.3e-5)^n_bins is 2.2% for the JAX-vs-C++ comparison
  alone and 4.3% over the 698 bins of the real comparison plus the false-alarm
  control. The overall false-alarm rate of the whole battery is therefore about
  1% (KS + success) plus about 4% (S(k)), not 1%; the thresholds were fixed before
  seeing data and were not changed.
- first g2-peak height = highest bin of the per-config g2 on the fixed grid
  r in [0, L/2], width 0.1; RMS displacement is minimum-image Cartesian; the
  S(k) per-config value is the mean over the k-vectors of a |k|^2 shell.
- -log10 Phi among successes is compared for reporting only (R22).

Resolution of the chi = 0.45 two-sided success difference (JAX 1200/1200 vs C++
1071/1200, two-sided p = 1.7e-31): C++ NLopt has a precision floor on the
tolerance (see "C++ energy floor" per chi); all C++ failures lie in
[1.0e-20, 2.0e-20]. JAX (energy_tol 0) continues to ~1e-30, so success is one-sided
by user decision. All five chi PASS, both controls behave, and the power control
now also asserts that S(k) and at least one KS test detect the wrong k-set.

### Controls

| control | chi | expected | outcome | behaves |
|---|---|---|---|---|
| false alarm (C++ vs C++) | 0.05 | PASS | PASS | True |
| false alarm (C++ vs C++) | 0.15 | PASS | PASS | True |
| false alarm (C++ vs C++) | 0.25 | PASS | PASS | True |
| false alarm (C++ vs C++) | 0.35 | PASS | PASS | True |
| false alarm (C++ vs C++) | 0.45 | PASS | PASS | True |
| power (JAX K2 x 0.9 vs C++) | 0.25 | FAIL | FAIL | True |
| power (JAX K2 x 0.9 vs C++) | 0.45 | FAIL | FAIL | True |

### Per-chi results (A = JAX, B = C++ unless stated)

#### chi = 0.05 (M = 13, chi_actual = 0.043771, K2 = 2.60930)

**JAX vs C++: PASS**

| statistic | A | B | z or KS D | p raw | p corrected | verdict |
|---|---|---|---|---|---|---|
| success fraction (less) | 1.0000 | 1.0000 | 0.000 | 1 | 1 | PASS |
| success fraction (two-sided, reported) | | | | 1 | 1 | n/a |
| KS min NN distance | | | 0.0150 | 0.999 | 1 | PASS |
| KS mean NN distance | | | 0.0250 | 0.848 | 1 | PASS |
| KS first g2-peak height | | | 0.0275 | 0.755 | 1 | PASS |
| KS RMS displacement | | | 0.0425 | 0.229 | 1 | PASS |
| S(k) worst of 25 bins | | | 0.680 (at |k|=4.061) | | | PASS (|z| < 4.0) |
| -log10 Phi KS (report only, R22) | | | | 0 | | n/a |

False-alarm control (C++ 0000-0002 vs C++ 0010-0012): **PASS**

- C++ energy floor: Phi_final quantiles {'median': 1.0764900356183804e-21, 'q90': 2.7379279198115737e-21, 'q99': 4.7617919680186445e-21, 'max': 7.698507976952976e-21}; C++ successes at 1e-20 / 2e-20 / 5e-20: [1200, 1200, 1200] of 1200
- status counts (JAX): {'STATUS_LINESEARCH_FAIL': 1200}; successes JAX 1200/1200, C++ 1200/1200
- -log10 Phi among successes (median [Q1, Q3]): JAX 30.11 [29.99, 30.24], C++ 20.97 [20.74, 21.29]
- timings: JAX relax 13.8 s (cached: False), analysis 8.0 s

#### chi = 0.15 (M = 40, chi_actual = 0.134680, K2 = 3.76326)

**JAX vs C++: PASS**

| statistic | A | B | z or KS D | p raw | p corrected | verdict |
|---|---|---|---|---|---|---|
| success fraction (less) | 1.0000 | 1.0000 | 0.000 | 1 | 1 | PASS |
| success fraction (two-sided, reported) | | | | 1 | 1 | n/a |
| KS min NN distance | | | 0.0242 | 0.875 | 1 | PASS |
| KS mean NN distance | | | 0.0267 | 0.787 | 1 | PASS |
| KS first g2-peak height | | | 0.0150 | 0.999 | 1 | PASS |
| KS RMS displacement | | | 0.0492 | 0.11 | 1 | PASS |
| S(k) worst of 53 bins | | | 0.648 (at |k|=5.581) | | | PASS (|z| < 4.0) |
| -log10 Phi KS (report only, R22) | | | | 0 | | n/a |

False-alarm control (C++ 0000-0002 vs C++ 0010-0012): **PASS**

- C++ energy floor: Phi_final quantiles {'median': 9.740356529326968e-22, 'q90': 2.1964520541495497e-21, 'q99': 3.385576338604157e-21, 'max': 4.536933568808567e-21}; C++ successes at 1e-20 / 2e-20 / 5e-20: [1200, 1200, 1200] of 1200
- status counts (JAX): {'STATUS_LINESEARCH_FAIL': 1200}; successes JAX 1200/1200, C++ 1200/1200
- -log10 Phi among successes (median [Q1, Q3]): JAX 28.97 [28.89, 29.04], C++ 21.01 [20.80, 21.24]
- timings: JAX relax 15.7 s (cached: False), analysis 5.5 s

#### chi = 0.25 (M = 73, chi_actual = 0.245791, K2 = 4.46184)

**JAX vs C++: PASS**

| statistic | A | B | z or KS D | p raw | p corrected | verdict |
|---|---|---|---|---|---|---|
| success fraction (less) | 1.0000 | 1.0000 | 0.000 | 1 | 1 | PASS |
| success fraction (two-sided, reported) | | | | 1 | 1 | n/a |
| KS min NN distance | | | 0.0375 | 0.368 | 1 | PASS |
| KS mean NN distance | | | 0.0217 | 0.941 | 1 | PASS |
| KS first g2-peak height | | | 0.0325 | 0.551 | 1 | PASS |
| KS RMS displacement | | | 0.0375 | 0.368 | 1 | PASS |
| S(k) worst of 73 bins | | | 1.170 (at |k|=10.997) | | | PASS (|z| < 4.0) |
| -log10 Phi KS (report only, R22) | | | | 0 | | n/a |

False-alarm control (C++ 0000-0002 vs C++ 0010-0012): **PASS**

- C++ energy floor: Phi_final quantiles {'median': 1.0613111444161545e-21, 'q90': 2.1563910029968327e-21, 'q99': 3.3912028331399694e-21, 'max': 5.669870588648582e-21}; C++ successes at 1e-20 / 2e-20 / 5e-20: [1200, 1200, 1200] of 1200
- status counts (JAX): {'STATUS_LINESEARCH_FAIL': 1200}; successes JAX 1200/1200, C++ 1200/1200
- -log10 Phi among successes (median [Q1, Q3]): JAX 28.33 [28.27, 28.39], C++ 20.97 [20.80, 21.16]
- timings: JAX relax 31.6 s (cached: False), analysis 6.5 s

#### chi = 0.35 (M = 101, chi_actual = 0.340067, K2 = 4.99141)

**JAX vs C++: PASS**

| statistic | A | B | z or KS D | p raw | p corrected | verdict |
|---|---|---|---|---|---|---|
| success fraction (less) | 1.0000 | 1.0000 | 0.000 | 1 | 1 | PASS |
| success fraction (two-sided, reported) | | | | 1 | 1 | n/a |
| KS min NN distance | | | 0.0292 | 0.687 | 1 | PASS |
| KS mean NN distance | | | 0.0175 | 0.993 | 1 | PASS |
| KS first g2-peak height | | | 0.0183 | 0.988 | 1 | PASS |
| KS RMS displacement | | | 0.0525 | 0.0732 | 1 | PASS |
| S(k) worst of 91 bins | | | 1.767 (at |k|=13.469) | | | PASS (|z| < 4.0) |
| -log10 Phi KS (report only, R22) | | | | 0 | | n/a |

False-alarm control (C++ 0000-0002 vs C++ 0010-0012): **PASS**

- C++ energy floor: Phi_final quantiles {'median': 1.9152870999305516e-21, 'q90': 3.364484477397599e-21, 'q99': 4.974454138665385e-21, 'max': 6.4150849822514376e-21}; C++ successes at 1e-20 / 2e-20 / 5e-20: [1200, 1200, 1200] of 1200
- status counts (JAX): {'STATUS_LINESEARCH_FAIL': 1200}; successes JAX 1200/1200, C++ 1200/1200
- -log10 Phi among successes (median [Q1, Q3]): JAX 27.83 [27.79, 27.87], C++ 20.72 [20.58, 20.85]
- timings: JAX relax 75.5 s (cached: False), analysis 5.5 s

#### chi = 0.45 (M = 128, chi_actual = 0.430976, K2 = 5.42756)

**JAX vs C++: PASS**

| statistic | A | B | z or KS D | p raw | p corrected | verdict |
|---|---|---|---|---|---|---|
| success fraction (less) | 1.0000 | 0.8925 | 11.676 | 1 | 1 | PASS |
| success fraction (two-sided, reported) | | | | 1.69e-31 | 8.46e-30 | n/a |
| KS min NN distance | | | 0.0317 | 0.584 | 1 | PASS |
| KS mean NN distance | | | 0.0283 | 0.721 | 1 | PASS |
| KS first g2-peak height | | | 0.0267 | 0.787 | 1 | PASS |
| KS RMS displacement | | | 0.0517 | 0.0812 | 1 | PASS |
| S(k) worst of 107 bins | | | 2.792 (at |k|=15.611) | | | PASS (|z| < 4.0) |
| -log10 Phi KS (report only, R22) | | | | 0 | | n/a |

False-alarm control (C++ 0000-0002 vs C++ 0010-0012): **PASS**

- C++ energy floor: Phi_final quantiles {'median': 6.1395323584721924e-21, 'q90': 1.0235030652424365e-20, 'q99': 1.5887230250102887e-20, 'max': 2.015534065962283e-20}; C++ successes at 1e-20 / 2e-20 / 5e-20: [1071, 1199, 1200] of 1200
- status counts (JAX): {'STATUS_LINESEARCH_FAIL': 1200}; successes JAX 1200/1200, C++ 1071/1200
- -log10 Phi among successes (median [Q1, Q3]): JAX 27.26 [27.18, 27.34], C++ 20.23 [20.14, 20.35]
- timings: JAX relax 101.7 s (cached: False), analysis 5.5 s

### Power control (JAX with K2 x 0.9 vs C++)

- chi = 0.25 (400 JAX configs): **FAIL** (expected FAIL); success 0.000 vs 1.000 (p 0); failing KS: ['nn_min', 'nn_mean', 'g2_peak', 'rms_disp']; S(k) worst |z| = 8.6
  - KS p-values: nn_min p=9.39e-67, nn_mean p=1.51e-286, g2_peak p=1.49e-36, rms_disp p=7.11e-106; success two-sided p 0
- chi = 0.45 (400 JAX configs): **FAIL** (expected FAIL); success 0.000 vs 0.892 (p 8.22e-237); failing KS: ['nn_min', 'nn_mean', 'g2_peak', 'rms_disp']; S(k) worst |z| = 21.6
  - KS p-values: nn_min p=1.51e-286, nn_mean p=3.46e-323, g2_peak p=7.86e-201, rms_disp p=8.09e-107; success two-sided p 8.22e-237


### Mapping and window verification

- chi 0.05: ic/chunk_NNNN vs task_NNNN/_InitConfig, 6 tasks x 400 configs: max wrapped diff 0.0e+00, bit-identical True; K2 = 2.609300 (log 2.6093), chi = 0.043771 (log 0.043771), M = 13: window and chi match all 6 engine.logs
- chi 0.15: ic/chunk_NNNN vs task_NNNN/_InitConfig, 6 tasks x 400 configs: max wrapped diff 0.0e+00, bit-identical True; K2 = 3.763262 (log 3.76326), chi = 0.134680 (log 0.13468), M = 40: window and chi match all 6 engine.logs
- chi 0.25: ic/chunk_NNNN vs task_NNNN/_InitConfig, 6 tasks x 400 configs: max wrapped diff 0.0e+00, bit-identical True; K2 = 4.461841 (log 4.46184), chi = 0.245791 (log 0.245791), M = 73: window and chi match all 6 engine.logs
- chi 0.35: ic/chunk_NNNN vs task_NNNN/_InitConfig, 6 tasks x 400 configs: max wrapped diff 0.0e+00, bit-identical True; K2 = 4.991412 (log 4.99141), chi = 0.340067 (log 0.340067), M = 101: window and chi match all 6 engine.logs
- chi 0.45: ic/chunk_NNNN vs task_NNNN/_InitConfig, 6 tasks x 400 configs: max wrapped diff 0.0e+00, bit-identical True; K2 = 5.427564 (log 5.42756), chi = 0.430976 (log 0.430976), M = 128: window and chi match all 6 engine.logs
