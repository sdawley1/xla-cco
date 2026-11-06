# Parity results

## Level 3: ensemble parity vs C++ NLopt L-BFGS (T12)

Test: `tests/parity/test_ensemble_parity.py` (marks `della`, `slow`); run with
`sbatch slurm/parity_ensemble.slurm`. Statistics: `tests/ensemble_stats.py`.

Run: SLURM job 16986040 (gputest, 1 GPU, COMPLETED exit 0, 13 passed, 197 s test
session; relaxations recomputed on the GPU in this job), git base 41d9baf plus
the T12 commits; JAX 0.12.0.1; 400 configs per task (tasks 0000-0002 = 1200 per
chi); success = Phi < 9.99999968e-21. An earlier run (job 16982731, identical
relaxations) gave the same statistics except for the two-sided success test.

Acceptance rules:

- family-wise alpha = 0.01, Bonferroni over 50 p-tests (5 chi x 5 tests x
  {JAX vs C++, C++ false-alarm control}), p-test threshold 2e-4;
- KS threshold = min(KS_PMIN, alpha / n_tests) = 2e-4 (ruling R27);
- S(k): every shell bin in (K2, 3 K2] must have |z| < 4;
- success fraction, JAX vs C++: ONE-SIDED (H1: JAX < C++), because the port
  must not reach ground states less often than C++. The two-sided p is also
  reported (tables). The false-alarm control (C++ vs C++) stays two-sided;
- first g2-peak height = highest bin of the per-config g2 on the fixed grid
  r in [0, L/2], width 0.1; RMS displacement is minimum-image Cartesian;
  S(k) per-config value = mean over the k-vectors of a |k|^2 shell;
- -log10 Phi among successes is compared for reporting only (R22).

Resolution of the chi = 0.45 two-sided success difference (JAX 1200/1200 vs C++
1071/1200, two-sided p = 1.7e-31): C++ NLopt has a precision floor on the
tolerance (see "C++ energy floor" per chi); all C++ failures lie in
[1.0e-20, 2.0e-20]. JAX (energy_tol 0) continues to ~1e-30, so the success
fraction is one-sided by user decision. With it, all five chi PASS and both
controls behave.

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
- timings: JAX relax 13.9 s (cached: False), analysis 8.3 s

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
- timings: JAX relax 15.8 s (cached: False), analysis 5.8 s

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
- timings: JAX relax 31.8 s (cached: False), analysis 6.7 s

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
- -log10 Phi among successes (median [Q1, Q3]): JAX 27.86 [27.80, 27.93], C++ 20.72 [20.58, 20.85]
- timings: JAX relax 33.8 s (cached: False), analysis 5.9 s

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
- status counts (JAX): {'STATUS_STAGNATED': 21, 'STATUS_LINESEARCH_FAIL': 1179}; successes JAX 1200/1200, C++ 1071/1200
- -log10 Phi among successes (median [Q1, Q3]): JAX 27.05 [26.94, 27.15], C++ 20.23 [20.14, 20.35]
- timings: JAX relax 46.1 s (cached: False), analysis 5.8 s

### Power control (JAX with K2 x 0.9 vs C++)

- chi = 0.25 (400 JAX configs): **FAIL** (expected FAIL); success 0.000 vs 1.000 (p 0); failing KS: ['nn_min', 'nn_mean', 'g2_peak', 'rms_disp']; S(k) worst |z| = 8.6
- chi = 0.45 (400 JAX configs): **FAIL** (expected FAIL); success 0.000 vs 0.892 (p 8.22e-237); failing KS: ['nn_min', 'nn_mean', 'g2_peak', 'rms_disp']; S(k) worst |z| = 21.6


### Mapping and window verification

- chi 0.05: ic/chunk_NNNN vs task_NNNN/_InitConfig, 6 tasks x 400 configs: max wrapped diff 0.0e+00, bit-identical True; K2 = 2.609300 (log 2.6093), chi = 0.043771 (log 0.043771), M = 13: window and chi match all 6 engine.logs
- chi 0.15: ic/chunk_NNNN vs task_NNNN/_InitConfig, 6 tasks x 400 configs: max wrapped diff 0.0e+00, bit-identical True; K2 = 3.763262 (log 3.76326), chi = 0.134680 (log 0.13468), M = 40: window and chi match all 6 engine.logs
- chi 0.25: ic/chunk_NNNN vs task_NNNN/_InitConfig, 6 tasks x 400 configs: max wrapped diff 0.0e+00, bit-identical True; K2 = 4.461841 (log 4.46184), chi = 0.245791 (log 0.245791), M = 73: window and chi match all 6 engine.logs
- chi 0.35: ic/chunk_NNNN vs task_NNNN/_InitConfig, 6 tasks x 400 configs: max wrapped diff 0.0e+00, bit-identical True; K2 = 4.991412 (log 4.99141), chi = 0.340067 (log 0.340067), M = 101: window and chi match all 6 engine.logs
- chi 0.45: ic/chunk_NNNN vs task_NNNN/_InitConfig, 6 tasks x 400 configs: max wrapped diff 0.0e+00, bit-identical True; K2 = 5.427564 (log 5.42756), chi = 0.430976 (log 0.430976), M = 128: window and chi match all 6 engine.logs
