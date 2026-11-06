# Parity results

## Level 3: ensemble parity vs C++ NLopt L-BFGS (T12)

Test: `tests/parity/test_ensemble_parity.py` (marks `della`, `slow`); run with
`sbatch slurm/parity_ensemble.slurm`. Statistics: `tests/ensemble_stats.py`.
Choices: the first g2-peak height is the maximum bin of the per-config g2 on the
fixed grid r in [0, L/2], width 0.1; RMS displacement is minimum-image
Cartesian; S(k) bins are the |k|^2 shells in (K2, 3 K2] with the per-config
shell mean, z = (m1 - m2)/sqrt(SEM1^2 + SEM2^2). Bonferroni family: 50 p-tests
(5 chi x 5 tests x {real, false-alarm control}); the -log10 Phi comparison is
report-only (R22).

Run: SLURM job 16982731 (gputest, della-l09g1, 1 GPU, exit 1 from the single failing test, 3m30s wall, 197 s test session), base git hash 41d9baf, JAX 0.12.0.1, 400 configs per task (tasks 0000-0002 = 1200 per chi), success = Phi < 9.99999968e-21, alpha = 0.01, p-test threshold 2e-4, |z|max = 4.

**Open finding (escalated, nothing tuned):** at chi = 0.45 the success fraction
differs (JAX 1200/1200 vs C++ 1071/1200, z = 11.7); all geometric statistics
and S(k) pass at every chi and both controls behave. C++ final energies stop
near the threshold (failures lie in 1.0e-20 to 2.0e-20, successes have
-log10 Phi in 20.0 to 20.8) while JAX with energy_tol 0 continues to
-log10 Phi ~ 30.

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
| success fraction | 1.0000 | 1.0000 | 0.000 | 1 | 1 | PASS |
| KS min NN distance | | | 0.0150 | 0.999 | 1 | PASS |
| KS mean NN distance | | | 0.0250 | 0.848 | 1 | PASS |
| KS first g2-peak height | | | 0.0275 | 0.755 | 1 | PASS |
| KS RMS displacement | | | 0.0425 | 0.229 | 1 | PASS |
| S(k) worst of 25 bins | | | 0.680 (at |k|=4.061) | | | PASS (|z| < 4.0) |
| -log10 Phi KS (report only, R22) | | | | 0 | | n/a |

False-alarm control (C++ 0000-0002 vs C++ 0010-0012): **PASS**

- status counts (JAX): {'STATUS_LINESEARCH_FAIL': 1200}; successes JAX 1200/1200, C++ 1200/1200
- -log10 Phi among successes (median [Q1, Q3]): JAX 30.11 [29.99, 30.24], C++ 20.97 [20.74, 21.29]
- timings: JAX relax 14.2 s (cached: False), analysis 8.7 s

#### chi = 0.15 (M = 40, chi_actual = 0.134680, K2 = 3.76326)

**JAX vs C++: PASS**

| statistic | A | B | z or KS D | p raw | p corrected | verdict |
|---|---|---|---|---|---|---|
| success fraction | 1.0000 | 1.0000 | 0.000 | 1 | 1 | PASS |
| KS min NN distance | | | 0.0242 | 0.875 | 1 | PASS |
| KS mean NN distance | | | 0.0267 | 0.787 | 1 | PASS |
| KS first g2-peak height | | | 0.0150 | 0.999 | 1 | PASS |
| KS RMS displacement | | | 0.0492 | 0.11 | 1 | PASS |
| S(k) worst of 53 bins | | | 0.648 (at |k|=5.581) | | | PASS (|z| < 4.0) |
| -log10 Phi KS (report only, R22) | | | | 0 | | n/a |

False-alarm control (C++ 0000-0002 vs C++ 0010-0012): **PASS**

- status counts (JAX): {'STATUS_LINESEARCH_FAIL': 1200}; successes JAX 1200/1200, C++ 1200/1200
- -log10 Phi among successes (median [Q1, Q3]): JAX 28.97 [28.89, 29.04], C++ 21.01 [20.80, 21.24]
- timings: JAX relax 15.9 s (cached: False), analysis 5.7 s

#### chi = 0.25 (M = 73, chi_actual = 0.245791, K2 = 4.46184)

**JAX vs C++: PASS**

| statistic | A | B | z or KS D | p raw | p corrected | verdict |
|---|---|---|---|---|---|---|
| success fraction | 1.0000 | 1.0000 | 0.000 | 1 | 1 | PASS |
| KS min NN distance | | | 0.0375 | 0.368 | 1 | PASS |
| KS mean NN distance | | | 0.0217 | 0.941 | 1 | PASS |
| KS first g2-peak height | | | 0.0325 | 0.551 | 1 | PASS |
| KS RMS displacement | | | 0.0375 | 0.368 | 1 | PASS |
| S(k) worst of 73 bins | | | 1.170 (at |k|=10.997) | | | PASS (|z| < 4.0) |
| -log10 Phi KS (report only, R22) | | | | 0 | | n/a |

False-alarm control (C++ 0000-0002 vs C++ 0010-0012): **PASS**

- status counts (JAX): {'STATUS_LINESEARCH_FAIL': 1200}; successes JAX 1200/1200, C++ 1200/1200
- -log10 Phi among successes (median [Q1, Q3]): JAX 28.33 [28.27, 28.39], C++ 20.97 [20.80, 21.16]
- timings: JAX relax 31.9 s (cached: False), analysis 6.5 s

#### chi = 0.35 (M = 101, chi_actual = 0.340067, K2 = 4.99141)

**JAX vs C++: PASS**

| statistic | A | B | z or KS D | p raw | p corrected | verdict |
|---|---|---|---|---|---|---|
| success fraction | 1.0000 | 1.0000 | 0.000 | 1 | 1 | PASS |
| KS min NN distance | | | 0.0292 | 0.687 | 1 | PASS |
| KS mean NN distance | | | 0.0175 | 0.993 | 1 | PASS |
| KS first g2-peak height | | | 0.0183 | 0.988 | 1 | PASS |
| KS RMS displacement | | | 0.0525 | 0.0732 | 1 | PASS |
| S(k) worst of 91 bins | | | 1.767 (at |k|=13.469) | | | PASS (|z| < 4.0) |
| -log10 Phi KS (report only, R22) | | | | 0 | | n/a |

False-alarm control (C++ 0000-0002 vs C++ 0010-0012): **PASS**

- status counts (JAX): {'STATUS_LINESEARCH_FAIL': 1200}; successes JAX 1200/1200, C++ 1200/1200
- -log10 Phi among successes (median [Q1, Q3]): JAX 27.86 [27.80, 27.93], C++ 20.72 [20.58, 20.85]
- timings: JAX relax 33.8 s (cached: False), analysis 5.7 s

#### chi = 0.45 (M = 128, chi_actual = 0.430976, K2 = 5.42756)

**JAX vs C++: FAIL**

| statistic | A | B | z or KS D | p raw | p corrected | verdict |
|---|---|---|---|---|---|---|
| success fraction | 1.0000 | 0.8925 | 11.676 | 1.69e-31 | 8.46e-30 | FAIL |
| KS min NN distance | | | 0.0317 | 0.584 | 1 | PASS |
| KS mean NN distance | | | 0.0283 | 0.721 | 1 | PASS |
| KS first g2-peak height | | | 0.0267 | 0.787 | 1 | PASS |
| KS RMS displacement | | | 0.0517 | 0.0812 | 1 | PASS |
| S(k) worst of 107 bins | | | 2.792 (at |k|=15.611) | | | PASS (|z| < 4.0) |
| -log10 Phi KS (report only, R22) | | | | 0 | | n/a |

False-alarm control (C++ 0000-0002 vs C++ 0010-0012): **PASS**

- status counts (JAX): {'STATUS_STAGNATED': 21, 'STATUS_LINESEARCH_FAIL': 1179}; successes JAX 1200/1200, C++ 1071/1200
- -log10 Phi among successes (median [Q1, Q3]): JAX 27.05 [26.94, 27.15], C++ 20.23 [20.14, 20.35]
- timings: JAX relax 46.1 s (cached: False), analysis 5.7 s

### Power control (JAX with K2 x 0.9 vs C++)

- chi = 0.25 (400 JAX configs): **FAIL** (expected FAIL); success 0.000 vs 1.000 (p 0); failing KS: ['nn_min', 'nn_mean', 'g2_peak', 'rms_disp']; S(k) worst |z| = 8.6
- chi = 0.45 (400 JAX configs): **FAIL** (expected FAIL); success 0.000 vs 0.892 (p 8.22e-237); failing KS: ['nn_min', 'nn_mean', 'g2_peak', 'rms_disp']; S(k) worst |z| = 21.6


### Mapping and window verification

- chi 0.05: ic/chunk_NNNN vs task_NNNN/_InitConfig, 6 tasks x 400 configs: max wrapped diff 0.0e+00, bit-identical True; K2 = 2.609300 (log 2.6093), chi = 0.043771 (log 0.043771), M = 13: window and chi match all 6 engine.logs
- chi 0.15: ic/chunk_NNNN vs task_NNNN/_InitConfig, 6 tasks x 400 configs: max wrapped diff 0.0e+00, bit-identical True; K2 = 3.763262 (log 3.76326), chi = 0.134680 (log 0.13468), M = 40: window and chi match all 6 engine.logs
- chi 0.25: ic/chunk_NNNN vs task_NNNN/_InitConfig, 6 tasks x 400 configs: max wrapped diff 0.0e+00, bit-identical True; K2 = 4.461841 (log 4.46184), chi = 0.245791 (log 0.245791), M = 73: window and chi match all 6 engine.logs
- chi 0.35: ic/chunk_NNNN vs task_NNNN/_InitConfig, 6 tasks x 400 configs: max wrapped diff 0.0e+00, bit-identical True; K2 = 4.991412 (log 4.99141), chi = 0.340067 (log 0.340067), M = 101: window and chi match all 6 engine.logs
- chi 0.45: ic/chunk_NNNN vs task_NNNN/_InitConfig, 6 tasks x 400 configs: max wrapped diff 0.0e+00, bit-identical True; K2 = 5.427564 (log 5.42756), chi = 0.430976 (log 0.430976), M = 128: window and chi match all 6 engine.logs
