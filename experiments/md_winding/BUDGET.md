# GPU budget ledger: low-temperature MD winding experiment

The hard cap is 100 GPU-hours on Della A100s. The orchestrator updates this ledger after every job, using:

`sacct -X -n -S 2026-11-08 --name=<job> --format=JobID,JobName,ElapsedRaw,AllocTRES`

with GPU-h = ElapsedRaw × n_gpu / 3600. A job is not submitted if its projected cost would exceed what remains on its line.

| Line | Allocation (GPU-h) | Used (GPU-h) |
|---|---|---|
| Verification and benchmarks (V3, GPU tests) | 8 | 3.46 |
| Pilot | 4 | 0.56 |
| Production (E1, E1b) | 60 | 12.93 |
| Langevin subset (E4) | 12 | 2.81 |
| Reserve | 16 | 0.00 |
| **Total** | **100** | **19.77** |

## Jobs

| Date | JobID | Name | Line | ElapsedRaw (s) | GPUs | GPU-h |
|---|---|---|---|---|---|---|
| 2026-11-08 | 17117101 | mdw_bench (V3) | Verification | 52 | 1 | 0.01 |
| 2026-11-08 | 17117175 | mdw_bench (V3) | Verification | 1823 | 1 | 0.51 |
| 2026-11-08 | 17117177 | mdw_bench (V3) | Verification | 993 | 1 | 0.28 |
| 2026-11-08 | 17117178 | mdw_bench (V3) | Verification | 1987 | 1 | 0.55 |
| 2026-11-08 | 17119585 | mdw_bench (V3) | Verification | 1328 | 1 | 0.37 |
| 2026-11-08 | 17119586 | mdw_bench (V3) | Verification | 1852 | 1 | 0.51 |
| 2026-11-08 | 17121008 | mdw_bench (V3) | Verification | 512 | 1 | 0.14 |
| 2026-11-08 | 17121010 | mdw_bench (V3) | Verification | 965 | 1 | 0.27 |
| 2026-11-08 | 17137898 | xla-cco-test-gpu (V4, 83 failed) | Verification | 1454 | 1 | 0.40 |
| 2026-11-09 | 17148922 | mdw_pilot (E1 pilot: flows2d chi=1-10, 40 lanes, full protocol, equilibration extended) | Pilot | 2033 | 1 | 0.56 |
| 2026-11-09 | 17143149 | xla-cco-test-gpu (V4 fixes, 780 passed) | Verification | 1495 | 1 | 0.42 |
| 2026-11-09 | 17163785 | mdw_e1_flows2d_2-5 | Production | 7884 | 1 | 2.19 |
| 2026-11-09 | 17163786 | mdw_e1_gau2d_1-10 | Production | 1995 | 1 | 0.55 |
| 2026-11-09 | 17163787 | mdw_e1_gau2d_2-5 | Production | 6872 | 1 | 1.91 |
| 2026-11-09 | 17163788 | mdw_e1_url2d_1-10 | Production | 2029 | 1 | 0.56 |
| 2026-11-09 | 17163789 | mdw_e1_url2d_2-5 | Production | 7732 | 1 | 2.15 |
| 2026-11-09 | 17163790 | mdw_e1_rsa2d_1-10 | Production | 2032 | 1 | 0.56 |
| 2026-11-09 | 17163791 | mdw_e1_rsa2d_2-5 | Production | 7437 | 1 | 2.07 |
| 2026-11-09 | 17163792 | mdw_e1_hip2d_1-10 | Production | 1998 | 1 | 0.56 |
| 2026-11-09 | 17163793 | mdw_e1_hip2d_2-5 | Production | 7589 | 1 | 2.11 |
| 2026-11-09 | 17163794 | mdw_e1b_flows2d_1-10 | Production | 69 | 1 | 0.02 |
| 2026-11-09 | 17163795 | mdw_e1b_flows2d_2-5 | Production | 110 | 1 | 0.03 |
| 2026-11-09 | 17163796 | mdw_e1b_gau2d_1-10 | Production | 105 | 1 | 0.03 |
| 2026-11-09 | 17163797 | mdw_e1b_gau2d_2-5 | Production | 151 | 1 | 0.04 |
| 2026-11-09 | 17163798 | mdw_e1b_url2d_1-10 | Production | 69 | 1 | 0.02 |
| 2026-11-09 | 17163799 | mdw_e1b_url2d_2-5 | Production | 108 | 1 | 0.03 |
| 2026-11-09 | 17163800 | mdw_e1b_rsa2d_1-10 | Production | 70 | 1 | 0.02 |
| 2026-11-09 | 17163801 | mdw_e1b_rsa2d_2-5 | Production | 106 | 1 | 0.03 |
| 2026-11-09 | 17163802 | mdw_e1b_hip2d_1-10 | Production | 100 | 1 | 0.03 |
| 2026-11-09 | 17163803 | mdw_e1b_hip2d_2-5 | Production | 102 | 1 | 0.03 |
| 2026-11-09 | 17163804 | mdw_e4_flows2d_1-10_g0.003 | Langevin | 694 | 1 | 0.19 |
| 2026-11-09 | 17163805 | mdw_e4_flows2d_1-10_g0.03 | Langevin | 705 | 1 | 0.20 |
| 2026-11-09 | 17163806 | mdw_e4_flows2d_2-5_g0.003 | Langevin | 1769 | 1 | 0.49 |
| 2026-11-09 | 17163807 | mdw_e4_flows2d_2-5_g0.03 | Langevin | 1759 | 1 | 0.49 |
| 2026-11-09 | 17163808 | mdw_e4_hip2d_1-10_g0.003 | Langevin | 718 | 1 | 0.20 |
| 2026-11-09 | 17163809 | mdw_e4_hip2d_1-10_g0.03 | Langevin | 708 | 1 | 0.20 |
| 2026-11-09 | 17163810 | mdw_e4_hip2d_2-5_g0.003 | Langevin | 1870 | 1 | 0.52 |
| 2026-11-09 | 17163811 | mdw_e4_hip2d_2-5_g0.03 | Langevin | 1881 | 1 | 0.52 |
