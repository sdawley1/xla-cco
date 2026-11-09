# GPU budget ledger: low-temperature MD winding experiment

The hard cap is 100 GPU-hours on Della A100s. The orchestrator updates this ledger after every job, using:

`sacct -X -n -S 2026-11-08 --name=<job> --format=JobID,JobName,ElapsedRaw,AllocTRES`

with GPU-h = ElapsedRaw × n_gpu / 3600. A job is not submitted if its projected cost would exceed what remains on its line.

| Line | Allocation (GPU-h) | Used (GPU-h) |
|---|---|---|
| Verification and benchmarks (V3, GPU tests) | 8 | 3.05 |
| Pilot | 4 | 0.56 |
| Production (E1, E1b) | 60 | 0.00 |
| Langevin subset (E4) | 12 | 0.00 |
| Reserve | 16 | 0.00 |
| **Total** | **100** | **3.61** |

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
