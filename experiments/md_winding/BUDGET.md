# GPU budget ledger: low-temperature MD winding experiment

The hard cap is 100 GPU-hours on Della A100s. The orchestrator updates this ledger after every job, using:

`sacct -X -n -S 2026-11-08 --name=<job> --format=JobID,JobName,ElapsedRaw,AllocTRES`

with GPU-h = ElapsedRaw × n_gpu / 3600. A job is not submitted if its projected cost would exceed what remains on its line.

| Line | Allocation (GPU-h) | Used (GPU-h) |
|---|---|---|
| Verification and benchmarks (V3, GPU tests) | 8 | 0.00 |
| Pilot | 4 | 0.00 |
| Production (E1, E1b) | 60 | 0.00 |
| Langevin subset (E4) | 12 | 0.00 |
| Reserve | 16 | 0.00 |
| **Total** | **100** | **0.00** |

## Jobs

| Date | JobID | Name | Line | ElapsedRaw (s) | GPUs | GPU-h |
|---|---|---|---|---|---|---|
