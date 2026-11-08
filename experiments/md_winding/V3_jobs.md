# V3 GPU jobs (line "Verification and benchmarks")

These rows use the `BUDGET.md` format; the orchestrator merges them into the ledger. GPU-h = ElapsedRaw × GPUs / 3600, from

`sacct -X -n -j <id> --format=JobID,JobName,ElapsedRaw,AllocTRES`

The V3 cap is 4 GPU-h. All jobs ran `experiments/slurm/batten2008_md_gpu.slurm` with STEP=bench and logged to `/scratch/gpfs/TORQUATO/sd2402/xla_cco_runs/v3/slurm-mdw_bench-<id>.out` (copies in `experiments/md_winding/logs/`, the script's default log directory).

| Date | JobID | Name | Line | ElapsedRaw (s) | GPUs | GPU-h |
|---|---|---|---|---|---|---|
| 2026-11-08 | 17117101 | mdw_bench | Verification and benchmarks | 52 | 1 | 0.014 |
| 2026-11-08 | 17117175 | mdw_bench | Verification and benchmarks | 1823 | 1 | 0.506 |
| 2026-11-08 | 17117177 | mdw_bench | Verification and benchmarks | 993 | 1 | 0.276 |
| 2026-11-08 | 17117178 | mdw_bench | Verification and benchmarks | 1987 | 1 | 0.552 |
| 2026-11-08 | 17119585 | mdw_bench | Verification and benchmarks | 1328 | 1 | 0.369 |
| 2026-11-08 | 17119586 | mdw_bench | Verification and benchmarks | 1852 | 1 | 0.514 |
| 2026-11-08 | 17121008 | mdw_bench | Verification and benchmarks | 512 | 1 | 0.142 |
| 2026-11-08 | 17121010 | mdw_bench | Verification and benchmarks | 965 | 1 | 0.268 |
| | | | **V3 total** | **9512** | | **2.642** |

What each job did:

| JobID | Mode | Content | Outcome |
|---|---|---|---|
| 17117101 | throughput | Sanity check: flows2d χ = 0.4, 40 lanes, \|k\| < 2K, wind_every 1 and 5 (A100-PCIE-40GB) | OK. This run sets the 1.25 hardware margin. |
| 17117175 | grid | Throughput grid: 52 groups, 100 rows (A100-SXM4-80GB) | OK |
| 17117177 | calib χ = 0.1 | First criterion (c): mean of per-bin rates, 8 lanes, t = 200 | Superseded by 17119585 and 17121008 |
| 17117178 | calib χ = 0.4 | Same first criterion; ran on a MIG 3g.40gb slice | No dt passed (noise in criterion (c)). The run was not saved. |
| 17119585 | calib χ = 0.1 | Matched-pair criterion with 16 lanes, t = 400, raw winding saved | Chose dt = 0.01 under the mean-of-ratios estimator (D-070); superseded. Its ladder is reused by 17121008. |
| 17119586 | calib χ = 0.4 | Same | No dt passed under the mean-of-ratios estimator (Jensen bias, D-070). Ladder and raw winding were saved. |
| 17121008 | calib χ = 0.1 `--ladder-from` | Ratio-of-means criterion re-evaluated, then the 1e6-step confirmation and wind_every | dt = 0.04, wind_every = 5 |
| 17121010 | calib χ = 0.4 `--ladder-from` | Same | dt = 0.02, wind_every = 2 |
