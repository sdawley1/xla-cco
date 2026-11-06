# Experiments

## Dawley & Torquato (2025) trajectories, χ = 0.1

This experiment reproduces part of *Evolution of various initial many-particle configurations to disordered stealthy hyperuniform ground states*, PRE **112**, 045310 (2025). It runs collective-coordinate optimization from Poisson (POS) and Gaussian-pair-statistics (GAU, hyperuniform) initial conditions to χ = 0.1 stealthy ground states with `xla_cco`'s adaptive gradient descent. Each run is stored at full resolution.

Only scripts are tracked. All generated data lives in git-ignored directories:

| Directory | Contents |
|---|---|
| `initial_conditions/` | `POS-2D-N1024` (100 Poisson configs, seed 20251106); `GAU-2D-N1000` (50 configs copied from `CONFIGURATIONS/N1000_configs_GaussianPairStat`, basis rescaled to unit volume); `provenance.json` |
| `trajectories/chi0.1/<ENS>/` | `traj_XXXX.{positions.f64,npz,json}` for each trajectory, plus `manifest.json`; `trajectories/chi0.1/summary.npz` (ensemble statistics); SLURM logs |
| `figures/` | `chi0.1_pos_gau.{png,pdf}` |

### Pipeline (run from the repository root)

```bash
export UV_CACHE_DIR=/scratch/gpfs/TORQUATO/sd2402/.uv-cache
uv run --no-sync python -I experiments/scripts/make_initial_conditions.py
sbatch experiments/slurm/run_trajectories_gpu.slurm      # or run_trajectories.slurm (CPU)
JAX_PLATFORMS=cpu uv run --no-sync python -I experiments/scripts/analyze_trajectories.py
/home/sd2402/torquato-scratch-sd2402/conda_envs/plottime/bin/python -I experiments/scripts/plot_figure.py
```

### Conventions

- **Descent rule** (paper, Sec. IV). The descent uses the normalized gradient. The step doubles after an accepted downhill move and halves otherwise; acceptance is as in the C++ `LocalGradientDescent`, with rel_tol = 0.01. The initial step is γρ^{1/2} = 1e-3 in Cartesian units, which is a fractional step of 1e-3/L.
- **χ = 0.1** uses K = 2.241996. That gives M = 210 and χ = 0.10264 for POS (N = 1024), and M = 200 and χ = 0.10010 for GAU (N = 1000).
- **Ground state.** Each run continues until Φ < 1e-22 (the paper's tolerance) or until the step floor. The step floor is where float64 can no longer resolve energy changes, at Φ ≈ 1e-22. Any terminal Φ < 1e-20 counts as a ground state.
  - All 150 trajectories reach a ground state.
  - 49/100 POS and 29/50 GAU runs reach Φ < 1e-22.
  - Each run's `status` and `final_energy` are in its JSON file.
- **Time.** An iteration contributes t_m = γρ^{1/2} (its Cartesian step) if accepted and 0 otherwise. T is the total, and the figures use t/T.
- **Proximity metric** (Eq. 10). p_0(t) = ρ^{1/2} [Σ_j |r_j(t) − r_j(0)|²]^{1/2}, computed on unwrapped paths. The plotted distance to the manifold is p_0(∞) − p_0(t).

### Loading a trajectory for further analysis

```python
from pathlib import Path
from xla_cco.io.trajstore import open_trajectory
from xla_cco.analysis.trajectory import unwrap_fractional, proximity_metric
import numpy as np

frames, arrays, meta = open_trajectory(Path("experiments/trajectories/chi0.1/POS/traj_0000"))
# frames: memmap (F, N, 2), wrapped fractional positions; frame 0 is the initial condition,
#         followed by the state after every ACCEPTED iteration
# arrays: per-iteration 'iteration', 'energy', 'step_used', 'step_used_cartesian',
#         'accepted', 'frame_index' (frame holding the state after that iteration), 'time', 'energy0'
# meta:   N, d, L, basis, chi_actual, M, K, settings, status, final_energy, git_commit, ...
paths = unwrap_fractional(np.asarray(frames))
p0 = proximity_metric(paths, paths[0], np.asarray(meta["basis"]), meta["density"])
positions_at_iteration_m = frames[arrays["frame_index"][m - 1]]   # m >= 1
```

Rejected iterations leave positions unchanged, so storing frames only at accepted iterations is lossless. Every iteration is still recoverable through `frame_index`.
