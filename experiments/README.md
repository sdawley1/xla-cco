# Experiments

## Dawley & Torquato (2025) trajectories, χ = 0.1 and χ = 0.4

This experiment reproduces part of *Evolution of various initial many-particle configurations to disordered stealthy hyperuniform ground states*, PRE **112**, 045310 (2025). It runs collective-coordinate optimization from Poisson (POS) and Gaussian-pair-statistics (GAU, hyperuniform) initial conditions to χ = 0.1 stealthy ground states with `xla_cco`'s adaptive gradient descent. Each run is stored at full resolution.

Only scripts are tracked. All generated data lives in git-ignored directories:

| Directory | Contents |
|---|---|
| `initial_conditions/` | `POS-2D-N1024` (100 Poisson configs, seed 20251106); `GAU-2D-N1000` (50 configs copied from `CONFIGURATIONS/N1000_configs_GaussianPairStat`, basis rescaled to unit volume); `provenance.json` |
| `trajectories/chi<χ>/<ENS>/` | `traj_XXXX.{positions.f64,npz,json}` for each trajectory, plus `manifest.json`; `trajectories/chi<χ>/summary.npz` (ensemble statistics); SLURM logs |
| `figures/` | `chi<χ>_pos_gau.{png,pdf}` (S(k) of initial conditions; distance and energy vs t/T, log-log) and `chi<χ>_ground_state_sk.{png,pdf}` (S(k) of the converged ground states) |

### Pipeline (run from the repository root)

```bash
export UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/gpfs/TORQUATO/${USER}/.uv-cache}"  # any directory you can write
uv sync --dev --extra plot --locked            # once: adds matplotlib for the plot scripts
uv run --no-sync python -I experiments/scripts/make_initial_conditions.py \
    --gau-source <prefix of the GAUSSIAN-2D-N_1000 ConfigPack>   # also creates experiments/trajectories/
sbatch experiments/slurm/run_trajectories_gpu.slurm      # chi = 0.1 (default); CPU: run_trajectories.slurm
sbatch --export=ALL,CHI=0.4 experiments/slurm/run_trajectories_gpu.slurm
# for chi = 0.4 pass --traj experiments/trajectories/chi0.4 to the analysis and
# --summary .../chi0.4/summary.npz --out experiments/figures/chi0.4_<name> to the plot scripts
JAX_PLATFORMS=cpu uv run --no-sync python -I experiments/scripts/analyze_trajectories.py
uv run --no-sync python -I experiments/scripts/plot_figure.py
uv run --no-sync python -I experiments/scripts/plot_ground_state_sk.py
```

### Conventions

- **Descent rule** (paper, Sec. IV). The descent uses the normalized gradient. The step doubles after an accepted downhill move and halves otherwise; acceptance is as in the C++ `LocalGradientDescent`, with rel_tol = 0.01. The initial step is γρ^{1/2} = 1e-3 in Cartesian units, which is a fractional step of 1e-3/L.
- **χ = 0.4** uses K = 4.483993. That gives M = 820 and χ = 0.40078 for POS, and M = 802 and χ = 0.40140 for GAU. The GPU job was 17000661.
- **χ = 0.1** uses K = 2.241996. That gives M = 210 and χ = 0.10264 for POS (N = 1024), and M = 200 and χ = 0.10010 for GAU (N = 1000).
- **Ground state.** Each run continues until Φ < 1e-22 (the paper's tolerance) or until the step floor. The step floor is where float64 can no longer resolve energy changes, at Φ ≈ 1e-22. Any terminal Φ < 1e-20 counts as a ground state.
  - All 150 trajectories reach a ground state.
  - 49/100 POS and 29/50 GAU runs reach Φ < 1e-22.
  - Each run's `status` and `final_energy` are in its JSON file.
  - At χ = 0.4 the float64 floor is higher: all 150 runs stop at the step floor with Φ between 4.9e-20 and 1.4e-18 (median about 2.5e-19). The floor rises with M, which is about 4x larger here, so the manifests' `n_ground_states` (Φ < 1e-20) reads 0.
    - These endpoints are still ground states to within about 1e-9. Polishing POS and GAU `traj_0000` with the library's L-BFGS takes Φ from about 1e-19 to 6e-26 and moves the whole configuration by only 3-4e-9 in total, against p_0(∞) values of 26 and 11.
    - In ground-state S(k), S ≤ 1.5e-19 for k < K − 0.1. The 0.1-wide bin straddling K (mean |k| = 4.45) mixes vectors inside and outside the window, so do not read it as part of the window.
- **Time.** An iteration contributes t_m = γρ^{1/2} (its Cartesian step) if accepted and 0 otherwise. T is the total, and the figures use t/T.
- **Proximity metric** (Eq. 10). p_0(t) = ρ^{1/2} [Σ_j |r_j(t) − r_j(0)|²]^{1/2}, computed on unwrapped paths. The plotted distance to the manifold is p_0(∞) − p_0(t).

### Loading a trajectory for further analysis

```python
from pathlib import Path
from xla_cco.io.trajstore import open_trajectory
from xla_cco.analysis.trajectory import unwrap_fractional, proximity_metric
import numpy as np

frames, arrays, meta = open_trajectory(
    Path("experiments/trajectories/chi0.1/POS/traj_0000")
)
# frames: memmap (F, N, 2), wrapped fractional positions; frame 0 is the initial condition,
#         followed by the state after every ACCEPTED iteration
# arrays: per-iteration 'iteration', 'energy', 'step_used', 'step_used_cartesian',
#         'accepted', 'frame_index' (frame holding the state after that iteration), 'time', 'energy0'
# meta:   N, d, L, basis, chi_actual, M, K, settings, status, final_energy, git_commit, ...
paths = unwrap_fractional(np.asarray(frames))
p0 = proximity_metric(paths, paths[0], np.asarray(meta["basis"]), meta["density"])
positions_at_iteration_m = frames[arrays["frame_index"][m - 1]]  # m >= 1
```

Rejected iterations leave positions unchanged, so storing frames only at accepted iterations is lossless. Every iteration is still recoverable through `frame_index`.

## Wasserstein-2 distance along the trajectories

This section compares the proximity metric with a label-free distance: the periodic Wasserstein-2 (optimal-transport) distance between N-particle configurations. It uses the uniform empirical measures on the torus and the minimum-image cost.

**Definition.** W(a, b) = ρ^{1/2} [min over permutations σ of Σ_i |a_i − b_σ(i)|²]^{1/2}. It is scaled like p_0, so W ≤ p_0, because the identity labelling is one of the admissible matchings. Two quantities are tracked per trajectory:

- W_0(t) = W(r(0), r(t)), the analogue of p_0(t). It is plotted as W_0(∞) − W_0(t).
- W_∞(t) = W(r(t), r(∞)), the distance to the ground state actually reached. Its labelled counterpart is q_∞(t) = ρ^{1/2} |r(t) − r(∞)|.

The library code is in `xla_cco.analysis.transport`:

- the cost matrix;
- log-domain Sinkhorn with eps annealing, warm starts and an optional stop on a bijective rounding;
- rounding with greedy repair.

### Two computations

- **Exact** (`wasserstein_exact.py`, CPU): Hungarian assignment (`scipy.optimize.linear_sum_assignment`) on 201 frames per trajectory, at t = 0 and 200 times log-spaced in t/T ∈ [1e-6, 1]. The labelled p_0 and q_∞ are stored on the same frames.
  - Output: `wasserstein/exact_<ENS>.npz`, with arrays of shape (trajectory, frame).
- **Sinkhorn** (`wasserstein_sinkhorn.py`, GPU): runs at every stored frame of every trajectory.
  - Parameters: eps = 1e-2 (Cartesian² units, ρ = 1).
  - Lanes: each (trajectory, reference) pair is an independent lane that warm-starts from its previous frame.
  - Stopping: a frame stops once the plan's row-argmax rounding is a bijection, once its marginal error is below tol = 1e-4, or after 100 iterations.
  - Repair: a rounding that is still not a bijection is repaired greedily and flagged (`is_perm = 0`). Such frames are not certified; the exact fallback below replaces them.
  - Reported value: `w_round`, the cost of that matching. It is a feasible matching, so `w_round` ≥ exact W for finite inputs.
  - Diagnostic only: `w_ent`, the entropic plan's own cost. It carries a bias of order eps, which swamps W near t → T.
  - Output: `wasserstein/sinkhorn_<ENS>[_<start>].npz`, with per-frame arrays concatenated over trajectories and delimited by `offsets`. The last axis is (r(0), r(∞)). Slices from `--start/--limit` are merged by the analysis.

### Accuracy of Sinkhorn against exact, and the exact fallback

- **Certified frames.** A frame is trusted only if the plan's row-argmax rounding is a bijection. All 76k compared bijective frames equalled the exact optimum (worst relative error 3e-16).
- **Uncertified frames.** Where the rounding collides, the greedy repair can be suboptimal, even when the plan has converged (D-014). This happens at genuine near-ties, where two close particles' partner swap costs less than ~eps, and at plans still unconverged at the 100-iteration cap, at t/T ≈ 0.7–0.9. The repair's error reached 2.8e-3 relative on converged frames and 100% on unconverged ones.
- **Exact fallback.** `wasserstein_fallback.py` solves the exact assignment at every uncertified (frame, reference) entry, about 3–5% of POS r(0) entries and essentially no GAU ones. It writes `wasserstein/fallback_<ENS>.npz`. `analyze_wasserstein.py` substitutes these values and stops with an error if any are missing, so every curve value is exact.
- **Diagnostics.** `summary.npz` stores the worst errors per class (`<ENS>/disc_rel_certified`, `disc_rel_repaired`, `disc_rel_final`, `disc_rel_ent`) and the fallback fraction (`disc_frac_fallback`).
- **Why not a higher cap.** A 2000-iteration cap removes most collisions but stalls for hours on persistent near-ties, because GPU lanes wait on a few hard frames.

### Results

- **χ = 0.1:** the optimal matching is the identity at every frame, for both ensembles, so W ≡ p_0 and W_∞ ≡ q_∞ exactly.
- **χ = 0.4, POS:** on average 30 of 1024 particles end up matched to a different initial particle. ⟨W_0(∞)⟩ = 25.540 against ⟨p_0(∞)⟩ = 25.605, a gap of ≤ 0.6%.
- **χ = 0.4, GAU:** about 1 particle is re-matched. ⟨W_0(∞)⟩ = 11.646 against ⟨p_0(∞)⟩ = 11.648.
- **Re-matching over time:** relative to r(∞), re-matching vanishes over the last decade of t/T. The labelled q_∞ therefore equals W_∞ there.

### Running

```bash
sbatch --export=ALL,CHI=0.1 experiments/slurm/wasserstein_gpu.slurm      # exact (CPU cores) + Sinkhorn (GPU)
# chi = 0.4 needs slices to fit gputest's 59 min:
sbatch --export=ALL,CHI=0.4,ENSEMBLES=POS,START=0,LIMIT=20,EXACT=1 experiments/slurm/wasserstein_gpu.slurm
sbatch --export=ALL,CHI=0.4 experiments/slurm/wasserstein_fallback.slurm  # after all Sinkhorn slices; CPU only
JAX_PLATFORMS=cpu uv run --no-sync python -I experiments/scripts/analyze_wasserstein.py --traj experiments/trajectories/chi0.4
uv run --no-sync python -I experiments/scripts/plot_wasserstein.py \
    --summary experiments/trajectories/chi0.4/wasserstein/summary.npz --out experiments/figures/chi0.4
```

### Figures

The figures are `figures/chi<χ>_w2.{png,pdf}` (the Wasserstein mirror of `chi<χ>_pos_gau`) and `figures/chi<χ>_w2_vs_p0.{png,pdf}` (exact W against labelled p on the same frames). The docstring of `plot_wasserstein.py` lists the panels.
