# Analysis notes (not amendments)

These notes are about how the analysis is read. They do not change HYPOTHESES.md, its amendments, or any pre-registered rule or verdict.

## 1. Per-lane Phi scatter is a canonical fluctuation, so the remedy is more Andersen cycles (AUDIT-E1 C1)

Near a ground state at harmonic T, NVE is close to integrable. The time average of Phi over a production segment is therefore about half the harmonic energy that the 2M constrained modes hold when the segment starts, and only the thermostat changes that energy. Count-mode Andersen at p = 0.01 relaxes it on a scale of roughly N/p ≈ 1e5 steps. In the pilot (flows2d, chi = 0.1, job 17148922), a lane's <Phi> is correlated 0.44 between its two segments.

The per-lane, per-segment sd of <Phi>/(M T) is 0.0505, close to the canonical value 1/sqrt(2M) = 0.0488 (M = 210). This scatter is the equilibrium energy fluctuation of a finite harmonic system. A longer NVE segment does not average it out; it only samples the same mode energies for longer. The same mechanism sets the noise of H1 <S>/T and of H2 rate / Rice per |k| bin. Waves in one bin share the soft modes, so their values are positively correlated, and pooling more waves reduces the SE less than 1/sqrt(W) would.

**Consequence for the decision table.** The INCONCLUSIVE remedy "rerun one cell at 2x length" should be read as **2x the number of cycles**, i.e. twice as many independent Andersen redraws of the mode energies. It should not be read as 2x n_nve. A rerun could also use Andersen phases of at least ~N/p steps between segments. Twice the cycles at fixed n_nve costs about the same GPU-h as twice n_nve, and it reduces the per-lane energy scatter by ~1/sqrt(2). Twice n_nve does not reduce it.

## 2. Exploratory pooled low-|k| bins (post hoc, not pre-registered)

`batten2008_md_stats.py` adds a separate `exploratory` section to verdicts.{json,md}. It appears after the unchanged verdict table and never enters a verdict. It merges adjacent constrained |k|/K bins, from low |k| upward, until each merged bin holds at least W_min waves, and it reports the H1 <S>/T and H2 rate / Rice median, mean, SE and z for each merged bin.

W_min = ceil(sigma_w^2 / (n SE^2)) with n = 10 ground states and SE = 3 %. sigma_w is the pilot's per-wave sd across ground states, measured over the waves with |k|/K <= 0.35 at harmonic T:
- S/T: sigma_w = 0.358, so W_min = 15;
- rate / Rice: sigma_w = 0.166, so W_min = 4.

The pilot output is in `experiments/trajectories/md_pilot_exploratory/`. The X1 figure is written by `batten2008_md_plot.py --exploratory`.
