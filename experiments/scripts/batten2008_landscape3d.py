#!/usr/bin/env python3

"""
The 1D stealthy landscape for every chi of N = 6, stacked in 3D.

The N = 3 figure (batten2008_N3) shows Phi(x_2, x_3) at x_1 = 0, which holds
only chi = 1/3 and 2/3 (chi = M / N needs an integer number M of constrained
waves). N = 6 has chi = 1/6, 1/3, 1/2, 2/3 and 5/6 (M = 1-5), so this
figure uses N = 6 (density 1, L = 6) and slices its 5D landscape: x_1 = 0,
x_4, x_5, x_6 at their lattice sites 3L/6, 4L/6, 5L/6, and (x_2, x_3) over
the cell. Phi = sum_{n=1}^{M} |sum_j exp(2 pi i n s_j)|^2 / N on fractional
positions s, as for the flows (batten2008_flows.build_energy).

Each chi is a plane of filled Blues contours at height chi (shared levels
and colorbar, as the 2D figure), with grey contour lines; no trajectories.
Writes <out>/batten2008_N6_landscape3d.png. NumPy and matplotlib only.

Samuel J. Dawley
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import argparse
import sys
from fractions import Fraction
from pathlib import Path

# ----- numerics -----
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batten2008_plot import finish, plt  # noqa: E402  (house rcParams)

# ============================================================
# CONSTANTS
# ============================================================

N = 6
FIXED = (3.0 / 6.0, 4.0 / 6.0, 5.0 / 6.0)  # x_4, x_5, x_6 (fractional)
N_LEVELS = 24  # filled contour levels, as batten2008_N3

# ============================================================
# HELPERS
# ============================================================


def landscape(u: np.ndarray, m: int) -> np.ndarray:
    """Phi on the (x_2, x_3) grid u x u, M = m modes; rows index x_2, columns x_3."""
    assert 1 <= m < N and u.ndim == 1, "need 1 <= M < N and a 1D grid"
    x2, x3 = np.meshgrid(u, u, indexing="ij")
    n = np.arange(1, m + 1)[:, None, None]
    rho = 1.0 + np.exp(2j * np.pi * n * x2) + np.exp(2j * np.pi * n * x3)
    rho = rho + sum(np.exp(2j * np.pi * n * f) for f in FIXED)
    out: np.ndarray = (np.abs(rho) ** 2).sum(axis=0) / N
    return out


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--grid", type=int, default=241)
    parser.add_argument("--out", type=Path, default=Path("experiments/figures"))
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    u = np.linspace(0.0, 1.0, args.grid)
    chis = [Fraction(m, N) for m in range(1, N)]
    phis = [landscape(u, c.numerator * (N // c.denominator)) for c in chis]
    levels = np.linspace(0.0, max(p.max() for p in phis), N_LEVELS + 1)
    x3, x2 = np.meshgrid(u, u)  # x axis x_3, y axis x_2 (rows of phi are x_2)
    fig = plt.figure(figsize=(9.0, 9.5))
    ax = fig.add_subplot(projection="3d")
    for chi, phi in zip(chis, phis, strict=True):
        z = float(chi)
        cf = ax.contourf(x3, x2, phi, levels=levels, cmap="Blues", zdir="z",
                         offset=z)  # fmt: skip
        ax.contour(x3, x2, phi, levels=levels[::2], colors="0.35", linewidths=0.5,
                   zdir="z", offset=z)  # fmt: skip
        print(f"chi = {chi}: Phi in [{phi.min():.3g}, {phi.max():.3g}]", flush=True)
    ax.set_zlim(0.0, 1.0)
    ax.set_zticks([float(c) for c in chis], [f"{c}" for c in chis])
    ax.set_xlabel(r"Position, $x_3$", labelpad=8)
    ax.set_ylabel(r"Position, $x_2$", labelpad=8)
    ax.set_zlabel(r"Stealthiness, $\chi$", labelpad=8)
    ax.view_init(elev=18, azim=-58)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.0)
    cbar = fig.colorbar(cf, ax=ax, label=r"Energy, $\Phi$", shrink=0.6, pad=0.08)
    cbar.set_ticks(np.arange(0.0, levels[-1] + 1e-9, 1.0))
    ax.set_title(r"$N = 6,\ x_1 = 0,\ (x_4, x_5, x_6) = (3, 4, 5)\,L/6$",
                 loc="left", fontsize=13)  # fmt: skip
    finish(fig, args.out / "batten2008_N6_landscape3d", args.dpi)
    print("Done.", flush=True)
