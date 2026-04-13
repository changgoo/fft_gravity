"""
performance_benchmark.py — Timing all four 1D Poisson solvers vs. N.

Produces performance_benchmark.png with two panels:
  Left:  Raw wall-clock time vs N (log-log)
  Right: Time / (N log2 N) vs N — flat line confirms O(N log N) scaling

Usage:
    python examples/performance_benchmark.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import timeit
import numpy as np
import matplotlib.pyplot as plt

from src.poisson1d import solve_periodic, solve_dirichlet, solve_neumann, solve_isolated
from src.utils import (
    make_grid_periodic,
    make_grid_dirichlet,
    make_grid_neumann,
    make_grid_isolated,
)

Ns = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
L = 1.0
REPEATS = 200

def time_solver(solver_fn, rho, L, repeats=REPEATS):
    """Return mean wall-clock time per call in seconds."""
    t = timeit.timeit(lambda: solver_fn(rho, L), number=repeats)
    return t / repeats

print(f"Timing solvers (N = {Ns[0]}..{Ns[-1]}, {REPEATS} repeats each) ...")
print(f"{'N':>6}  {'Periodic':>10}  {'Dirichlet':>10}  {'Neumann':>10}  {'Isolated':>10}  (µs/call)")
print("-" * 62)

times = {name: [] for name in ["Periodic", "Dirichlet", "Neumann", "Isolated"]}

for N in Ns:
    x_per,  h = make_grid_periodic(N, L)
    x_dir,  _ = make_grid_dirichlet(N, L)
    x_neu,  _ = make_grid_neumann(N, L)
    x_iso,  _ = make_grid_isolated(N, L)

    rho_per = np.sin(2 * np.pi * x_per / L)
    rho_dir = np.sin(np.pi * x_dir / L)
    rho_neu = np.cos(np.pi * x_neu / L)
    rho_iso = np.sin(2 * np.pi * x_iso / L)

    t_per = time_solver(solve_periodic,  rho_per, L)
    t_dir = time_solver(solve_dirichlet, rho_dir, L)
    t_neu = time_solver(solve_neumann,   rho_neu, L)
    t_iso = time_solver(solve_isolated,  rho_iso, L)

    times["Periodic"].append(t_per)
    times["Dirichlet"].append(t_dir)
    times["Neumann"].append(t_neu)
    times["Isolated"].append(t_iso)

    print(f"{N:>6}  {t_per*1e6:>10.2f}  {t_dir*1e6:>10.2f}  {t_neu*1e6:>10.2f}  {t_iso*1e6:>10.2f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
Ns_arr = np.array(Ns, dtype=float)
nlogn  = Ns_arr * np.log2(Ns_arr)

fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=(12, 5))

colors = ["C0", "C1", "C2", "C3"]
for (name, ts), color in zip(times.items(), colors):
    ts_arr = np.array(ts)
    ax_raw.loglog(Ns_arr, ts_arr * 1e6, "o-", color=color, label=name, lw=1.5)
    ax_norm.semilogx(Ns_arr, ts_arr / nlogn * 1e9, "o-", color=color, label=name, lw=1.5)

# Reference O(N log N) line on raw plot
ref = times["Periodic"][0] * nlogn / nlogn[0]
ax_raw.loglog(Ns_arr, ref * 1e6, "k--", lw=1, label=r"$\mathcal{O}(N \log N)$")

ax_raw.set_xlabel("N (grid points)")
ax_raw.set_ylabel("Wall time (µs/call)")
ax_raw.set_title("Raw timing")
ax_raw.legend(fontsize=9)
ax_raw.grid(True, which="both", alpha=0.3)

ax_norm.set_xlabel("N (grid points)")
ax_norm.set_ylabel(r"Time / $(N \log_2 N)$  (ns)")
ax_norm.set_title(r"Normalised by $N \log_2 N$  (flat = $\mathcal{O}(N \log N)$)")
ax_norm.legend(fontsize=9)
ax_norm.grid(True, which="both", alpha=0.3)

fig.suptitle("1D Poisson Solver Performance", fontsize=13)
fig.tight_layout()

out = os.path.join(os.path.dirname(__file__), "performance_benchmark.png")
fig.savefig(out, dpi=150)
print(f"\nSaved: {out}")
plt.show()
