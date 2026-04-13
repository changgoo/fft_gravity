"""
demo_1d.py — Four-panel demonstration of all 1D Poisson solvers.

Produces demo_1d.png with one subplot per boundary condition type.
Each panel shows rho (dashed) and the computed phi vs the analytic
exact solution (solid/dotted).

Usage:
    python examples/demo_1d.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.poisson1d import solve_periodic, solve_dirichlet, solve_neumann, solve_isolated
from src.utils import (
    make_grid_periodic,
    make_grid_dirichlet,
    make_grid_neumann,
    make_grid_isolated,
    linf_error,
)

N = 128
L = 1.0

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes = axes.flatten()

# ---------------------------------------------------------------------------
# Panel 0: Periodic
# ---------------------------------------------------------------------------
ax = axes[0]
x, h = make_grid_periodic(N, L)
rho = np.sin(2 * np.pi * x / L)
phi_exact = -(L / (2 * np.pi)) ** 2 * rho
phi = solve_periodic(rho, L)

ax.plot(x, rho, "--", color="C1", lw=1.2, label=r"$\rho = \sin(2\pi x/L)$")
ax.plot(x, phi_exact, "k-", lw=2.0, label=r"$\phi_{\rm exact}$")
ax.plot(x, phi, "C0--", lw=1.5, label=r"$\phi_{\rm FFT}$")
ax.set_title(f"Periodic  [max err = {linf_error(phi, phi_exact):.1e}]")
ax.set_xlabel("x")
ax.legend(fontsize=8)

# ---------------------------------------------------------------------------
# Panel 1: Dirichlet
# ---------------------------------------------------------------------------
ax = axes[1]
x, h = make_grid_dirichlet(N, L)
rho = np.sin(np.pi * x / L)
phi_exact = -(L / np.pi) ** 2 * rho
phi = solve_dirichlet(rho, L)

# Add boundary zeros for visual continuity
x_full = np.concatenate([[0.0], x, [L]])
phi_full = np.concatenate([[0.0], phi, [0.0]])
phi_ex_full = np.concatenate([[0.0], phi_exact, [0.0]])

ax.plot(x, rho, "--", color="C1", lw=1.2, label=r"$\rho = \sin(\pi x/L)$")
ax.plot(x_full, phi_ex_full, "k-", lw=2.0, label=r"$\phi_{\rm exact}$")
ax.plot(x_full, phi_full, "C0--", lw=1.5, label=r"$\phi_{\rm DST}$")
ax.set_title(f"Dirichlet  [max err = {linf_error(phi, phi_exact):.1e}]")
ax.set_xlabel("x")
ax.legend(fontsize=8)

# ---------------------------------------------------------------------------
# Panel 2: Neumann
# ---------------------------------------------------------------------------
ax = axes[2]
x, h = make_grid_neumann(N, L)
rho = np.cos(np.pi * x / L)
phi_exact = -(L / np.pi) ** 2 * rho
phi = solve_neumann(rho, L)

ax.plot(x, rho, "--", color="C1", lw=1.2, label=r"$\rho = \cos(\pi x/L)$")
ax.plot(x, phi_exact, "k-", lw=2.0, label=r"$\phi_{\rm exact}$")
ax.plot(x, phi, "C0--", lw=1.5, label=r"$\phi_{\rm DCT}$")
ax.set_title(f"Neumann  [max err = {linf_error(phi, phi_exact):.1e}]")
ax.set_xlabel("x")
ax.legend(fontsize=8)

# ---------------------------------------------------------------------------
# Panel 3: Isolated (Gaussian source, compare to direct quadrature reference)
# ---------------------------------------------------------------------------
ax = axes[3]
L_iso = 4.0
x, h = make_grid_isolated(N, L_iso)
sigma = 0.4
rho = np.exp(-0.5 * ((x - L_iso / 2) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

phi = solve_isolated(rho, L_iso, baseline="zero_first")

# Direct quadrature reference
phi_ref = np.array([np.sum(np.abs(x[i] - x) / 2 * rho) * h for i in range(N)])
phi_ref -= phi_ref[0]

ax.plot(x, rho / rho.max() * np.abs(phi).max(), "--", color="C1",
        lw=1.2, label=r"$\rho$ (scaled)")
ax.plot(x, phi_ref, "k-", lw=2.0, label=r"$\phi_{\rm direct}$")
ax.plot(x, phi, "C0--", lw=1.5, label=r"$\phi_{\rm FFT}$")
ax.set_title(f"Isolated  [max err = {linf_error(phi, phi_ref):.1e}]")
ax.set_xlabel("x")
ax.legend(fontsize=8)

# ---------------------------------------------------------------------------
# Finish
# ---------------------------------------------------------------------------
for ax in axes:
    ax.axhline(0, color="gray", lw=0.5, ls=":")

fig.suptitle("1D Poisson Solvers: FFT vs Analytic Solution", fontsize=13)
fig.tight_layout()

out = os.path.join(os.path.dirname(__file__), "demo_1d.png")
fig.savefig(out, dpi=150)
print(f"Saved: {out}")
plt.show()
