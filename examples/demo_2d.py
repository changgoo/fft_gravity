"""
demo_2d.py — 2D FFT-based Poisson solver demonstrations.

Produces a 2×2 panel figure (demo_2d.png) showing:

  Row 1 — Periodic BC:  source ρ (left) and solution φ (right)
           Source: ρ(x,y) = sin(2πx/Lx)·cos(4πy/Ly)
           Analytic: φ = −ρ / (kx² + ky²)

  Row 2 — Isolated (free-space) BC:  source ρ (left) and solution φ (right)
           Source: 2D Gaussian, representing a point-like mass sheet in gravity

Run:
    python examples/demo_2d.py
Output:
    demo_2d.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.poissonnd import solve_periodic_nd, solve_isolated_nd


# ---------------------------------------------------------------------------
# Grid setup
# ---------------------------------------------------------------------------

Nx, Ny = 128, 128
Lx, Ly = 1.0, 1.0
hx, hy = Lx / Nx, Ly / Ny

x = np.linspace(0, Lx, Nx, endpoint=False)
y = np.linspace(0, Ly, Ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing="ij")


# ---------------------------------------------------------------------------
# Periodic source and analytic solution
# ---------------------------------------------------------------------------

kx = 2.0 * np.pi / Lx          # fundamental in x
ky = 4.0 * np.pi / Ly          # 2nd harmonic in y

rho_per = np.sin(kx * X) * np.cos(ky * Y)
phi_per_exact = -rho_per / (kx ** 2 + ky ** 2)
phi_per = solve_periodic_nd(rho_per, (Lx, Ly))
err_per = np.max(np.abs(phi_per - phi_per_exact))


# ---------------------------------------------------------------------------
# Isolated (free-space) source and solution
# ---------------------------------------------------------------------------

sigma = 0.08
x0, y0 = Lx / 2, Ly / 2
rho_iso = np.exp(-0.5 * ((X - x0) ** 2 + (Y - y0) ** 2) / sigma ** 2)
rho_iso /= 2.0 * np.pi * sigma ** 2   # normalise to unit integral

phi_iso = solve_isolated_nd(rho_iso, (Lx, Ly), baseline="zero_mean")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("2D FFT-based Poisson solvers  ($\\nabla^2\\phi = \\rho$)", fontsize=14)

extent = [0, Lx, 0, Ly]

# --- Row 0: Periodic ---
im00 = axes[0, 0].imshow(
    rho_per.T, origin="lower", extent=extent, aspect="equal", cmap="RdBu_r"
)
axes[0, 0].set_title(r"Periodic: source $\rho$")
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("y")
fig.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.04)

# show both numeric and analytic (they overlie exactly)
im01 = axes[0, 1].imshow(
    phi_per.T, origin="lower", extent=extent, aspect="equal", cmap="RdBu_r"
)
axes[0, 1].set_title(
    rf"Periodic: $\phi$ (max err = {err_per:.1e})"
)
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("y")
fig.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.04)

# --- Row 1: Isolated ---
im10 = axes[1, 0].imshow(
    rho_iso.T, origin="lower", extent=extent, aspect="equal", cmap="hot_r"
)
axes[1, 0].set_title(r"Isolated: Gaussian source $\rho$")
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("y")
fig.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04)

# Use diverging colormap centred at 0 for the potential
vmax = np.max(np.abs(phi_iso))
im11 = axes[1, 1].imshow(
    phi_iso.T,
    origin="lower",
    extent=extent,
    aspect="equal",
    cmap="RdBu_r",
    norm=mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
)
axes[1, 1].set_title(r"Isolated: potential $\phi$ (zero-mean)")
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("y")
fig.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("demo_2d.png", dpi=150)
print("Saved demo_2d.png")
print(f"  Periodic max error vs analytic: {err_per:.2e}  (expect ~ machine eps)")
