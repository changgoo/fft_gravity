"""
demo_3d.py — 3D FFT-based Poisson solver demonstrations.

Produces a 2×3 panel figure (demo_3d.png).  Each row shows one BC type;
each column shows the mid-plane slice (z = Lz/2) of:
  col 0 — source ρ  (xy mid-plane)
  col 1 — solution φ (xy mid-plane)
  col 2 — solution φ (xz mid-plane)

Row 0 — Periodic BC
  Source: ρ = sin(2πx/Lx)·cos(2πy/Ly)·sin(2πz/Lz)
  Analytic: φ = −ρ / (kx² + ky² + kz²)

Row 1 — Isolated (free-space) BC
  Source: 3D isotropic Gaussian

Run:
    python examples/demo_3d.py
Output:
    demo_3d.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.poissonnd import solve_periodic_nd, solve_isolated_nd


# ---------------------------------------------------------------------------
# Grid setup
# ---------------------------------------------------------------------------

N = 64
L = 1.0
h = L / N

x = np.linspace(0, L, N, endpoint=False)
X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

# Mid-plane index
mid = N // 2


# ---------------------------------------------------------------------------
# Periodic source and analytic solution
# ---------------------------------------------------------------------------

kx = 2.0 * np.pi / L
ky = 2.0 * np.pi / L
kz = 2.0 * np.pi / L

rho_per = np.sin(kx * X) * np.cos(ky * Y) * np.sin(kz * Z)
phi_per_exact = -rho_per / (kx ** 2 + ky ** 2 + kz ** 2)
phi_per = solve_periodic_nd(rho_per, L)
err_per = np.max(np.abs(phi_per - phi_per_exact))


# ---------------------------------------------------------------------------
# Isolated (free-space) source and solution
# ---------------------------------------------------------------------------

sigma = 0.06
x0 = L / 2
rho_iso = np.exp(
    -0.5 * ((X - x0) ** 2 + (Y - x0) ** 2 + (Z - x0) ** 2) / sigma ** 2
)
rho_iso /= (2.0 * np.pi * sigma ** 2) ** 1.5   # unit-integral normalisation

phi_iso = solve_isolated_nd(rho_iso, L, baseline="zero_mean")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def slice_xy(arr):
    """Return the xy mid-plane slice: arr[:, :, mid]."""
    return arr[:, :, mid]


def slice_xz(arr):
    """Return the xz mid-plane slice: arr[:, mid, :]."""
    return arr[:, mid, :]


extent_xy = [0, L, 0, L]

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
fig.suptitle(
    "3D FFT-based Poisson solvers  ($\\nabla^2\\phi = \\rho$, mid-plane slices)",
    fontsize=13,
)


def add_panel(ax, data, title, cmap, norm=None, extent=extent_xy):
    im = ax.imshow(
        data.T, origin="lower", extent=extent, aspect="equal", cmap=cmap, norm=norm
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y" if extent is extent_xy else "z")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


# --- Row 0: Periodic ---
add_panel(axes[0, 0], slice_xy(rho_per), r"Periodic: $\rho$ (z=L/2)", "RdBu_r")
add_panel(
    axes[0, 1],
    slice_xy(phi_per),
    rf"Periodic: $\phi$ (z=L/2)  err={err_per:.1e}",
    "RdBu_r",
)
add_panel(
    axes[0, 2],
    slice_xz(phi_per),
    r"Periodic: $\phi$ (y=L/2)",
    "RdBu_r",
)

# --- Row 1: Isolated ---
add_panel(axes[1, 0], slice_xy(rho_iso), r"Isolated: $\rho$ (z=L/2)", "hot_r")

vmax_iso = np.max(np.abs(phi_iso))
iso_norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-vmax_iso, vmax=vmax_iso)

add_panel(
    axes[1, 1],
    slice_xy(phi_iso),
    r"Isolated: $\phi$ (z=L/2, zero-mean)",
    "RdBu_r",
    norm=iso_norm,
)
add_panel(
    axes[1, 2],
    slice_xz(phi_iso),
    r"Isolated: $\phi$ (y=L/2)",
    "RdBu_r",
    norm=iso_norm,
    extent=[0, L, 0, L],
)
axes[1, 2].set_ylabel("z")

plt.tight_layout()
plt.savefig("demo_3d.png", dpi=150)
print("Saved demo_3d.png")
print(f"  Periodic max error vs analytic: {err_per:.2e}  (expect ~ machine eps)")
