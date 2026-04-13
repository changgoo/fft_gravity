"""
convergence_study.py — Convergence analysis for all four 1D Poisson solvers.

Produces convergence_study.png with two panels:
  Left:  Eigenfunction tests (machine-precision floor, flat lines)
  Right: Algebraic-convergence tests (O(h^2) slope = -2)

Also prints a summary table of measured convergence slopes.

Usage:
    python examples/convergence_study.py
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
    convergence_rate,
)

Ns = [8, 16, 32, 64, 128, 256, 512]
L = 1.0

# ---------------------------------------------------------------------------
# Eigenfunction tests (machine-precision floor)
# ---------------------------------------------------------------------------
def errors_periodic_eigenfunction(Ns, L):
    errors = []
    for N in Ns:
        x, _ = make_grid_periodic(N, L)
        rho = np.sin(2 * np.pi * x / L)
        phi = solve_periodic(rho, L)
        phi_exact = -(L / (2 * np.pi)) ** 2 * rho
        errors.append(linf_error(phi, phi_exact))
    return errors

def errors_dirichlet_eigenfunction(Ns, L):
    errors = []
    for N in Ns:
        x, _ = make_grid_dirichlet(N, L)
        rho = np.sin(np.pi * x / L)
        phi = solve_dirichlet(rho, L, modified_wavenumber=False)
        phi_exact = -(L / np.pi) ** 2 * rho
        errors.append(linf_error(phi, phi_exact))
    return errors

def errors_neumann_eigenfunction(Ns, L):
    errors = []
    for N in Ns:
        x, _ = make_grid_neumann(N, L)
        rho = np.cos(np.pi * x / L)
        phi = solve_neumann(rho, L)
        phi_exact = -(L / np.pi) ** 2 * rho
        errors.append(linf_error(phi, phi_exact))
    return errors

# ---------------------------------------------------------------------------
# Algebraic-convergence tests (O(h^2))
# ---------------------------------------------------------------------------
def errors_dirichlet_modified(Ns, L):
    """Modified wavenumber Dirichlet: O(h^2) for multi-mode smooth source."""
    errors = []
    for N in Ns:
        x, _ = make_grid_dirichlet(N, L)
        n1, n2 = 3, 5
        rho = np.sin(n1 * np.pi * x / L) + np.sin(n2 * np.pi * x / L)
        phi = solve_dirichlet(rho, L, modified_wavenumber=True)
        phi_exact = (
            -(L / (n1 * np.pi)) ** 2 * np.sin(n1 * np.pi * x / L)
            - (L / (n2 * np.pi)) ** 2 * np.sin(n2 * np.pi * x / L)
        )
        errors.append(linf_error(phi, phi_exact))
    return errors

def errors_isolated(Ns, L_iso=4.0, sigma=0.3):
    """Isolated solver: O(h^2) vs direct quadrature on fine reference grid."""
    # Fine reference
    N_ref = 1024
    x_ref, h_ref = make_grid_isolated(N_ref, L_iso)
    rho_ref = np.exp(-0.5 * ((x_ref - L_iso / 2) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    phi_ref = np.array([np.sum(np.abs(x_ref[i] - x_ref) / 2 * rho_ref) * h_ref for i in range(N_ref)])
    phi_ref -= phi_ref[0]

    errors = []
    for N in Ns:
        x, h = make_grid_isolated(N, L_iso)
        rho = np.exp(-0.5 * ((x - L_iso / 2) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        phi = solve_isolated(rho, L_iso, baseline="zero_first")
        stride = N_ref // N
        phi_ref_sub = phi_ref[::stride][:N]
        errors.append(linf_error(phi, phi_ref_sub))
    return errors

# ---------------------------------------------------------------------------
# Collect all results
# ---------------------------------------------------------------------------
print("Computing errors ...")
e_per   = errors_periodic_eigenfunction(Ns, L)
e_dir   = errors_dirichlet_eigenfunction(Ns, L)
e_neu   = errors_neumann_eigenfunction(Ns, L)
e_dir_m = errors_dirichlet_modified(Ns, L)
e_iso   = errors_isolated(Ns)

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
def safe_slope(Ns, errors):
    # Avoid log(0) for machine-precision floors
    valid = [(n, e) for n, e in zip(Ns, errors) if e > 1e-15]
    if len(valid) < 2:
        return float("nan")
    ns, es = zip(*valid)
    slope, _ = convergence_rate(list(ns), list(es))
    return slope

print("\n" + "=" * 60)
print(f"{'Solver':<35} {'Slope':>8}  {'Expected':>10}")
print("=" * 60)
rows = [
    ("Periodic (spectral, eigenfunction)", e_per, "~0 (machine eps)"),
    ("Dirichlet (spectral, eigenfunction)", e_dir, "~0 (machine eps)"),
    ("Neumann (spectral, eigenfunction)", e_neu, "~0 (machine eps)"),
    ("Dirichlet (modified, multi-mode)", e_dir_m, "-2"),
    ("Isolated (Gaussian)", e_iso, "-2"),
]
for name, errors, expected in rows:
    s = safe_slope(Ns, errors)
    print(f"  {name:<33} {s:>8.2f}  {expected:>10}")
print("=" * 60)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5))
Ns_arr = np.array(Ns, dtype=float)

# Left: eigenfunction (machine eps floor)
for label, errors, color in [
    ("Periodic (spectral)", e_per, "C0"),
    ("Dirichlet (spectral)", e_dir, "C1"),
    ("Neumann (spectral)", e_neu, "C2"),
]:
    ax_left.loglog(Ns_arr, errors, "o-", color=color, label=label, lw=1.5)

ax_left.axhline(1e-15, color="gray", ls="--", lw=1, label="Machine epsilon")
ax_left.set_xlabel("N (grid points)")
ax_left.set_ylabel(r"$L_\infty$ error")
ax_left.set_title("Eigenfunction tests (spectral accuracy)")
ax_left.legend(fontsize=9)
ax_left.grid(True, which="both", alpha=0.3)

# Right: algebraic convergence
for label, errors, color in [
    ("Dirichlet (modified wavenumber)", e_dir_m, "C3"),
    ("Isolated (Gaussian)", e_iso, "C4"),
]:
    ax_right.loglog(Ns_arr, errors, "o-", color=color, label=label, lw=1.5)

# Reference O(h^2) line
ref = e_dir_m[0] * (Ns_arr[0] / Ns_arr) ** 2
ax_right.loglog(Ns_arr, ref, "k--", lw=1, label=r"$\mathcal{O}(N^{-2})$")

ax_right.set_xlabel("N (grid points)")
ax_right.set_ylabel(r"$L_\infty$ error")
ax_right.set_title(r"Algebraic convergence ($\mathcal{O}(h^2)$)")
ax_right.legend(fontsize=9)
ax_right.grid(True, which="both", alpha=0.3)

fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "convergence_study.png")
fig.savefig(out, dpi=150)
print(f"\nSaved: {out}")
plt.show()
