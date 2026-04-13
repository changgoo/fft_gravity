"""
fft_gravity — FFT-based solvers for Poisson's equation.

Submodules
----------
poisson1d : four 1D solvers (periodic, Dirichlet, Neumann, isolated)
utils     : grid constructors, error norms, convergence helpers
"""

from .poisson1d import (
    solve_periodic,
    solve_dirichlet,
    solve_neumann,
    solve_isolated,
)
from .utils import (
    make_grid_periodic,
    make_grid_dirichlet,
    make_grid_neumann,
    make_grid_isolated,
    l2_error,
    linf_error,
    convergence_rate,
    richardson_extrapolate,
)

__all__ = [
    "solve_periodic",
    "solve_dirichlet",
    "solve_neumann",
    "solve_isolated",
    "make_grid_periodic",
    "make_grid_dirichlet",
    "make_grid_neumann",
    "make_grid_isolated",
    "l2_error",
    "linf_error",
    "convergence_rate",
    "richardson_extrapolate",
]
