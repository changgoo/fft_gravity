"""
fft_gravity — FFT-based solvers for Poisson's equation.

Submodules
----------
poisson1d  : four 1D solvers (periodic, Dirichlet, Neumann, isolated)
poissonnd  : N-dimensional solvers (periodic and isolated, d=1,2,3)
utils      : grid constructors, error norms, convergence helpers
"""

from .poisson1d import (
    solve_periodic,
    solve_dirichlet,
    solve_neumann,
    solve_isolated,
)
from .poissonnd import (
    solve_periodic_nd,
    solve_isolated_nd,
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
    "solve_periodic_nd",
    "solve_isolated_nd",
    "make_grid_periodic",
    "make_grid_dirichlet",
    "make_grid_neumann",
    "make_grid_isolated",
    "l2_error",
    "linf_error",
    "convergence_rate",
    "richardson_extrapolate",
]
