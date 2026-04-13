"""
utils.py — Grid constructors, error norms, and convergence helpers.

All grids use double precision (float64) by default.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Grid constructors
# ---------------------------------------------------------------------------


def make_grid_periodic(N: int, L: float) -> tuple[np.ndarray, float]:
    """Return (x, h) for an N-point vertex-centred periodic grid on [0, L).

    Points: x_j = j * h,  j = 0, 1, ..., N-1
    Spacing: h = L / N  (endpoint NOT included — the grid wraps around)

    Parameters
    ----------
    N : int
        Number of grid points.
    L : float
        Domain length.

    Returns
    -------
    x : ndarray, shape (N,)
    h : float
    """
    h = L / N
    x = np.arange(N) * h
    return x, h


def make_grid_dirichlet(N: int, L: float) -> tuple[np.ndarray, float]:
    """Return (x, h) for the N-1 interior points of a Dirichlet grid on [0, L].

    The cell count is N (so h = L/N), and the N+1 grid vertices are
    x_j = j*h for j = 0, ..., N.  This function returns only the
    N-1 *interior* points x_j for j = 1, ..., N-1.

    Parameters
    ----------
    N : int
        Number of cells (boundary points are x=0 and x=L).
    L : float
        Domain length.

    Returns
    -------
    x : ndarray, shape (N-1,)   — interior points only
    h : float
    """
    h = L / N
    x = np.arange(1, N) * h
    return x, h


def make_grid_neumann(N: int, L: float) -> tuple[np.ndarray, float]:
    """Return (x, h) for an N-point cell-centred Neumann grid on [0, L].

    Points: x_j = (j + 0.5) * h,  j = 0, 1, ..., N-1
    Spacing: h = L / N

    Parameters
    ----------
    N : int
        Number of grid points (cells).
    L : float
        Domain length.

    Returns
    -------
    x : ndarray, shape (N,)
    h : float
    """
    h = L / N
    x = (np.arange(N) + 0.5) * h
    return x, h


def make_grid_isolated(N: int, L: float) -> tuple[np.ndarray, float]:
    """Return (x, h) for an N-point vertex-centred isolated-BC grid on [0, L).

    Identical layout to the periodic grid but with no periodicity assumption.

    Points: x_j = j * h,  j = 0, 1, ..., N-1
    Spacing: h = L / N

    Parameters
    ----------
    N : int
        Number of grid points.
    L : float
        Domain length.

    Returns
    -------
    x : ndarray, shape (N,)
    h : float
    """
    return make_grid_periodic(N, L)


# ---------------------------------------------------------------------------
# Error norms
# ---------------------------------------------------------------------------


def l2_error(phi: np.ndarray, phi_exact: np.ndarray) -> float:
    """RMS (L2) error between phi and phi_exact.

    Returns sqrt(mean((phi - phi_exact)**2)).
    """
    return float(np.sqrt(np.mean((phi - phi_exact) ** 2)))


def linf_error(phi: np.ndarray, phi_exact: np.ndarray) -> float:
    """L-infinity error between phi and phi_exact.

    Returns max|phi - phi_exact|.
    """
    return float(np.max(np.abs(phi - phi_exact)))


# ---------------------------------------------------------------------------
# Convergence analysis
# ---------------------------------------------------------------------------


def convergence_rate(
    Ns: np.ndarray | list[int],
    errors: np.ndarray | list[float],
) -> tuple[float, np.ndarray]:
    """Estimate the convergence order from a grid-refinement study.

    Fits a line to log(errors) vs log(Ns) using least squares.  For a
    solver with order p, errors ~ C * N^(-p), so the fitted slope should
    be approximately -p.

    Parameters
    ----------
    Ns : array-like of int
        Grid sizes (must be increasing).
    errors : array-like of float
        Corresponding error values.

    Returns
    -------
    slope : float
        Estimated convergence order as a signed number (e.g. -2.0 for 2nd order).
    coeffs : ndarray, shape (2,)
        Polynomial coefficients [slope, intercept] from np.polyfit.
    """
    Ns = np.asarray(Ns, dtype=float)
    errors = np.asarray(errors, dtype=float)
    coeffs = np.polyfit(np.log(Ns), np.log(errors), 1)
    return float(coeffs[0]), coeffs


def richardson_extrapolate(
    phi_fine: np.ndarray,
    phi_coarse: np.ndarray,
    order: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Richardson extrapolation from a coarse and fine grid solution.

    Assumes the fine grid has twice the resolution of the coarse grid and
    that the solver has convergence order `order`.

    Parameters
    ----------
    phi_fine : ndarray
        Solution on the fine grid (every other point, same shape as phi_coarse).
    phi_coarse : ndarray
        Solution on the coarse grid.
    order : int
        Convergence order of the solver (default 2).

    Returns
    -------
    phi_extrap : ndarray
        Richardson-extrapolated estimate (same shape as phi_coarse).
    err_estimate : ndarray
        Estimated pointwise error in phi_coarse: |phi_fine - phi_coarse|.
    """
    r = 2 ** order
    phi_extrap = (r * phi_fine - phi_coarse) / (r - 1)
    err_estimate = np.abs(phi_fine - phi_coarse)
    return phi_extrap, err_estimate
