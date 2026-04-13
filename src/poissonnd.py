"""
poissonnd.py — FFT-based Poisson solvers for N-dimensional domains.

All solvers find phi satisfying

    ∇²phi = rho(x)   (i.e. sum_d d²phi/dx_d² = rho)

on a rectangular domain with either periodic or isolated (free-space)
boundary conditions.

Supported dimensions: 1, 2, or 3 (the code is dimension-agnostic but the
Green's function is hard-coded per dimension).

Sign convention
---------------
Same as poisson1d.py:  ∇²phi = rho.
For gravity, pass  4*pi*G * rho_mass  as the rho argument.

Green's functions (free-space)
-------------------------------
The Green's function G satisfies ∇²G = δ^d(r) in d dimensions:

  d=1:  G(x)   = |x| / 2
  d=2:  G(r)   = ln(r) / (2π)      [diverges at r=0 and r→∞]
  d=3:  G(r)   = -1 / (4π r)       [G → 0 as r → ∞]

At r = 0 the Green's function is singular.  We set G[0,...,0] = 0, which
is consistent with zeroing the mean of phi (gauge freedom) and matches
what production PM codes do.

  - 3D: the missing self-interaction term is O(h²), same as the overall
    quadrature error — no convergence impact.
  - 2D: the missing term is O(h²|ln h|), introducing a logarithmic
    correction to the O(h²) convergence.  In practice this is invisible
    at the grid sizes used here.

Quadrature weight
-----------------
The FFT convolution discretises the integral ∫ G(r−r') ρ(r') dⁿr' as a
Riemann sum.  The weight is the cell volume  dV = hx·hy·... = prod(hs).
This is computed as np.prod(hs) so it works for anisotropic grids.

References
----------
Hockney, R. W. & Eastwood, J. W. (1988).
    Computer Simulation Using Particles. CRC Press.
    Chapter 6: PM method, isolated boundary conditions, optimal Green's function.
"""

import numpy as np
from scipy.fft import fftn, ifftn


# ---------------------------------------------------------------------------
# Periodic boundary conditions — N-dimensional
# ---------------------------------------------------------------------------


def solve_periodic_nd(
    rho: np.ndarray,
    Ls: float | tuple[float, ...],
) -> np.ndarray:
    """Solve ∇²phi = rho with periodic boundary conditions in d dimensions.

    Works for d = 1, 2, 3 (or any positive integer).

    Grid: vertex-centred in each dimension.
      x_j = j * hx,  j = 0, ..., Nx-1,  hx = Lx / Nx  (and similarly for y, z)

    The N-dimensional DFT diagonalises the Laplacian:

        phi_hat[n] = rho_hat[n] / lambda[n],
        lambda[n]  = -(kx_n1² + ky_n2² + ... )

    The zero mode (all wavenumbers = 0) is singular; we zero it out to
    enforce zero-mean phi.

    Parameters
    ----------
    rho : ndarray, shape (Nx,) or (Nx, Ny) or (Nx, Ny, Nz)
        Source term on the nd-dimensional periodic grid.
    Ls : float or sequence of float
        Domain length(s).  A scalar applies the same length to all dimensions.

    Returns
    -------
    phi : ndarray, same shape as rho, real
        Potential with zero mean.
    """
    rho = np.asarray(rho, dtype=float)
    ndim = rho.ndim
    shape = rho.shape
    Ls = _broadcast_Ls(Ls, ndim)

    # Forward n-dimensional FFT
    rho_hat = fftn(rho)

    # Build k² = kx² + ky² + ... via broadcasting
    k2 = np.zeros(shape)
    for d, (N, L) in enumerate(zip(shape, Ls)):
        h = L / N
        freq = np.fft.fftfreq(N, d=h)
        k = 2.0 * np.pi * freq
        # Reshape k so it broadcasts along axis d
        slices = [np.newaxis] * ndim
        slices[d] = slice(None)
        k2 = k2 + k[tuple(slices)] ** 2

    lambda_k = -k2

    # Gauge fix: zero the (0,0,...,0) mode
    rho_hat.flat[0] = 0.0
    lambda_k.flat[0] = 1.0

    return np.real(ifftn(rho_hat / lambda_k))


# ---------------------------------------------------------------------------
# Isolated (free-space) boundary conditions — N-dimensional
# ---------------------------------------------------------------------------


def solve_isolated_nd(
    rho: np.ndarray,
    Ls: float | tuple[float, ...],
    baseline: str = "zero_corner",
) -> np.ndarray:
    """Solve ∇²phi = rho with isolated (open/free-space) BCs in d dimensions.

    Works for d = 1, 2, 3.

    Implementation follows Hockney & Eastwood (1988), Ch. 6:

    1. Zero-pad rho from (Nx, ...) to (2Nx, ...) along every axis.
    2. Build the Green's function G on the extended periodic grid using the
       cyclic nearest-image distance (min-image convention).
    3. Compute phi_ext = IFFT(FFT(G) · FFT(rho_ext)) · prod(hs)
       where prod(hs) = hx·hy·... is the cell-volume quadrature weight.
    4. Extract the first (Nx, ...) points along every axis.
    5. Apply baseline subtraction.

    Green's function (per dimension):
      d=1:  G(r) = r / 2
      d=2:  G(r) = ln(r) / (2π),  G[0,0] = 0
      d=3:  G(r) = -1 / (4π r),   G[0,0,0] = 0

    Parameters
    ----------
    rho : ndarray, shape (Nx,) or (Nx, Ny) or (Nx, Ny, Nz)
        Source term.
    Ls : float or sequence of float
        Domain length(s).  Scalar for isotropic (same length in all dims).
    baseline : {'zero_corner', 'zero_mean', 'none'}
        How to fix the additive constant:
        * 'zero_corner' : subtract phi[0, ..., 0]  (default).
        * 'zero_mean'   : subtract mean(phi).
        * 'none'        : no subtraction.

    Returns
    -------
    phi : ndarray, same shape as rho, real

    Notes
    -----
    The Green's function singularity at r=0 is handled by setting G[0,...,0]=0.
    In 3D this introduces an O(h²) error (same as overall quadrature accuracy).
    In 2D it introduces an O(h²|ln h|) error, slightly sub-quadratic but
    convergent and invisible at typical grid sizes.
    """
    rho = np.asarray(rho, dtype=float)
    ndim = rho.ndim
    if ndim not in (1, 2, 3):
        raise ValueError(f"solve_isolated_nd supports d=1,2,3; got d={ndim}")
    shape = rho.shape
    Ls = _broadcast_Ls(Ls, ndim)
    hs = tuple(L / N for L, N in zip(Ls, shape))

    # Extended shape: double every dimension
    ext_shape = tuple(2 * N for N in shape)

    # Zero-pad rho onto the extended grid
    rho_ext = np.zeros(ext_shape)
    slices = tuple(slice(0, N) for N in shape)
    rho_ext[slices] = rho

    # Build Green's function on the extended grid
    G = _build_green_function(ext_shape, hs, ndim)

    # FFT convolution: phi_ext = IFFT(FFT(G) * FFT(rho_ext)) * dV
    dV = np.prod(hs)
    phi_ext = np.real(ifftn(fftn(G) * fftn(rho_ext))) * dV

    # Extract physical domain
    phi = phi_ext[slices]

    # Baseline subtraction
    if baseline == "zero_corner":
        phi = phi - phi.flat[0]
    elif baseline == "zero_mean":
        phi = phi - np.mean(phi)
    elif baseline == "none":
        pass
    else:
        raise ValueError(
            f"baseline must be 'zero_corner', 'zero_mean', or 'none'; got {baseline!r}"
        )

    return phi


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _broadcast_Ls(
    Ls: float | tuple[float, ...],
    ndim: int,
) -> list[float]:
    """Return Ls as a list of length ndim, broadcasting a scalar if needed."""
    if np.isscalar(Ls):
        return [float(Ls)] * ndim
    Ls = list(Ls)
    if len(Ls) != ndim:
        raise ValueError(
            f"len(Ls)={len(Ls)} does not match rho.ndim={ndim}"
        )
    return [float(L) for L in Ls]


def _build_green_function(
    ext_shape: tuple[int, ...],
    hs: tuple[float, ...],
    ndim: int,
) -> np.ndarray:
    """Build the nD Green's function on the extended periodic grid.

    Uses the cyclic nearest-image distance in each dimension:
      dist_d[k] = min(k, M_d - k) * h_d

    Then r = sqrt(sum_d dist_d²), and G(r) is the dimension-appropriate
    free-space Green's function with G[0,...,0] = 0.
    """
    # Squared distance from the origin using nearest-image convention
    r2 = np.zeros(ext_shape)
    for d, (M, h) in enumerate(zip(ext_shape, hs)):
        idx = np.arange(M)
        dist_d = np.minimum(idx, M - idx) * h
        # Broadcast dist_d along axis d
        slices = [np.newaxis] * ndim
        slices[d] = slice(None)
        r2 = r2 + dist_d[tuple(slices)] ** 2

    r = np.sqrt(r2)   # shape ext_shape; r[0,...,0] = 0

    if ndim == 1:
        G = r / 2.0
    elif ndim == 2:
        with np.errstate(divide="ignore"):
            G = np.where(r > 0.0, np.log(r) / (2.0 * np.pi), 0.0)
    elif ndim == 3:
        with np.errstate(divide="ignore", invalid="ignore"):
            G = np.where(r > 0.0, -1.0 / (4.0 * np.pi * r), 0.0)
    else:
        raise ValueError(f"ndim={ndim} not supported")

    return G
