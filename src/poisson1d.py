"""
poisson1d.py — FFT-based solvers for the 1D Poisson equation.

All solvers find phi satisfying

    d²phi/dx² = rho(x)

on a 1D domain [0, L] with different boundary conditions.

Sign convention
---------------
The equation is written as  d²phi/dx² = rho.
For gravitational physics, the caller is responsible for including the
4*pi*G factor: pass rho_phys * 4 * pi * G as the rho argument.

SciPy FFT normalization strategy
---------------------------------
Throughout this module we use *unnormalized* forward transforms
(scipy.fft default: norm=None) and rely on the corresponding *inverse*
transform to carry the normalization factor.  This is the safest
approach because it avoids having to track extra factors manually:

  - scipy.fft.fft / ifft  :  forward sums without 1/N; ifft divides by N.
  - scipy.fft.dst(type=1) :  unnormalized forward; idst(type=1) divides by 2(N+1).
  - scipy.fft.dct(type=2) :  unnormalized forward; idct(type=2) divides by 2N.

Do NOT reconstruct the inverse by hand (e.g. dst(phi_hat) / (2*(N+1)));
always call idst / idct to avoid off-by-one normalization errors.

Gauge freedom
-------------
Periodic and Neumann solvers both have a zero eigenvalue (k=0 or n=0
mode).  The potential is only defined up to an additive constant.  We
fix the gauge by zeroing the mean of phi (i.e. setting rho_hat[0] = 0
before the division).  This is done silently — no exception is raised
when mean(rho) != 0.

References
----------
Hockney, R. W. & Eastwood, J. W. (1988).
    Computer Simulation Using Particles. CRC Press.
    Chapter 6: The particle-mesh (PM) method and isolated boundary conditions.
"""

import numpy as np
from scipy.fft import fft, ifft, dst, idst, dct, idct


# ---------------------------------------------------------------------------
# Periodic boundary conditions
# ---------------------------------------------------------------------------


def solve_periodic(rho: np.ndarray, L: float) -> np.ndarray:
    """Solve d²phi/dx² = rho with periodic boundary conditions.

    Grid: vertex-centred, N points, x_j = j * L/N (j = 0, ..., N-1).

    The DFT decomposes rho into plane waves e^{2*pi*i*n*x/L}.  Each
    Fourier mode satisfies  d²/dx²  ->  -(2*pi*n/L)^2, so

        phi_hat[n] = rho_hat[n] / (-(2*pi*n/L)^2)

    The n=0 mode is singular (zero eigenvalue).  We zero it out, which
    is equivalent to enforcing mean(phi) = 0.

    Parameters
    ----------
    rho : ndarray, shape (N,)
        Source term on the N-point periodic grid.
    L : float
        Domain length.

    Returns
    -------
    phi : ndarray, shape (N,), real
        Potential with zero mean.
    """
    rho = np.asarray(rho, dtype=float)
    N = len(rho)
    h = L / N

    # Forward FFT
    rho_hat = fft(rho)

    # Wavenumbers: k_n = 2*pi*n/L  (using fftfreq's normalised frequency * 2*pi)
    freq = np.fft.fftfreq(N, d=h)          # cycles per unit length
    k = 2.0 * np.pi * freq                  # angular wavenumber

    # Eigenvalues of the 1D spectral Laplacian: lambda_n = -k_n^2
    lambda_k = -(k ** 2)

    # Gauge fix: zero out the k=0 mode (enforces zero mean of phi)
    rho_hat[0] = 0.0
    lambda_k[0] = 1.0                       # avoid division by zero; numerator is 0

    phi_hat = rho_hat / lambda_k

    return np.real(ifft(phi_hat))


# ---------------------------------------------------------------------------
# Dirichlet boundary conditions  (phi = 0 at x=0 and x=L)
# ---------------------------------------------------------------------------


def solve_dirichlet(
    rho: np.ndarray,
    L: float,
    modified_wavenumber: bool = False,
) -> np.ndarray:
    """Solve d²phi/dx² = rho with homogeneous Dirichlet boundary conditions.

    Grid: N-1 interior points  x_j = j * h,  j = 1, ..., N-1,  h = L/N.
    The caller passes rho on these N-1 interior points.  Boundary values
    phi(0) = phi(L) = 0 are implicit.

    The DST-I is the natural transform because the sine functions
    sin(n*pi*x/L)  (n = 1, ..., N-1) are eigenfunctions of the 1D
    Laplacian with zero-Dirichlet BCs:

        d²/dx² sin(n*pi*x/L) = -(n*pi/L)^2 * sin(n*pi*x/L)

    so  phi_hat[n] = rho_hat[n] / lambda_n.

    Two eigenvalue options:
      * Spectral  (default): lambda_n = -(n*pi/L)^2
        Gives exponential convergence for smooth rho; machine precision
        when rho is a single sine mode.
      * Modified wavenumber: lambda_n = -(2/h * sin(n*pi*h / (2*L)))^2
        Matches the second-order finite-difference Laplacian exactly,
        giving O(h^2) convergence regardless of rho smoothness.

    Parameters
    ----------
    rho : ndarray, shape (N-1,)
        Source term at the N-1 interior grid points.
    L : float
        Domain length (boundary points at x=0 and x=L).
    modified_wavenumber : bool, optional
        If True, use the FD-consistent modified wavenumber (O(h^2)).
        Default False (spectral eigenvalues).

    Returns
    -------
    phi : ndarray, shape (N-1,), real
        Potential at the N-1 interior points.
    """
    rho = np.asarray(rho, dtype=float)
    Nm1 = len(rho)          # number of interior points
    N = Nm1 + 1             # number of cells; h = L/N
    h = L / N

    # Forward DST-I
    rho_hat = dst(rho, type=1)              # shape (N-1,)

    # Mode indices n = 1, 2, ..., N-1
    n = np.arange(1, N)                     # length N-1

    if modified_wavenumber:
        # FD-consistent: matches the 3-point finite-difference Laplacian
        lambda_n = -(2.0 / h * np.sin(n * np.pi * h / (2.0 * L))) ** 2
    else:
        # Spectral (exact continuous eigenvalues)
        lambda_n = -(n * np.pi / L) ** 2

    phi_hat = rho_hat / lambda_n

    return idst(phi_hat, type=1)


# ---------------------------------------------------------------------------
# Neumann boundary conditions  (d phi/dx = 0 at x=0 and x=L)
# ---------------------------------------------------------------------------


def solve_neumann(rho: np.ndarray, L: float) -> np.ndarray:
    """Solve d²phi/dx² = rho with homogeneous Neumann boundary conditions.

    Grid: cell-centred, N points,  x_j = (j + 0.5) * h,  j = 0, ..., N-1.

    The DCT-II is the natural transform because cosine functions
    cos(n*pi*x/L)  (n = 0, 1, ..., N-1) are eigenfunctions of the 1D
    Laplacian with zero-Neumann BCs:

        d²/dx² cos(n*pi*x/L) = -(n*pi/L)^2 * cos(n*pi*x/L)
        d/dx   cos(n*pi*x/L)|_{x=0, L} = 0  (automatically satisfied)

    The n=0 mode is the mean of rho/phi and has eigenvalue 0.  We fix the
    gauge by setting phi_hat[0] = 0 (mean(phi) = 0).

    Parameters
    ----------
    rho : ndarray, shape (N,)
        Source term on the N-point cell-centred grid.
    L : float
        Domain length.

    Returns
    -------
    phi : ndarray, shape (N,), real
        Potential with zero mean.
    """
    rho = np.asarray(rho, dtype=float)
    N = len(rho)
    h = L / N

    # Forward DCT-II
    rho_hat = dct(rho, type=2)              # shape (N,)

    # Mode indices n = 0, 1, ..., N-1
    n = np.arange(N, dtype=float)

    # Eigenvalues of the spectral Laplacian
    lambda_n = -(n * np.pi / L) ** 2

    # Gauge fix: zero out the n=0 mode
    rho_hat[0] = 0.0
    lambda_n[0] = 1.0                       # avoid division by zero

    phi_hat = rho_hat / lambda_n

    return idct(phi_hat, type=2)


# ---------------------------------------------------------------------------
# Isolated / free-space boundary conditions
# ---------------------------------------------------------------------------


def solve_isolated(
    rho: np.ndarray,
    L: float,
    baseline: str = "zero_first",
) -> np.ndarray:
    """Solve d²phi/dx² = rho with isolated (open/free-space) boundary conditions.

    No artificial periodicity is imposed.  The solution is the convolution
    of rho with the 1D free-space Green's function

        G(x) = |x| / 2        [satisfies  d²G/dx² = delta(x)]

    Implementation follows Hockney & Eastwood (1988), Ch. 6 (PM method):

    1. Zero-pad rho from N to 2N on an extended periodic domain.
    2. Build G on the 2N grid using the cyclic nearest-image distance:
          G_k = min(k, 2N-k) * h / 2  for k = 0, ..., 2N-1.
    3. Compute phi_ext = IFFT(FFT(G) * FFT(rho_ext)) * h
       (the factor h is the Riemann quadrature weight for the integral).
    4. Extract the first N points; the remaining N points are contaminated
       by wrap-around and must be discarded.

    The potential is defined only up to an additive constant.  The
    `baseline` parameter controls how this constant is fixed.

    Parameters
    ----------
    rho : ndarray, shape (N,)
        Source term on the N-point grid.
    L : float
        Domain length.
    baseline : {'zero_first', 'zero_mean', 'none'}
        How to fix the additive constant in phi:
        * 'zero_first' : subtract phi[0]  (default).
        * 'zero_mean'  : subtract mean(phi).
        * 'none'       : no subtraction; absolute value is arbitrary.

    Returns
    -------
    phi : ndarray, shape (N,), real
        Potential after baseline subtraction.

    Notes
    -----
    In 1D the free-space potential grows with distance (G ~ |x|/2), so
    phi(x) → ±∞ as |x| → ∞ unless the total "charge" int rho dx = 0.
    When mean(rho) != 0, the result is still well-defined as a relative
    potential; use baseline='zero_first' or 'zero_mean' to fix the origin.
    """
    rho = np.asarray(rho, dtype=float)
    N = len(rho)
    h = L / N
    M = 2 * N                               # extended domain size

    # Extended source: zero-pad rho to length 2N
    rho_ext = np.zeros(M)
    rho_ext[:N] = rho

    # Green's function on the extended 2N periodic grid.
    # The cyclic nearest-image distance from index 0 to index k on a
    # periodic grid of M points with spacing h is min(k, M-k)*h.
    k_idx = np.arange(M)
    dist = np.minimum(k_idx, M - k_idx) * h
    G = dist / 2.0                          # G(x) = |x| / 2

    # FFT convolution  (circular on the 2N domain)
    phi_ext = np.real(ifft(fft(G) * fft(rho_ext))) * h

    # Extract the physical domain (first N points)
    phi = phi_ext[:N]

    # Baseline subtraction
    if baseline == "zero_first":
        phi = phi - phi[0]
    elif baseline == "zero_mean":
        phi = phi - np.mean(phi)
    elif baseline == "none":
        pass
    else:
        raise ValueError(
            f"baseline must be 'zero_first', 'zero_mean', or 'none'; got {baseline!r}"
        )

    return phi
