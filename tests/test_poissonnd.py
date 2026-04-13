"""
tests/test_poissonnd.py — Unit tests for N-dimensional Poisson solvers.

Test philosophy mirrors test_poisson1d.py:
* Eigenfunction tests: machine precision for pure sinusoidal sources.
* Isolated vs direct quadrature: FFT convolution matches O(N²) reference.
* Baseline consistency: different baseline options differ by a constant.

Dimensions covered: 1-D (regression), 2-D, 3-D.
"""

import numpy as np
import pytest

from src.poissonnd import solve_periodic_nd, solve_isolated_nd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def gaussian_2d(x, y, x0, y0, sigma):
    """2D isotropic Gaussian normalised so it integrates to 1 over all space."""
    return (
        np.exp(-0.5 * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)
        / (2.0 * np.pi * sigma ** 2)
    )


def gaussian_3d(x, y, z, x0, y0, z0, sigma):
    """3D isotropic Gaussian normalised so it integrates to 1."""
    return (
        np.exp(-0.5 * ((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) / sigma ** 2)
        / (2.0 * np.pi * sigma ** 2) ** 1.5
    )


def direct_quadrature_2d(rho, hx, hy):
    """Reference 2D isolated potential: direct O(N^4) convolution with G[0,0]=0."""
    Nx, Ny = rho.shape
    phi = np.zeros((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            r = np.zeros((Nx, Ny))
            for ii in range(Nx):
                for jj in range(Ny):
                    dx = (i - ii) * hx
                    dy = (j - jj) * hy
                    r[ii, jj] = np.sqrt(dx ** 2 + dy ** 2)
            with np.errstate(divide="ignore", invalid="ignore"):
                G = np.where(r > 0.0, np.log(r) / (2.0 * np.pi), 0.0)
            phi[i, j] = np.sum(G * rho) * hx * hy
    return phi


def direct_quadrature_3d(rho, hx, hy, hz):
    """Reference 3D isolated potential: direct O(N^6) convolution with G[0,0,0]=0."""
    Nx, Ny, Nz = rho.shape
    phi = np.zeros((Nx, Ny, Nz))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                r = np.zeros((Nx, Ny, Nz))
                for ii in range(Nx):
                    for jj in range(Ny):
                        for kk in range(Nz):
                            dx = (i - ii) * hx
                            dy = (j - jj) * hy
                            dz = (k - kk) * hz
                            r[ii, jj, kk] = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                with np.errstate(divide="ignore", invalid="ignore"):
                    G = np.where(r > 0.0, -1.0 / (4.0 * np.pi * r), 0.0)
                phi[i, j, k] = np.sum(G * rho) * hx * hy * hz
    return phi


# ---------------------------------------------------------------------------
# Periodic BC — N-dimensional
# ---------------------------------------------------------------------------


class TestPeriodicND:
    """Tests for solve_periodic_nd."""

    def test_1d_regression(self):
        """1D call matches the 1D-only solver result (regression guard)."""
        from src.poisson1d import solve_periodic

        N, L = 64, 1.0
        x = np.linspace(0, L, N, endpoint=False)
        rho = np.sin(2.0 * np.pi * x / L)

        phi_1d = solve_periodic(rho, L)
        phi_nd = solve_periodic_nd(rho, L)

        assert np.allclose(phi_nd, phi_1d, atol=1e-12)

    def test_2d_eigenfunction(self):
        """sin(2πx/Lx)·sin(2πy/Ly) is an exact eigenfunction; machine precision."""
        Nx, Ny = 32, 32
        Lx, Ly = 1.0, 1.0
        x = np.linspace(0, Lx, Nx, endpoint=False)
        y = np.linspace(0, Ly, Ny, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing="ij")

        kx = 2.0 * np.pi / Lx
        ky = 2.0 * np.pi / Ly
        rho = np.sin(kx * X) * np.sin(ky * Y)
        phi_exact = -rho / (kx ** 2 + ky ** 2)

        phi = solve_periodic_nd(rho, (Lx, Ly))

        assert np.allclose(phi, phi_exact, atol=1e-12), (
            f"Max error = {np.max(np.abs(phi - phi_exact)):.2e}"
        )

    def test_2d_mixed_modes(self):
        """Sum of two 2D modes; check against analytic superposition."""
        Nx, Ny = 64, 64
        Lx, Ly = 2.0, 1.0
        x = np.linspace(0, Lx, Nx, endpoint=False)
        y = np.linspace(0, Ly, Ny, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing="ij")

        kx1, ky1 = 2.0 * np.pi / Lx, 2.0 * np.pi / Ly
        kx2, ky2 = 4.0 * np.pi / Lx, 2.0 * np.pi / Ly

        rho = np.sin(kx1 * X) * np.sin(ky1 * Y) + np.sin(kx2 * X) * np.cos(ky2 * Y)
        phi_exact = (
            -np.sin(kx1 * X) * np.sin(ky1 * Y) / (kx1 ** 2 + ky1 ** 2)
            - np.sin(kx2 * X) * np.cos(ky2 * Y) / (kx2 ** 2 + ky2 ** 2)
        )

        phi = solve_periodic_nd(rho, (Lx, Ly))

        assert np.allclose(phi, phi_exact, atol=1e-11)

    def test_2d_zero_mean_enforced(self):
        """2D periodic solver enforces zero-mean phi regardless of mean(rho)."""
        Nx, Ny = 32, 32
        rho = np.ones((Nx, Ny)) + np.random.default_rng(42).standard_normal((Nx, Ny))
        phi = solve_periodic_nd(rho, 1.0)
        assert np.isclose(np.mean(phi), 0.0, atol=1e-12)

    def test_3d_eigenfunction(self):
        """3D eigenfunction sin(kx·x)·sin(ky·y)·sin(kz·z): machine precision."""
        Nx, Ny, Nz = 16, 16, 16
        Lx, Ly, Lz = 1.0, 1.0, 1.0
        x = np.linspace(0, Lx, Nx, endpoint=False)
        y = np.linspace(0, Ly, Ny, endpoint=False)
        z = np.linspace(0, Lz, Nz, endpoint=False)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        kx = 2.0 * np.pi / Lx
        ky = 2.0 * np.pi / Ly
        kz = 2.0 * np.pi / Lz
        rho = np.sin(kx * X) * np.sin(ky * Y) * np.sin(kz * Z)
        phi_exact = -rho / (kx ** 2 + ky ** 2 + kz ** 2)

        phi = solve_periodic_nd(rho, (Lx, Ly, Lz))

        assert np.allclose(phi, phi_exact, atol=1e-12), (
            f"Max error = {np.max(np.abs(phi - phi_exact)):.2e}"
        )

    def test_scalar_L_broadcasts(self):
        """A scalar Ls applies the same length to all dimensions."""
        Nx, Ny = 16, 16
        L = 1.0
        x = np.linspace(0, L, Nx, endpoint=False)
        y = np.linspace(0, L, Ny, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing="ij")

        k = 2.0 * np.pi / L
        rho = np.sin(k * X) + np.sin(k * Y)
        phi_scalar = solve_periodic_nd(rho, L)
        phi_tuple = solve_periodic_nd(rho, (L, L))

        assert np.allclose(phi_scalar, phi_tuple, atol=1e-14)

    def test_Ls_length_mismatch_raises(self):
        """Mismatched Ls length raises ValueError."""
        rho = np.ones((8, 8))
        with pytest.raises(ValueError, match="len\\(Ls\\)"):
            solve_periodic_nd(rho, (1.0, 1.0, 1.0))


# ---------------------------------------------------------------------------
# Isolated BC — N-dimensional
# ---------------------------------------------------------------------------


class TestIsolatedND:
    """Tests for solve_isolated_nd."""

    def test_1d_regression(self):
        """1D call is consistent with 1D-only solve_isolated."""
        from src.poisson1d import solve_isolated

        N, L = 32, 4.0
        x = np.linspace(0, L, N, endpoint=False)
        sigma = 0.3
        rho = np.exp(-0.5 * ((x - L / 2) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

        phi_1d = solve_isolated(rho, L, baseline="zero_first")
        phi_nd = solve_isolated_nd(rho, L, baseline="zero_corner")

        # Both zero the first element; results should match
        assert np.allclose(phi_nd, phi_1d, atol=1e-12)

    def test_2d_vs_direct_quadrature(self):
        """2D FFT convolution matches direct O(N^4) summation (small grid)."""
        N = 10
        L = 3.0
        h = L / N
        x = np.linspace(0, L, N, endpoint=False)
        X, Y = np.meshgrid(x, x, indexing="ij")
        sigma = 0.4
        rho = gaussian_2d(X, Y, L / 2, L / 2, sigma)

        phi_fft = solve_isolated_nd(rho, L, baseline="zero_corner")
        phi_ref = direct_quadrature_2d(rho, h, h)
        phi_ref -= phi_ref.flat[0]   # match zero_corner baseline

        assert np.allclose(phi_fft, phi_ref, atol=1e-8), (
            f"Max error = {np.max(np.abs(phi_fft - phi_ref)):.2e}"
        )

    def test_3d_vs_direct_quadrature(self):
        """3D FFT convolution matches direct O(N^6) summation (tiny grid)."""
        N = 6
        L = 2.0
        h = L / N
        x = np.linspace(0, L, N, endpoint=False)
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        sigma = 0.3
        rho = gaussian_3d(X, Y, Z, L / 2, L / 2, L / 2, sigma)

        phi_fft = solve_isolated_nd(rho, L, baseline="zero_corner")
        phi_ref = direct_quadrature_3d(rho, h, h, h)
        phi_ref -= phi_ref.flat[0]

        assert np.allclose(phi_fft, phi_ref, atol=1e-8), (
            f"Max error = {np.max(np.abs(phi_fft - phi_ref)):.2e}"
        )

    def test_2d_baseline_options_differ_by_constant(self):
        """zero_corner vs zero_mean baselines differ by a spatially uniform shift."""
        N = 16
        L = 2.0
        x = np.linspace(0, L, N, endpoint=False)
        X, Y = np.meshgrid(x, x, indexing="ij")
        rho = gaussian_2d(X, Y, L / 2, L / 2, 0.3)

        phi_zc = solve_isolated_nd(rho, L, baseline="zero_corner")
        phi_zm = solve_isolated_nd(rho, L, baseline="zero_mean")
        phi_no = solve_isolated_nd(rho, L, baseline="none")

        assert np.std(phi_zc - phi_zm) < 1e-12
        assert np.std(phi_zc - phi_no) < 1e-12

    def test_3d_baseline_options_differ_by_constant(self):
        """Same baseline consistency check in 3D."""
        N = 8
        L = 2.0
        x = np.linspace(0, L, N, endpoint=False)
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        rho = gaussian_3d(X, Y, Z, L / 2, L / 2, L / 2, 0.3)

        phi_zc = solve_isolated_nd(rho, L, baseline="zero_corner")
        phi_zm = solve_isolated_nd(rho, L, baseline="zero_mean")

        assert np.std(phi_zc - phi_zm) < 1e-12

    def test_invalid_ndim_raises(self):
        """4D input raises ValueError (only d=1,2,3 supported)."""
        rho = np.ones((4, 4, 4, 4))
        with pytest.raises(ValueError, match="d=1,2,3"):
            solve_isolated_nd(rho, 1.0)

    def test_invalid_baseline_raises(self):
        """Unknown baseline string raises ValueError."""
        rho = np.ones((8, 8))
        with pytest.raises(ValueError, match="baseline"):
            solve_isolated_nd(rho, 1.0, baseline="bad")

    def test_2d_convergence(self):
        """2D isolated solver converges at ~ O(h^2) using a fine FFT reference."""
        L = 3.0
        sigma = 0.4
        # Powers of 2 so sub-sampling strides are exact integers
        Ns = [8, 16, 32, 64]

        # Fine FFT reference
        N_ref = 128
        x_ref = np.linspace(0, L, N_ref, endpoint=False)
        X_ref, Y_ref = np.meshgrid(x_ref, x_ref, indexing="ij")
        phi_ref = solve_isolated_nd(
            gaussian_2d(X_ref, Y_ref, L / 2, L / 2, sigma), L, baseline="zero_corner"
        )

        errors = []
        for N in Ns:
            x = np.linspace(0, L, N, endpoint=False)
            X, Y = np.meshgrid(x, x, indexing="ij")
            phi = solve_isolated_nd(
                gaussian_2d(X, Y, L / 2, L / 2, sigma), L, baseline="zero_corner"
            )
            stride = N_ref // N
            phi_sub = phi_ref[::stride, ::stride][:N, :N]
            errors.append(np.max(np.abs(phi - phi_sub)))

        # Fit convergence slope on log-log (log h vs log error)
        log_hs = np.log(np.array([L / N for N in Ns]))
        log_errs = np.log(np.array(errors))
        slope = np.polyfit(log_hs, log_errs, 1)[0]
        assert slope > 1.5, f"Expected slope > 1.5 (~ O(h^2)), got {slope:.2f}"
