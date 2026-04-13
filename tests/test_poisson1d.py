"""
tests/test_poisson1d.py — Unit tests for the four 1D Poisson solvers.

Test philosophy
---------------
* Eigenfunction tests: when rho is a single eigenfunction of the Laplacian
  with the given BC, the FFT solve is a single division in spectral space.
  These tests pass at machine precision (atol ~ 1e-12).

* Convergence tests: for non-eigenfunction sources and for the modified-
  wavenumber / isolated solvers, we measure the L-infinity error against
  the analytic solution across a range of grid sizes and assert that the
  fitted convergence slope is within tolerance of the expected order.

* Linearity and gauge tests: sanity checks on linearity and the zero-mean
  gauge enforcement.

Sign convention: d²phi/dx² = rho (see src/poisson1d.py module docstring).
"""

import numpy as np
import pytest

from src.poisson1d import (
    solve_periodic,
    solve_dirichlet,
    solve_neumann,
    solve_isolated,
)
from src.utils import (
    make_grid_periodic,
    make_grid_dirichlet,
    make_grid_neumann,
    make_grid_isolated,
    l2_error,
    linf_error,
    convergence_rate,
)


# ---------------------------------------------------------------------------
# Periodic BC tests
# ---------------------------------------------------------------------------


class TestPeriodic:
    """Tests for solve_periodic."""

    def test_eigenfunction(self):
        """sin(2*pi*x/L) is an exact eigenfunction; error should be machine eps."""
        N, L = 64, 1.0
        x, h = make_grid_periodic(N, L)
        rho = np.sin(2.0 * np.pi * x / L)
        phi_exact = -rho / (2.0 * np.pi / L) ** 2

        phi = solve_periodic(rho, L)

        assert np.allclose(phi, phi_exact, atol=1e-12), (
            f"Max error = {linf_error(phi, phi_exact):.2e}"
        )

    def test_zero_mean_enforced_silent(self):
        """Non-zero mean rho is silently projected; mean(phi) == 0."""
        N, L = 64, 1.0
        x, _ = make_grid_periodic(N, L)
        rho = np.sin(2.0 * np.pi * x / L) + 1.0   # non-zero mean

        phi = solve_periodic(rho, L)

        assert np.isclose(np.mean(phi), 0.0, atol=1e-13)

    def test_linearity(self):
        """phi(rho1 + rho2) == phi(rho1) + phi(rho2)."""
        N, L = 64, 2.0
        x, _ = make_grid_periodic(N, L)
        rho1 = np.sin(2.0 * np.pi * x / L)
        rho2 = np.sin(4.0 * np.pi * x / L)

        phi_sum = solve_periodic(rho1 + rho2, L)
        phi1 = solve_periodic(rho1, L)
        phi2 = solve_periodic(rho2, L)

        assert np.allclose(phi_sum, phi1 + phi2, atol=1e-13)

    def test_multi_mode(self):
        """Sum of two modes has analytic solution; check accuracy."""
        N, L = 128, 1.0
        x, _ = make_grid_periodic(N, L)
        k1, k2 = 2.0 * np.pi / L, 4.0 * np.pi / L
        rho = np.sin(k1 * x) + np.sin(k2 * x)
        phi_exact = -np.sin(k1 * x) / k1 ** 2 - np.sin(k2 * x) / k2 ** 2

        phi = solve_periodic(rho, L)
        assert np.allclose(phi, phi_exact, atol=1e-12)


# ---------------------------------------------------------------------------
# Dirichlet BC tests
# ---------------------------------------------------------------------------


class TestDirichlet:
    """Tests for solve_dirichlet."""

    def test_eigenfunction_spectral(self):
        """sin(pi*x/L) is the n=1 eigenfunction; spectral mode gives machine eps."""
        N, L = 64, 1.0
        x, _ = make_grid_dirichlet(N, L)
        rho = np.sin(np.pi * x / L)
        phi_exact = -(L / np.pi) ** 2 * rho

        phi = solve_dirichlet(rho, L, modified_wavenumber=False)

        assert np.allclose(phi, phi_exact, atol=1e-12), (
            f"Max error = {linf_error(phi, phi_exact):.2e}"
        )

    def test_eigenfunction_modified_wavenumber(self):
        """Modified wavenumber has O(h^2) error even for the n=1 eigenfunction.

        For N=64, h=1/64, the modified eigenvalue differs from the spectral
        one by O(h^2) ~ 2e-4, giving a relative potential error of the same
        order.  We confirm the error is O(h^2) by checking it scales correctly
        with N.
        """
        L = 1.0
        errors = {}
        for N in [64, 128, 256]:
            x, h = make_grid_dirichlet(N, L)
            rho = np.sin(np.pi * x / L)
            phi_exact = -(L / np.pi) ** 2 * rho
            phi = solve_dirichlet(rho, L, modified_wavenumber=True)
            errors[N] = linf_error(phi, phi_exact)

        # Error should roughly halve for each doubling of N (O(h^2))
        ratio_64_128 = errors[64] / errors[128]
        ratio_128_256 = errors[128] / errors[256]
        assert ratio_64_128 > 3.0, f"Error ratio 64→128 = {ratio_64_128:.2f} (expected ~4)"
        assert ratio_128_256 > 3.0, f"Error ratio 128→256 = {ratio_128_256:.2f} (expected ~4)"

    def test_output_length(self):
        """Solver returns N-1 interior values for a grid with N cells."""
        N, L = 32, 1.0
        x, _ = make_grid_dirichlet(N, L)      # length N-1
        rho = np.sin(np.pi * x / L)
        phi = solve_dirichlet(rho, L)
        assert phi.shape == (N - 1,)

    def test_two_mode_analytic(self):
        """Analytic solution for sum of two DST eigenfunctions."""
        N, L = 128, 1.0
        x, _ = make_grid_dirichlet(N, L)
        n1, n2 = 3, 5
        rho = np.sin(n1 * np.pi * x / L) + np.sin(n2 * np.pi * x / L)
        phi_exact = (
            -(L / (n1 * np.pi)) ** 2 * np.sin(n1 * np.pi * x / L)
            - (L / (n2 * np.pi)) ** 2 * np.sin(n2 * np.pi * x / L)
        )

        phi = solve_dirichlet(rho, L)
        assert np.allclose(phi, phi_exact, atol=1e-12)

    @pytest.mark.parametrize("N", [16, 32, 64, 128, 256])
    def test_modified_wavenumber_convergence(self, N):
        """Build errors for O(h^2) convergence check (parametrized over N)."""
        # This test only checks that the solver runs; the slope is checked
        # in test_modified_wavenumber_convergence_slope below.
        L = 1.0
        x, h = make_grid_dirichlet(N, L)
        rho = np.sin(3 * np.pi * x / L) + np.sin(5 * np.pi * x / L)
        phi = solve_dirichlet(rho, L, modified_wavenumber=True)
        assert phi.shape == (N - 1,)

    def test_modified_wavenumber_convergence_slope(self):
        """Modified wavenumber Dirichlet solver converges at ~ O(h^2)."""
        L = 1.0
        Ns = [16, 32, 64, 128, 256]
        errors = []
        for N in Ns:
            x, _ = make_grid_dirichlet(N, L)
            rho = np.sin(3 * np.pi * x / L) + np.sin(5 * np.pi * x / L)
            phi_exact = (
                -(L / (3 * np.pi)) ** 2 * np.sin(3 * np.pi * x / L)
                - (L / (5 * np.pi)) ** 2 * np.sin(5 * np.pi * x / L)
            )
            phi = solve_dirichlet(rho, L, modified_wavenumber=True)
            errors.append(linf_error(phi, phi_exact))

        slope, _ = convergence_rate(Ns, errors)
        assert slope < -1.8, f"Expected slope < -1.8 (O(h^2)), got {slope:.2f}"


# ---------------------------------------------------------------------------
# Neumann BC tests
# ---------------------------------------------------------------------------


class TestNeumann:
    """Tests for solve_neumann."""

    def test_eigenfunction(self):
        """cos(pi*x/L) is the n=1 eigenfunction; error should be machine eps."""
        N, L = 64, 1.0
        x, _ = make_grid_neumann(N, L)
        rho = np.cos(np.pi * x / L)
        phi_exact = -(L / np.pi) ** 2 * rho

        phi = solve_neumann(rho, L)

        assert np.allclose(phi, phi_exact, atol=1e-12), (
            f"Max error = {linf_error(phi, phi_exact):.2e}"
        )

    def test_zero_mean_phi(self):
        """Gauge fix: mean(phi) should be zero regardless of mean(rho)."""
        N, L = 64, 1.0
        x, _ = make_grid_neumann(N, L)
        rho = np.cos(np.pi * x / L) + 2.0   # non-zero mean

        phi = solve_neumann(rho, L)

        assert np.isclose(np.mean(phi), 0.0, atol=1e-13)

    def test_constant_source(self):
        """Constant rho (non-zero mean) is silently handled; mean(phi) == 0."""
        N, L = 64, 1.0
        x, _ = make_grid_neumann(N, L)
        rho = np.ones(N)

        phi = solve_neumann(rho, L)

        assert np.isclose(np.mean(phi), 0.0, atol=1e-13)

    def test_two_mode_analytic(self):
        """Sum of two DCT eigenfunctions has analytic solution."""
        N, L = 128, 1.0
        x, _ = make_grid_neumann(N, L)
        n1, n2 = 2, 4
        rho = np.cos(n1 * np.pi * x / L) + np.cos(n2 * np.pi * x / L)
        phi_exact = (
            -(L / (n1 * np.pi)) ** 2 * np.cos(n1 * np.pi * x / L)
            - (L / (n2 * np.pi)) ** 2 * np.cos(n2 * np.pi * x / L)
        )

        phi = solve_neumann(rho, L)
        assert np.allclose(phi, phi_exact, atol=1e-12)


# ---------------------------------------------------------------------------
# Isolated BC tests
# ---------------------------------------------------------------------------


def _gaussian_source(x, x0, sigma):
    """Normalised Gaussian: integrates to 1."""
    return np.exp(-0.5 * ((x - x0) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def _direct_quadrature(rho, x, h):
    """Reference solution via direct summation: phi[i] = sum_j G(x[i]-x[j])*rho[j]*h."""
    N = len(x)
    phi = np.zeros(N)
    for i in range(N):
        r = np.abs(x[i] - x)
        phi[i] = np.sum(r / 2.0 * rho) * h
    return phi


class TestIsolated:
    """Tests for solve_isolated."""

    def test_vs_direct_quadrature(self):
        """FFT convolution matches direct O(N^2) quadrature to 1e-6."""
        N, L = 64, 4.0
        x, h = make_grid_isolated(N, L)
        sigma = 0.3
        rho = _gaussian_source(x, L / 2, sigma)

        phi_fft = solve_isolated(rho, L, baseline="zero_first")
        phi_ref = _direct_quadrature(rho, x, h)
        phi_ref -= phi_ref[0]               # match baseline

        assert np.allclose(phi_fft, phi_ref, atol=1e-6), (
            f"Max error = {linf_error(phi_fft, phi_ref):.2e}"
        )

    def test_convergence(self):
        """Isolated solver converges at ~ O(h^2) vs. direct quadrature."""
        L = 4.0
        sigma = 0.3
        Ns = [32, 64, 128, 256]
        errors = []

        # Use the finest grid direct quadrature as the reference
        N_ref = 512
        x_ref, h_ref = make_grid_isolated(N_ref, L)
        rho_ref = _gaussian_source(x_ref, L / 2, sigma)
        phi_ref_fine = _direct_quadrature(rho_ref, x_ref, h_ref)
        phi_ref_fine -= phi_ref_fine[0]

        for N in Ns:
            x, h = make_grid_isolated(N, L)
            rho = _gaussian_source(x, L / 2, sigma)
            phi = solve_isolated(rho, L, baseline="zero_first")

            # Interpolate reference to coarser grid for comparison
            # (simply sub-sample since N_ref is a multiple of all Ns)
            stride = N_ref // N
            phi_ref_sub = phi_ref_fine[::stride][:N]
            errors.append(linf_error(phi, phi_ref_sub))

        slope, _ = convergence_rate(Ns, errors)
        assert slope < -1.5, f"Expected slope < -1.5, got {slope:.2f}"

    def test_baseline_options_differ_by_constant(self):
        """All baseline options give results that differ by a constant only."""
        N, L = 64, 4.0
        x, h = make_grid_isolated(N, L)
        rho = _gaussian_source(x, L / 2, 0.3)

        phi_zf = solve_isolated(rho, L, baseline="zero_first")
        phi_zm = solve_isolated(rho, L, baseline="zero_mean")
        phi_no = solve_isolated(rho, L, baseline="none")

        # Differences should be spatially constant (std ~ 0)
        assert np.std(phi_zf - phi_zm) < 1e-12
        assert np.std(phi_zf - phi_no) < 1e-12

    def test_invalid_baseline_raises(self):
        """An unknown baseline value raises ValueError."""
        N, L = 32, 1.0
        x, _ = make_grid_isolated(N, L)
        rho = np.ones(N)
        with pytest.raises(ValueError, match="baseline"):
            solve_isolated(rho, L, baseline="bad_option")


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------


class TestUtils:
    """Tests for grid constructors, error norms, and convergence_rate."""

    def test_convergence_rate_known_order(self):
        """convergence_rate recovers slope = -2 for errors ~ C/N^2."""
        Ns = np.array([10, 20, 40, 80], dtype=float)
        C = 5.0
        errors = C / Ns ** 2
        slope, coeffs = convergence_rate(Ns, errors)
        assert abs(slope - (-2.0)) < 0.05, f"Slope = {slope:.4f}, expected -2.0"

    def test_grid_periodic_length(self):
        x, h = make_grid_periodic(64, 1.0)
        assert len(x) == 64
        assert np.isclose(h, 1.0 / 64)
        assert np.isclose(x[0], 0.0)
        assert x[-1] < 1.0              # endpoint NOT included

    def test_grid_dirichlet_length(self):
        """make_grid_dirichlet(N, L) returns N-1 interior points."""
        x, h = make_grid_dirichlet(64, 1.0)
        assert len(x) == 63
        assert np.isclose(h, 1.0 / 64)
        assert np.isclose(x[0], h)     # first interior point at h, not 0

    def test_grid_neumann_length_and_offset(self):
        x, h = make_grid_neumann(64, 1.0)
        assert len(x) == 64
        assert np.isclose(x[0], 0.5 * h)   # cell-centred offset

    def test_grid_isolated_same_as_periodic(self):
        x_iso, h_iso = make_grid_isolated(64, 1.0)
        x_per, h_per = make_grid_periodic(64, 1.0)
        assert np.allclose(x_iso, x_per)
        assert np.isclose(h_iso, h_per)

    def test_l2_error_zero(self):
        phi = np.array([1.0, 2.0, 3.0])
        assert l2_error(phi, phi) == 0.0

    def test_linf_error_known(self):
        phi = np.array([0.0, 0.0, 1.0])
        phi_exact = np.zeros(3)
        assert np.isclose(linf_error(phi, phi_exact), 1.0)
