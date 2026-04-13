# 7. Error Analysis and Convergence

This document explains how to measure, interpret, and predict the accuracy
of the four FFT Poisson solvers as the grid is refined.

---

## 7.1 Types of convergence

### Spectral (exponential) convergence

For a function with infinitely many continuous derivatives (an *analytic*
function), the Fourier coefficients decay *exponentially* with mode number:
$|\hat{f}_n| \lesssim C e^{-\alpha n}$.  Truncating at $n = N/2$ therefore
introduces an error that also decays exponentially:

$$
\|\phi - \phi_{\rm exact}\|_{\infty} \lesssim C_1 e^{-\alpha N}
$$

In practice, for the eigenfunction test cases ($\rho = \sin$ or $\cos$) there
is **only one nonzero mode**, so the truncation error is *identically zero*.
The only error is floating-point rounding ($\sim 10^{-15}$), independent of
grid size.

### Algebraic ($\mathcal{O}(h^p)$) convergence

When the solver introduces a *discretization approximation* — either through
modified wavenumbers (matching a finite-difference Laplacian) or through the
trapezoidal-rule quadrature in the isolated solver — the error decays
algebraically:

$$
\|\phi - \phi_{\rm exact}\|_{\infty} \sim C h^p = C (L/N)^p
$$

On a log-log plot of error vs $N$, the slope is $-p$.

---

## 7.2 Expected convergence rates

| Solver | Source type | Convergence |
|---|---|---|
| Periodic (spectral $\lambda$) | Single sine mode | Machine precision ($\sim 10^{-15}$), $N$-independent |
| Periodic (spectral $\lambda$) | Multi-mode analytic | Exponential in $N$ |
| Dirichlet (spectral $\lambda$) | Single sine mode | Machine precision |
| Dirichlet (modified $\lambda$) | Any smooth $\rho$ | $\mathcal{O}(h^2) = \mathcal{O}(N^{-2})$ |
| Neumann (spectral $\lambda$) | Single cosine mode | Machine precision |
| Isolated | Smooth $\rho$ | $\mathcal{O}(h^2) = \mathcal{O}(N^{-2})$ |

---

## 7.3 Why eigenfunctions give machine precision

Consider the periodic solver with $\rho = \sin(2\pi x / L)$.  This has
exactly two nonzero DFT coefficients: $\hat{\rho}_1 = -iN/2$ and
$\hat{\rho}_{N-1} = iN/2$ (in NumPy's ordering).  The division step is

$$
\hat{\phi}_1 = \frac{\hat{\rho}_1}{-(2\pi/L)^2}, \qquad \hat{\phi}_{N-1} = \frac{\hat{\rho}_{N-1}}{-(2\pi/L)^2}
$$

These are single floating-point multiplications.  The inverse FFT then
reconstructs $\sin(2\pi x/L)$ from two nonzero bins — again exactly, up to
the rounding in the FFT butterfly operations ($\sim 10^{-16}$).

The same argument applies to the DST-I and DCT-II solvers with single-mode
inputs.

---

## 7.4 Measuring convergence empirically

Given a set of grid sizes $\{N_i\}$ and corresponding errors $\{e_i\}$:

1. Compute errors: `e_i = linf_error(solve(...), phi_exact)` for each $N_i$
2. Fit a line to $\log e_i$ vs $\log N_i$ using least squares
3. The slope is $-p$ (the convergence order)

```python
from src.utils import convergence_rate

Ns     = [16, 32, 64, 128, 256]
errors = [...]                    # compute for each N

slope, coeffs = convergence_rate(Ns, errors)
print(f"Convergence order: {-slope:.1f}")
```

A slope of $-2$ (order 2) means halving $h$ reduces the error by a factor of
4.  A slope of $-3$ means a factor of 8, and so on.

---

## 7.5 The Gibbs phenomenon (a cautionary note)

If $\rho$ has a *discontinuity* (e.g. a step function), the Fourier
coefficients decay only as $|\hat{\rho}_n| \sim 1/n$ (slowly), and the
spectral solver exhibits the **Gibbs phenomenon**: near the discontinuity,
the approximation overshoots by $\sim 9\%$ regardless of $N$.

For such sources, the spectral solver converges only as $\mathcal{O}(h)$ in
the max norm (first order), much slower than the $\mathcal{O}(h^2)$ one
might expect from a second-order finite-difference method.

> This repository focuses on smooth sources.  For discontinuous $\rho$,
> consider: (a) smoothing the source before solving, (b) using a finite-
> difference solver, or (c) working in weak/integral form.

---

## 7.6 Verification via the convergence study example

```bash
python examples/convergence_study.py
```

This script generates a log-log plot (`convergence_study.png`) with:
- A **flat line** near machine epsilon for eigenfunction tests (all four solvers)
- A **slope $-2$ line** for the modified-wavenumber Dirichlet and isolated solvers
- A printed table of measured slopes for comparison with theory

---

## 7.7 Richardson extrapolation

When no analytic solution is available, Richardson extrapolation estimates the
error from two numerical solutions on grids of size $N$ and $2N$:

$$
\phi_{\rm extrap} = \frac{2^p \phi_{2N} - \phi_N}{2^p - 1} + \mathcal{O}(h^{2p})
$$

where $p$ is the convergence order.  For our $\mathcal{O}(h^2)$ solvers this
doubles the effective order.

```python
from src.utils import richardson_extrapolate

phi_N   = solve_isolated(rho_N, L)
phi_2N  = solve_isolated(rho_2N, L)[::2]  # sub-sample to coarse grid

phi_extrap, err = richardson_extrapolate(phi_2N, phi_N, order=2)
```
