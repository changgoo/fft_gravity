# 3. 1D Poisson Solver — Periodic Boundary Conditions

## Setup

**Domain:** $[0, L)$ (the left and right endpoints are identified — the grid
wraps around).

**Grid:** $N$ uniformly-spaced *vertex-centred* points

$$
x_j = j\, h, \quad j = 0, 1, \ldots, N-1, \quad h = \frac{L}{N}
$$

Because $x_j$ and $x_{j+N}$ represent the same physical point, the function
values satisfy $f_0 = f_N$ automatically — the endpoint is *not* stored.

**Equation:**

$$
\frac{d^2\phi}{dx^2} = \rho(x), \qquad \phi(0) = \phi(L), \quad \phi'(0) = \phi'(L)
$$

---

## Derivation

Apply the DFT to both sides of $d^2\phi/dx^2 = \rho$.  The derivative theorem
(Section 2 of [02_fourier_fundamentals.md](02_fourier_fundamentals.md)) gives

$$
-k_n^2\, \hat{\phi}_n = \hat{\rho}_n
$$

where the angular wavenumbers are

$$
k_n = \frac{2\pi n}{L}, \quad n = 0, \pm 1, \pm 2, \ldots, \pm\frac{N}{2}
$$

(In practice NumPy stores these as $n = 0, 1, \ldots, N/2-1, -N/2, \ldots, -1$.)

Solving for $\hat{\phi}_n$:

$$
\hat{\phi}_n = \frac{\hat{\rho}_n}{-k_n^2}
$$

Applying the inverse DFT recovers $\phi_j$.

---

## The zero mode and gauge freedom

At $n = 0$ we have $k_0 = 0$, so the denominator $-k_0^2 = 0$ and the
equation is singular.

Physically, the $n=0$ mode corresponds to the *spatially uniform* part of
$\phi$.  The Poisson equation $d^2\phi/dx^2 = \rho$ cannot determine this
constant — if $\phi(x)$ is a solution then so is $\phi(x) + C$ for any
constant $C$.  This is **gauge freedom**.

There is also a *solvability condition*: for a periodic $\phi$ to exist,
the average source must vanish:

$$
\int_0^L \rho(x)\, dx = 0 \quad \Longleftrightarrow \quad \hat{\rho}_0 = 0
$$

**Implementation:** rather than raising an exception when $\hat{\rho}_0 \neq 0$,
we zero out the $n=0$ mode before dividing:

```python
rho_hat[0] = 0.0   # project onto zero-mean subspace AND fix gauge
lambda_k[0] = 1.0  # avoid ZeroDivisionError; numerator is 0
```

This simultaneously enforces $\langle\rho\rangle = 0$ and $\langle\phi\rangle = 0$.

---

## Step-by-step algorithm

1. **Allocate grid:** $x_j = jh$, $h = L/N$
2. **Evaluate source:** compute $\rho_j = \rho(x_j)$
3. **Forward FFT:** $\hat{\rho}_n = \text{FFT}(\rho)$
4. **Build wavenumbers:** $k_n = 2\pi \cdot \texttt{fftfreq}(N, d=h)$
5. **Build eigenvalues:** $\lambda_n = -k_n^2$; set $\lambda_0 = 1$, $\hat{\rho}_0 = 0$
6. **Divide:** $\hat{\phi}_n = \hat{\rho}_n / \lambda_n$
7. **Inverse FFT:** $\phi_j = \text{Re}\bigl[\text{IFFT}(\hat{\phi})\bigr]$

> **Why take `np.real`?** For a real-valued source $\rho$, the exact solution
> $\phi$ is real.  Floating-point rounding leaves imaginary residuals of order
> $10^{-16}$ in `ifft`'s output; calling `.real` removes these cleanly without
> hiding any real bugs.

---

## Python implementation

```python
from scipy.fft import fft, ifft
import numpy as np

def solve_periodic(rho, L):
    N = len(rho)
    h = L / N
    rho_hat   = fft(rho)
    k         = 2 * np.pi * np.fft.fftfreq(N, d=h)
    lambda_k  = -(k ** 2)
    rho_hat[0] = 0.0
    lambda_k[0] = 1.0
    return np.real(ifft(rho_hat / lambda_k))
```

See `src/poisson1d.py:solve_periodic` for the full implementation with
docstring and type annotations.

---

## Analytic test case

Choose a source that is an exact eigenfunction of the periodic Laplacian:

$$
\rho(x) = \sin\!\left(\frac{2\pi x}{L}\right)
$$

The exact solution is

$$
\phi_{\rm exact}(x) = -\frac{\sin(2\pi x/L)}{(2\pi/L)^2} = -\left(\frac{L}{2\pi}\right)^2 \sin\!\left(\frac{2\pi x}{L}\right)
$$

Because this source has a single Fourier mode ($n=1$), the FFT represents it
*exactly* (up to floating-point precision), and the entire solve reduces to
one division.  The error is therefore at machine precision ($\sim 10^{-15}$),
independent of $N$.

```python
N, L = 64, 1.0
x    = np.arange(N) * L/N
rho  = np.sin(2*np.pi*x/L)
phi  = solve_periodic(rho, L)
phi_exact = -(L/(2*np.pi))**2 * np.sin(2*np.pi*x/L)
print(np.max(np.abs(phi - phi_exact)))   # ~1e-15
```
