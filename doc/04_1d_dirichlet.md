# 4. 1D Poisson Solver — Dirichlet Boundary Conditions

## Setup

**Boundary conditions:**

$$
\phi(0) = 0, \qquad \phi(L) = 0
$$

**Grid:** $N-1$ *interior* vertex-centred points

$$
x_j = j\,h, \quad j = 1, \ldots, N-1, \quad h = \frac{L}{N}
$$

The boundary values $\phi(0) = \phi(L) = 0$ are *implicit* — the solver
operates only on the $N-1$ interior values of $\rho$ (and returns $N-1$
interior values of $\phi$).

**Equation:**

$$
\frac{d^2\phi}{dx^2} = \rho(x), \qquad \phi(0) = 0,\ \phi(L) = 0
$$

---

## Eigenfunctions of the Dirichlet Laplacian

Which functions satisfy both $d^2f/dx^2 = \lambda f$ and $f(0) = f(L) = 0$?

The answer is the sine family:

$$
f_n(x) = \sin\!\left(\frac{n\pi x}{L}\right), \quad n = 1, 2, 3, \ldots
$$

with eigenvalues

$$
\lambda_n = -\left(\frac{n\pi}{L}\right)^2
$$

These functions form a complete orthogonal basis on $(0, L)$ with the Dirichlet
condition built in.  Any function satisfying $f(0) = f(L) = 0$ can be expanded
in this basis.

---

## From continuous to discrete: the DST-I

On a grid of $N-1$ interior points (with $N$ cells of width $h = L/N$), the
discrete version of the sine expansion is the **Discrete Sine Transform type I
(DST-I)**:

$$
\hat{f}_n = \sum_{j=1}^{N-1} f_j \sin\!\left(\frac{\pi j n}{N}\right), \quad n = 1, \ldots, N-1
$$

The inverse relation is

$$
f_j = \frac{1}{N} \sum_{n=1}^{N-1} \hat{f}_n \sin\!\left(\frac{\pi j n}{N}\right)
$$

(with the $1/N$ normalization handled automatically by `scipy.fft.idst(type=1)`).

---

## Solving Poisson's equation

Expanding $\phi$ and $\rho$ in the DST-I basis:

$$
-\left(\frac{n\pi}{L}\right)^2 \hat{\phi}_n = \hat{\rho}_n
\quad \Longrightarrow \quad
\hat{\phi}_n = \frac{\hat{\rho}_n}{\lambda_n}
$$

There is **no gauge ambiguity** here: all eigenvalues $\lambda_n < 0$ are
non-zero (the $n=0$ constant mode does not exist with Dirichlet BCs).

---

## Spectral vs modified wavenumber eigenvalues

### Spectral eigenvalues (default)

$$
\lambda_n = -\left(\frac{n\pi}{L}\right)^2, \quad n = 1, \ldots, N-1
$$

These are the exact eigenvalues of the continuous Laplacian.  When $\rho$ is a
single sine mode, the solve is exact at machine precision.  For smooth
multi-mode sources, the error decays *exponentially* with $N$.

### Modified wavenumber (FD-consistent)

The 3-point finite-difference Laplacian acting on $\sin(n\pi x/L)$ gives

$$
\frac{\phi_{j+1} - 2\phi_j + \phi_{j-1}}{h^2} = -\left(\frac{2}{h}\sin\frac{n\pi h}{2L}\right)^2 \phi_j
$$

Using the modified eigenvalue

$$
\lambda_n^{\rm mod} = -\left(\frac{2}{h}\sin\frac{n\pi h}{2L}\right)^2 = -\left(\frac{2}{h}\sin\frac{n\pi}{2N}\right)^2
$$

makes the FFT solve *exactly* equivalent to inverting the standard
second-order finite-difference matrix.  This gives $\mathcal{O}(h^2)$
convergence for arbitrary smooth $\rho$.

> **When to use which?**  Use spectral eigenvalues when you want the best
> possible accuracy.  Use modified wavenumbers when you want your FFT result
> to match a finite-difference simulation exactly (e.g. for comparing codes).

---

## Step-by-step algorithm

1. **Grid:** $x_j = jh$, $j = 1, \ldots, N-1$, $h = L/N$
2. **Evaluate source:** $\rho_j = \rho(x_j)$ — array of length $N-1$
3. **Forward DST-I:** $\hat{\rho}_n = \texttt{dst}(\rho,\, \text{type=1})$
4. **Build eigenvalues:** $n = [1, 2, \ldots, N-1]$
   - Spectral: $\lambda_n = -(n\pi/L)^2$
   - Modified: $\lambda_n = -(2/h \cdot \sin(n\pi/(2N)))^2$
5. **Divide:** $\hat{\phi}_n = \hat{\rho}_n / \lambda_n$
6. **Inverse DST-I:** $\phi_j = \texttt{idst}(\hat{\phi},\, \text{type=1})$

---

## Python implementation

```python
from scipy.fft import dst, idst
import numpy as np

def solve_dirichlet(rho, L, modified_wavenumber=False):
    Nm1 = len(rho)          # number of interior points
    N   = Nm1 + 1           # number of cells
    h   = L / N
    n   = np.arange(1, N)   # mode indices 1 ... N-1

    rho_hat = dst(rho, type=1)

    if modified_wavenumber:
        lambda_n = -(2 / h * np.sin(n * np.pi / (2 * N)))**2
    else:
        lambda_n = -(n * np.pi / L)**2

    return idst(rho_hat / lambda_n, type=1)
```

See `src/poisson1d.py:solve_dirichlet` for the full implementation.

---

## Important scipy normalization note

`scipy.fft.dst(x, type=1)` applies the **unnormalized** forward transform.
`scipy.fft.idst(x, type=1)` divides by $2(N+1)$, correctly recovering the
original array.

Do **not** reconstruct the inverse by calling `dst` again and dividing
manually — the factor $2(N+1)$ depends on the array length and is easy to get
wrong.  Always use `idst`.

---

## Analytic test case

Source:

$$
\rho(x) = \sin\!\left(\frac{\pi x}{L}\right)
$$

Exact solution:

$$
\phi_{\rm exact}(x) = -\left(\frac{L}{\pi}\right)^2 \sin\!\left(\frac{\pi x}{L}\right)
$$

This is the $n=1$ eigenfunction; the spectral solver returns machine precision.

```python
N, L = 64, 1.0
h    = L / N
x    = np.arange(1, N) * h       # interior points
rho  = np.sin(np.pi * x / L)
phi  = solve_dirichlet(rho, L)
phi_exact = -(L/np.pi)**2 * np.sin(np.pi * x / L)
print(np.max(np.abs(phi - phi_exact)))   # ~1e-14
```

---

## Convergence summary

| Source | Eigenvalues | Expected error |
|---|---|---|
| Single sine mode $\sin(n\pi x/L)$ | Spectral | $\sim 10^{-15}$ (machine eps) |
| Multi-mode smooth $\rho$ | Spectral | Exponential in $N$ |
| Any smooth $\rho$ | Modified | $\mathcal{O}(h^2)$ |
