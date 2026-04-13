# 5. 1D Poisson Solver — Neumann Boundary Conditions

## Setup

**Boundary conditions:**

$$
\frac{d\phi}{dx}\bigg|_{x=0} = 0, \qquad \frac{d\phi}{dx}\bigg|_{x=L} = 0
$$

**Grid:** $N$ *cell-centred* points

$$
x_j = \left(j + \tfrac{1}{2}\right) h, \quad j = 0, 1, \ldots, N-1, \quad h = \frac{L}{N}
$$

Cell-centred points are the natural choice for Neumann BCs because they place
no grid point at the boundary — the flux condition is implicit in the
even-extension symmetry used by the DCT-II.

**Equation:**

$$
\frac{d^2\phi}{dx^2} = \rho(x), \qquad \phi'(0) = 0,\ \phi'(L) = 0
$$

---

## Eigenfunctions of the Neumann Laplacian

Which functions satisfy $d^2f/dx^2 = \lambda f$ with $f'(0) = f'(L) = 0$?

The cosine family:

$$
f_n(x) = \cos\!\left(\frac{n\pi x}{L}\right), \quad n = 0, 1, 2, \ldots
$$

with eigenvalues

$$
\lambda_n = -\left(\frac{n\pi}{L}\right)^2
$$

The $n=0$ mode $f_0 = 1$ is the spatially uniform (mean) component, with
$\lambda_0 = 0$ — the same gauge singularity we encountered in the periodic case.

---

## From continuous to discrete: the DCT-II

On the $N$-point cell-centred grid, the natural transform is the
**Discrete Cosine Transform type II (DCT-II)**:

$$
\hat{f}_n = 2\sum_{j=0}^{N-1} f_j \cos\!\left(\frac{\pi (2j+1) n}{2N}\right), \quad n = 0, \ldots, N-1
$$

The inverse (DCT-III, often called IDCT-II) is

$$
f_j = \frac{1}{2N}\left[\hat{f}_0 + 2\sum_{n=1}^{N-1} \hat{f}_n \cos\!\left(\frac{\pi (2j+1) n}{2N}\right)\right]
$$

with the $1/(2N)$ normalization handled by `scipy.fft.idct(type=2)`.

---

## Why a cell-centred grid?

The DCT-II implicitly assumes that the function is *even* about $x = -h/2$
and $x = L + h/2$ (the cell edges).  An even extension automatically
satisfies $f'(0) = 0$ and $f'(L) = 0$ without any extra work.

A *vertex-centred* grid (like the periodic and Dirichlet grids) would put
points at $x=0$ and $x=L$.  For Neumann BCs the derivative condition would
then need to be enforced explicitly; the cell-centred grid makes it automatic.

---

## Solving Poisson's equation

The DST-I expansion of $\phi$ and $\rho$ in the cosine basis:

$$
-\left(\frac{n\pi}{L}\right)^2 \hat{\phi}_n = \hat{\rho}_n
\quad \Longrightarrow \quad
\hat{\phi}_n = \frac{\hat{\rho}_n}{\lambda_n}
$$

**Gauge fix:** at $n=0$, $\lambda_0 = 0$.  We set $\hat{\phi}_0 = 0$
(equivalently, $\hat{\rho}_0 = 0$), enforcing $\langle\phi\rangle = 0$.

---

## Step-by-step algorithm

1. **Grid:** $x_j = (j+\tfrac{1}{2})h$, $j = 0, \ldots, N-1$, $h = L/N$
2. **Evaluate source:** $\rho_j = \rho(x_j)$ — array of length $N$
3. **Forward DCT-II:** $\hat{\rho}_n = \texttt{dct}(\rho,\, \text{type=2})$
4. **Build eigenvalues:** $n = [0, 1, \ldots, N-1]$, $\lambda_n = -(n\pi/L)^2$
5. **Gauge fix:** $\hat{\rho}_0 = 0$, $\lambda_0 = 1$ (avoid division by zero)
6. **Divide:** $\hat{\phi}_n = \hat{\rho}_n / \lambda_n$
7. **Inverse DCT-II:** $\phi_j = \texttt{idct}(\hat{\phi},\, \text{type=2})$

---

## Python implementation

```python
from scipy.fft import dct, idct
import numpy as np

def solve_neumann(rho, L):
    N   = len(rho)
    n   = np.arange(N, dtype=float)

    rho_hat     = dct(rho, type=2)
    lambda_n    = -(n * np.pi / L)**2
    rho_hat[0]  = 0.0
    lambda_n[0] = 1.0            # gauge fix

    return idct(rho_hat / lambda_n, type=2)
```

See `src/poisson1d.py:solve_neumann` for the full implementation.

---

## scipy normalization note

`scipy.fft.dct(x, type=2)` is unnormalized: the $n=0$ term picks up a
factor of $1$, higher terms a factor of $2$, but there is no $1/N$.
`scipy.fft.idct(x, type=2)` divides by $2N$.

Always use `idct` for the inverse — do not manually apply the $1/(2N)$
factor, which is easy to confuse with $1/(2(N+1))$ from the DST-I.

---

## Analytic test case

Source:

$$
\rho(x) = \cos\!\left(\frac{\pi x}{L}\right)
$$

Exact solution:

$$
\phi_{\rm exact}(x) = -\left(\frac{L}{\pi}\right)^2 \cos\!\left(\frac{\pi x}{L}\right)
$$

This is the $n=1$ eigenfunction evaluated at cell-centred points.

```python
N, L = 64, 1.0
h    = L / N
x    = (np.arange(N) + 0.5) * h
rho  = np.cos(np.pi * x / L)
phi  = solve_neumann(rho, L)
phi_exact = -(L/np.pi)**2 * np.cos(np.pi * x / L)
print(np.max(np.abs(phi - phi_exact)))   # ~1e-14
```

---

## Comparison with the Dirichlet solver

| | Dirichlet | Neumann |
|---|---|---|
| BC | $\phi(0)=\phi(L)=0$ | $\phi'(0)=\phi'(L)=0$ |
| Grid | Vertex-centred interior | Cell-centred |
| Transform | DST-I | DCT-II |
| Eigenfunctions | $\sin(n\pi x/L)$ | $\cos(n\pi x/L)$ |
| Zero mode | None (all $\lambda_n < 0$) | $\lambda_0=0$ (gauge fix) |
| Gauge ambiguity | No | Yes ($\phi + C$ is also a solution) |
