# 9. Multi-Dimensional Poisson Solver — Periodic Boundary Conditions

## From 1D to $d$ Dimensions

The 1D derivation in [03_1d_periodic.md](03_1d_periodic.md) used a single
fact: the DFT diagonalises $d^2/dx^2$.  In $d$ dimensions, the
$d$-dimensional DFT diagonalises the full Laplacian

$$
\nabla^2 \phi = \sum_{i=1}^{d} \frac{\partial^2 \phi}{\partial x_i^2} = \rho(\mathbf{x})
$$

because the Laplacian is a *sum* of one-dimensional second derivatives, and
the $d$-dimensional DFT is a *product* of 1D DFTs applied along each axis
independently.

---

## Setup

**Domain:** $[0, L_x) \times [0, L_y) \times \cdots$ — periodic in every direction.

**Grid:** $N_x \times N_y \times \cdots$ vertex-centred points with spacings

$$
h_i = \frac{L_i}{N_i}, \qquad x_j^{(i)} = j\,h_i, \quad j = 0,\ldots,N_i - 1
$$

In 2D this gives a rectangular grid $(x_j, y_k)$; in 3D a cuboid grid
$(x_j, y_k, z_l)$.  Isotropy ($L_x = L_y = \cdots$, $N_x = N_y = \cdots$) is
**not** required.

---

## Derivation

Apply the $d$-dimensional DFT to both sides of $\nabla^2\phi = \rho$.
The DFT is separable along each axis, so

$$
\widehat{\nabla^2\phi}_{\mathbf{n}} = \left(-k_{n_x}^2 - k_{n_y}^2 - \cdots\right) \hat\phi_{\mathbf{n}} = \hat\rho_{\mathbf{n}}
$$

where the multi-index $\mathbf{n} = (n_x, n_y, \ldots)$ and the angular
wavenumbers are

$$
k_{n_i} = \frac{2\pi\, n_i}{L_i}, \qquad n_i = 0, \pm 1, \ldots, \pm\frac{N_i}{2}
$$

Defining the **total squared wavenumber**

$$
|\mathbf{k}|^2 = k_{n_x}^2 + k_{n_y}^2 + \cdots
$$

the spectral division becomes identical to the 1D case:

$$
\hat\phi_{\mathbf{n}} = \frac{\hat\rho_{\mathbf{n}}}{-|\mathbf{k}|^2}
$$

The inverse $d$-dimensional DFT then recovers $\phi$ on the grid.

---

## Gauge freedom in $d$ dimensions

The mode $\mathbf{n} = \mathbf{0}$ (all wavenumbers zero) gives
$|\mathbf{k}|^2 = 0$.  As in 1D, this is a singularity corresponding to the
undetermined additive constant in $\phi$.  We apply the same fix:

```python
rho_hat.flat[0] = 0.0   # project out the mean source
lambda_k.flat[0] = 1.0  # avoid division by zero
```

This enforces $\langle\phi\rangle = 0$ — the zero-mean gauge.

---

## Building $|\mathbf{k}|^2$ via broadcasting

The key numerical step is constructing the array $|\mathbf{k}|^2$ without
nested loops or meshgrid calls for every pair of axes.  We accumulate the
contribution from each axis using NumPy broadcasting:

```python
k2 = np.zeros(shape)
for d, (N, L) in enumerate(zip(shape, Ls)):
    h    = L / N
    freq = np.fft.fftfreq(N, d=h)      # 1D array of length N
    k    = 2.0 * np.pi * freq
    # Reshape k to broadcast along axis d only
    slices       = [np.newaxis] * ndim
    slices[d]    = slice(None)
    k2 = k2 + k[tuple(slices)] ** 2
```

For a 2D grid of shape $(N_x, N_y)$, the loop runs twice:
- $d=0$: `k` has shape $(N_x,)$, reshaped to $(N_x, 1)$ — broadcasts over $y$
- $d=1$: `k` has shape $(N_y,)$, reshaped to $(1, N_y)$ — broadcasts over $x$

The result is the same as `np.add.outer(kx**2, ky**2)` in 2D, but generalises
to any number of dimensions without code changes.

---

## Step-by-step algorithm

1. **Forward nD FFT:** $\hat\rho = \text{FFTN}(\rho)$
2. **Build $|\mathbf{k}|^2$** via the broadcasting loop above
3. **Set eigenvalue array:** $\lambda = -|\mathbf{k}|^2$
4. **Gauge fix:** `rho_hat.flat[0] = 0`, `lambda.flat[0] = 1`
5. **Divide in spectral space:** $\hat\phi = \hat\rho / \lambda$
6. **Inverse nD FFT:** $\phi = \text{Re}\bigl[\text{IFFTN}(\hat\phi)\bigr]$

---

## Python implementation

```python
from scipy.fft import fftn, ifftn
import numpy as np

def solve_periodic_nd(rho, Ls):
    rho   = np.asarray(rho, dtype=float)
    ndim  = rho.ndim
    shape = rho.shape
    Ls    = [Ls] * ndim if np.isscalar(Ls) else list(Ls)

    rho_hat = fftn(rho)

    k2 = np.zeros(shape)
    for d, (N, L) in enumerate(zip(shape, Ls)):
        h    = L / N
        freq = np.fft.fftfreq(N, d=h)
        k    = 2.0 * np.pi * freq
        slices       = [np.newaxis] * ndim
        slices[d]    = slice(None)
        k2 += k[tuple(slices)] ** 2

    lambda_k          = -k2
    rho_hat.flat[0]   = 0.0
    lambda_k.flat[0]  = 1.0

    return np.real(ifftn(rho_hat / lambda_k))
```

See `src/poissonnd.py:solve_periodic_nd` for the full implementation with
docstring and type annotations.

---

## Analytic test case — 2D

Choose a source that is a tensor product of 1D eigenfunctions:

$$
\rho(x,y) = \sin\!\left(\frac{2\pi x}{L_x}\right) \cos\!\left(\frac{4\pi y}{L_y}\right)
$$

With $k_x = 2\pi/L_x$ and $k_y = 4\pi/L_y$, the exact solution is

$$
\phi_{\rm exact}(x,y) = \frac{-\rho(x,y)}{k_x^2 + k_y^2}
$$

Because this source occupies a single Fourier mode, the FFT represents it
exactly and the solver achieves machine precision independent of $N_x, N_y$.

```python
Nx, Ny = 64, 64
Lx, Ly = 1.0, 1.0
x = np.linspace(0, Lx, Nx, endpoint=False)
y = np.linspace(0, Ly, Ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')

kx = 2*np.pi/Lx;  ky = 4*np.pi/Ly
rho = np.sin(kx*X) * np.cos(ky*Y)
phi_exact = -rho / (kx**2 + ky**2)

phi = solve_periodic_nd(rho, (Lx, Ly))
print(np.max(np.abs(phi - phi_exact)))   # ~ 1e-17
```

---

## Analytic test case — 3D

The extension to 3D is identical:

$$
\rho(x,y,z) = \sin(k_x x)\sin(k_y y)\sin(k_z z)
\qquad\Longrightarrow\qquad
\phi_{\rm exact} = \frac{-\rho}{k_x^2 + k_y^2 + k_z^2}
$$

```python
Nx = Ny = Nz = 32;  L = 1.0
x = np.linspace(0, L, Nx, endpoint=False)
X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
k = 2*np.pi/L

rho = np.sin(k*X) * np.sin(k*Y) * np.sin(k*Z)
phi = solve_periodic_nd(rho, L)
print(np.max(np.abs(phi - (-rho / (3*k**2)))))   # ~ 1e-17
```

---

## Convergence

For a smooth, periodic source composed of many Fourier modes, the periodic
solver converges **spectrally** (exponentially fast in $N$) — just as in 1D.
For a single-mode source, the error is at machine precision regardless of $N$
because the DFT represents band-limited functions exactly.

For non-smooth sources (e.g., a step function or a sharp spike), Gibbs-type
oscillations appear and convergence degrades to algebraic rates; see
[07_convergence.md](07_convergence.md) for a detailed discussion.

---

## Complexity and memory

| Grid | Points | FFT complexity | Memory |
|---|---|---|---|
| 1D | $N$ | $\mathcal{O}(N\log N)$ | $\mathcal{O}(N)$ |
| 2D | $N^2$ | $\mathcal{O}(N^2\log N)$ | $\mathcal{O}(N^2)$ |
| 3D | $N^3$ | $\mathcal{O}(N^3\log N)$ | $\mathcal{O}(N^3)$ |

In 3D, a $256^3$ grid holds $\sim 16\times 10^6$ cells.  With two
double-precision arrays ($\rho$ and $\phi$) plus a complex workspace, this
requires roughly 400 MB — manageable on a modern workstation.  For
$512^3$ ($\sim 3\times 10^8$ cells) you need $\sim 3$ GB.

`scipy.fft.fftn` uses the same `pocketfft` backend as `numpy.fft` but
accepts a `workers` argument for multi-threaded execution:

```python
from scipy.fft import fftn, ifftn
phi_hat = fftn(rho, workers=-1)   # use all available CPU cores
```
