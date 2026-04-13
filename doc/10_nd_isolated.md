# 10. Multi-Dimensional Poisson Solver — Isolated (Free-Space) Boundary Conditions

## Motivation

The 1D isolated solver ([06_1d_isolated.md](06_1d_isolated.md)) used zero-padding
and a 1D free-space Green's function.  The same strategy extends to $d$
dimensions: zero-pad $\rho$ along every axis, convolve with the $d$-dimensional
free-space Green's function using the nD FFT, and extract the physical subdomain.

This is the **Particle-Mesh (PM)** approach to computing gravitational
potentials, described in detail by Hockney & Eastwood (1988, Ch. 6).  It is
the foundation of $\mathcal{O}(N \log N)$ gravity solvers in cosmological
$N$-body codes.

---

## Free-Space Green's Functions

The Green's function $G(\mathbf{r})$ satisfies

$$
\nabla^2 G(\mathbf{r}) = \delta^{(d)}(\mathbf{r})
$$

Its form depends on dimension:

| $d$ | $G(\mathbf{r})$ | Behaviour at $r \to \infty$ |
|---|---|---|
| 1 | $\displaystyle\frac{|x|}{2}$ | grows (no decay) |
| 2 | $\displaystyle\frac{\ln r}{2\pi}$ | grows logarithmically |
| 3 | $\displaystyle-\frac{1}{4\pi r}$ | decays to zero |

where $r = \|\mathbf{r}\|$.

**Derivation sketch (3D).** In spherical coordinates, $\nabla^2(1/r) = 0$
for $r > 0$.  Gauss's theorem over a small sphere of radius $\epsilon$ gives
$\oint \nabla(1/r) \cdot \hat{n}\, dS = -4\pi$, so $G = -1/(4\pi r)$
satisfies $\nabla^2 G = \delta^{(3)}$ with the correct unit residue.

**Derivation sketch (2D).** In polar coordinates,
$\nabla^2(\ln r) = \nabla^2\ln r = 0$ for $r > 0$.  A line-integral
argument (the 2D analogue of the Gauss law above) gives $G = \ln r / (2\pi)$.

---

## The Singularity at $r = 0$

All three Green's functions are singular at the origin.  When discretised on
a grid of spacing $h$, the origin corresponds to the single cell at
$(j_1, j_2, \ldots) = (0, 0, \ldots)$, which sits at distance zero from
itself.

**Standard PM-code treatment:** set $G[\mathbf{0}] = 0$ and proceed.

This is justified as follows:

- **3D:** the missing self-interaction term contributes an $\mathcal{O}(h^2)$
  error, identical to the quadrature error of the trapezoidal rule.  Setting
  $G[\mathbf{0}] = 0$ therefore has no impact on the overall convergence order.
- **2D:** the missing term is $\mathcal{O}(h^2 |\ln h|)$, which is slightly
  sub-quadratic but still converges to zero and is invisible at the grid sizes
  used in practice.
- **1D:** $G(0) = 0$ exactly (the formula $|x|/2$ at $x=0$ gives 0), so no
  special treatment is needed.

In all cases, setting $G[\mathbf{0}] = 0$ is consistent with fixing the
additive gauge constant of $\phi$.

---

## Zero-Padding in $d$ Dimensions

Extend the grid by a factor of 2 along **every** axis:

$$
(N_x, N_y, \ldots) \;\longrightarrow\; (2N_x, 2N_y, \ldots)
$$

Place $\rho$ in the lower-left hypercorner and fill the rest with zeros:

$$
\rho_{\rm ext}[\mathbf{j}] = \begin{cases} \rho[\mathbf{j}] & 0 \le j_i < N_i \text{ for all } i \\ 0 & \text{otherwise} \end{cases}
$$

**Why this works.**  A linear convolution of two length-$N$ sequences has
length $2N - 1$.  By zero-padding both inputs to length $2N$, circular
(periodic) convolution on the $2N$ grid equals linear convolution for the
first $N$ output points — exactly.  The same argument holds dimension by
dimension for separable operations, making it valid in any number of
dimensions.

The padding factor of 2 is a minimum.  Larger padding factors are correct
but wasteful; smaller ones introduce aliasing errors.

---

## Building the Green's Function on the Extended Grid

On the extended grid of shape $(M_1, M_2, \ldots) = (2N_1, 2N_2, \ldots)$,
define the **nearest-image distance** from the origin to cell
$\mathbf{k} = (k_1, k_2, \ldots)$:

$$
d_i(k_i) = \min(k_i,\; M_i - k_i) \cdot h_i
$$

This is the cyclic distance on the extended periodic grid — the same
min-image convention used in molecular dynamics.  The Euclidean distance is

$$
r(\mathbf{k}) = \sqrt{\sum_i d_i(k_i)^2}
$$

and the Green's function is evaluated at $r(\mathbf{k})$, with
$G[\mathbf{0}] = 0$ as discussed above.

```python
r2 = np.zeros(ext_shape)
for d, (M, h) in enumerate(zip(ext_shape, hs)):
    idx    = np.arange(M)
    dist_d = np.minimum(idx, M - idx) * h
    # Broadcast dist_d along axis d
    slices       = [np.newaxis] * ndim
    slices[d]    = slice(None)
    r2 += dist_d[tuple(slices)] ** 2

r = np.sqrt(r2)   # r[0,...,0] = 0
```

---

## The nD FFT Convolution Formula

The potential is the $d$-dimensional convolution

$$
\phi(\mathbf{x}) = \int G(\mathbf{x} - \mathbf{x}')\, \rho(\mathbf{x}')\, d^d x'
$$

Discretised as a Riemann sum with cell volume $\Delta V = h_1 h_2 \cdots h_d$:

$$
\phi_{\rm ext} = \text{IFFTN}\!\left(\text{FFTN}(G) \cdot \text{FFTN}(\rho_{\rm ext})\right) \cdot \Delta V
$$

The factor $\Delta V = \prod_i h_i$ is the **quadrature weight** for the
$d$-dimensional integral.  For an isotropic grid ($h_i = h$ for all $i$),
this is $h^d$; for an anisotropic grid it is the product of all spacings.

**Important:** do not replace $\prod h_i$ with $h^d$ — this silently breaks
for non-cubic grids.

---

## Step-by-step algorithm

1. **Grid:** $x_j^{(i)} = j h_i$, $h_i = L_i / N_i$
2. **Zero-pad:** embed $\rho$ in shape $(2N_1, 2N_2, \ldots)$; zeros elsewhere
3. **Build $G$:** compute $r(\mathbf{k})$ via nearest-image distances,
   apply the dimension-appropriate formula, set $G[\mathbf{0}] = 0$
4. **FFT convolution:**
   $\phi_{\rm ext} = \text{Re}\bigl[\text{IFFTN}(\text{FFTN}(G) \cdot \text{FFTN}(\rho_{\rm ext}))\bigr] \cdot \prod_i h_i$
5. **Extract:** take the first $N_i$ indices along every axis
6. **Baseline subtraction:** subtract $\phi[\mathbf{0}]$ or $\langle\phi\rangle$
   (not needed in 3D if $\phi \to 0$ is a valid convention, but harmless)

---

## Python implementation

```python
from scipy.fft import fftn, ifftn
import numpy as np

def solve_isolated_nd(rho, Ls, baseline='zero_corner'):
    rho   = np.asarray(rho, dtype=float)
    ndim  = rho.ndim                     # 1, 2, or 3
    shape = rho.shape
    Ls    = [Ls]*ndim if np.isscalar(Ls) else list(Ls)
    hs    = tuple(L/N for L,N in zip(Ls, shape))

    ext_shape = tuple(2*N for N in shape)
    rho_ext   = np.zeros(ext_shape)
    slices    = tuple(slice(0, N) for N in shape)
    rho_ext[slices] = rho

    # Green's function on the extended grid
    r2 = np.zeros(ext_shape)
    for d, (M, h) in enumerate(zip(ext_shape, hs)):
        idx    = np.arange(M)
        dist_d = np.minimum(idx, M - idx) * h
        s = [np.newaxis]*ndim; s[d] = slice(None)
        r2 += dist_d[tuple(s)]**2
    r = np.sqrt(r2)

    if ndim == 1:
        G = r / 2.0
    elif ndim == 2:
        G = np.where(r > 0, np.log(r) / (2*np.pi), 0.0)
    elif ndim == 3:
        G = np.where(r > 0, -1.0 / (4*np.pi*r), 0.0)

    dV      = np.prod(hs)
    phi_ext = np.real(ifftn(fftn(G) * fftn(rho_ext))) * dV
    phi     = phi_ext[slices]

    if baseline == 'zero_corner':
        phi -= phi.flat[0]
    elif baseline == 'zero_mean':
        phi -= np.mean(phi)
    return phi
```

See `src/poissonnd.py:solve_isolated_nd` for the full implementation.

---

## Baseline subtraction in $d$ dimensions

| $d$ | $G(r \to \infty)$ | Need baseline? |
|---|---|---|
| 1 | $\to +\infty$ | Always — $\phi$ has no absolute reference |
| 2 | $\to +\infty$ | Usually — $\phi$ grows logarithmically for non-zero total mass |
| 3 | $\to 0$ | No — $\phi \to 0$ is a natural normalisation |

In 3D gravity, the standard convention is $\phi \to 0$ as $r \to \infty$, so
`baseline='none'` is physically meaningful.  The `'zero_corner'` and
`'zero_mean'` options are provided for consistency with the 1D and 2D cases.

---

## Accuracy and convergence

The solver computes the **exact** circular convolution on the extended grid.
All numerical error comes from the **quadrature approximation** of the
continuous integral $\int G\,\rho\,d^dr$ by the Riemann sum with cell volume
$\Delta V$:

- **3D isolated:** converges at $\mathcal{O}(h^2)$ for smooth $\rho$.  The
  $G[\mathbf{0}] = 0$ approximation contributes an $\mathcal{O}(h^2)$ error —
  the same order as the quadrature error, so it does not degrade convergence.

- **2D isolated:** converges at $\mathcal{O}(h^2 |\ln h|)$, slightly slower
  than second order due to the $G[\mathbf{0}] = 0$ correction.  In practice
  this logarithmic factor is negligible: at $N = 256$ the effective convergence
  rate is indistinguishable from $\mathcal{O}(h^2)$.

The tests in `tests/test_poissonnd.py` verify these rates against direct
$\mathcal{O}(N^{2d})$ summation references for $d = 2, 3$.

---

## 2D Example: Gaussian source

```python
import numpy as np
from src.poissonnd import solve_isolated_nd

N, L = 128, 1.0
x    = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, x, indexing='ij')

sigma = 0.08
rho   = np.exp(-0.5*((X - L/2)**2 + (Y - L/2)**2)/sigma**2)
rho  /= 2*np.pi*sigma**2       # unit-integral normalisation

phi = solve_isolated_nd(rho, L, baseline='zero_mean')
```

Run `python examples/demo_2d.py` for a full 2×2 figure showing $\rho$ and
$\phi$ for both periodic and isolated solvers.

---

## 3D Example: Isolated gravitational potential

```python
N, L = 64, 1.0
x    = np.linspace(0, L, N, endpoint=False)
X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

sigma = 0.06
rho   = np.exp(-0.5*((X-L/2)**2 + (Y-L/2)**2 + (Z-L/2)**2)/sigma**2)
rho  /= (2*np.pi*sigma**2)**1.5

phi = solve_isolated_nd(rho, L, baseline='zero_mean')
# phi[:, :, N//2] is the xy mid-plane potential
```

Run `python examples/demo_3d.py` for a 2×3 figure showing xy and xz
mid-plane slices.

---

## Memory and performance

The extended grid has $2^d$ times as many cells as the physical grid.  In 3D
this is an 8× memory overhead:

| Physical grid | Extended grid | Approx. memory (double) |
|---|---|---|
| $64^3$ | $128^3$ | $\sim$ 130 MB |
| $128^3$ | $256^3$ | $\sim$ 1 GB |
| $256^3$ | $512^3$ | $\sim$ 8 GB |

The Green's function $G$ on the extended grid is real and does not depend on
$\rho$ — it can be precomputed and its FFT cached if many solves are needed
with the same grid.

For the FFT itself, `scipy.fft.fftn` supports multi-threading via the
`workers` argument (see [08_performance.md](08_performance.md)).

---

## Connection to cosmological gravity solvers

This algorithm is the core of the **Particle-Mesh (PM) $N$-body method**
(Hockney & Eastwood 1988, Ch. 6).  In a cosmological simulation:

1. *Mass assignment:* deposit particle masses onto the 3D mesh to form $\rho$.
2. *Poisson solve:* call `solve_isolated_nd` (or its gravitational variant
   with a $4\pi G$ prefactor) to obtain $\phi$.
3. *Force interpolation:* compute $\mathbf{g} = -\nabla\phi$ on the mesh
   (via finite differences or spectral differentiation), then interpolate
   back to particle positions.

The total cost per time-step is $\mathcal{O}(N^3 \log N)$ — dominated by the
three 3D FFTs — compared to $\mathcal{O}(N^6)$ for direct summation.  This
favourable scaling made PM codes practical for cosmological volumes long before
tree or multipole methods became widespread.
