# 1. Overview: Poisson's Equation and the FFT Approach

## What is Poisson's equation?

Poisson's equation is one of the most ubiquitous partial differential equations
in mathematical physics:

$$
\nabla^2 \phi = \rho
$$

In one dimension this reads

$$
\frac{d^2\phi}{dx^2} = \rho(x), \quad x \in [0, L]
$$

where $\rho(x)$ is a known *source* (or *right-hand side*) and $\phi(x)$ is the
unknown *potential*.

**Gravitational physics.**  In Newtonian gravity, $\phi$ is the gravitational
potential and $\rho$ is the mass density.  The full 3D equation is

$$
\nabla^2 \phi = 4\pi G \rho
$$

where $G$ is Newton's constant.  A 1D slab geometry reduces this to the equation
above (with the $4\pi G$ factor absorbed into $\rho$).

**Electrostatics.**  In SI units, Gauss's law in differential form gives

$$
\nabla^2 \phi = -\frac{\rho_{\rm charge}}{\epsilon_0}
$$

where $\rho_{\rm charge}$ is the charge density and $\epsilon_0$ is the
permittivity of free space.

> **Sign convention used throughout this repository.**
> We work with
> $$\frac{d^2\phi}{dx^2} = \rho(x)$$
> The factor $4\pi G$ or $-1/\epsilon_0$ is the *caller's responsibility*.

---

## Why spectral methods?

The naive approach to solving $d^2\phi/dx^2 = \rho$ on a grid of $N$ points
would be to assemble and invert an $N \times N$ tridiagonal matrix — an
$\mathcal{O}(N)$ operation (for a tridiagonal system), but one that must be
set up and factored each time the grid changes.

A spectral (FFT-based) approach exploits the fact that differentiation becomes
*multiplication* in Fourier space:

$$
\frac{d^2}{dx^2} \xrightarrow{\mathcal{F}} -k^2
$$

So Poisson's equation $d^2\phi/dx^2 = \rho$ becomes, in Fourier space,

$$
-k^2 \hat{\phi}(k) = \hat{\rho}(k)
\quad\Longrightarrow\quad
\hat{\phi}(k) = \frac{\hat{\rho}(k)}{-k^2}
$$

The algorithm is:

1. Compute $\hat{\rho}$ via FFT — $\mathcal{O}(N \log N)$
2. Divide pointwise by $-k^2$ — $\mathcal{O}(N)$
3. Compute $\phi$ via inverse FFT — $\mathcal{O}(N \log N)$

Total cost: $\mathcal{O}(N \log N)$, with a very small constant.  Moreover,
for *smooth* source functions the solution is *spectrally accurate* — the
error decays exponentially with $N$, far faster than any finite-difference
scheme.

---

## The four boundary condition families

This repository covers four distinct boundary condition (BC) types for the
1D Poisson equation.  Each requires a slightly different transform.

### Periodic BCs

$$
\phi(0) = \phi(L), \quad \frac{d\phi}{dx}\bigg|_0 = \frac{d\phi}{dx}\bigg|_L
$$

Used in cosmological simulations and periodic boxes.  The natural transform
is the **Discrete Fourier Transform (DFT)**, implemented by `scipy.fft.fft`.

$\to$ [doc/03_1d_periodic.md](03_1d_periodic.md)

### Dirichlet BCs (homogeneous)

$$
\phi(0) = 0, \quad \phi(L) = 0
$$

Used when the potential vanishes at conducting walls.  The natural transform
is the **Discrete Sine Transform type I (DST-I)**, implemented by
`scipy.fft.dst(type=1)`.

$\to$ [doc/04_1d_dirichlet.md](04_1d_dirichlet.md)

### Neumann BCs (homogeneous)

$$
\frac{d\phi}{dx}\bigg|_{x=0} = 0, \quad \frac{d\phi}{dx}\bigg|_{x=L} = 0
$$

Used for insulating/reflecting walls where the flux vanishes.  The natural
transform is the **Discrete Cosine Transform type II (DCT-II)**, implemented
by `scipy.fft.dct(type=2)`.

$\to$ [doc/05_1d_neumann.md](05_1d_neumann.md)

### Isolated (free-space / open) BCs

No walls at all.  The source $\rho$ has compact support in $[0, L]$, and the
potential $\phi$ extends to an unbounded domain.  The solution is a convolution
with the 1D free-space **Green's function** $G(x) = |x|/2$.  In practice this
is computed via the **zero-padding FFT trick** from Hockney & Eastwood (1988).

$\to$ [doc/06_1d_isolated.md](06_1d_isolated.md)

---

## Repository navigation

```
fft_gravity/
├── doc/               ← you are here (tutorials, start with 01 → 08)
├── src/
│   ├── poisson1d.py   ← four solver functions
│   └── utils.py       ← grid constructors, error norms, convergence helpers
├── tests/
│   └── test_poisson1d.py
└── examples/
    ├── demo_1d.py
    ├── convergence_study.py
    └── performance_benchmark.py
```

Recommended reading order:

1. [01_overview.md](01_overview.md) — this file
2. [02_fourier_fundamentals.md](02_fourier_fundamentals.md) — DFT, DST, DCT theory
3. [03_1d_periodic.md](03_1d_periodic.md) — simplest solver
4. [04_1d_dirichlet.md](04_1d_dirichlet.md) — DST-I solver
5. [05_1d_neumann.md](05_1d_neumann.md) — DCT-II solver
6. [06_1d_isolated.md](06_1d_isolated.md) — Green's function and zero-padding
7. [07_convergence.md](07_convergence.md) — error analysis
8. [08_performance.md](08_performance.md) — performance and scaling

---

## Prerequisites

- Fourier series and the continuous Fourier transform at the level of a
  graduate mechanics or mathematical methods course.
- Basic Python/NumPy — array operations, `np.fft.fftfreq`, indexing.
- No prior knowledge of spectral methods is assumed; all transforms are
  introduced from first principles in [doc/02](02_fourier_fundamentals.md).
