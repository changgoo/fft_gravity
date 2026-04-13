# 6. 1D Poisson Solver — Isolated (Free-Space) Boundary Conditions

## Motivation

In the periodic, Dirichlet, and Neumann solvers, the domain $[0, L]$ is a
*closed* box.  But in many astrophysical problems — computing the
gravitational potential of an isolated galaxy, or a molecular cloud — the
source $\rho$ is localized and the potential should extend to an *open*
domain with $\phi \to 0$ (or at least well-defined) far from the source.

Simply applying an FFT on $[0, L]$ would give a *periodic* result: the
solver would treat the source as if it were repeated with period $L$.
The artificial copies (images) at $x = \pm L, \pm 2L, \ldots$ would
contaminate the potential.

The **isolated** (or **free-space**) solver removes these images.

---

## The 1D Free-Space Green's Function

The solution to an isolated Poisson equation can be written as a
*convolution*:

$$
\phi(x) = \int_{-\infty}^{\infty} G(x - x')\, \rho(x')\, dx'
$$

where $G$ is the **free-space Green's function** — the response to a unit
point source:

$$
\frac{d^2 G}{dx^2} = \delta(x)
$$

**Derivation.** Integrate once:

$$
\frac{dG}{dx} = \begin{cases} +\tfrac{1}{2} & x > 0 \\ -\tfrac{1}{2} & x < 0 \end{cases}
= \frac{\text{sign}(x)}{2}
$$

Integrate again (choosing $G(0) = 0$):

$$
\boxed{G(x) = \frac{|x|}{2}}
$$

We can verify: $d^2G/dx^2 = \frac{1}{2}\frac{d}{dx}\text{sign}(x) = \delta(x)$. ✓

> **1D vs 3D.**  In 3D, the free-space Green's function is $G(r) = -1/(4\pi r)$
> (which vanishes as $r \to \infty$).  In 1D, $G(x) = |x|/2$ *grows* with
> distance — the 1D Poisson equation does not have a solution that vanishes
> at infinity for a non-zero total source.  We will deal with this via
> baseline subtraction.

---

## Why Periodic FFT Fails

If we naively apply the periodic FFT to $\rho$ (which lives on $[0, L]$),
we are implicitly assuming that $\rho$ is periodic with period $L$.
The Fourier-space Green's function would be

$$
\hat{G}_n^{\rm per} = \frac{1}{-k_n^2}
$$

which corresponds to the *periodic* Green's function (the infinite
periodic sum of copies of $G(x)$), not the free-space $G(x) = |x|/2$.
The result is wrong for isolated sources.

---

## The Zero-Padding Strategy

The idea, due to Hockney & Eastwood (1988, Ch. 6), is:

1. **Double the domain.** Create an extended grid of $2N$ points on
   $[0, 2L)$.
2. **Zero-pad the source.** Place $\rho$ in the first $N$ cells; set
   $\rho_j = 0$ for $j = N, \ldots, 2N-1$.
3. **Use the exact Green's function.** Build $G_k$ on the $2N$ grid using
   the actual free-space Green's function.
4. **FFT convolution.** On the $2N$ periodic domain, circular convolution
   equals linear convolution provided the source is zero-padded to at least
   twice its original length.
5. **Extract.** The first $N$ points of the result are the correct isolated
   potential; the last $N$ points are contaminated by the periodic wrap and
   are discarded.

---

## Building the Green's Function on the Extended Grid

On a grid of $M = 2N$ points with spacing $h$, the cyclic index $k$ runs
from $0$ to $2N-1$.  The *nearest-image distance* from index $0$ to index
$k$ on this periodic grid is

$$
d_k = \min(k,\; 2N - k) \cdot h
$$

So the Green's function values are

$$
G_k = \frac{d_k}{2} = \frac{\min(k,\; 2N-k)}{2} \cdot h, \quad k = 0, \ldots, 2N-1
$$

This is exactly $G(x) = |x|/2$ evaluated at the nearest-image distance on
the extended periodic domain.

---

## The FFT Convolution Formula

The discrete convolution

$$
\phi_j = h \sum_{j'=0}^{N-1} G(x_j - x_{j'})\, \rho_{j'}
$$

is computed via the **convolution theorem**:

$$
\phi_{\rm ext} = \text{IFFT}\!\left(\text{FFT}(G) \cdot \text{FFT}(\rho_{\rm ext})\right) \cdot h
$$

The factor $h$ is the **Riemann sum quadrature weight** — it converts the
discrete sum $\sum_j$ into an approximation to the integral $\int dx$.
Omitting this factor is a common source of error.

---

## Step-by-step algorithm

1. **Grid:** $x_j = jh$, $j = 0, \ldots, N-1$, $h = L/N$
2. **Evaluate source:** $\rho_j = \rho(x_j)$ — array of length $N$
3. **Zero-pad:** $\rho_{\rm ext}[0:N] = \rho$, $\rho_{\rm ext}[N:2N] = 0$
4. **Build $G$ on extended grid:** $G_k = \min(k, 2N-k) \cdot h / 2$
5. **FFT convolution:**
   $\phi_{\rm ext} = \text{IFFT}\bigl(\text{FFT}(G) \cdot \text{FFT}(\rho_{\rm ext})\bigr) \cdot h$
6. **Extract:** $\phi = \text{Re}(\phi_{\rm ext}[0:N])$
7. **Baseline subtraction** (optional): subtract $\phi[0]$ or $\langle\phi\rangle$

---

## Python implementation

```python
from scipy.fft import fft, ifft
import numpy as np

def solve_isolated(rho, L, baseline='zero_first'):
    N  = len(rho)
    h  = L / N
    M  = 2 * N

    rho_ext       = np.zeros(M)
    rho_ext[:N]   = rho

    k_idx = np.arange(M)
    dist  = np.minimum(k_idx, M - k_idx) * h
    G     = dist / 2.0

    phi_ext = np.real(ifft(fft(G) * fft(rho_ext))) * h
    phi     = phi_ext[:N]

    if baseline == 'zero_first':
        phi -= phi[0]
    elif baseline == 'zero_mean':
        phi -= np.mean(phi)
    return phi
```

See `src/poisson1d.py:solve_isolated` for the full implementation.

---

## Baseline subtraction

In 1D, $G(x) = |x|/2$ grows without bound, so the convolution integral
diverges for a non-zero-mean source.  The *difference* $\phi(x) - \phi(x_0)$
is always well-defined.  We offer three options:

| `baseline` | Effect |
|---|---|
| `'zero_first'` | $\phi \leftarrow \phi - \phi[0]$ (default) |
| `'zero_mean'` | $\phi \leftarrow \phi - \langle\phi\rangle$ |
| `'none'` | No subtraction; absolute value is arbitrary |

All three differ by a constant, so $d^2\phi/dx^2$ is unchanged.

---

## Connection to Hockney & Eastwood (1988)

This approach is the 1D version of the **Particle-Mesh (PM) method** described
in Chapter 6 of Hockney & Eastwood.  In 3D, the same strategy — zero-padding
plus FFT convolution with the free-space Green's function — is the basis of
all modern $\mathcal{O}(N \log N)$ gravity solvers used in cosmological
$N$-body codes.

In 3D, the situation is simpler in one respect: $G(r) = -1/(4\pi r)$ decays
at infinity, so there is no baseline ambiguity.  In 1D, the growing Green's
function means we must always specify a reference point.

---

## Accuracy and limitations

The FFT convolution computes the *exact* circular convolution on the $2N$
grid with spacing $h$.  The error relative to the continuous integral
$\int G(x-x')\rho(x') dx'$ comes from the **quadrature error** of the
trapezoidal rule, which is $\mathcal{O}(h^2)$ for smooth $\rho$.

The **zero-padding guarantee** — that circular convolution on $2N$ equals
linear convolution for the first $N$ points — holds *exactly* (not just
approximately) as long as the source is zero-padded by at least a factor of 2.
The $\mathcal{O}(h^2)$ error comes entirely from the quadrature approximation.
