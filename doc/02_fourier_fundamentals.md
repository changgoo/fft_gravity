# 2. Fourier Fundamentals

This document introduces the mathematical tools behind all four Poisson solvers:
the DFT, DST, and DCT, their normalization conventions in SciPy, and the
all-important **derivative theorem**.

---

## 2.1 Continuous Fourier transform

We use the *physics convention* (positive exponent for forward transform):

$$
\hat{f}(k) = \int_{-\infty}^{\infty} f(x)\, e^{-ikx}\, dx, \qquad
f(x) = \frac{1}{2\pi} \int_{-\infty}^{\infty} \hat{f}(k)\, e^{ikx}\, dk
$$

With this convention the **derivative theorem** reads

$$
\widehat{\frac{d^n f}{dx^n}}(k) = (ik)^n\, \hat{f}(k)
$$

In particular, for the second derivative:

$$
\widehat{\frac{d^2 f}{dx^2}}(k) = -k^2\, \hat{f}(k)
$$

This is the key identity that turns the Poisson equation into a pointwise
division.

---

## 2.2 Discrete Fourier Transform (DFT)

Given $N$ samples $f_j$ at $x_j = jh$ ($j = 0, \ldots, N-1$, $h = L/N$), the
**forward DFT** is

$$
\hat{f}_n = \sum_{j=0}^{N-1} f_j\, e^{-2\pi i jn / N}, \quad n = 0, \ldots, N-1
$$

and the **inverse DFT** is

$$
f_j = \frac{1}{N} \sum_{n=0}^{N-1} \hat{f}_n\, e^{2\pi i jn / N}
$$

Note: `scipy.fft.fft` computes the forward sum (no $1/N$ factor);
`scipy.fft.ifft` divides by $N$.

**Wavenumber array.** The discrete angular wavenumbers corresponding to the
DFT indices are

$$
k_n = \frac{2\pi n}{L}, \quad n = 0, 1, \ldots, \frac{N}{2}-1, -\frac{N}{2}, \ldots, -1
$$

In NumPy this is obtained via

```python
freq = np.fft.fftfreq(N, d=h)   # cycles per unit length, range [-1/(2h), 1/(2h))
k    = 2 * np.pi * freq          # angular wavenumber
```

The negative-frequency components arise from the periodicity of the DFT.
For a real-valued input $f_j$, we have $\hat{f}_{N-n} = \hat{f}_n^*$, so
the negative frequencies carry no additional information.

**Nyquist frequency.** The highest representable frequency is $k_{\rm Nyq} = \pi/h$.
Features with $|k| > k_{\rm Nyq}$ are *aliased* onto lower frequencies.
This sets the resolution limit of any FFT-based solver.

---

## 2.3 The derivative theorem (discrete version)

On a periodic grid of $N$ points with spacing $h$, applying the forward DFT
to $f_{j+1} - 2f_j + f_{j-1}$ (the standard second-order finite-difference
Laplacian) yields the **modified wavenumber**

$$
\tilde{k}_n^2 = \left(\frac{2}{h} \sin\frac{k_n h}{2}\right)^2
$$

For the *spectral* (exact continuous) derivative we use $k_n^2$ directly.
The choice between them is a trade-off:

| Eigenvalue used | Convergence | When to use |
|---|---|---|
| $-k_n^2$ (spectral) | Exponential for smooth $\rho$ | Spectral accuracy desired |
| $-\tilde{k}_n^2$ (modified) | $\mathcal{O}(h^2)$ | Matching a finite-difference code |

---

## 2.4 Discrete Sine Transform (DST-I)

The **DST-I** of a sequence $f_j$ of length $M$ is

$$
\hat{f}_n = 2\sum_{j=1}^{M} f_j \sin\!\left(\frac{\pi j n}{M+1}\right),
\quad n = 1, \ldots, M
$$

Key properties:
- The DST-I is its own inverse up to a factor of $1/(2(M+1))$:
  $\text{IDST-I}(\hat{f}) = f$.  `scipy.fft.idst(type=1)` applies this
  normalization automatically — **always use `idst`, never divide manually**.
- The functions $\sin(n\pi x/L)$ are eigenfunctions of $d^2/dx^2$ with
  zero-Dirichlet BCs on $[0, L]$:

$$
\frac{d^2}{dx^2}\sin\!\left(\frac{n\pi x}{L}\right) = -\left(\frac{n\pi}{L}\right)^2 \sin\!\left(\frac{n\pi x}{L}\right)
$$

This is why the DST-I diagonalizes the Laplacian with Dirichlet BCs.

**scipy convention** (`norm=None`, the default):

```python
from scipy.fft import dst, idst

rho_hat = dst(rho, type=1)            # forward, unnormalized
phi_hat = rho_hat / eigenvalues
phi     = idst(phi_hat, type=1)       # inverse, handles 1/(2*(N+1)) internally
```

---

## 2.5 Discrete Cosine Transform (DCT-II)

The **DCT-II** of a sequence $f_j$ of length $M$ is

$$
\hat{f}_n = 2\sum_{j=0}^{M-1} f_j \cos\!\left(\frac{\pi(2j+1)n}{2M}\right),
\quad n = 0, \ldots, M-1
$$

Key properties:
- The DCT-II inverse (DCT-III) is related by $\text{IDCT-II}(\hat{f}) = f$.
  `scipy.fft.idct(type=2)` applies the $1/(2N)$ normalization automatically.
- The functions $\cos(n\pi x/L)$ are eigenfunctions of $d^2/dx^2$ with
  zero-Neumann BCs:

$$
\frac{d}{dx}\cos\!\left(\frac{n\pi x}{L}\right)\bigg|_{x=0, L} = 0, \quad
\frac{d^2}{dx^2}\cos\!\left(\frac{n\pi x}{L}\right) = -\left(\frac{n\pi}{L}\right)^2 \cos\!\left(\frac{n\pi x}{L}\right)
$$

- The $n=0$ mode is the mean: $\hat{f}_0 \propto \sum_j f_j$.  Its eigenvalue
  is zero (singular), so we zero it out (gauge fix).

**scipy convention** (`norm=None`, the default):

```python
from scipy.fft import dct, idct

rho_hat = dct(rho, type=2)            # forward, unnormalized
phi_hat = rho_hat / eigenvalues
phi     = idct(phi_hat, type=2)       # inverse, handles 1/(2N) internally
```

---

## 2.6 Normalization reference table

| Transform | Forward (scipy default) | Inverse |
|---|---|---|
| `fft` | $\sum_j f_j e^{-2\pi i jn/N}$ (no $1/N$) | `ifft` divides by $N$ |
| `dst(type=1)` | $2\sum_j f_j \sin(\pi j n/(N+1))$ | `idst(type=1)` divides by $2(N+1)$ |
| `dct(type=2)` | $2\sum_j f_j \cos(\pi(2j+1)n/(2N))$ | `idct(type=2)` divides by $2N$ |

> **Rule of thumb:** always call `idst` / `idct` for the inverse — do not
> reconstruct it by hand.  The `norm="ortho"` option gives symmetric
> (unitary) transforms but changes the relationship between $\hat{\phi}_n$
> and the eigenvalue equation; we avoid it here to keep the math transparent.

---

## 2.7 Parseval's theorem

For the DFT:

$$
\sum_{j=0}^{N-1} |f_j|^2 = \frac{1}{N} \sum_{n=0}^{N-1} |\hat{f}_n|^2
$$

This is useful for checking that the FFT preserves total energy, and for
deriving error bounds in spectral space.
