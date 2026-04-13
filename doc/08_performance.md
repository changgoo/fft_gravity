# 8. Performance and Scaling

## 8.1 Algorithmic complexity

All four solvers have the same asymptotic complexity:

| Step | Cost |
|---|---|
| Forward FFT / DST / DCT | $\mathcal{O}(N \log N)$ |
| Eigenvalue division (pointwise) | $\mathcal{O}(N)$ |
| Inverse transform | $\mathcal{O}(N \log N)$ |
| **Total** | $\mathcal{O}(N \log N)$ |

The isolated solver uses a $2N$-point FFT (twice the domain size), so it
costs roughly $2\times$ more than the periodic solver at the same $N$.  But
the complexity is still $\mathcal{O}(N \log N)$.

For comparison:
- Direct matrix inversion: $\mathcal{O}(N^3)$
- Tridiagonal solve (Thomas algorithm): $\mathcal{O}(N)$ — but the FFT has
  a much smaller constant for large $N$ and is trivially parallelisable.

---

## 8.2 scipy.fft vs numpy.fft

Both `scipy.fft` and `numpy.fft` wrap FFTPACK (or FFTW on some builds), but
there are practical differences:

| Feature | `numpy.fft` | `scipy.fft` |
|---|---|---|
| DST / DCT | Not available | `dst`, `dct`, `idst`, `idct` |
| Multithreading | No | Yes: `workers` parameter |
| Planning | No | Optional backend (pyfftw) |
| Default backend | FFTPACK / NumPy | FFTPACK / pocketfft |

We use `scipy.fft` throughout because:
1. It provides DST and DCT (required for Dirichlet and Neumann solvers).
2. The `workers=-1` option uses all available CPU cores.

```python
from scipy.fft import fft, ifft

# Single-threaded (default)
phi = np.real(ifft(fft(rho) / lambda_k))

# Multi-threaded
phi = np.real(ifft(fft(rho, workers=-1) / lambda_k, workers=-1))
```

For 1D problems, the overhead of multithreading is significant at small $N$
(the thread creation cost dominates); the break-even point is typically
$N \gtrsim 10^4$.

---

## 8.3 Memory considerations

For a real-valued input of length $N$:
- The FFT output occupies $N$ complex doubles = $16N$ bytes.
- For the isolated solver the extended array has $2N$ complex doubles = $32N$ bytes.
- At $N = 10^6$: ~16 MB for periodic, ~32 MB for isolated.

To reduce memory usage, `scipy.fft.rfft` / `irfft` exploits Hermitian
symmetry of real inputs and stores only $N/2 + 1$ complex values.  For
pedagogical clarity, we use the full-complex `fft`/`ifft` throughout, but the
real-FFT variants are drop-in replacements for the periodic solver.

---

## 8.4 Profiling guidance

Use `timeit` to measure wall-clock time:

```bash
python examples/performance_benchmark.py
```

This script times all four solvers across `Ns = [64, 128, ..., 8192]` using
100 repeats each, normalises by $N \log_2 N$, and plots the result.  A flat
normalised-timing curve confirms $\mathcal{O}(N \log N)$ scaling.

Quick one-liner profiling in a script or notebook:

```python
import timeit
N, L = 1024, 1.0
x, h = make_grid_periodic(N, L)
rho  = np.sin(2 * np.pi * x / L)

t = timeit.timeit(lambda: solve_periodic(rho, L), number=1000)
print(f"N={N}: {t/1000*1e6:.1f} µs per call")
```

---

## 8.5 JAX extension

[JAX](https://jax.readthedocs.io) provides NumPy-compatible array operations
with automatic GPU/TPU dispatch and JIT compilation.  The periodic solver is
a near-direct port:

```python
import jax.numpy as jnp
from jax.numpy.fft import fft, ifft

def solve_periodic_jax(rho, L):
    N        = rho.shape[0]
    h        = L / N
    rho_hat  = fft(rho)
    k        = 2 * jnp.pi * jnp.fft.fftfreq(N, d=h)
    lambda_k = -(k ** 2).at[0].set(1.0)
    rho_hat  = rho_hat.at[0].set(0.0)
    return jnp.real(ifft(rho_hat / lambda_k))
```

To JIT-compile:

```python
from jax import jit
solve_periodic_fast = jit(solve_periodic_jax)

# First call triggers compilation (slow)
phi = solve_periodic_fast(rho, L)
# Subsequent calls use cached compiled code (fast)
phi = solve_periodic_fast(rho, L)
```

> **Warm-up note.** Always discard the first JAX call when benchmarking —
> it includes compilation time.  Use `phi.block_until_ready()` to force
> synchronization before timing.

DST and DCT in JAX are available via `jax.scipy.fft` in JAX ≥ 0.4.14.  For
earlier versions, the DST-I and DCT-II can be emulated by padding and applying
the standard FFT (at the cost of a factor of ~2 in memory and a small overhead).

---

## 8.6 Tips for large-scale problems

1. **Use `numpy.float32`** for GPU work — the memory bandwidth advantage
   often outweighs the precision loss for exploratory simulations.
2. **Prefer powers of 2** for $N$ — FFT is fastest when $N = 2^k$.  The
   next fastest sizes are $N = 2^a \cdot 3^b \cdot 5^c$.
3. **Batch multiple solves** — if you need to solve many Poisson problems
   with the same grid, precompute the eigenvalue array once and reuse it.
4. **In-place operations** — `scipy.fft` supports in-place transforms via
   the `overwrite_x=True` flag, reducing memory allocation overhead.
5. **3D extension** — extending to 3D is straightforward: replace 1D FFT
   with `numpy.fft.fftn` and the 1D wavenumber array with a 3D meshgrid.
   The isolated BC in 3D uses the 3D Green's function $G(r) = -1/(4\pi r)$.
