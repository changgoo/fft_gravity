# Plan: fft_gravity — FFT-Based 1D Poisson Solvers (Educational Repository)

## Context

Build a greenfield educational repository that teaches FFT-based solutions to the 1D Poisson equation `d²φ/dx² = ρ(x)`. The audience is astrophysics grad students new to spectral methods. The repo has two pillars: pedagogical step-by-step derivations in `doc/` (Markdown with LaTeX math), and clean Python implementations with pytest unit tests in `src/` and `tests/`. Reference: Hockney & Eastwood, *Computer Simulation Using Particles* (esp. Ch. 6 for isolated BCs).

**Sign convention throughout:** `d²φ/dx² = ρ` (not `∇²φ = -4πGρ`). The caller applies the `4πG` factor.  
**Stack:** NumPy + SciPy (scipy.fft) in v1; JAX is discussed in docs and `pyproject.toml` optional extras but not implemented in v1.

---

## Repository Layout

```
fft_gravity/
├── README.md
├── plan.md                 ← this file
├── pyproject.toml
├── doc/
│   ├── 01_overview.md
│   ├── 02_fourier_fundamentals.md
│   ├── 03_1d_periodic.md
│   ├── 04_1d_dirichlet.md
│   ├── 05_1d_neumann.md
│   ├── 06_1d_isolated.md
│   ├── 07_convergence.md
│   └── 08_performance.md
├── src/
│   ├── __init__.py
│   ├── poisson1d.py
│   └── utils.py
├── examples/
│   ├── demo_1d.py
│   ├── convergence_study.py
│   └── performance_benchmark.py
└── tests/
    └── test_poisson1d.py
```

---

## Documentation Outlines (`doc/`)

Each file is a standalone pedagogical tutorial with LaTeX math in fenced `$$` blocks.

### `01_overview.md`
- What is Poisson's equation; role in gravity and electrostatics
- Why spectral methods: O(N log N) vs O(N²); spectral accuracy
- The four BC families (one-paragraph physical motivation each)
- Repository navigation; prerequisites

### `02_fourier_fundamentals.md`
- Continuous FT conventions (physics: e^{-ikx})
- DFT definition, FFT algorithm overview
- `np.fft.fftfreq` wavenumber array, Nyquist frequency
- **Derivative theorem:** `d^n/dx^n → (ik)^n` — the core of all four solvers
- DST-I and DCT-II definitions and physical motivation (eigenfunctions of Laplacian with respective BCs)
- **scipy.fft normalization table** (norm=None vs norm="ortho" for fft, dst, dct)

### `03_1d_periodic.md`
- Grid: vertex-centred, x_j = j·L/N, j = 0…N-1
- Derivation: apply FFT → φ̂_n = ρ̂_n / (−k_n²)
- Wavenumbers via `np.fft.fftfreq(N, d=h) * 2π`
- **Gauge fix:** λ_0 = 0 is singular; set ρ̂_0 = 0 before dividing (enforces zero-mean φ)
- Analytic test: ρ = sin(2πx/L) → φ_exact = −(L/2π)² sin(2πx/L)

### `04_1d_dirichlet.md`
- Grid: N-1 interior points, x_j = j·h, j = 1…N-1, h = L/N
- Odd extension → DST-I eigenfunctions
- `scipy.fft.dst(type=1)` convention; `idst(type=1)` handles inverse normalization automatically
- Eigenvalues: spectral λ_n = −(nπ/L)²; modified λ_n = −(2/h · sin(nπ/(2N)))² (FD-consistent)
- Analytic test: ρ = sin(πx/L) → φ_exact = −(L/π)² sin(πx/L)

### `05_1d_neumann.md`
- Grid: cell-centred, x_j = (j+½)·h, j = 0…N-1
- Even extension → DCT-II eigenfunctions
- `scipy.fft.dct(type=2)` / `idct(type=2)` convention
- Eigenvalues: λ_n = −(nπ/L)²; gauge fix: set φ̂_0 = 0
- Analytic test: ρ = cos(πx/L) → φ_exact = −(L/π)² cos(πx/L)

### `06_1d_isolated.md`
- Why periodisation introduces images at ±L
- 1D free-space Green's function: G(x) = |x|/2 (derivation: integrate d²G/dx² = δ(x))
- **Zero-padding trick:** extend ρ from N to 2N; linear convolution = circular convolution on 2N domain
- G on extended grid: `G_k = min(k, 2N-k) · h/2` (cyclic nearest-image distance)
- Algorithm: `φ_ext = IFFT(FFT(G) · FFT(ρ_ext)) · h`; take first N points
- Baseline subtraction options; reference to Hockney & Eastwood Ch. 6

### `07_convergence.md`
- Spectral convergence: exponential decay for analytic functions; why eigenfunctions give machine precision
- Polynomial convergence: O(h²) for modified wavenumbers and isolated solver
- Expected rates table; Gibbs phenomenon note
- How to interpret output of `examples/convergence_study.py`

### `08_performance.md`
- O(N log N) complexity for all solvers
- scipy.fft vs numpy.fft; `workers` parameter
- JAX extension: `jax.numpy.fft.fft` drop-in; JIT compilation; GPU dispatch
- Profiling patterns from `examples/performance_benchmark.py`

---

## `src/poisson1d.py` — Public API

All functions: input `rho` (1D ndarray on natural grid), `L` (domain length). Return `phi` on same grid. Derive `N = len(rho)`, `h = L/N` internally.

```python
solve_periodic(rho, L) -> np.ndarray
```
- Gauge: sets `rho_hat[0] = 0` (projects to zero-mean; no exception on non-zero mean)
- Returns `np.real(ifft(fft(rho) / lambda_n))`

```python
solve_dirichlet(rho, L, modified_wavenumber=False) -> np.ndarray
```
- Input: `rho` of length N-1 (interior points only); N = len(rho)+1, h = L/N
- Eigenvalues: spectral `λ_n = -(nπ/L)²`, or modified `-(2/h·sin(nπ/(2N)))²`
- Uses `scipy.fft.dst(rho, type=1)` / `scipy.fft.idst(phi_hat, type=1)`
- `n = np.arange(1, len(rho)+1)`

```python
solve_neumann(rho, L) -> np.ndarray
```
- Input: `rho` of length N (cell-centred)
- Gauge: sets `phi_hat[0] = 0`
- Uses `scipy.fft.dct(rho, type=2)` / `scipy.fft.idct(phi_hat, type=2)`

```python
solve_isolated(rho, L, baseline='zero_first') -> np.ndarray
```
- `baseline`: `'zero_first'` (subtract phi[0]), `'zero_mean'` (subtract mean), `'none'`
- Zero-pads to 2N; builds G; FFT convolution with `· h` quadrature factor; returns first N points

**Key pitfalls documented in module docstring:**
1. Use `idst`/`idct` (not manual `/2(N+1)`) for correct normalization
2. Set `rho_hat[0] = 0` (not divide by sentinel) for gauge fix
3. The `· h` factor in isolated convolution is mandatory (Riemann quadrature weight)
4. Take `.real` of IFFT output to suppress floating-point imaginary residuals

---

## `src/utils.py` — Public API

```python
make_grid_periodic(N, L) -> (x, h)   # N points, endpoint=False
make_grid_dirichlet(N, L) -> (x, h)  # N-1 interior points; N = cell count
make_grid_neumann(N, L) -> (x, h)    # N cell-centred points
make_grid_isolated(N, L) -> (x, h)   # N vertex points (same as periodic grid)

l2_error(phi, phi_exact) -> float    # RMS: sqrt(mean((phi-phi_exact)²))
linf_error(phi, phi_exact) -> float  # L∞: max|phi - phi_exact|

convergence_rate(Ns, errors) -> (slope, coeffs)  # np.polyfit on log-log
richardson_extrapolate(phi_fine, phi_coarse) -> (phi_extrap, err)
```

---

## `tests/test_poisson1d.py` — Test Cases

### `TestPeriodic`
- **eigenfunction:** ρ = sin(2πx/L), N=64 → `np.allclose(phi, phi_exact, atol=1e-12)`
- **zero_mean_enforced:** ρ = sin(2πx/L)+1 → assert `np.isclose(mean(phi), 0)`
- **linearity:** phi(ρ1+ρ2) == phi(ρ1) + phi(ρ2)

### `TestDirichlet`
- **eigenfunction_spectral:** ρ = sin(πx/L), N=64 → `atol=1e-12`
- **eigenfunction_modified:** same source, `modified_wavenumber=True` → `atol=1e-10`
- **convergence_modified:** multi-mode source (sin(3πx/L)+sin(5πx/L)) with analytic φ_exact; Ns=[16,32,64,128,256]; assert `convergence_rate(Ns, errors).slope < -1.8`

### `TestNeumann`
- **eigenfunction:** ρ = cos(πx/L), N=64 → `atol=1e-12`
- **zero_mean_phi:** after solve, assert `np.isclose(mean(phi), 0)`

### `TestIsolated`
- **vs_quadrature:** Gaussian source, N=128, L=4.0, σ=0.3; reference = direct quadrature `sum_j G(x_i-x_j)·ρ_j·h`; assert `atol=1e-6` after baseline subtraction
- **convergence:** Ns=[32,64,128,256]; assert slope < -1.5
- **baseline_options:** assert `np.std(phi_zero_first - phi_zero_mean) < 1e-12`

### `TestUtils`
- **convergence_rate_known:** Ns=[10,20,40,80], errors=C/Ns²; assert slope ≈ -2.0
- **grid_shapes:** assert correct array lengths for all four grid functions

---

## Example Scripts

| Script | Purpose | Output |
|---|---|---|
| `examples/demo_1d.py` | 4-panel figure: ρ and φ vs φ_exact for all four BCs | `demo_1d.png` |
| `examples/convergence_study.py` | Log-log error vs N for all solvers; print measured slopes | `convergence_study.png` |
| `examples/performance_benchmark.py` | Timing vs N for all solvers; normalise by N log N | `performance_benchmark.png` |

---

## `pyproject.toml` Key Dependencies

```toml
[project]
requires-python = ">=3.10"
dependencies = ["numpy>=1.24", "scipy>=1.10"]

[project.optional-dependencies]
jax = ["jax>=0.4.14", "jaxlib>=0.4.14"]
examples = ["matplotlib>=3.7"]
dev = ["pytest>=7.4", "pytest-cov>=4.1", "matplotlib>=3.7"]
```

---

## Verification

1. `pip install -e ".[dev]"` — install package in editable mode
2. `pytest tests/ -v` — all unit tests pass; eigenfunction tests at `atol=1e-12`; convergence slope tests pass
3. `python examples/demo_1d.py` — produces `demo_1d.png`
4. `python examples/convergence_study.py` — log-log convergence plot
5. `python examples/performance_benchmark.py` — N log N normalised timing

---

## Implementation Order

1. `pyproject.toml` + `src/__init__.py`
2. `src/utils.py`
3. `src/poisson1d.py`
4. `tests/test_poisson1d.py`
5. `doc/` files (01 → 08)
6. `examples/` scripts
7. `README.md`
