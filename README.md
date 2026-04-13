# fft_gravity

An educational repository for solving **Poisson's equation in 1D** using
the Fast Fourier Transform (FFT).  Targeted at astrophysics graduate students
who are new to spectral methods.

$$
\frac{d^2\phi}{dx^2} = \rho(x)
$$

Four boundary condition types are covered, each with a complete mathematical
derivation, a clean Python implementation, and unit tests.

---

## Quick start

```bash
git clone https://github.com/changgoo/fft_gravity.git
cd fft_gravity
pip install -e ".[dev]"   # installs numpy, scipy, pytest, matplotlib

# Run the unit tests
pytest tests/ -v

# Run the demo
python examples/demo_1d.py
```

---

## Boundary conditions covered

| BC type | Transform | Source file |
|---|---|---|
| Periodic | DFT (`scipy.fft.fft`) | `src/poisson1d.py:solve_periodic` |
| Dirichlet ($\phi=0$ at walls) | DST-I (`scipy.fft.dst`) | `src/poisson1d.py:solve_dirichlet` |
| Neumann ($\phi'=0$ at walls) | DCT-II (`scipy.fft.dct`) | `src/poisson1d.py:solve_neumann` |
| Isolated / free-space | FFT + zero-padding | `src/poisson1d.py:solve_isolated` |

---

## Repository layout

```
fft_gravity/
├── doc/                    ← tutorial documents (start here)
│   ├── 01_overview.md
│   ├── 02_fourier_fundamentals.md
│   ├── 03_1d_periodic.md
│   ├── 04_1d_dirichlet.md
│   ├── 05_1d_neumann.md
│   ├── 06_1d_isolated.md
│   ├── 07_convergence.md
│   └── 08_performance.md
├── src/
│   ├── poisson1d.py        ← four solver functions
│   └── utils.py            ← grid constructors, error norms, convergence helpers
├── tests/
│   └── test_poisson1d.py
├── examples/
│   ├── demo_1d.py          ← four-panel demonstration figure
│   ├── convergence_study.py
│   └── performance_benchmark.py
└── pyproject.toml
```

---

## Sign convention

Throughout this repository the equation is written as

$$
\frac{d^2\phi}{dx^2} = \rho
$$

For gravitational physics the caller is responsible for including the
$4\pi G$ factor (pass `4*pi*G*rho_mass` as the `rho` argument).

---

## Usage example

```python
import numpy as np
from src.poisson1d import solve_periodic
from src.utils import make_grid_periodic

N, L = 256, 1.0
x, h = make_grid_periodic(N, L)

# Source: single sine mode (analytic solution known exactly)
rho = np.sin(2 * np.pi * x / L)
phi = solve_periodic(rho, L)

phi_exact = -(L / (2 * np.pi))**2 * np.sin(2 * np.pi * x / L)
print(f"Max error: {np.max(np.abs(phi - phi_exact)):.2e}")   # ~1e-15
```

---

## Documentation

Read the tutorials in `doc/` in order:

1. [Overview](doc/01_overview.md) — what problem we solve and why FFT
2. [Fourier fundamentals](doc/02_fourier_fundamentals.md) — DFT, DST, DCT, derivative theorem
3. [Periodic BC](doc/03_1d_periodic.md)
4. [Dirichlet BC](doc/04_1d_dirichlet.md)
5. [Neumann BC](doc/05_1d_neumann.md)
6. [Isolated BC](doc/06_1d_isolated.md) — Green's function, zero-padding (Hockney & Eastwood)
7. [Convergence](doc/07_convergence.md) — spectral vs algebraic accuracy
8. [Performance](doc/08_performance.md) — O(N log N) scaling, JAX extension

---

## Running the examples

```bash
# Four-panel figure comparing all solvers to analytic solutions
python examples/demo_1d.py

# Log-log convergence plots
python examples/convergence_study.py

# Wall-clock timing vs N
python examples/performance_benchmark.py
```

---

## Dependencies

| Package | Version | Required for |
|---|---|---|
| `numpy` | ≥ 1.24 | Core |
| `scipy` | ≥ 1.10 | FFT, DST, DCT |
| `matplotlib` | ≥ 3.7 | Examples (optional) |
| `pytest` | ≥ 7.4 | Tests (optional) |
| `jax` | ≥ 0.4.14 | GPU extension (optional) |

Install all optional extras:

```bash
pip install -e ".[dev]"     # numpy + scipy + matplotlib + pytest
pip install -e ".[jax]"     # adds JAX
```

---

## Reference

Hockney, R. W. & Eastwood, J. W. (1988).
*Computer Simulation Using Particles*. CRC Press.
Chapter 6: The particle-mesh (PM) method and free-space boundary conditions.
