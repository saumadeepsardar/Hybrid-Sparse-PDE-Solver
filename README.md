# Hybrid Adaptive Multilevel Sparse PDE Solver (HSPS)

## Overview

HSPS is a prototype C++17 framework for solving sparse linear systems arising
from PDE discretisations. It implements the **online adaptive ladder** described
in the design spec: start cheap and escalate only when needed, minimising energy
per successful solve.

```
Sparse PDE
    │
    ▼
Discretisation (poisson_2d / convection_diffusion_2d)
    │
    ▼
CSR Sparse Matrix A
    │
    ▼
Feature Extraction ──────────────────────────────► (future: GNN warm-start)
    │  (n, nnz, density, diag-dominance, SPD flag, cond-estimate)
    ▼
Adaptive Decision Engine (AdaptiveSelector)
    │
    ├─ EASY     ── CG      + Jacobi   ◄ default start
    ├─ MODERATE ── FGMRES  + ILU(0)  ◄ on stall
    └─ HARD     ── FGMRES  + AMG     ◄ on stiff system
    │
    ▼
Energy-Aware Runtime Controller (EnergyMonitor)
    │  proxy model: E = α·FLOP + β·MEM_BYTES
    │  RAPL hardware counters (Linux/Intel, falls back to proxy)
    ▼
SolverStats { iterations, rel_res, energy_joules, flop_count, mem_bytes }
```

---

## Build

### Requirements

| Tool       | Minimum version |
|------------|----------------|
| g++        | 9.0  (C++17)   |
| OpenMP     | 4.5            |
| GNU Make   | 3.81           |

Optional (auto-detected):
- Intel RAPL (`/sys/class/powercap/…`) for hardware energy counters.

### Quick start

```bash
make              # build library + examples + tests (default)
make tests        # build and run all 53 unit tests
make poisson N=64 # run 2-D Poisson example on 64×64 grid
make convdiff     # run convection-diffusion regime sweep
make bench        # run energy benchmark (N=16,32,64)
make clean        # remove all build artefacts
```

### Build variants

```bash
make DEBUG=1      # -O0 -g (development)
make ASAN=1       # AddressSanitizer + UBSan
make OMP=0        # disable OpenMP (serial reference build)
make OPT="-O3 -march=native"   # custom optimisation flags
```

---

## Project Structure

```
hybrid_sparse_pde_solver/
├── include/
│   ├── core/
│   │   ├── types.hpp              # Real, Index, enums, SolverParams, SolverStats
│   │   ├── vector.hpp             # Dense vector (OpenMP BLAS-1)
│   │   └── sparse_matrix.hpp     # CSR matrix, SpMV, factory methods
│   ├── solvers/
│   │   ├── solver_base.hpp        # Abstract Krylov interface
│   │   ├── cg_solver.hpp          # Preconditioned Conjugate Gradient
│   │   └── fgmres_solver.hpp      # Flexible GMRES(m)
│   ├── preconditioners/
│   │   ├── preconditioner_base.hpp
│   │   ├── jacobi.hpp             # Diagonal (CHEAP energy tier)
│   │   ├── ilu.hpp                # ILU(0)   (MODERATE energy tier)
│   │   └── amg.hpp                # Smoothed-Aggregation AMG (EXPENSIVE)
│   ├── energy/
│   │   └── energy_monitor.hpp     # FLOP/MEM proxy + RAPL interface
│   ├── adaptive/
│   │   └── adaptive_selector.hpp  # Online adaptive ladder
│   └── utils/
│       ├── logger.hpp             # Structured logging
│       └── timer.hpp              # RAII wall-clock timer
│
├── src/                           # Implementations (mirrors include/)
│   ├── core/
│   ├── solvers/
│   ├── preconditioners/
│   ├── energy/
│   ├── adaptive/
│   └── utils/
│
├── examples/
│   ├── poisson_2d.cpp             # 2-D Poisson, all solver pairs + adaptive
│   ├── convection_diffusion.cpp   # Regime sweep, escalation demo
│   └── energy_benchmark.cpp       # Side-by-side energy comparison table
│
├── tests/
│   ├── test_framework.hpp         # Minimal single-header harness
│   ├── test_sparse_matrix.cpp     #  9 tests
│   ├── test_preconditioners.cpp   #  8 tests
│   ├── test_solvers.cpp           # 13 tests
│   ├── test_adaptive.cpp          # 12 tests
│   └── test_energy.cpp            # 11 tests   total: 53
│
├── Makefile
└── README.md
```

---

## Numerical Backbone

### Krylov Solvers

| Solver  | Requirements       | Notes                                        |
|---------|--------------------|----------------------------------------------|
| CG      | Symmetric PD       | PCG — Saad Alg. 6.18; stall detection built-in |
| FGMRES  | General            | Saad 1993 FGMRES(m); flexible preconditioner per inner iter |

### Preconditioners

| Preconditioner | Energy tier | Best for                              |
|----------------|-------------|---------------------------------------|
| Jacobi         | CHEAP       | Initial iterations, low-cond systems  |
| ILU(0)         | MODERATE    | General SPD/non-SPD; moderate nnz     |
| AMG            | EXPENSIVE   | Stiff elliptic problems; scales to large n |

AMG uses **smoothed aggregation**:
- Greedy aggregation with strength-of-connection threshold θ (default 0.25)
- Piecewise-constant prolongation P; restriction R = Pᵀ
- Galerkin coarse operator Aᶜ = R A P
- Jacobi smoother (ω = 2/3); direct solve at coarsest level

### Adaptive Ladder

```
                  ┌─────────────────────────────────┐
  Start here ───► │  EASY:  CG + Jacobi              │
                  │  Escalate if:                    │
                  │   • residual flat for 15 iters   │
                  │   • >40% budget spent            │
                  └──────────────┬──────────────────┘
                                 │  stall
                                 ▼
                  ┌─────────────────────────────────┐
                  │  MODERATE: FGMRES + ILU(0)       │
                  │  Escalate if:                    │
                  │   • stall or >70% budget         │
                  └──────────────┬──────────────────┘
                                 │  stall
                                 ▼
                  ┌─────────────────────────────────┐
                  │  HARD: FGMRES + AMG              │
                  │  (ceiling — no further escalation│
                  └─────────────────────────────────┘
```

**Warm start**: current solution `x` is passed to the next state as initial guess.

### Energy Model (proxy)

```
E [J] = α_flop × FLOP_count  +  α_mem × mem_bytes

α_flop = 2.0e-10 J/FLOP  (≈200 pJ — typical double-precision FMA)
α_mem  = 5.0e-9  J/byte   (≈5 nJ  — DRAM access)
```

On Linux/Intel machines with root access, RAPL hardware counters override
the proxy automatically.

---

## Example Output (n=64, energy benchmark)

```
Grid: 64x64  (N=4096 DOFs)
Pair               Conv  Iters  Setup(s)   Solve(s)     E(J)    GFLOP/s
CG + Jacobi         YES    119  1.6e-05   3.1e-03   4.48e-01      3.15
CG + ILU(0)         YES     52  7.6e-05   5.9e-03   1.95e-01      0.71
FGMRES + Jacobi     YES    269  1.5e-05   3.4e-02   6.16e-01      0.45
FGMRES + ILU(0)     YES     51  8.5e-05   9.9e-03   1.17e-01      0.29
FGMRES + AMG        YES     29  3.5e-03   9.5e-03   6.64e-02      0.17

→ Minimum-energy converged: FGMRES + AMG  (6.640e-02 J)
```

AMG gives fewest iterations and **lowest energy** despite the most expensive
setup — exactly matching the design objective.

---

## Suggested Performance Improvements

See `docs/IMPROVEMENTS.md` for the full annotated list.

---

## References

1. Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems* (2nd ed.). SIAM.
2. Trottenberg, U., Oosterlee, C., & Schüller, A. (2001). *Multigrid*. Academic Press.
3. Notay, Y. (2010). An aggregation-based algebraic multigrid method.
   *Electronic Transactions on Numerical Analysis*, 37, 123–146.
4. Intel RAPL documentation: https://www.intel.com/content/www/us/en/developer/articles/technical/software-security-guidance/advisory-guidance/running-average-power-limit-energy-reporting.html
