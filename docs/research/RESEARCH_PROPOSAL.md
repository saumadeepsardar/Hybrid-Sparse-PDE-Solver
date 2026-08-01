# HAMLSS: Hybrid Adaptive Machine-Learning Sparse Solver
## A Research Extension of the Hybrid Adaptive Multilevel Sparse PDE Solver (HSPS)

**Full Research Proposal — Version 1.0**

---

## Abstract

Solving large sparse linear systems arising from PDE discretisations is the
dominant cost in scientific computing, engineering simulation, and climate
modelling. The HSPS prototype demonstrates that an adaptive online solver ladder
(CG+Jacobi → FGMRES+ILU → FGMRES+AMG) combined with an energy proxy model can
select configurations that converge reliably while minimising energy per solve.

This proposal extends HSPS into a full research programme, **HAMLSS**, with
four tightly coupled research thrusts:

1. **GNN-Guided Configuration Learning** — replace heuristic routing with a
   Graph Neural Network trained on (matrix features, best configuration, energy)
   triples, targeting the minimum-energy successful configuration.

2. **Mathematically Rigorous Energy Modelling** — replace the proxy model with
   hardware-calibrated power measurements, roofline-based analytic bounds, and
   a learned correction term, producing verifiable energy-to-solution estimates.

3. **Communication-Optimal Distributed Krylov Methods** — implement and
   theoretically analyse s-step Krylov, asynchronous preconditioned iterations,
   and pipelined GMRES with non-blocking collectives to reduce per-iteration
   communication from O(log P) to O(1) all-reduces.

4. **Heterogeneous Execution on Multi-GPU/Multi-Node Systems** — design a
   task graph runtime that dynamically migrates work between CPU cores and GPUs
   based on real-time occupancy and power measurements, targeting exascale-class
   machines.

The research outputs are: (a) a published open-source solver library,
(b) 3–4 journal/conference papers, (c) a comprehensive benchmark suite covering
50+ real PDE problems from SuiteSparse, and (d) reproducible energy-to-solution
metrics on at least two HPC platforms.

---

## 1. Motivation and Problem Statement

### 1.1 The Energy-Convergence Gap

State-of-the-art PDE solvers (PETSc, Trilinos, hypre) are designed to minimise
**wall-clock time**. As HPC systems approach exascale, energy consumption has
become the binding constraint: the cost of electricity at a large data centre
now exceeds the capital cost of hardware over a 4-year lifecycle. A solver that
converges 10% faster but uses 30% more energy is strictly worse at scale.

No current production solver has an explicit energy objective. HSPS demonstrated
that a proxy model (E = α·FLOP + β·MEM) can rank configurations by energy.
The open questions are:

- How accurate is the proxy? When does it fail?
- Can a learned model generalise across matrix families and machines?
- What is the theoretical minimum energy-to-solution for a given PDE?

### 1.2 The Configuration Space is Too Large to Search

A practical solver has ~12 tunable parameters:

| Parameter          | Range                          | Impact |
|--------------------|-------------------------------|--------|
| Solver type        | CG, FGMRES, BiCGSTAB, GMRES  | High   |
| Preconditioner     | Jacobi, ILU, AMG, block-Jacobi| High   |
| Restart size m     | 10 – 300                       | High   |
| ILU drop tolerance | 1e-6 – 1e-2                   | Medium |
| AMG strength θ     | 0.05 – 0.75                   | Medium |
| AMG levels         | 2 – 15                        | Medium |
| AMG smoothing steps| 1 – 4 pre + 1 – 4 post        | Medium |
| OMP thread count   | 1 – max_cores                 | Medium |
| MPI rank layout    | row-wise, column-wise, 2D     | Low    |
| SpMV format        | CSR, SELL-C-σ, BSR            | Medium |
| Precond frequency  | every 1 – 10 outer iters      | Low    |
| Block size (BSR)   | 1 – 8                         | Low    |

Full grid search on a single matrix would require O(10,000) solves. The GNN
learns a mapping from matrix structure to near-optimal configuration in a single
forward pass (~1 ms), amortising the search cost across all future solves.

### 1.3 Distributed Computing Bottleneck

In HSPS the MPI backend uses `MPI_Allgatherv` for SpMV — copying the entire
vector to every rank. This is O(N) communication per SpMV, which dominates
computation at N > 10⁶ on P > 256 processes. The theoretical lower bound
is O(√N/P) for 2D problems. Reaching it requires:
- Nearest-neighbour halo exchange (implemented in stub form in HSPS)
- s-step Krylov methods (batch multiple steps before communicating)
- Asynchronous overlapping of communication and computation

This gap between the current stub and the theoretical optimum is a concrete,
publishable research target.

---

## 2. Research Thrusts

---

### Thrust 1: GNN-Guided Configuration Learning

**Research question:** Can a GNN trained on (sparse matrix, optimal
configuration) pairs predict minimum-energy configurations for unseen matrices
from different PDE families, with better generalisation than feature-vector
baselines?

#### 1.1 Graph Representation of Sparse Matrices

A sparse matrix A naturally forms a directed weighted graph G = (V, E, xᵥ, xₑ):

- **Nodes** V = {0, …, n−1} — one per DOF
- **Edges** E = {(i,j) : aᵢⱼ ≠ 0} — one per non-zero
- **Node features** xᵥᵢ = [degree, |aᵢᵢ|, Σⱼ|aᵢⱼ|, aᵢᵢ/Σⱼ≠ᵢ|aᵢⱼ|, row_norm]
- **Edge features** xₑᵢⱼ = [|aᵢⱼ|/row_norm_i, sign(aᵢⱼ), |aᵢⱼ−aⱼᵢ|]

For large matrices (n > 100,000) full-graph GNN is infeasible. Two strategies:

**Option A — Sparsified neighbourhood graph.** Sample a random 5,000-node
subgraph with BFS expansion. Theoretical justification: AMG aggregation patterns
are local, so local structure predicts global solvability.

**Option B — Hierarchical coarsening.** Run 2 levels of AMG aggregation to
produce a coarse graph of ~200 nodes. Run GNN on the coarse graph. The
coarsening itself is a provably structure-preserving compression.

#### 1.2 GNN Architecture

```
Input graph G
    │
    ▼  × 3 layers
GraphSAGE layer
  h_v^{l+1} = MLP( CONCAT( h_v^l, MEAN_{u∈N(v)} h_u^l ) )
  Activation: LeakyReLU(0.2)
  Hidden dim: 128
    │
    ▼
Global mean pooling → 128-dim graph embedding z
    │
    ├── Softmax head → P(initial_state ∈ {EASY, MODERATE, HARD})
    ├── Regression head → restart_size ∈ [10, 300]
    ├── Regression head → drop_tol ∈ [1e-6, 1e-2]  (log-scale)
    └── Regression head → amg_strength ∈ [0.05, 0.75]
```

**Loss function:**
```
L = λ₁ · CrossEntropy(state) + λ₂ · MSE(restart) + λ₃ · MSE(drop_tol)
  + λ₄ · E_predicted / E_minimum_observed   ← energy-weighted term
```

The energy-weighted term penalises configurations that converge but at high
energy cost, directly encoding the minimum-energy objective from the design.

#### 1.3 Data Collection Pipeline

The `LoggingAdvisor` in HSPS already writes a CSV with
(n, nnz, density, diag_dominance, estimated_cond, is_spd, initial_state).
Extension: after every solve, record the full (matrix_path, config, iters,
rel_res, energy_joules) tuple. Script:

```bash
for mat in data/*.mtx; do
    for state in 0 1 2; do          # EASY, MODERATE, HARD
        for restart in 20 50 100; do
            bin/run_dataset $mat --force-state $state --restart $restart
        done
    done
done
```

Target: 50,000 (matrix, config, outcome) triples across 500+ matrices from
SuiteSparse groups: Bai, Boeing, DNVS, GHS_indef, Norris, FEMLAB, 2cubes.

#### 1.4 Baseline Comparisons

| Model | Description |
|-------|-------------|
| Random | Uniform random from {EASY, MODERATE, HARD} |
| Heuristic | Current `HeuristicAdvisor` (3-branch rule) |
| Feature | Current `FeatureAdvisor` (12-feature decision tree) |
| Linear | Logistic regression on 12 matrix features |
| MLP | 3-layer MLP on 12 matrix features |
| GNN-flat | Full-graph GNN (baseline for subgraph variants) |
| **GNN-BFS** | **Proposed: BFS-subgraph GNN** |
| **GNN-AMG** | **Proposed: AMG-coarsened GNN** |

**Metrics:**
- Configuration accuracy (correct initial_state)
- Energy ratio: E_chosen / E_optimal (1.0 = perfect)
- Convergence rate (fraction of predicted configs that converge)
- Inference time (must be < 10ms for practical use)

#### 1.5 Implementation Plan

```
Week 1–2:   Extend run_dataset --force-state --restart flags
            Collect 10,000 training triples from bundled + SuiteSparse matrices
Week 3–4:   Implement GNN in PyTorch Geometric (Python)
            Export trained model to TorchScript for C++ inference
Week 5–6:   Implement GNNAdvisor in C++ (loads TorchScript .pt file)
            Wire into AdaptiveSelector::set_ml_advisor()
Week 7–8:   Ablation study: subgraph size, hidden dim, number of layers
            Comparison against baselines on held-out test set
Week 9–10:  Write-up: Section 4 of paper
```

**Deliverable:** `include/adaptive/gnn_advisor.hpp` + `src/adaptive/gnn_advisor.cpp`
+ Python training script `scripts/train_gnn.py` + pre-trained model `models/gnn_v1.pt`.

---

### Thrust 2: Mathematically Rigorous Energy Modelling

**Research question:** Can we produce an energy-to-solution estimate for a
Krylov solver that is accurate to within 5% on real hardware, and can we derive
a theoretical lower bound for the minimum energy required to solve a given
linear system to a given tolerance?

#### 2.1 Limitations of the Current Proxy Model

The current HSPS model E = α·FLOP + β·MEM assumes:
1. Energy scales linearly with FLOPs and memory traffic
2. FLOP energy α and MEM energy β are hardware constants
3. There is no idle/startup energy term

All three assumptions are violated in practice:
- SpMV on irregular sparse data has low arithmetic intensity → memory-bound,
  not compute-bound. FLOP energy is negligible vs. MEM energy.
- β depends on cache hit rate, which varies with matrix sparsity pattern.
- A 30-iteration CG solve on a warm machine is faster (and uses less energy)
  than a cold-start because of OS scheduling and cache state.

#### 2.2 Roofline Energy Model

The roofline model (Williams et al., 2009) characterises each kernel by its
arithmetic intensity I = FLOPs / bytes:

```
E_kernel = max( FLOPs / peak_GFLOPS, bytes / peak_GB_s ) × P_idle
         + (FLOPs × α_compute) + (bytes × α_mem)
```

For SpMV on a modern CPU:
- Arithmetic intensity I ≈ 0.125 (1 FLOP per 8 bytes for CSR double)
- Memory-bandwidth bound when I < ridge_point ≈ 8 (for most CPUs)
- Therefore SpMV is always memory-bound → E dominated by α_mem

**Research contribution:** A per-kernel roofline energy model calibrated by
micro-benchmark, replacing the single (α, β) pair with a per-kernel parameter
set. The calibration uses STREAM (bandwidth), DGEMM (compute), and custom
sparse-kernel benchmarks.

#### 2.3 RAPL Integration and Hardware Validation

Intel RAPL (Running Average Power Limit) provides per-socket energy counters
at ~1 ms resolution. The HSPS stub already opens `/sys/class/powercap/…`.

Extension:
1. **Per-iteration RAPL sampling** — sample RAPL at start and end of each CG
   iteration, not just at solve boundaries. Requires <1% overhead (reading
   RAPL is a memory-mapped register read, ~10 ns).
2. **Calibration protocol** — for each matrix size, run: (a) RAPL-idle
   baseline, (b) SpMV benchmark, (c) dot-product benchmark, (d) full solve.
   Fit (α_spmv, α_dot, α_axpy, P_idle) by least-squares.
3. **Cross-platform portability** — AMD uProf PMC, ARM Energy Meters,
   NVIDIA NVML for GPU. Abstract behind `EnergyHardwareInterface`.

#### 2.4 Theoretical Lower Bound

**Theorem (proposed):** For a symmetric positive definite system Ax = b with
condition number κ, tolerance ε, and n unknowns, the minimum energy of any
Krylov solver using exact arithmetic is:

```
E_min ≥ C(κ, ε) × n × E_matvec

where C(κ, ε) = ⌈ (1/2) log(2/ε) / log((√κ+1)/(√κ−1)) ⌉
```

This is the CG iteration lower bound (Shewchuk, 1994) multiplied by the minimum
energy of a single SpMV. The theorem requires proving that (a) no Krylov method
can converge in fewer iterations than CG on SPD systems (Voevodin, 1983), and
(b) no SpMV implementation can use fewer than Ω(nnz) energy.

**Research contribution:** Formal proof + experimental validation on pde225,
pde900, pde2961 showing that the HAMLSS solver achieves within 1.5× of E_min
on well-conditioned SPD systems.

#### 2.5 Implementation Plan

```
Week 1–3:   Implement per-kernel RAPL sampling (1-line change per solver loop)
            Run calibration protocol on development machine
            Fit (α_spmv, α_dot, α_axpy) via numpy.linalg.lstsq
Week 4–5:   Extend EnergyMonitor with hardware-calibrated model
            Implement EnergyHardwareInterface abstract class
            Port to AMD uProf (if AMD machine available)
Week 6–8:   Prove E_min theorem (draft proof, review with supervisor)
            Verify bound experimentally on all 5 bundled matrices
Week 9–10:  Write-up: energy modelling section of paper
```

**Deliverable:** `include/energy/energy_hardware_interface.hpp`,
`include/energy/roofline_model.hpp`, calibration script `scripts/calibrate_energy.py`,
formal theorem statement in `docs/energy_lower_bound.pdf`.

---

### Thrust 3: Communication-Optimal Distributed Krylov Methods

**Research question:** Can s-step Krylov methods with pipelined collectives
achieve communication cost within a constant factor of the theoretical
minimum for distributed sparse PDE solvers, and what is the tradeoff between
communication reduction and numerical stability?

#### 3.1 The Communication Bottleneck at Scale

For a standard GMRES with restart m on P processes:
- Per iteration: 1 SpMV + (j+1) dot products for Arnoldi step j
- Each dot product requires one `MPI_Allreduce` = O(log P) latency
- For m=50: ~1,300 Allreduces per restart on P=1024 processes
- At 1 μs per latency hop: 1.3 ms/restart purely in barriers

The HSPS pipelined FGMRES overlaps the SpMV with one Allreduce, reducing to
1 Allreduce per iteration. s-step methods go further: batch k iterations before
communicating, reducing to 1 Allreduce per k iterations.

#### 3.2 s-Step CG (CA-CG)

Communication-avoiding CG (Hoemmen, 2010) computes k steps simultaneously
by pre-computing the Krylov basis {r, Ar, A²r, …, Aᵏr} in one SpMV sequence:

```
Standard CG: k iterations → k SpMVs + k Allreduces
s-step CG:   k iterations → k SpMVs + 1 Allreduce (+ basis orthogonalisation)
```

The basis is stored in a matrix B = [r, Ar, …, Aᵏr] (the "Newton basis"
or "monomial basis"). Orthogonalisation is local (no communication).

**Key challenge:** The monomial basis is ill-conditioned for large k. The
Newton or Chebyshev basis improves conditioning at the cost of requiring a
spectral radius estimate before the iteration starts.

**HSPS connection:** The Lanczos estimator already in `adaptive_selector.cpp`
provides the spectral radius estimate needed for Chebyshev basis construction.

**Research contribution:** Implement s-step CG with Chebyshev basis in
`src/solvers/sstep_cg_solver.cpp`, measure condition number of basis matrix
vs. k, and derive an adaptive k-selection strategy that keeps the condition
number below a threshold τ.

#### 3.3 Asynchronous Preconditioned Iterations

In hybrid MPI+OMP systems, the standard pattern is:
1. All ranks compute local part of SpMV
2. `MPI_Allreduce` for dot products (global synchronisation)
3. Apply preconditioner (local, but can be slow for AMG)

Asynchronous variants (Anzt et al., 2014) allow each rank to proceed with
stale values from neighbouring ranks, eliminating the barrier entirely:

```cpp
// Standard:
A.local_spmv(x_local, y_local);
MPI_Allreduce(&local_dot, &global_dot, ...);  // wait for all
update(alpha, x);

// Asynchronous:
A.local_spmv(x_local, y_local);
MPI_Iallreduce(&local_dot, &global_dot, ...);  // non-blocking
do_local_work();                                // overlap
MPI_Wait(&req);                                 // only wait here
update(alpha, x);
```

The theoretical concern: using a stale global_dot means each rank solves a
slightly different system. Convergence theory (Frommer & Szyld, 2000) shows
that under mild conditions (spectral radius of the error propagation matrix < 1)
asynchronous iterations converge to the correct solution.

**Research contribution:**
1. Prove a convergence bound for asynchronous preconditioned Krylov with a
   Jacobi preconditioner, quantifying the convergence rate in terms of the
   maximum staleness τ and the condition number of A.
2. Implement async GMRES in `src/solvers/async_fgmres_solver.cpp`.
3. Measure empirically: at what P does async overtake synchronous on Poisson?

#### 3.4 Near-Neighbour Halo Exchange

The `DistributedMatrix::spmv` currently uses `MPI_Allgatherv` (global
scatter-gather). Replace with nearest-neighbour `MPI_Isend`/`MPI_Irecv`:

```
Phase 1: Send local halo values to neighbours (non-blocking)
Phase 2: Compute local diagonal block SpMV (no communication needed)
Phase 3: Wait for neighbour halo receives (MPI_Waitall)
Phase 4: Compute off-diagonal block SpMV using received halos
```

Communication volume: O(boundary DOFs) = O(n^{(d-1)/d}) for d-dimensional
problems. For 2D: O(√n) vs. O(n) for Allgatherv. At n=10⁶, P=1024: this
is a factor of 1000× reduction in bytes communicated per SpMV.

**Implementation target:** Complete the `HaloDescriptor` stub in
`distributed_matrix.cpp` and validate with a 2D Poisson solve at P=4,8,16,32.

#### 3.5 Implementation Plan

```
Week 1–2:   Complete nearest-neighbour halo exchange in distributed_matrix.cpp
            Validate correctness with mpirun -np 4 on pde2961.mtx
Week 3–5:   Implement s-step CG with Chebyshev basis (k=2,3,4,5)
            Measure basis condition number vs. k for multiple matrices
            Derive adaptive k-selection criterion
Week 6–8:   Implement async FGMRES with MPI_Iallreduce
            Prove convergence bound for Jacobi-preconditioned case
Week 9–10:  Strong-scaling experiments: 1, 2, 4, 8, 16, 32 ranks on pde2961
            Measure: time/iter, energy/iter, communication fraction
Week 11–12: Write-up: communication-reduction section of paper
```

**Deliverable:** `src/solvers/sstep_cg_solver.cpp`, `src/solvers/async_fgmres_solver.cpp`,
corrected `distributed_matrix.cpp` halo exchange, scaling plots.

---

### Thrust 4: Heterogeneous Execution on Multi-GPU Systems

**Research question:** Can a task graph runtime that dynamically partitions
work between CPUs and GPUs — guided by real-time power measurements — achieve
better energy-to-solution than static CPU-only or GPU-only assignments?

#### 4.1 The Heterogeneous Opportunity

The CUDA backend in HSPS runs the entire solve on GPU. But:
- Small matrices (n < 10,000): CPU is faster due to GPU launch overhead
- AMG setup (SpGEMM, aggregation): better on CPU (irregular memory patterns)
- Krylov inner loop (SpMV, BLAS-1): better on GPU (regular parallel patterns)
- Preconditioner apply (ILU triangular solve): better on CPU (sequential)

A static assignment leaves performance on the table. A dynamic task graph
can route each operation to the fastest/most-energy-efficient processor.

#### 4.2 Task Graph Design

Each Krylov iteration is decomposed into a directed acyclic graph of tasks:

```
[SpMV on GPU] ─────────────────────────────────►[axpy on GPU]
                                                      │
[Precond apply on CPU]──► [dot product on CPU/GPU]──►[scale on GPU]
```

Each task is annotated with:
- Estimated FLOP count
- Estimated memory traffic (bytes)
- Estimated energy (from roofline model, Thrust 2)
- Dependencies (edges in the task graph)

The runtime scheduler uses a simple greedy policy:
1. When a task becomes ready (all dependencies met), assign it to the device
   with the lowest estimated energy for that task type and the shortest queue.
2. If GPU power exceeds P_TDP − margin, route the next task to CPU.
3. If CPU is fully subscribed, buffer and wait for GPU.

#### 4.3 Device Coordination

```cpp
class HeterogeneousContext {
    CUDABackend    gpu;
    OMPBackend     cpu;
    EnergyMonitor  power_monitor;

    // Returns CUDA or OMP based on task type and current power
    BackendBase& route(TaskType t) {
        if (power_monitor.gpu_watts() > gpu_threshold_) return cpu;
        if (t == TaskType::SPMV && n_ > gpu_threshold_n_) return gpu;
        return cpu;
    }
};
```

#### 4.4 Unified Memory for CPU-GPU Data Movement

CUDA Unified Memory (cudaMallocManaged) allows the same pointer to be accessed
from CPU and GPU. The driver migrates pages on demand. For the Krylov vectors
(x, r, p, z) which are accessed from both CPU (preconditioner) and GPU (SpMV):

```cpp
// Instead of explicit upload/download:
cudaMallocManaged(&x.d_data, n * sizeof(double));
// Now CPU can read/write x.d_data and GPU can too — driver handles migration
```

This removes the explicit `upload_vector`/`download_vector` calls in the
hot loop, reducing overhead from O(n) bytes per transfer to page-fault-driven
demand migration.

**Research contribution:** Measure the overhead of demand-paging vs. explicit
transfers for different matrix sizes. Find the crossover point where demand
migration wins.

#### 4.5 Implementation Plan

```
Week 1–3:   Implement HeterogeneousContext routing logic
            Test with matrix sizes spanning CPU-optimal to GPU-optimal
Week 4–5:   Implement Unified Memory path in CUDABackend
            Measure page-migration overhead vs. explicit transfers
Week 6–8:   Implement real-time GPU power measurement via NVML
            Wire into HeterogeneousContext routing policy
Week 9–10:  End-to-end experiment: static CPU vs. static GPU vs. dynamic routing
            on 10 SuiteSparse matrices across 3 size ranges
Week 11–12: Write-up: heterogeneous execution section of paper
```

**Deliverable:** `include/parallel/heterogeneous_context.hpp`,
`src/parallel/heterogeneous_context.cpp`, NVML integration in `energy_monitor`.

---

## 3. Evaluation Plan

### 3.1 Benchmark Suite

| Category | Matrices | Source | n range | Characteristics |
|----------|----------|--------|---------|-----------------|
| 2D Elliptic | pde225, pde900, pde2961, FEMLAB/poisson2D | SuiteSparse Bai/FEMLAB | 225–75k | SPD, uniform condition |
| 3D Structural | bcsstk13, bcsstk38, G3_circuit | SuiteSparse Boeing/AMD | 2k–1.5M | SPD, high condition |
| Convection-Diffusion | olm500, olm5000, convdiff_upwind_484 | SuiteSparse Bai + HSPS | 484–5k | Non-symmetric, stiff |
| Helmholtz | helm2d03, helmholtz_400 | SuiteSparse/HSPS | 400–392k | Indefinite, complex-shifted |
| Thermal/Heat | heart1, bfly | SuiteSparse Norris/Oberwolfach | 3.5k–10k | SPD, moderate condition |
| Fluid Dynamics | s3rmt3m3, ship_003, Cylinder* | SuiteSparse Cylshell/DNVS | 5k–121k | Non-symmetric, ill-conditioned |

### 3.2 Evaluation Metrics

For each (matrix, solver configuration) pair, record:

| Metric | Symbol | Unit |
|--------|--------|------|
| Iterations to convergence | K | count |
| Wall-clock time | T | seconds |
| Energy to solution (proxy) | E_proxy | Joules |
| Energy to solution (RAPL) | E_hw | Joules |
| Energy efficiency ratio | E_hw / E_min | dimensionless (≥1) |
| Configuration prediction accuracy | acc | % |
| GNN inference time | t_infer | milliseconds |
| Communication fraction | comm% | % of wall time |
| Memory bandwidth utilisation | BW% | % of peak |

### 3.3 Hardware Platforms

| Platform | CPU | GPU | MPI nodes | Notes |
|----------|-----|-----|-----------|-------|
| Dev workstation | Intel Core i9-13900 | NVIDIA RTX 4090 | 1 | RAPL available |
| HPC cluster node | 2× AMD EPYC 9654 | 4× A100-80GB | 1–32 | AMD uProf power |
| Cloud (optional) | AWS HPC7g (Graviton3) | N/A | 1–64 | ARM energy counters |

### 3.4 Statistical Protocol

- Each (matrix, config) combination run 5 times; report median ± IQR
- Warm machine: 3 warm-up runs before timing
- Cold energy: measure total including AMG setup (not just Krylov phase)
- Report both energy-to-first-convergence and energy-to-ε-tolerance

---

## 4. Theoretical Contributions (Proof Targets)

| # | Statement | Technique | Difficulty |
|---|-----------|-----------|------------|
| T1 | E_min lower bound for SPD Krylov (§2.4) | Voevodin + FLOP lower bounds | Medium |
| T2 | s-step CG Chebyshev basis conditioning: κ(B_k) ≤ (κ(A))^k | Chebyshev approximation theory | Hard |
| T3 | Adaptive k selection keeps κ(B_k) < τ with O(log n) overhead | Lanczos + condition monitoring | Medium |
| T4 | Convergence of async Jacobi-preconditioned GMRES with staleness τ | Frommer-Szyld framework | Hard |
| T5 | GNN generalisation bound: PAC-learning on graph-structured data | Probably Approximately Correct theory | Hard |
| T6 | Near-neighbour SpMV communication complexity: Θ(n^{(d-1)/d} / P^{(d-1)/d}) | PDE discretisation + parallel I/O theory | Easy |

T1, T3, T6 are publishable standalone; T2, T4, T5 are longer-term.

---

## 5. Publication Plan

### Paper 1: The Core System (Conference — SC or ISC)
**Title:** "HAMLSS: Hybrid Adaptive Machine-Learning Sparse Solver with
Hardware-Accurate Energy Modelling"

**Sections:**
1. Introduction: energy as first-class objective
2. Architecture: adaptive ladder + backend abstraction
3. GNN configuration predictor (Thrust 1, partial)
4. Roofline energy model + RAPL validation (Thrust 2)
5. Evaluation: 30 SuiteSparse matrices, energy vs. PETSc baseline
6. Conclusion

**Target:** SC24 or ISC High Performance 2025
**Expected contribution:** First solver to report energy-to-solution with
hardware validation, with ML-guided configuration.

---

### Paper 2: Communication Reduction (Journal — SIAM SISC)
**Title:** "Communication-Optimal s-Step Krylov Methods with Adaptive Basis
Conditioning for Distributed PDE Systems"

**Sections:**
1. Background: communication complexity of Krylov methods
2. Chebyshev basis s-step CG with adaptive k selection (Thrust 3)
3. Convergence theory for asynchronous preconditioned iterations (T4)
4. Halo exchange implementation and validation
5. Strong-scaling results: 1–32 MPI ranks on 3 matrix families

**Target:** SIAM Journal on Scientific Computing (SISC) 2025
**Expected contribution:** Proof of convergence for asynchronous GMRES (T4)
+ adaptive k selection theory (T3) + scaling experiments.

---

### Paper 3: GNN Generalisation (ML Conference — NeurIPS / ICLR)
**Title:** "Predicting Minimum-Energy Solver Configurations for Sparse Linear
Systems via Graph Neural Networks"

**Sections:**
1. PDE solving as a combinatorial optimisation problem
2. Graph representation of sparse matrices
3. GNN architecture and training (50k samples)
4. Generalisation experiments: seen vs. unseen PDE families
5. PAC-learning bound (T5, if proven)
6. Comparison with AutoML baselines (SMAC, Optuna)

**Target:** NeurIPS 2025 or ICLR 2026
**Expected contribution:** First GNN trained on sparse matrix graphs to predict
solver configurations. Dataset released open-source (50k training triples).

---

### Paper 4: Heterogeneous Execution (Conference — PPoPP or ICS)
**Title:** "Dynamic CPU-GPU Task Routing for Heterogeneous Sparse Linear Solvers
Guided by Real-Time Power Measurements"

**Sections:**
1. The heterogeneous opportunity in Krylov methods
2. Task graph decomposition and routing policy
3. CUDA Unified Memory crossover analysis
4. NVML-guided dynamic routing
5. Energy efficiency vs. static assignment on 10 matrices, 3 platforms

**Target:** PPoPP 2026 or ICS 2026
**Expected contribution:** First demonstration of real-time power-guided task
routing in a production sparse solver.

---

## 6. Software Engineering Plan

### 6.1 Code Organisation

All new research code lives in new subdirectories of the existing structure:

```
include/
  ml/
    gnn_advisor.hpp        (Thrust 1 — C++ TorchScript inference)
    dataset_collector.hpp  (data collection harness)
  energy/
    roofline_model.hpp     (Thrust 2 — per-kernel energy model)
    energy_hw_interface.hpp (hardware abstraction: RAPL/uProf/NVML)
  solvers/
    sstep_cg_solver.hpp    (Thrust 3 — s-step CG)
    async_fgmres_solver.hpp (Thrust 3 — async GMRES)
  parallel/
    heterogeneous_context.hpp (Thrust 4 — CPU-GPU routing)
    nvml_monitor.hpp          (Thrust 4 — GPU power via NVML)

scripts/
  train_gnn.py             (PyTorch Geometric training)
  calibrate_energy.py      (roofline calibration)
  collect_dataset.sh       (batch matrix solve + CSV export)
  download_suitesparse.sh  (download 50+ curated matrices)
  plot_scaling.py          (strong-scaling plots for Paper 2)
  plot_energy_comparison.py (energy vs. PETSc for Paper 1)

models/
  gnn_v1.pt                (trained TorchScript model)
  energy_coeffs.json       (calibrated roofline parameters per platform)

benchmarks/
  suite/                   (50+ SuiteSparse .mtx files — gitignored, downloaded)
  results/                 (CSV results — committed)
```

### 6.2 Reproducibility Requirements

Every experiment result in every paper must be reproducible from a single
command. A `Makefile` target `reproduce_paper1` etc. will:

1. Download required matrices from SuiteSparse
2. Run all experiments (can take hours on HPC)
3. Generate all figures as PDF
4. Compute all numbers cited in the paper body

### 6.3 Open-Source Release

The project will be released under the MIT licence on GitHub with:
- Fully automated CI (GitHub Actions) running all 132 tests on push
- Docker image for reproducibility
- Zenodo DOI for each paper's experiment artefacts
- Preprint on arXiv before each conference submission

---

## 7. Timeline (18 months)

```
Month 1–2   │ Thrust 1: GNN data collection + PyTorch prototype
Month 2–3   │ Thrust 2: RAPL integration + roofline calibration
Month 3–4   │ Thrust 3: Halo exchange completion + s-step CG
Month 4–5   │ Thrust 1: GNN training + C++ inference integration
Month 5–6   │ Thrust 2: E_min theorem proof (T1, T6)
Month 6–7   │ Thrust 3: Async GMRES + convergence proof draft (T4)
Month 7–8   │ Paper 1 write-up and submission (SC/ISC)
Month 8–9   │ Thrust 4: Heterogeneous context + NVML integration
Month 9–10  │ Thrust 3: Strong-scaling experiments (1–32 ranks)
Month 10–11 │ Paper 2 write-up and submission (SISC)
Month 11–12 │ Thrust 1: GNN ablation + generalisation experiments
Month 12–13 │ Paper 3 write-up and submission (NeurIPS/ICLR)
Month 13–14 │ Thrust 4: CPU-GPU routing experiments, all platforms
Month 14–15 │ Paper 4 write-up and submission (PPoPP/ICS)
Month 15–16 │ Revisions for Papers 1 and 2
Month 16–17 │ Open-source release + documentation
Month 17–18 │ Thesis write-up (if PhD context)
```

---

## 8. Resources Required

| Resource | Quantity | Purpose |
|----------|----------|---------|
| HPC allocation | 200,000 CPU-hours | Strong-scaling experiments |
| GPU cluster access | 4× A100 for 3 months | GNN training + GPU solve experiments |
| SuiteSparse access | Free download | ~50 matrices, ~5 GB |
| PyTorch Geometric | Open-source | GNN implementation |
| Intel VTune | Free academic | Cache miss + energy profiling |
| NVIDIA Nsight | Free | CUDA kernel profiling |

---

## 9. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| GNN does not generalise beyond training distribution | Medium | High | Diverse training set; ensemble; fall back to FeatureAdvisor |
| RAPL not available on test machine (AMD/ARM) | Medium | Medium | AMD uProf fallback already planned |
| s-step CG numerically unstable for k > 3 | High | Medium | Limit k adaptively using Lanczos condition estimate |
| Async GMRES diverges on indefinite problems | Low | High | Safety: detect divergence and restart synchronously |
| CUDA Unified Memory overhead too high | Medium | Low | Fall back to explicit transfer; still publishable |
| HPC allocation insufficient for scaling experiments | Low | Medium | Use cloud (AWS HPC7g) as fallback |

---

## 10. Related Work

**Adaptive solver selection:**
- Bhowmick et al. (2006): SciDAC solver selection via decision trees
- George & Liu (2012): Sparse direct solver selection heuristics
- Xu et al. (2019): Performance prediction for Krylov solvers

**ML for numerical methods:**
- Luz et al. (2020): Learning meshless simulation via GNN
- Greenfeld et al. (2019): Learning to optimise multigrid solvers (NeurIPS)
- Taghibakhshi et al. (2021): Reinforcement learning for AMG
- **Gap:** None predict minimum-energy configurations

**Energy-efficient computing:**
- Rünger & Schwind (2015): Energy model for MPI-parallel programs
- Hähnel et al. (2012): Energy consumption of iterative solvers
- **Gap:** No hardware-validated energy lower bound for Krylov solvers

**s-step methods:**
- Hoemmen (2010): Communication-avoiding Krylov subspace methods (PhD thesis)
- Carson et al. (2015): Avoiding communication in nonsymmetric Lanczos
- **Gap:** No adaptive k selection using real-time condition monitoring

**Heterogeneous solvers:**
- Anzt et al. (2020): Ginkgo: heterogeneous sparse linear algebra library
- Yamazaki et al. (2022): Hierarchical solver for heterogeneous systems
- **Gap:** No real-time power-guided task routing

---

## 11. Expected Impact

**Academic:** 4 papers in top venues (SC, SISC, NeurIPS/ICLR, PPoPP/ICS),
2 theorems (T1, T4), 1 open dataset (50k training triples + 50+ matrices).

**Industrial:** A drop-in replacement for PETSc's KSP solver that reduces
energy consumption by an estimated 20–40% on well-characterised PDE classes
(based on HSPS benchmark results showing FGMRES+AMG uses 93% less energy
than CG+Jacobi on a 64×64 Poisson problem).

**Scientific computing community:** The energy lower bound (T1) and GNN
training dataset will be stand-alone contributions usable by other solver
researchers regardless of the HAMLSS implementation.
