# Suggested Performance Improvements

Annotated list of concrete changes to make to existing source files, ordered
by expected impact on the design goals: **robustness → scalability → energy efficiency**.

---

## 1. `src/adaptive/adaptive_selector.cpp` — Condition-Number Estimator

**Problem**: `spectral_radius_estimate` uses a naive power iteration that can
stall on unlucky initial vectors and produce wildly inaccurate estimates
(observed: `cond ≈ 3.4e16` for a well-conditioned n=32 Poisson matrix).

**Fix A — Lanczos-based estimator** (best accuracy):
```cpp
// Replace spectral_radius_estimate(15) with a 20-step Lanczos tridiagonalisation.
// Eigenvalue bounds of the Lanczos tridiagonal give tight enclosures of λ_min, λ_max.
```

**Fix B — Cheap Gershgorin bound** (current intent, implementation was wrong):
```cpp
// The Gershgorin lower bound can be negative for Poisson because some rows have
// sum-of-off-diagonals > diagonal (e.g. corner nodes: diag=4, off-sum=3, but
// for ε-scaled matrices the bound is exact).
// Guard: take max(Gershgorin_lower, smallest_positive_diagonal * 0.1).
Real lam_min = std::max(gershgorin_lower, 0.1 * min_positive_diag);
```

**Impact**: Fixes misrouting of well-conditioned problems to HARD state.
Routing accuracy directly affects average energy per solve.

---

## 2. `src/preconditioners/amg.cpp` — Prolongation Smoothing

**Problem**: Piecewise-constant prolongation (tent functions) gives slow
convergence for anisotropic problems. Iteration counts are 2–4× higher
than with smoothed prolongation.

**Fix — Jacobi-smoothed prolongation** (standard SA-AMG):
```cpp
// After building P_0 (piecewise constant), smooth it:
//   P = (I - ω D^{-1} A) P_0
// where ω = 4/3 * 1/λ_max  (spectral radius of D^{-1} A).
// Implemented as: P_smooth[i,j] = P_0[i,j] - ω * Σ_k (a_ik/a_ii) * P_0[k,j]
// Only 1–2 smoothing steps needed. Doubles the AMG setup cost but
// typically halves the V-cycle count.
```

**File changes**: `include/preconditioners/amg.hpp` add `smooth_prolongation()`,
call it inside `setup()` after `build_prolongation()`.

**Impact**: 2–5× faster convergence on anisotropic / stretched-grid problems.

---

## 3. `src/solvers/fgmres_solver.cpp` — s-Step / Pipelined GMRES

**Problem**: Standard FGMRES requires one global all-reduce per inner
iteration (from `w.dot(V[i])`). At scale (MPI), this dominates runtime
and energy.

**Fix — Pipelined GMRES** (Ghysels & Vanroose, 2014):
```cpp
// Overlap the SpMV and the dot-product reductions:
//   1. Compute w = A z_j  AND  begin async-reduce of all inner products
//      { <w, v_0>, <w, v_1>, …, <w, v_j> } in a single MPI_Iallreduce.
//   2. While reduction completes, apply previous Givens rotations.
//   3. Use reduction result for current Arnoldi step.
// Cost: 1 all-reduce per iteration instead of j+1.
// Communication savings: O(m) → O(1) collectives per restart.
```

**File changes**: `include/solvers/fgmres_solver.hpp` add `PipelinedFGMRESSolver`
subclass; enable with `params_.pipeline = true`.

**Impact**: Critical for MPI scaling. At 1000+ cores, communication can be
80%+ of runtime. Pipelining cuts energy per iteration by ~30–50% at scale.

---

## 4. `src/core/sparse_matrix.cpp` — SELL-C-σ Format for GPU/SIMD

**Problem**: CSR has irregular memory access patterns that underutilise
SIMD width on CPUs and warp efficiency on GPUs.

**Fix — Add SELL-C-σ (Sliced ELL) format**:
```cpp
// Partition rows into chunks of C (e.g. C=32 for AVX-512, C=32 for CUDA warp).
// Sort rows within each chunk by nnz_per_row descending (σ = sort permutation).
// Pad each chunk to chunk_max_nnz.
// SpMV is then stride-1 in inner loop → vectorises cleanly.
//
// Conversion: O(nnz) one-time cost (acceptable per design note).
// Recommended trigger: add SELL-C-σ path when n > 10000 AND running on GPU.
```

**File changes**: New files `include/core/sell_cs_matrix.hpp` +
`src/core/sell_cs_matrix.cpp`. `SparseMatrix::to_sell_cs()` conversion method.

**Impact**: 2–4× SpMV throughput on AVX-512 CPUs; 3–8× on NVIDIA GPUs.
SpMV dominates Krylov inner-loop cost so this is the single highest-leverage
compute optimisation.

---

## 5. `src/preconditioners/ilu.cpp` — Level-k Fill and Threshold Dropping

**Problem**: ILU(0) retains only the original sparsity pattern. For
convection-dominated problems, ILU(0) preconditions poorly (condition
number of M⁻¹A can remain O(n)).

**Fix A — ILU(k)**: allow fill up to level k from the original graph.
```cpp
// Level-k fill: entry (i,j) is at level ℓ if min over paths i→j of
// (sum of levels of edges) = ℓ. Include all entries at level ≤ k.
// k=1 doubles nnz(L+U) but often halves iteration count.
```

**Fix B — ILUT(τ, p)**: threshold-based dropping + maximum fill per row.
```cpp
// After computing fill entry a_ij, keep it only if |a_ij| > τ * ‖row‖.
// Also keep at most p entries per row (beyond the original pattern).
// Expose as params_.drop_tol (already in SolverParams!) and params_.fill_p.
```

**File changes**: `include/preconditioners/ilu.hpp` add `set_drop_tol()`,
`set_fill_level()`; update `setup()` in `src/preconditioners/ilu.cpp`.

**Impact**: 2–10× iteration reduction for convection-dominated problems.
The `drop_tol` field is already in `SolverParams` — just wire it up.

---

## 6. `src/energy/energy_monitor.cpp` — Calibrated Energy Model

**Problem**: The proxy coefficients `α_flop = 2e-10, α_mem = 5e-9` are
hardware-generic. On modern CPUs with L3 caches, effective α_mem can
be 100× lower for matrix data that fits in cache.

**Fix — Runtime micro-benchmark calibration**:
```cpp
// In EnergyMonitor::calibrate():
//   1. Run STREAM Triad to measure achieved memory bandwidth B [GB/s].
//      alpha_mem = P_idle / B   where P_idle from OS or RAPL baseline.
//   2. Run DGEMM to measure peak FLOP/s F.
//      alpha_flop = P_active / F.
//   3. For matrices fitting in L3 cache, use alpha_mem_l3 ≈ alpha_mem * 0.1.
// Store in EnergyModelCoeffs and select per-solve based on matrix mem footprint.
```

**File changes**: `src/energy/energy_monitor.cpp` — fill in the `calibrate()`
stub (currently returns defaults).

**Impact**: Makes energy estimates accurate to within ~10% vs ~10× currently.
Required for the ML model to learn a meaningful "minimum-energy" objective.

---

## 7. `include/adaptive/adaptive_selector.hpp` — GNN Warm-Start Hook

**Problem**: The current feature-to-state mapping is a simple heuristic
(3 branches based on SPD + cond estimate). The design calls for GNN inference.

**Suggested interface** (drop-in replacement for `select_initial_state`):
```cpp
class MLAdvisor {
public:
    virtual ~MLAdvisor() = default;
    // Given matrix features → return (solver, precond, params) triple.
    virtual SolverParams advise(const AdaptiveSelector::MatrixFeatures& f) const = 0;
};

// In AdaptiveSelector:
void set_ml_advisor(std::shared_ptr<MLAdvisor> ml) { ml_advisor_ = ml; }
```

**GNN implementation notes**:
- Node features: `[row_nnz, diag_val, off_diag_sum, row_norm]` per node
- Edge features: `[abs(a_ij), a_ij/a_ii]` per non-zero
- Message passing: 3–4 layers, mean aggregation
- Output head: softmax over `{EASY, MODERATE, HARD}` + regression for
  `restart_size`, `drop_tol`, `amg_strength`
- Training: collect `(features, best_config)` pairs from `energy_benchmark`
  runs; minimise cross-entropy + energy-weighted regression loss

**File changes**: new `include/adaptive/ml_advisor.hpp`,
`src/adaptive/ml_advisor.cpp`. Add `ml_advisor_` member to `AdaptiveSelector`.

---

## 8. `src/core/sparse_matrix.cpp` — Parallel `mat_mat_product`

**Problem**: Galerkin coarse operator `Ac = R*A*P` in AMG setup is
sequential. For large matrices (n > 100k), this is the AMG setup bottleneck.

**Fix — row-parallel SpGEMM**:
```cpp
// The outer loop `for (i = 0; i < m; ++i)` in mat_mat_product is
// embarrassingly parallel. Each row of C is computed independently.
// OpenMP thread-local hash maps replace the shared dense accumulator:
#pragma omp parallel for schedule(dynamic, 64)
for (Index i = 0; i < m; ++i) {
    std::unordered_map<Index, Real> row_map;
    // ... same logic per row ...
}
// Requires thread-local dense scratch buffers to avoid false sharing.
```

**File changes**: `src/core/sparse_matrix.cpp` — parallelise `mat_mat_product`.

**Impact**: AMG setup time (currently sequential) scales linearly with cores.
For n=100k, setup time drops from ~10s to ~1s on 16 cores.

---

## 9. `src/solvers/cg_solver.cpp` — Early-Breakdown Recovery

**Problem**: CG terminates early when `|ρ_old| < ε`, which can happen before
reaching the tolerance when `r·z ≈ 0` numerically (near-breakdown). The
current code breaks silently, leaving the solver in a non-converged state.

**Fix — Restart on breakdown**:
```cpp
if (std::abs(rho_old) < REAL_EPS * rho_prev) {
    // Breakdown: restart with r as new search direction
    HSPS_LOG_WARN("CG: breakdown at iter ", iter, ", restarting");
    z = r;  // reset preconditioned direction
    if (precond_) precond_->apply(r, z);
    p = z;
    rho = r.dot(z);
    rho_old = rho;
    continue;
}
```

**File changes**: `src/solvers/cg_solver.cpp` — replace `break` with restart.
Track restart count in `SolverStats::restarts` (field already exists).

---

## 10. `Makefile` — MPI Support

**Problem**: The design calls for MPI parallelism but the current build only
supports OpenMP (shared memory).

**Fix — Add MPI build target**:
```makefile
MPI ?= 0
ifeq ($(MPI),1)
    CXX      = mpic++
    CXXFLAGS += -DHSPS_MPI
    LDFLAGS  += -DHSPS_MPI
endif
```

Wrap collective operations in `#ifdef HSPS_MPI` guards:
- `include/core/vector.hpp`: `dot()` → `MPI_Allreduce` for distributed dot
- `include/core/sparse_matrix.hpp`: distributed SpMV with halo exchange
- `src/solvers/fgmres_solver.cpp`: single `MPI_Allreduce` per Arnoldi step

Recommended: add a `DistributedVector` and `DistributedSparseMatrix` wrapper
rather than modifying the serial classes directly.

**Impact**: Unlocks s-step Krylov and asynchronous methods (design note:
"communication costs huge energy → research s-step Krylov, pipelined GMRES,
asynchronous methods").

---

## Priority Summary

| Priority | File(s)                          | Change                        | Expected gain           |
|----------|----------------------------------|-------------------------------|-------------------------|
| 1 (High) | adaptive_selector.cpp            | Fix condition estimator       | Correct adaptive routing|
| 2 (High) | fgmres_solver.cpp                | Pipelined GMRES               | 30–50% at MPI scale     |
| 3 (High) | amg.cpp                          | Smoothed prolongation         | 2–5× AMG convergence    |
| 4 (Med)  | sparse_matrix.cpp                | SELL-C-σ format               | 2–8× SpMV throughput    |
| 5 (Med)  | ilu.cpp                          | ILUT(τ, p)                    | 2–10× for conv-diff     |
| 6 (Med)  | energy_monitor.cpp               | Calibrated coefficients       | Accurate energy model   |
| 7 (Med)  | adaptive_selector.hpp            | GNN advisor hook              | ML-guided routing       |
| 8 (Low)  | sparse_matrix.cpp                | Parallel SpGEMM               | AMG setup scaling       |
| 9 (Low)  | cg_solver.cpp                    | Breakdown restart             | Robustness              |
|10 (Low)  | Makefile                         | MPI build target              | Distributed scale-out   |
