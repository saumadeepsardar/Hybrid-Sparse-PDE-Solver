#pragma once

// =============================================================================
// heterogeneous_context.hpp  —  Dynamic CPU-GPU task routing (Thrust 4)
//
// Routes each Krylov task to the most energy-efficient processor:
//
//   Task type          Preferred device       Reason
//   ──────────────────────────────────────────────────────────────────────
//   SpMV (n > 10k)     GPU                    Regular parallel memory access
//   SpMV (n ≤ 10k)     CPU                    GPU launch overhead dominates
//   dot / axpy          GPU (if n > 5k)        High memory bandwidth
//   Jacobi apply        GPU                    Embarrassingly parallel
//   ILU triangular      CPU                    Sequential dependency chain
//   AMG setup           CPU                    Irregular SpGEMM patterns
//   AMG V-cycle         GPU (coarse)+ CPU (fine) Mixed — routed per level
//   MPI halo exchange   CPU                    CPU-side networking
//
// Power-aware routing
// -------------------
// If gpu_power_w > gpu_power_limit * power_safety_factor (default 0.9),
// the next task is routed to CPU regardless of size. This prevents GPU
// thermal throttling which would increase energy-per-FLOP.
//
// Unified Memory
// --------------
// When HSPS_USE_CUDA is defined and unified_memory = true:
//   - Krylov vectors allocated with cudaMallocManaged
//   - CPU and GPU access the same physical memory (demand-paged by driver)
//   - Eliminates explicit upload/download calls in the hot loop
//   - Crossover: efficient for n > ~50k where page-fault overhead < transfer overhead
//
// Build
// -----
//   make BACKEND=CUDA NVML=1
//   make BACKEND=MPI_CUDA NVML=1  (multi-node multi-GPU)
// =============================================================================

#include "omp_backend.hpp"
#include "cuda_backend.hpp"
#include "nvml_monitor.hpp"
#include "parallel_context.hpp"
#include <memory>
#include <string>

namespace hsps {

// ---------------------------------------------------------------------------
// Task types for routing decisions
// ---------------------------------------------------------------------------
enum class TaskType {
    SPMV,           ///< Sparse matrix-vector product
    DOT,            ///< Vector dot product
    AXPY,           ///< Vector axpy / axpby
    SCALE,          ///< Vector scale
    PRECOND_JACOBI, ///< Jacobi preconditioner apply
    PRECOND_ILU,    ///< ILU preconditioner apply (sequential — prefer CPU)
    PRECOND_AMG,    ///< AMG V-cycle (mixed CPU/GPU per level)
    AMG_SETUP,      ///< AMG hierarchy construction (irregular — prefer CPU)
    MPI_EXCHANGE,   ///< Halo exchange (always CPU)
    COARSE_SOLVE,   ///< Small direct solve (always CPU)
};

// ---------------------------------------------------------------------------
// Routing policy configuration
// ---------------------------------------------------------------------------
struct RoutingPolicy {
    Index  gpu_spmv_threshold  = 10000;  ///< Min n for GPU SpMV
    Index  gpu_blas1_threshold = 5000;   ///< Min n for GPU BLAS-1
    double power_safety_factor = 0.90;   ///< Route to CPU if GPU power > limit * this
    bool   unified_memory      = false;  ///< Use cudaMallocManaged
    bool   verbose             = false;  ///< Log routing decisions
};

// ---------------------------------------------------------------------------
// HeterogeneousContext
// ---------------------------------------------------------------------------
class HeterogeneousContext {
public:
    HeterogeneousContext(const ParallelContext& ctx,
                          RoutingPolicy policy = {});

    // ------------------------------------------------------------------
    // Core routing decision
    // Route a task of type t on problem size n to the best device.
    // Returns a reference to either cpu_ or gpu_ (or cpu_ fallback).
    // ------------------------------------------------------------------
    BackendBase& route(TaskType t, Index problem_size = 0) const;

    // ------------------------------------------------------------------
    // Convenience: route with automatic task type inference based on n
    // ------------------------------------------------------------------
    BackendBase& route_spmv (Index n) const;
    BackendBase& route_blas1(Index n) const;

    // ------------------------------------------------------------------
    // Direct backend access
    // ------------------------------------------------------------------
    OMPBackend&  cpu() { return cpu_; }
    BackendBase& gpu_or_cpu() {
        return cuda_available_ ? static_cast<BackendBase&>(gpu_) : cpu_;
    }

    // ------------------------------------------------------------------
    // Power monitoring
    // ------------------------------------------------------------------
    double gpu_power_w()      const;
    double gpu_power_limit_w()const;
    double gpu_headroom_w()   const;
    bool   gpu_near_tdp()     const;  ///< true if power > limit * safety_factor

    // ------------------------------------------------------------------
    // Statistics
    // ------------------------------------------------------------------
    struct RoutingStats {
        long long gpu_tasks = 0;
        long long cpu_tasks = 0;
        long long gpu_redirects = 0;  ///< Tasks routed from GPU to CPU due to power
        double    gpu_energy_j  = 0.0;
        double    cpu_energy_j  = 0.0;
    };
    const RoutingStats& stats() const { return stats_; }
    void print_routing_report(std::ostream& os) const;

    // ------------------------------------------------------------------
    // Unified Memory management
    // ------------------------------------------------------------------
    void alloc_unified(Index n, Real val, Vector& v) const;

private:
    const ParallelContext& ctx_;
    RoutingPolicy          policy_;
    bool                   cuda_available_;

    OMPBackend  cpu_;
    CUDABackend gpu_;
    NVMLMonitor nvml_;

    mutable RoutingStats stats_;

    // Log routing decision (when verbose)
    void log_route(TaskType t, Index n, bool to_gpu) const;
};

} // namespace hsps
