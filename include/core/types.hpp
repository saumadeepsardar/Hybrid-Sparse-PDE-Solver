#pragma once

// =============================================================================
// types.hpp  —  Fundamental types, enumerations, and configuration structs
//               for the Hybrid Adaptive Multilevel Sparse PDE Solver (HSPS)
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace hsps {

// ---------------------------------------------------------------------------
// Scalar / index aliases  (change Real → float for single precision trials)
// ---------------------------------------------------------------------------
using Real  = double;
using Index = int;          // 32-bit row/col indices; switch to int64_t for >2B nnz
using Size  = std::size_t;

constexpr Real REAL_ZERO = Real(0);
constexpr Real REAL_ONE  = Real(1);
constexpr Real REAL_EPS  = std::numeric_limits<Real>::epsilon();
constexpr Real REAL_INF  = std::numeric_limits<Real>::infinity();

// ---------------------------------------------------------------------------
// Solver / preconditioner identity
// ---------------------------------------------------------------------------
enum class SolverType {
    CG,           ///< Preconditioned Conjugate Gradient  (symmetric SPD only)
    FGMRES,       ///< Flexible Generalised Minimum Residual (general systems)
    BICGSTAB,     ///< BiCGSTAB — comparison baseline (Paper 1)
    SSTEP_CG,     ///< Communication-avoiding s-step CG (Thrust 3)
    ASYNC_FGMRES  ///< Asynchronous non-blocking FGMRES (Thrust 3)
};

enum class PrecondType {
    NONE,         ///< No preconditioning  (identity)
    JACOBI,       ///< Diagonal Jacobi      — cheapest, highly parallel
    ILU,          ///< ILU(0)/ILUT          — moderate cost, good convergence
    AMG,          ///< Algebraic Multigrid  — expensive, best for stiff systems
    BLOCK_JACOBI, ///< Block-diagonal Jacobi — better GPU utilisation (Thrust 4)
    CHEBYSHEV     ///< Polynomial precond    — s-step compatible (Thrust 3)
};

// ---------------------------------------------------------------------------
// Adaptive engine states (maps to the online-adaptation ladder in the design)
// ---------------------------------------------------------------------------
enum class AdaptiveState {
    EASY,      ///< CG  + Jacobi   — low energy, fast iterations
    MODERATE,  ///< FGMRES + ILU   — triggered on CG stall
    HARD       ///< FGMRES + AMG   — triggered when system is highly stiff
};

inline const char* to_string(SolverType s) {
    switch (s) {
        case SolverType::CG:           return "CG";
        case SolverType::FGMRES:       return "FGMRES";
        case SolverType::BICGSTAB:     return "BiCGSTAB";
        case SolverType::SSTEP_CG:     return "s-step CG";
        case SolverType::ASYNC_FGMRES: return "Async-FGMRES";
        default:                       return "Unknown";
    }
}

inline const char* to_string(PrecondType p) {
    switch (p) {
        case PrecondType::NONE:         return "None";
        case PrecondType::JACOBI:       return "Jacobi";
        case PrecondType::ILU:          return "ILU(0)/ILUT";
        case PrecondType::AMG:          return "AMG";
        case PrecondType::BLOCK_JACOBI: return "Block-Jacobi";
        case PrecondType::CHEBYSHEV:    return "Chebyshev";
        default:                        return "Unknown";
    }
}

inline const char* to_string(AdaptiveState st) {
    switch (st) {
        case AdaptiveState::EASY:     return "EASY  [CG+Jacobi]";
        case AdaptiveState::MODERATE: return "MODERATE [FGMRES+ILU]";
        case AdaptiveState::HARD:     return "HARD  [FGMRES+AMG]";
        default:                      return "Unknown";
    }
}

// ---------------------------------------------------------------------------
// Per-solve statistics collected by every solver
// ---------------------------------------------------------------------------
struct SolverStats {
    int         iterations     = 0;
    Real        initial_residual = REAL_ZERO;
    Real        final_residual  = REAL_ZERO;
    bool        converged      = false;
    double      solve_time_s   = 0.0;
    double      setup_time_s   = 0.0;    ///< Preconditioner build time
    double      energy_joules  = 0.0;    ///< Estimated (proxy model)
    long long   flop_count     = 0;      ///< Floating-point operations
    long long   mem_bytes      = 0;      ///< Memory traffic (bytes)
    SolverType  solver_used    = SolverType::CG;
    PrecondType precond_used   = PrecondType::JACOBI;
    AdaptiveState state_used   = AdaptiveState::EASY;
    int         restarts       = 0;      ///< FGMRES restart count
    // Thrust 2 — hardware energy measurements
    double      energy_joules_hw  = 0.0; ///< RAPL/NVML hardware reading
    double      rapl_start_j      = 0.0; ///< RAPL snapshot at solve start
    double      rapl_end_j        = 0.0; ///< RAPL snapshot at solve end
    // Thrust 3 — communication-optimal fields
    int         sstep_k           = 0;   ///< s-step batch size used
    bool        async_mode        = false;///< whether async iterations were used
    long long   comm_bytes        = 0;   ///< MPI communication bytes
    double      comm_time_s       = 0.0; ///< Time spent in MPI collectives
};

// ---------------------------------------------------------------------------
// Tunable parameters (filled by Adaptive Selector; future: ML guidance)
// ---------------------------------------------------------------------------
struct SolverParams {
    Real  tol            = 1e-8;    ///< Relative residual tolerance
    int   max_iter       = 2000;    ///< Maximum total Krylov iterations
    int   restart_size   = 50;      ///< FGMRES restart dimension (m)
    Real  drop_tol       = 1e-4;    ///< Reserved: ILU(τ) threshold
    int   amg_levels     = 10;      ///< Max AMG hierarchy depth
    Real  amg_strength   = 0.25;    ///< Strength-of-connection threshold
    Real  amg_strong     = 0.25;    ///< Alias used by adaptive selector
    int   amg_smooth_pre = 2;       ///< Pre-smoothing steps
    int   amg_smooth_post= 2;       ///< Post-smoothing steps
    int   stall_window   = 15;      ///< Iterations without sufficient reduction
    Real  stall_threshold= 0.95;    ///< Residual ratio to declare a stall
    bool  verbose        = false;   ///< Per-iteration output
    int   print_every    = 25;      ///< Print residual every N iterations
    // Thrust 3 — s-step and async parameters
    int   sstep_k         = 1;      ///< s-step batch size (1 = standard CG/GMRES)
    Real  sstep_cond_tol  = 1e12;   ///< Max basis condition number before reducing k
    bool  async_enabled   = false;  ///< Enable asynchronous non-blocking reductions
    int   async_max_stale = 3;      ///< Max staleness τ for async iterations
    // Thrust 1 — extra tuning exposed to GNN advisor
    int   fill_per_row    = 0;      ///< ILUT fill beyond sparsity pattern
    int   block_size      = 1;      ///< Block size for Block-Jacobi preconditioner
    // Thrust 2 — energy measurement
    bool  record_hw_energy = false; ///< Enable per-iteration RAPL/NVML snapshots
};

} // namespace hsps
