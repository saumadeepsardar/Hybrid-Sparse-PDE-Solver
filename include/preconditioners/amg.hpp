#pragma once

// =============================================================================
// amg.hpp  —  Algebraic Multigrid (Smoothed Aggregation) preconditioner
//
// Algorithm overview
// ------------------
//   Setup   : Build hierarchy via greedy aggregation → prolongation P →
//             restriction R = P^T → coarse operator A_c = R A P (Galerkin).
//             Recurse to a direct solve at the coarsest level.
//   V-cycle : Pre-smooth (Jacobi), restrict residual, coarse solve,
//             prolongate correction, post-smooth (Jacobi).
//   Apply   : One V-cycle per call  →  FGMRES-compatible (non-stationary).
//
// Energy tier : EXPENSIVE (setup), but excellent convergence per iteration.
// Usage       : Triggered on highly stiff/poorly conditioned systems
//               — design ladder HARD state.
//               "AMG — excellent convergence, expensive setup, best for hard
//               systems."
// =============================================================================

#include "preconditioner_base.hpp"
#include "../core/sparse_matrix.hpp"
#include "../core/vector.hpp"
#include <vector>
#include <memory>

namespace hsps {

// ---------------------------------------------------------------------------
// One level of the AMG hierarchy
// ---------------------------------------------------------------------------
struct AMGLevel {
    SparseMatrix A;          ///< Operator on this level
    SparseMatrix P;          ///< Prolongation  (fine ← coarse)
    SparseMatrix R;          ///< Restriction   (R = P^T)
    Vector       diag_inv;   ///< 1/diag(A) for Jacobi smoother
};

// ---------------------------------------------------------------------------
// AMG preconditioner
// ---------------------------------------------------------------------------
class AMGPreconditioner : public PreconditionerBase {
public:
    AMGPreconditioner() : PreconditionerBase(PrecondType::AMG) {}

    double setup(const SparseMatrix& A) override;
    void   apply(const Vector& r, Vector& z) const override;

    const char* energy_tier() const override { return "EXPENSIVE"; }

    // ------------------------------------------------------------------
    // Configuration (set before setup())
    // ------------------------------------------------------------------
    void set_max_levels     (int n)   { max_levels_      = n;   }
    void set_strength_thresh(Real t)  { strength_thresh_ = t;   }
    void set_smooth_pre     (int n)   { smooth_pre_       = n;   }
    void set_smooth_post    (int n)   { smooth_post_      = n;   }
    void set_coarse_limit   (int n)   { coarse_limit_     = n;   }
    /// Jacobi-smoothed prolongation (SA-AMG). Default: enabled, 1 step.
    void set_smooth_prolongation(bool on, int steps = 1) {
        smooth_p_ = on; smooth_p_steps_ = steps;
    }

    int levels() const { return static_cast<int>(hierarchy_.size()); }

private:
    // ------------------------------------------------------------------
    // Setup helpers
    // ------------------------------------------------------------------
    /// Greedy aggregation: assign each dof to an aggregate
    std::vector<Index> aggregate(const SparseMatrix& A) const;

    /// Jacobi-smooth prolongation: P = (I - omega*D^{-1}*A)^steps * P_raw
    SparseMatrix smooth_prolongation_op(const SparseMatrix& A,
                                        const SparseMatrix& P_raw) const;

    /// Build piecewise-constant prolongation from aggregate map
    SparseMatrix build_prolongation(Index n_fine,
                                    const std::vector<Index>& agg_map,
                                    Index n_coarse) const;

    // ------------------------------------------------------------------
    // V-cycle (recursive)
    // ------------------------------------------------------------------
    void v_cycle(int level, const Vector& rhs, Vector& x) const;

    /// Jacobi smoother:  x += ω * D^{-1} (rhs - A x)
    void smooth(int level, const Vector& rhs, Vector& x, int n_steps) const;

    // ------------------------------------------------------------------
    // Data
    // ------------------------------------------------------------------
    std::vector<AMGLevel> hierarchy_;

    int  max_levels_      = 10;
    Real strength_thresh_ = 0.25;
    int  smooth_pre_      = 2;
    int  smooth_post_     = 2;
    int  coarse_limit_    = 50;    ///< Direct solve below this size
    Real omega_           = 0.67;  ///< Jacobi damping  (2/3 for optimal)
    bool smooth_p_        = true;  ///< Jacobi-smoothed prolongation (SA-AMG)
    int  smooth_p_steps_  = 1;     ///< Number of smoothing sweeps on P
};

} // namespace hsps
