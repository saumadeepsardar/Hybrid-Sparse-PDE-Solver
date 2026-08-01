#pragma once

// =============================================================================
// ilu.hpp  —  Incomplete LU factorisation with zero fill (ILU(0))
//
// Factorises A into L·U with the same sparsity pattern as A.
// Apply: forward/backward substitution with L and U.
//
// Energy tier : MODERATE
// Usage       : Triggered when CG/Jacobi stalls — design ladder MODERATE state.
//               "ILU — moderate energy, good convergence."
// =============================================================================

#include "preconditioner_base.hpp"
#include <vector>

namespace hsps {

class ILUPreconditioner : public PreconditionerBase {
public:
    ILUPreconditioner() : PreconditionerBase(PrecondType::ILU) {}

    double setup(const SparseMatrix& A) override;
    void   apply(const Vector& r, Vector& z) const override;

    const char* energy_tier() const override { return "MODERATE"; }

    // ------------------------------------------------------------------
    // ILUT(τ, p) controls — set BEFORE calling setup()
    // ------------------------------------------------------------------
    /// Drop threshold τ: discard fill entries with |a_ij| < τ · ‖row_i‖
    void set_drop_tol(Real tau)   { drop_tol_ = tau;  }
    /// Max fill per row p (beyond original pattern). 0 = ILU(0) exactly.
    void set_fill_per_row(int p)  { fill_p_   = p;    }

private:
    // ILU(0)/ILUT factors in CSR layout
    Index              n_       = 0;
    std::vector<Index> row_ptr_;
    std::vector<Index> col_idx_;
    std::vector<Real>  lu_val_;

    Real drop_tol_ = 0.0;   ///< 0 → ILU(0); > 0 → ILUT threshold dropping
    int  fill_p_   = 0;     ///< Extra fill per row beyond pattern (ILUT)

    void forward_sub (const Vector& r, Vector& y) const;
    void backward_sub(const Vector& y, Vector& z) const;

    // ILUT setup: factorises with threshold dropping + fill limit
    double setup_ilut(const SparseMatrix& A);
    // ILU(0) setup: no fill, retains original sparsity pattern
    double setup_ilu0(const SparseMatrix& A);
};

} // namespace hsps
