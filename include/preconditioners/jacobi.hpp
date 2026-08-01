#pragma once

// =============================================================================
// jacobi.hpp  —  Diagonal (Jacobi) preconditioner
//
// z_i = r_i / a_{ii}
//
// Energy tier : CHEAP
// Parallelism : embarrassingly parallel (OpenMP)
// Usage       : Best early in solve, low-conditioned problems.
//               Design note: "Jacobi — very low energy, cheap, highly parallel,
//               useful early."
// =============================================================================

#include "preconditioner_base.hpp"

namespace hsps {

class JacobiPreconditioner : public PreconditionerBase {
public:
    JacobiPreconditioner() : PreconditionerBase(PrecondType::JACOBI) {}

    double setup(const SparseMatrix& A) override;
    void   apply(const Vector& r, Vector& z) const override;

    const char* energy_tier() const override { return "CHEAP"; }

    /// Damping factor ω (default 1.0 = standard Jacobi)
    void set_omega(Real omega) { omega_ = omega; }

private:
    Vector inv_diag_;   ///< 1 / a_{ii}  (pre-computed at setup)
    Real   omega_ = 1.0;
};

} // namespace hsps
