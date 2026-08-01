#pragma once

// =============================================================================
// cg_solver.hpp  —  Preconditioned Conjugate Gradient
//
// Requirements: A must be symmetric positive (semi-)definite.
// Default pair: CG + Jacobi (EASY adaptive state).
// =============================================================================

#include "solver_base.hpp"

namespace hsps {

class CGSolver : public SolverBase {
public:
    CGSolver() : SolverBase(SolverType::CG) {}

    bool solve(const SparseMatrix& A,
               const Vector&       b,
                     Vector&       x,
               SolverStats&        stats) override;
};

} // namespace hsps
