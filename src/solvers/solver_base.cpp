// =============================================================================
// solver_base.cpp
// =============================================================================

#include "../../include/solvers/solver_base.hpp"

namespace hsps {

Real SolverBase::relative_residual(const SparseMatrix& A,
                                    const Vector& b,
                                    const Vector& x) const {
    Vector r = A * x;
    Real bnorm = b.norm2();
    for (Index i = 0; i < b.size(); ++i)
        r[i] = b[i] - r[i];
    return (bnorm > REAL_EPS) ? (r.norm2() / bnorm) : r.norm2();
}

} // namespace hsps
