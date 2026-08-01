#pragma once

// =============================================================================
// fgmres_solver.hpp  —  Flexible Generalised Minimum Residual (FGMRES)
//
// Saad (1993): unlike standard GMRES, FGMRES allows the preconditioner to
// change at each step, making it compatible with AMG V-cycles and any other
// non-stationary preconditioner.
//
// Default pair: FGMRES + ILU  (MODERATE)
//               FGMRES + AMG  (HARD)
// =============================================================================

#include "solver_base.hpp"
#include <vector>

namespace hsps {

class FGMRESSolver : public SolverBase {
public:
    FGMRESSolver() : SolverBase(SolverType::FGMRES) {}

    bool solve(const SparseMatrix& A,
               const Vector&       b,
                     Vector&       x,
               SolverStats&        stats) override;

private:
    // ------------------------------------------------------------------
    // One restart cycle of FGMRES(m)
    // Returns the number of inner iterations performed and updates x.
    // ------------------------------------------------------------------
    int  fgmres_cycle(const SparseMatrix& A,
                      const Vector&       b,
                            Vector&       x,
                      Real                tol_abs,
                      int                 m,
                      Real&               res_norm) const;

    // Apply Givens rotation
    static void apply_givens(Real& dx, Real& dy, Real cs, Real sn);
    // Compute Givens rotation coefficients
    static void compute_givens(Real f, Real g, Real& cs, Real& sn, Real& r);

    // Solve upper Hessenberg least-squares via back substitution
    static void solve_hessenberg(const std::vector<std::vector<Real>>& H,
                                 const std::vector<Real>& g,
                                 int k, std::vector<Real>& y);
};

} // namespace hsps
