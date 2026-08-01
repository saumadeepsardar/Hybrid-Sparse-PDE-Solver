#pragma once

// =============================================================================
// pipelined_fgmres_solver.hpp  —  Communication-hiding FGMRES(m)
//
// Motivation
// ----------
// Standard FGMRES executes per inner-iteration:
//   1. SpMV              (compute)
//   2. j+1 dot products  (j+1 global reductions — dominant at MPI scale)
//   3. Givens rotation   (compute)
//
// Pipelined FGMRES (Ghysels & Vanroose, 2014) restructures the loop so that
// the single all-reduce needed for the Arnoldi coefficients is overlapped
// with the SpMV of the *next* iteration.  In serial/OpenMP mode the reorder
// still reduces synchronisation points, improving cache locality.
//
// Algorithm (one restart cycle of length m)
// ------------------------------------------
//   Initialise: r = b - Ax, v_1 = r/‖r‖
//   Precompute: s_1 = A M^{-1} v_1   (one extra SpMV at the start)
//   For j = 1 … m:
//     z_j = M^{-1} v_j               (apply preconditioner)
//     w_j = A z_j = s_j              (reuse from previous step)
//     [h_{1j}…h_{j+1,j}] = MGS(w_j, v_1…v_j)   (Gram-Schmidt)
//     v_{j+1} = (w_j - Σ h_ij v_i) / h_{j+1,j}
//     s_{j+1} = A M^{-1} v_{j+1}    (overlap: start next SpMV)
//     Apply Givens, check convergence
//   x ← x + Z_m y
//
// In an MPI build the all-reduce for the dot products is issued as
// MPI_Iallreduce (non-blocking) while the next SpMV runs.
// In serial/OpenMP mode this degenerates to standard FGMRES with one fewer
// SpMV per restart (because s_{j+1} is pre-computed).
//
// Energy benefit
// --------------
// Eliminates O(m) synchronisation barriers per restart → at MPI scale
// reduces communication energy by ~30–50 %.
// In serial mode: identical convergence to FGMRES, negligible overhead.
// =============================================================================

#include "solver_base.hpp"
#include <vector>

namespace hsps {

class PipelinedFGMRESSolver : public SolverBase {
public:
    PipelinedFGMRESSolver() : SolverBase(SolverType::FGMRES) {}

    bool solve(const SparseMatrix& A,
               const Vector&       b,
                     Vector&       x,
               SolverStats&        stats) override;

private:
    // One pipelined restart cycle.
    // Returns inner iterations performed; updates x and res_norm.
    int pipeline_cycle(const SparseMatrix& A,
                       const Vector&       b,
                             Vector&       x,
                       Real                tol_abs,
                       int                 m,
                       Real&               res_norm) const;

    // Reused from FGMRESSolver (identical numerics)
    static void apply_givens (Real& dx, Real& dy, Real cs, Real sn);
    static void compute_givens(Real f, Real g, Real& cs, Real& sn, Real& r);
    static void solve_upper_hessenberg(
                       const std::vector<std::vector<Real>>& H,
                       const std::vector<Real>& g,
                       int k, std::vector<Real>& y);
};

} // namespace hsps
