#pragma once

// =============================================================================
// sstep_cg_solver.hpp  —  Communication-Avoiding s-step Conjugate Gradient
//
// Thrust 3: reduces MPI_Allreduce calls from k per k-iterations to 1.
//
// Algorithm overview (Hoemmen 2010, "Communication-Avoiding Krylov Methods")
// ---------------------------------------------------------------------------
// Standard CG requires 1 global Allreduce per iteration.
// s-step CG batches k iterations:
//
//   Step 1: Build the Krylov basis B_k = [r, Ar, A²r, …, A^{k-1}r]
//           (k SpMVs, but fully local — no communication)
//
//   Step 2: Compute all k×k Gram matrices at once:
//             G = B_k^T B_k   (local computation)
//             One MPI_Allreduce to sum G across ranks
//
//   Step 3: Orthogonalise via classical Gram-Schmidt on G (local)
//           Update x using the k computed step vectors (local)
//
// Communication cost: 1 Allreduce per k iterations vs. k for standard CG.
// At k=4, P=256 processes: ~4× reduction in communication energy.
//
// Numerical stability
// -------------------
// The monomial basis {r, Ar, …, A^k r} is ill-conditioned for large k.
// We use the Newton basis with shifts from the Lanczos spectral estimate:
//   v_j = (A - θ_j I) v_{j-1}   where θ_j ∈ Chebyshev nodes on [λ_min, λ_max]
//
// The Chebyshev shifts are computed from spectral_interval() (already in
// SparseMatrix — added in Month 1 of this plan).
//
// Adaptive k selection
// --------------------
// Before each batch, we estimate the condition number of B_k:
//   if κ(B_k) > params_.sstep_cond_tol → reduce k by 1 (minimum k=1 = standard CG)
// This is computed cheaply via the singular value ratio of the Gram matrix.
//
// Key property for MPI
// --------------------
// The single Allreduce in Step 2 collects k×k = O(k²) doubles.
// For k=4, that is 16 doubles vs. k=4 separate reductions of 1 double each.
// Same communication volume, but 4× fewer round-trip latencies.
// =============================================================================

#include "../solvers/solver_base.hpp"
#include "../core/sparse_matrix.hpp"
#include "../core/vector.hpp"
#include "../parallel/parallel_context.hpp"
#include <vector>
#include <memory>

namespace hsps {

class SStepCGSolver : public SolverBase {
public:
    explicit SStepCGSolver(int k = 4)
        : SolverBase(SolverType::SSTEP_CG), k_(k) {}

    bool solve(const SparseMatrix& A,
               const Vector&       b,
                     Vector&       x,
               SolverStats&        stats) override;

    /// Set MPI context for global reductions (required for distributed runs)
    void set_parallel_context(const ParallelContext* ctx) { ctx_ = ctx; }

    /// Set step size k (batch size). Adaptive k reduces this if basis ill-conditioned.
    void set_k(int k) { k_ = k; }
    int  k()    const { return k_; }

private:
    // ------------------------------------------------------------------
    // One s-step batch
    // ------------------------------------------------------------------

    /// Build Newton-Chebyshev basis of length k starting from residual r.
    /// Returns B_k = [v_0, v_1, ..., v_{k-1}] where v_j = (A - θ_j) v_{j-1}.
    std::vector<Vector> build_basis(const SparseMatrix& A,
                                     const Vector& r,
                                     const std::vector<Real>& chebyshev_shifts) const;

    /// Compute Gram matrix G[i][j] = B_i · B_j (local dot products)
    void gram_matrix(const std::vector<Vector>& B,
                     std::vector<std::vector<Real>>& G) const;

    /// Global Allreduce of the k×k Gram matrix (one MPI call for k² doubles)
    void allreduce_gram(std::vector<std::vector<Real>>& G) const;

    /// Estimate condition number of Gram matrix via SVD-free ratio estimate
    double gram_condition_estimate(const std::vector<std::vector<Real>>& G) const;

    /// Compute Chebyshev nodes on interval [lam_min, lam_max] as shifts
    static std::vector<Real> chebyshev_shifts(Real lam_min, Real lam_max, int k);

    /// Solve the k×k local least-squares system for the update coefficients
    bool local_lstsq(const std::vector<std::vector<Real>>& G,
                     const std::vector<Real>& rhs,
                     std::vector<Real>& coeffs) const;

    // ------------------------------------------------------------------
    // Data
    // ------------------------------------------------------------------
    int                    k_   = 4;   ///< Current batch size
    const ParallelContext* ctx_ = nullptr;
};

} // namespace hsps
