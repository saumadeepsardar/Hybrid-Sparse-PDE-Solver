#pragma once

// =============================================================================
// async_fgmres_solver.hpp  —  Asynchronous non-blocking FGMRES(m)
//
// Thrust 3: overlaps MPI_Allreduce with the next SpMV computation.
//
// Standard FGMRES per inner iteration j:
//   1. z_j = M^{-1} v_j                  (local: precond apply)
//   2. w   = A z_j                        (local: SpMV)
//   3. h_ij = <w, v_i> for i=0..j        (GLOBAL: j+1 dot products → Allreduce)
//   4. Gram-Schmidt update                 (local)
//   5. Givens rotation                     (local)
//
// Async FGMRES restructuring (Ghysels & Vanroose 2014):
//   1. z_j = M^{-1} v_j
//   2. w   = A z_j
//   3. Issue MPI_Iallreduce for all h_ij values simultaneously (NON-BLOCKING)
//   4. Start computing z_{j+1} = M^{-1} v_{j+1}   ← OVERLAPS with reduction
//   5. Compute w_{j+1} = A z_{j+1}                 ← OVERLAPS with reduction
//   6. MPI_Wait for the Iallreduce from step 3 to complete
//   7. Continue Gram-Schmidt with received h values
//
// In serial/OMP mode (no MPI): steps 3-6 reduce to a standard synchronous
// dot product — identical numerics, negligible overhead.
//
// Energy benefit
// --------------
// At P=256 processes with network latency τ=5μs:
//   Standard FGMRES(50): 50 Allreduces × 5μs = 250μs pure waiting per restart
//   Async FGMRES(50):    ≤ 1 unmasked Allreduce latency per restart
//   Reduction: up to 49× fewer stall cycles per restart.
//   At P=1024: proportionally larger gain.
//
// Convergence
// -----------
// Async FGMRES is mathematically identical to standard FGMRES when the
// non-blocking reductions complete before their results are used.
// The "async" label refers to the overlap of communication with computation,
// NOT to using stale values (that would be a different algorithm).
// =============================================================================

#include "../solvers/solver_base.hpp"
#include "../core/sparse_matrix.hpp"
#include "../parallel/parallel_context.hpp"
#include <vector>
#include <memory>

namespace hsps {

class AsyncFGMRESSolver : public SolverBase {
public:
    explicit AsyncFGMRESSolver(const ParallelContext* ctx = nullptr)
        : SolverBase(SolverType::ASYNC_FGMRES), ctx_(ctx) {}

    bool solve(const SparseMatrix& A,
               const Vector&       b,
                     Vector&       x,
               SolverStats&        stats) override;

    void set_parallel_context(const ParallelContext* ctx) { ctx_ = ctx; }

private:
    // One restart cycle with non-blocking reductions
    int async_cycle(const SparseMatrix& A,
                    const Vector& b, Vector& x,
                    Real tol_abs, int m, Real& res_norm) const;

    // Issue non-blocking Allreduce for all inner products of column j
    // Returns a request handle (MPI_Request) cast to void*
    void* issue_iallreduce(std::vector<double>& local_dots,
                            std::vector<double>& global_dots,
                            int count) const;

    // Wait for previously issued Iallreduce
    void wait_iallreduce(void* request) const;

    static void compute_givens(Real f, Real g, Real& cs, Real& sn, Real& r);
    static void apply_givens  (Real& dx, Real& dy, Real cs, Real sn);
    static void solve_upper_hessenberg(
                    const std::vector<std::vector<Real>>& H,
                    const std::vector<Real>& g,
                    int k, std::vector<Real>& y);

    const ParallelContext* ctx_ = nullptr;
};

} // namespace hsps
