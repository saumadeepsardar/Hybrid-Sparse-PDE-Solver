#pragma once

// =============================================================================
// parallel_fgmres_solver.hpp  —  Backend-dispatched Flexible GMRES(m)
//
// All vector operations (dot, axpy, nrm2, spmv) are routed through the
// BackendBase interface, so this single implementation runs correctly on
// OMP / MPI / CUDA backends.
//
// For MPI: each Arnoldi Gram-Schmidt step issues one global_dot reduction.
//   With pipelined mode: the reduction is issued before the next SpMV so
//   it can overlap with computation (see PipelinedFGMRESSolver).
//
// For CUDA: all Krylov vectors (V, Z) are DeviceVectors; the solve loop
//   never copies data back to host until convergence.
// =============================================================================

#include "../solvers/solver_base.hpp"
#include "../parallel/backend_base.hpp"
#include <memory>
#include <vector>

namespace hsps {

class ParallelFGMRESSolver : public SolverBase {
public:
    explicit ParallelFGMRESSolver(std::shared_ptr<BackendBase> backend)
        : SolverBase(SolverType::FGMRES), backend_(std::move(backend)) {}

    bool solve(const SparseMatrix& A,
               const Vector&       b,
                     Vector&       x,
               SolverStats&        stats) override;

    void set_backend(std::shared_ptr<BackendBase> b) { backend_ = b; }

private:
    std::shared_ptr<BackendBase> backend_;

    int inner_cycle(const SparseMatrix& A,
                    const Vector& b, Vector& x,
                    Real tol_abs, int m, Real& res_norm) const;

    static void compute_givens(Real f, Real g,
                               Real& cs, Real& sn, Real& r);
    static void apply_givens  (Real& dx, Real& dy, Real cs, Real sn);
    static void solve_upper_hessenberg(
                    const std::vector<std::vector<Real>>& H,
                    const std::vector<Real>& g,
                    int k, std::vector<Real>& y);
};

} // namespace hsps
