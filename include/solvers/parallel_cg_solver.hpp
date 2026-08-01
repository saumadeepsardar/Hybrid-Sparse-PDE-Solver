#pragma once

// =============================================================================
// parallel_cg_solver.hpp  —  Backend-dispatched Preconditioned Conjugate Gradient
//
// This solver uses a BackendBase pointer for all vector operations, making it
// transparently parallel across OMP / MPI / CUDA backends.
//
// For MPI: dot products include a global Allreduce via backend->global_dot().
// For CUDA: all vectors live on device; host copies only at input/output.
//
// Algorithm: same PCG as cg_solver.cpp but all BLAS-1 and SpMV calls go
// through the backend interface.
// =============================================================================

#include "../solvers/solver_base.hpp"
#include "../parallel/backend_base.hpp"
#include <memory>

namespace hsps {

class ParallelCGSolver : public SolverBase {
public:
    explicit ParallelCGSolver(std::shared_ptr<BackendBase> backend)
        : SolverBase(SolverType::CG), backend_(std::move(backend)) {}

    bool solve(const SparseMatrix& A,
               const Vector&       b,
                     Vector&       x,
               SolverStats&        stats) override;

    void set_backend(std::shared_ptr<BackendBase> b) { backend_ = b; }
    const BackendBase& backend() const { return *backend_; }

private:
    std::shared_ptr<BackendBase> backend_;
};

} // namespace hsps
