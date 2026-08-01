#pragma once

// =============================================================================
// omp_backend.hpp  —  Shared-memory OpenMP backend
//
// All primitives operate on the host Vector (std::vector<Real>) using
// OpenMP parallel loops.  This is always available — it is the default
// backend when no accelerator is requested.
//
// Thread count:
//   Set via ParallelConfig::omp_threads or the OMP_NUM_THREADS env var.
//   A value of 0 defers to the OpenMP runtime default.
// =============================================================================

#include "backend_base.hpp"
#include "../core/vector.hpp"
#include "../core/sparse_matrix.hpp"

namespace hsps {

class OMPBackend : public BackendBase {
public:
    explicit OMPBackend(int num_threads = 0);

    const char* name()   const override { return "OpenMP"; }
    bool        is_gpu() const override { return false; }

    // ------------------------------------------------------------------
    // Vector helpers
    // ------------------------------------------------------------------
    void alloc(Index n, Real val, Vector& out) const override;

    // ------------------------------------------------------------------
    // BLAS-1  —  OpenMP-parallelised implementations
    // ------------------------------------------------------------------
    Real dot  (const Vector& x, const Vector& y) const override;
    Real nrm2 (const Vector& x)                  const override;
    void axpy (Real a, const Vector& x, Vector& y)     const override;
    void axpby(Real a, const Vector& x, Real b, Vector& y) const override;
    void copy (const Vector& src, Vector& dst)          const override;
    void scale(Real a, Vector& x)                       const override;
    void fill (Vector& x, Real val)                     const override;

    // ------------------------------------------------------------------
    // SpMV  — row-parallel CSR SpMV (schedule dynamic for load balance)
    // ------------------------------------------------------------------
    void spmv(const SparseMatrix& A,
              const Vector& x, Vector& y) const override;

    // ------------------------------------------------------------------
    // Thread count management
    // ------------------------------------------------------------------
    int  threads()     const { return threads_; }
    void set_threads(int n);

private:
    int threads_;
};

} // namespace hsps
