#pragma once

// =============================================================================
// backend_base.hpp  —  Pure-virtual interface for a parallel compute backend
//
// Every backend (OMP / MPI / CUDA) must implement this interface.
// The Krylov solvers call only these primitives, making them backend-agnostic.
//
// Operation naming follows BLAS-1 conventions:
//   dot(x, y)          — scalar inner product (with global reduction for MPI)
//   nrm2(x)            — Euclidean norm
//   axpy(a, x, y)      — y += a * x
//   axpby(a, x, b, y)  — y = a*x + b*y
//   copy(x, y)         — y = x
//   scale(a, x)        — x *= a
//   fill(x, v)         — x[:] = v
//   spmv(A, x, y)      — y = A * x  (sparse matrix-vector product)
//   precond_apply(M,r,z) — z = M^{-1} r
//
// All operations are synchronous from the caller's point of view.
// Internally, CUDA backends may use streams; they must synchronize before
// returning control to the host.
// =============================================================================

#include "../core/types.hpp"
#include <string>

namespace hsps {

// Forward declarations
class SparseMatrix;
class Vector;
class PreconditionerBase;

// ---------------------------------------------------------------------------
// Abstract backend interface
// ---------------------------------------------------------------------------
class BackendBase {
public:
    virtual ~BackendBase() = default;

    // ------------------------------------------------------------------
    // Identity
    // ------------------------------------------------------------------
    virtual const char* name()  const = 0;
    virtual bool        is_gpu() const { return false; }

    // ------------------------------------------------------------------
    // Vector creation helpers
    // ------------------------------------------------------------------
    /// Allocate a vector of length n, optionally filled with val.
    /// Backends that have device memory must allocate on the appropriate device.
    virtual void alloc(Index n, Real val, Vector& out) const = 0;
    virtual void free_vec(Vector& v) const { (void)v; } // default: no-op for CPU

    // ------------------------------------------------------------------
    // BLAS-1  (all operate on the backend's native storage)
    // ------------------------------------------------------------------
    virtual Real dot   (const Vector& x, const Vector& y) const = 0;
    virtual Real nrm2  (const Vector& x)                  const = 0;
    virtual void axpy  (Real a, const Vector& x, Vector& y)    const = 0;
    virtual void axpby (Real a, const Vector& x,
                        Real b, Vector& y)                      const = 0;
    virtual void copy  (const Vector& src, Vector& dst)         const = 0;
    virtual void scale (Real a, Vector& x)                      const = 0;
    virtual void fill  (Vector& x, Real val)                    const = 0;

    // ------------------------------------------------------------------
    // Sparse matrix-vector product  y = A * x
    // ------------------------------------------------------------------
    virtual void spmv(const SparseMatrix& A,
                      const Vector& x, Vector& y) const = 0;

    // ------------------------------------------------------------------
    // Preconditioner apply  z = M^{-1} r
    // ------------------------------------------------------------------
    virtual void precond_apply(const PreconditionerBase& M,
                               const Vector& r, Vector& z) const;

    // ------------------------------------------------------------------
    // Global reductions (no-op for single-process backends)
    // ------------------------------------------------------------------
    virtual Real global_dot (Real local_dot)  const { return local_dot; }
    virtual Real global_nrm2(Real local_sum2) const { return local_sum2; }

    // ------------------------------------------------------------------
    // Synchronisation  (no-op for CPU; cudaDeviceSynchronize for GPU)
    // ------------------------------------------------------------------
    virtual void sync() const {}
};

} // namespace hsps
