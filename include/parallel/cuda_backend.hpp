#pragma once

// =============================================================================
// cuda_backend.hpp  —  NVIDIA GPU backend via CUDA + cuBLAS + cuSPARSE
//
// Memory model
// ------------
//   • CPU Vector (host)  ←→  DeviceVector (device)
//   • The backend maintains a device CSR representation of each SparseMatrix.
//   • Explicit upload/download happens only at solve boundaries.
//   • All Krylov inner-loop operations run entirely on-device.
//
// Requires HSPS_USE_CUDA compile flag (set by Makefile BACKEND=CUDA).
// When HSPS_USE_CUDA is not defined, the class still compiles as a
// no-op stub that prints a warning and falls back to OMPBackend,
// allowing a single codebase to build for both CPU and GPU targets.
//
// cuSPARSE API notes
// ------------------
//   We target cuSPARSE ≥ 11.0 (CUDA ≥ 11.0) which uses the newer
//   cusparseSpMV_bufferSize / cusparseSpMV API.
//   For CUDA 10.x the legacy cusparseXcsrmv path is available via
//   the HSPS_CUDA_LEGACY_CUSPARSE compile flag.
// =============================================================================

#include "backend_base.hpp"
#include "../core/vector.hpp"
#include "../core/sparse_matrix.hpp"
#include "parallel_context.hpp"

#ifdef HSPS_USE_CUDA
#  include <cuda_runtime.h>
#  include <cublas_v2.h>
#  include <cusparse.h>
#endif

#include <memory>
#include <vector>

namespace hsps {

// ---------------------------------------------------------------------------
// DeviceVector: device-resident dense vector (RAII)
// ---------------------------------------------------------------------------
struct DeviceVector {
    Real*  d_data  = nullptr;  ///< Raw device pointer
    Index  n       = 0;        ///< Number of elements

    DeviceVector() = default;
    explicit DeviceVector(Index n, Real init_val = 0.0);
    ~DeviceVector();

    // Non-copyable, movable
    DeviceVector(const DeviceVector&)            = delete;
    DeviceVector& operator=(const DeviceVector&) = delete;
    DeviceVector(DeviceVector&&) noexcept;
    DeviceVector& operator=(DeviceVector&&) noexcept;

    void upload  (const Vector& host_vec);   ///< Host → Device (synchronous)
    void download(Vector& host_vec)   const; ///< Device → Host (synchronous)
    void fill    (Real val);
    bool valid() const { return d_data != nullptr; }
};

// ---------------------------------------------------------------------------
// DeviceSparseMatrix: device-resident CSR matrix for cuSPARSE SpMV
// ---------------------------------------------------------------------------
struct DeviceSparseMatrix {
    // CSR arrays on device
    int*   d_row_ptr  = nullptr;
    int*   d_col_idx  = nullptr;
    Real*  d_values   = nullptr;
    Index  rows       = 0;
    Index  cols       = 0;
    Index  nnz        = 0;

#ifdef HSPS_USE_CUDA
    cusparseSpMatDescr_t mat_descr   = nullptr;
    cusparseDnVecDescr_t vec_x_descr = nullptr;
    cusparseDnVecDescr_t vec_y_descr = nullptr;
    void*  spmv_buffer               = nullptr;
    size_t spmv_buffer_size          = 0;
#endif

    DeviceSparseMatrix() = default;
    ~DeviceSparseMatrix();

    DeviceSparseMatrix(const DeviceSparseMatrix&)            = delete;
    DeviceSparseMatrix& operator=(const DeviceSparseMatrix&) = delete;

    /// Upload a host SparseMatrix to the device and build cuSPARSE descriptors.
    void upload(const SparseMatrix& host_mat,
                const ParallelContext& ctx);
    bool valid() const { return d_row_ptr != nullptr; }
};

// ---------------------------------------------------------------------------
// CUDABackend
// ---------------------------------------------------------------------------
class CUDABackend : public BackendBase {
public:
    explicit CUDABackend(const ParallelContext& ctx);
    ~CUDABackend() override = default;

    const char* name()   const override { return "CUDA"; }
    bool        is_gpu() const override { return true;   }

    // ------------------------------------------------------------------
    // Vector helpers
    // ------------------------------------------------------------------
    void alloc(Index n, Real val, Vector& out) const override;

    // ------------------------------------------------------------------
    // BLAS-1  (via cuBLAS, operate on DeviceVectors internally)
    //
    // Design note: the public interface still takes host Vectors for
    // API uniformity.  Each call uploads/downloads as needed.
    // For maximum performance the caller should use the device-native
    // overloads below (operate directly on DeviceVector).
    // ------------------------------------------------------------------
    Real dot  (const Vector& x, const Vector& y) const override;
    Real nrm2 (const Vector& x)                  const override;
    void axpy (Real a, const Vector& x, Vector& y)    const override;
    void axpby(Real a, const Vector& x, Real b, Vector& y) const override;
    void copy (const Vector& src, Vector& dst)         const override;
    void scale(Real a, Vector& x)                      const override;
    void fill (Vector& x, Real val)                    const override;

    // ------------------------------------------------------------------
    // Device-native BLAS-1  (avoid host↔device copies in hot loops)
    // These are used by the GPU-aware Krylov solvers.
    // ------------------------------------------------------------------
    Real dot_device  (const DeviceVector& x, const DeviceVector& y) const;
    Real nrm2_device (const DeviceVector& x)                         const;
    void axpy_device (Real a, const DeviceVector& x, DeviceVector& y) const;
    void axpby_device(Real a, const DeviceVector& x,
                      Real b, DeviceVector& y)                         const;
    void copy_device (const DeviceVector& src, DeviceVector& dst)      const;
    void scale_device(Real a, DeviceVector& x)                         const;
    void fill_device (DeviceVector& x, Real val)                       const;

    // ------------------------------------------------------------------
    // SpMV  (cuSPARSE on device; host Vector API uploads/downloads)
    // ------------------------------------------------------------------
    void spmv(const SparseMatrix& A,
              const Vector& x, Vector& y) const override;

    /// Device-native SpMV — no host copies (use in hot loops)
    void spmv_device(const DeviceSparseMatrix& A,
                     const DeviceVector& x, DeviceVector& y) const;

    // ------------------------------------------------------------------
    // Convenience: upload host data → device, run Krylov, download result
    // ------------------------------------------------------------------
    void upload_matrix  (const SparseMatrix& A,  DeviceSparseMatrix& dA) const;
    void upload_vector  (const Vector& v,         DeviceVector& dv)       const;
    void download_vector(const DeviceVector& dv,  Vector& v)              const;

    // ------------------------------------------------------------------
    // Synchronisation
    // ------------------------------------------------------------------
    void sync() const override;

private:
    const ParallelContext& ctx_;

    // Check and throw on CUDA errors
    static void cuda_check(int code, const char* file, int line);
    static void cublas_check(int code, const char* file, int line);
    static void cusparse_check(int code, const char* file, int line);
};

// Convenience macros for error checking (only defined when CUDA is active)
#ifdef HSPS_USE_CUDA
#  define CUDA_CHECK(expr)     CUDABackend::cuda_check   ((expr), __FILE__, __LINE__)
#  define CUBLAS_CHECK(expr)   CUDABackend::cublas_check ((expr), __FILE__, __LINE__)
#  define CUSPARSE_CHECK(expr) CUDABackend::cusparse_check((expr),__FILE__, __LINE__)
#else
#  define CUDA_CHECK(expr)     (void)(expr)
#  define CUBLAS_CHECK(expr)   (void)(expr)
#  define CUSPARSE_CHECK(expr) (void)(expr)
#endif

} // namespace hsps
