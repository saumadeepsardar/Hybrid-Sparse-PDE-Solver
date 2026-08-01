#pragma once

// =============================================================================
// parallel_context.hpp  —  RAII singleton that owns all parallel resources
//
// Usage
// -----
//   // At program start (before any parallel operations):
//   ParallelContext ctx = ParallelContext::init(argc, argv, config);
//
//   // Access anywhere:
//   auto& ctx = ParallelContext::get();
//   ctx.barrier();           // MPI_Barrier or OMP barrier
//   ctx.allreduce_sum(val);  // MPI_Allreduce or no-op
//
//   // Destructor calls MPI_Finalize and cudaDeviceReset automatically.
// =============================================================================

#include "parallel_config.hpp"
#include <memory>
#include <ostream>
#include <functional>

// Forward declarations so header doesn't pull in heavy CUDA/MPI headers
// unless the corresponding feature is compiled in.
#ifdef HSPS_USE_MPI
#  include <mpi.h>
#endif

#ifdef HSPS_USE_CUDA
#  include <cuda_runtime.h>
#  include <cublas_v2.h>
#  include <cusparse.h>
#endif

namespace hsps {

class ParallelContext {
public:
    // ------------------------------------------------------------------
    // Lifecycle  (call init() exactly once before any parallel work)
    // ------------------------------------------------------------------

    /// Initialise all requested backends and return an owning context.
    /// Pass argc/argv for MPI_Init; they are ignored if MPI is disabled.
    static ParallelContext init(int argc, char** argv,
                                ParallelConfig config = {});

    /// Access the process-global context (must have called init() first).
    static ParallelContext& get();

    ~ParallelContext();

    // Non-copyable, movable
    ParallelContext(const ParallelContext&) = delete;
    ParallelContext& operator=(const ParallelContext&) = delete;
    ParallelContext(ParallelContext&&) noexcept;
    ParallelContext& operator=(ParallelContext&&) noexcept;

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------
    const ParallelConfig& config() const { return cfg_; }

    int rank()     const { return cfg_.mpi_rank;  }
    int nprocs()   const { return cfg_.mpi_size;  }
    bool is_root() const { return cfg_.mpi_rank == 0; }

    int omp_threads()  const;
    int cuda_device()  const { return cfg_.cuda_device; }

    // ------------------------------------------------------------------
    // Collective operations  (no-op for inactive backends)
    // ------------------------------------------------------------------

    /// Block until all processes/threads reach this point.
    void barrier() const;

    /// Sum a scalar across all MPI ranks.
    double allreduce_sum(double local_val) const;

    /// Sum an array across all MPI ranks (in-place on root, broadcast to all).
    void allreduce_sum(double* data, int count) const;

    // ------------------------------------------------------------------
    // CUDA handles (only valid if CUDA backend active)
    // ------------------------------------------------------------------
#ifdef HSPS_USE_CUDA
    cublasHandle_t  cublas_handle()  const { return cublas_handle_;  }
    cusparseHandle_t cusparse_handle() const { return cusparse_handle_; }
#endif

    // ------------------------------------------------------------------
    // Diagnostics
    // ------------------------------------------------------------------
    void print_info(std::ostream& os) const;

private:
    ParallelContext() = default;

    void init_omp();
    void init_mpi(int argc, char** argv);
    void init_cuda();

    void finalize_mpi();
    void finalize_cuda();

    ParallelConfig cfg_;
    bool           owns_mpi_  = false;
    bool           owns_cuda_ = false;

#ifdef HSPS_USE_CUDA
    cublasHandle_t   cublas_handle_   = nullptr;
    cusparseHandle_t cusparse_handle_ = nullptr;
#endif

    // Process-global singleton pointer (set by init())
    static ParallelContext* instance_;
};

} // namespace hsps
