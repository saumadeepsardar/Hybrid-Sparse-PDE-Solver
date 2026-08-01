#pragma once

// =============================================================================
// mpi_backend.hpp  —  Distributed-memory MPI backend
//
// Extends OMPBackend for local computation and adds:
//   • Global dot/norm reductions via MPI_Allreduce
//   • Distributed SpMV via DistributedMatrix (halo exchange)
//   • Collective barrier
//
// Requires HSPS_USE_MPI to be defined at compile time.
// If not defined, this header still compiles but all MPI calls are stubs
// that behave as if nprocs==1, enabling single-binary serial testing.
//
// Typical usage with MPI:
//   mpirun -np 4 ./bin/poisson_2d_mpi 64
//
// The MPI backend always layers ON TOP of OpenMP — each MPI rank runs
// omp_threads OpenMP threads (hybrid parallelism).
// =============================================================================

#include "omp_backend.hpp"
#include "parallel_context.hpp"
#include "distributed_vector.hpp"
#include "distributed_matrix.hpp"

namespace hsps {

class MPIBackend : public OMPBackend {
public:
    explicit MPIBackend(const ParallelContext& ctx, int omp_threads = 0);

    const char* name() const override { return "MPI+OpenMP"; }

    // ------------------------------------------------------------------
    // Global reductions (override pure-local versions in OMPBackend)
    // ------------------------------------------------------------------
    Real global_dot (Real local_dot)  const override;
    Real global_nrm2(Real local_sum2) const override;

    /// Full global dot over DistributedVectors (includes Allreduce)
    Real dot_distributed(const DistributedVector& x,
                         const DistributedVector& y) const;

    /// Full global norm over DistributedVector
    Real nrm2_distributed(const DistributedVector& x) const;

    // ------------------------------------------------------------------
    // Distributed SpMV
    // ------------------------------------------------------------------
    void spmv_distributed(const DistributedMatrix& A,
                           const DistributedVector& x,
                                 DistributedVector& y) const;

    // ------------------------------------------------------------------
    // Collective barrier across all ranks
    // ------------------------------------------------------------------

    void do_barrier() const;   ///< MPI_Barrier (non-override method)

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------
    int rank()   const { return ctx_.rank();   }
    int nprocs() const { return ctx_.nprocs(); }

private:
    const ParallelContext& ctx_;

    /// Internal: MPI_Allreduce SUM for a single double
    double mpi_allreduce_sum(double local) const;
};

} // namespace hsps
