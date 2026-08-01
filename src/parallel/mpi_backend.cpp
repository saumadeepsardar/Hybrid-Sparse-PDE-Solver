// =============================================================================
// mpi_backend.cpp
// =============================================================================

#include "../../include/parallel/mpi_backend.hpp"
#include <cmath>
#include <stdexcept>

#ifdef HSPS_USE_MPI
#  include <mpi.h>
#endif

namespace hsps {

MPIBackend::MPIBackend(const ParallelContext& ctx, int omp_threads)
    : OMPBackend(omp_threads), ctx_(ctx) {}

// ---------------------------------------------------------------------------
// Global dot/norm reductions
// ---------------------------------------------------------------------------
double MPIBackend::mpi_allreduce_sum(double local) const {
#ifdef HSPS_USE_MPI
    if (ctx_.nprocs() > 1) {
        double result = 0.0;
        MPI_Allreduce(&local, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return result;
    }
#endif
    return local;
}

Real MPIBackend::global_dot(Real local_dot) const {
    return static_cast<Real>(mpi_allreduce_sum(static_cast<double>(local_dot)));
}

Real MPIBackend::global_nrm2(Real local_sum2) const {
    double global_sum2 = mpi_allreduce_sum(static_cast<double>(local_sum2));
    return static_cast<Real>(std::sqrt(global_sum2));
}

// ---------------------------------------------------------------------------
// DistributedVector dot/nrm2
// ---------------------------------------------------------------------------
Real MPIBackend::dot_distributed(const DistributedVector& x,
                                  const DistributedVector& y) const {
    Real local_dot = OMPBackend::dot(x.local, y.local);
    return global_dot(local_dot);
}

Real MPIBackend::nrm2_distributed(const DistributedVector& x) const {
    Real local_dot = OMPBackend::dot(x.local, x.local);
    return global_nrm2(local_dot);
}

// ---------------------------------------------------------------------------
// Distributed SpMV (delegates to DistributedMatrix which handles halo exchange)
// ---------------------------------------------------------------------------
void MPIBackend::spmv_distributed(const DistributedMatrix& A,
                                   const DistributedVector& x,
                                         DistributedVector& y) const {
    A.spmv(x, y, ctx_);
}

// ---------------------------------------------------------------------------
// Barrier
// ---------------------------------------------------------------------------
void MPIBackend::do_barrier() const {
#ifdef HSPS_USE_MPI
    if (ctx_.config().mpi_inited)
        MPI_Barrier(MPI_COMM_WORLD);
#endif
}

} // namespace hsps
