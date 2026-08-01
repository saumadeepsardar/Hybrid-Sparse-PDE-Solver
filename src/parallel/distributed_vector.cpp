// =============================================================================
// distributed_vector.cpp
// =============================================================================

#include "../../include/parallel/distributed_vector.hpp"
#include <numeric>
#include <stdexcept>
#include <cmath>

#ifdef HSPS_USE_MPI
#  include <mpi.h>
#endif

namespace hsps {

// ---------------------------------------------------------------------------
// compute_split: divide N global rows evenly across nprocs ranks
// ---------------------------------------------------------------------------
void DistributedVector::compute_split(Index global_n, int nprocs,
                                       std::vector<Index>& counts,
                                       std::vector<Index>& offsets) {
    counts .resize(nprocs);
    offsets.resize(nprocs);
    const Index base = global_n / nprocs;
    const Index rem  = global_n % nprocs;
    Index off = 0;
    for (int r = 0; r < nprocs; ++r) {
        counts[r]  = base + (r < rem ? 1 : 0);
        offsets[r] = off;
        off       += counts[r];
    }
}

// ---------------------------------------------------------------------------
// create: allocate local segment for this rank
// ---------------------------------------------------------------------------
DistributedVector DistributedVector::create(Index global_n, Real init_val,
                                             const ParallelContext& ctx) {
    DistributedVector dv;
    dv.global_n = global_n;

    std::vector<Index> counts, offsets;
    compute_split(global_n, ctx.nprocs(), counts, offsets);

    const int rank = ctx.rank();
    dv.local_offset = offsets[rank];
    dv.local.resize(counts[rank], init_val);
    return dv;
}

// ---------------------------------------------------------------------------
// scatter: broadcast full vector (on rank 0) to all ranks
// ---------------------------------------------------------------------------
DistributedVector DistributedVector::scatter(const Vector& full,
                                              const ParallelContext& ctx) {
    DistributedVector dv;
    dv.global_n = (ctx.rank() == 0) ? full.size() : 0;

#ifdef HSPS_USE_MPI
    // Broadcast global length first
    Index gn = dv.global_n;
    MPI_Bcast(&gn, 1, MPI_INT, 0, MPI_COMM_WORLD);
    dv.global_n = gn;

    std::vector<Index> counts, offsets;
    compute_split(gn, ctx.nprocs(), counts, offsets);
    dv.local_offset = offsets[ctx.rank()];
    dv.local.resize(counts[ctx.rank()]);

    // Scatterv from rank 0
    // MPI requires int counts/displs
    std::vector<int> icounts(counts.begin(), counts.end());
    std::vector<int> idispls(offsets.begin(), offsets.end());

    MPI_Scatterv(
        ctx.rank() == 0 ? full.data() : nullptr,
        icounts.data(), idispls.data(), MPI_DOUBLE,
        dv.local.data(), static_cast<int>(dv.local.size()), MPI_DOUBLE,
        0, MPI_COMM_WORLD);
#else
    // Single process: local = full
    dv.global_n     = full.size();
    dv.local_offset = 0;
    dv.local        = full;
#endif
    return dv;
}

// ---------------------------------------------------------------------------
// gather: collect all local segments to rank 0
// ---------------------------------------------------------------------------
Vector DistributedVector::gather(const ParallelContext& ctx) const {
    Vector result;
#ifdef HSPS_USE_MPI
    std::vector<Index> counts, offsets;
    compute_split(global_n, ctx.nprocs(), counts, offsets);
    std::vector<int> icounts(counts.begin(), counts.end());
    std::vector<int> idispls(offsets.begin(), offsets.end());

    if (ctx.rank() == 0) result.resize(global_n);
    MPI_Gatherv(
        local.data(), static_cast<int>(local.size()), MPI_DOUBLE,
        ctx.rank() == 0 ? result.data() : nullptr,
        icounts.data(), idispls.data(), MPI_DOUBLE,
        0, MPI_COMM_WORLD);
#else
    result = local;
#endif
    return result;
}

// ---------------------------------------------------------------------------
// Global reductions
// ---------------------------------------------------------------------------
Real DistributedVector::global_dot(const DistributedVector& other,
                                    const ParallelContext& ctx) const {
    Real local_s = REAL_ZERO;
    const Index n = local.size();
    for (Index i = 0; i < n; ++i)
        local_s += local[i] * other.local[i];
    return static_cast<Real>(ctx.allreduce_sum(static_cast<double>(local_s)));
}

Real DistributedVector::global_nrm2(const ParallelContext& ctx) const {
    Real local_s = REAL_ZERO;
    for (Index i = 0; i < local.size(); ++i)
        local_s += local[i] * local[i];
    double g = ctx.allreduce_sum(static_cast<double>(local_s));
    return static_cast<Real>(std::sqrt(g));
}

} // namespace hsps
