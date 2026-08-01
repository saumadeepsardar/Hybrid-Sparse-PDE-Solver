#pragma once

// =============================================================================
// distributed_vector.hpp  —  Row-partitioned distributed dense vector
//
// Layout
// ------
//   Global vector of length N is split into contiguous blocks:
//     rank 0 owns rows  [0,         local_n[0])
//     rank 1 owns rows  [offset[1], offset[1]+local_n[1])
//     ...
//
//   Each rank holds only its local segment in `local` (a standard Vector).
//   Global operations (dot, norm) require an MPI_Allreduce.
//
// Construction
// ------------
//   // Even split:
//   DistributedVector v = DistributedVector::even_split(N, ctx);
//
//   // From a full vector on rank 0 (scattered to all ranks):
//   DistributedVector v = DistributedVector::scatter(full_vec, ctx);
//
//   // Gather back to rank 0:
//   Vector full = v.gather();
// =============================================================================

#include "../core/types.hpp"
#include "../core/vector.hpp"
#include "parallel_context.hpp"
#include <vector>

namespace hsps {

struct DistributedVector {
    Vector  local;          ///< This rank's local segment
    Index   global_n = 0;   ///< Total global length
    Index   local_offset = 0; ///< Global index of local[0]

    // ------------------------------------------------------------------
    // Factories
    // ------------------------------------------------------------------

    /// Create an uninitialized distributed vector with even row split
    static DistributedVector create(Index global_n, Real init_val,
                                    const ParallelContext& ctx);

    /// Scatter a full vector (on rank 0) to all ranks
    static DistributedVector scatter(const Vector& full,
                                     const ParallelContext& ctx);

    /// Gather all local segments to a full vector on rank 0
    Vector gather(const ParallelContext& ctx) const;

    /// Compute global lengths for an even N-way split
    static void compute_split(Index global_n, int nprocs,
                               std::vector<Index>& counts,
                               std::vector<Index>& offsets);

    // ------------------------------------------------------------------
    // Global reductions  (require MPI_Allreduce — use sparingly)
    // ------------------------------------------------------------------
    Real global_dot(const DistributedVector& other,
                    const ParallelContext& ctx) const;
    Real global_nrm2(const ParallelContext& ctx) const;

    // ------------------------------------------------------------------
    // Local operations (no communication — call these in inner loops)
    // ------------------------------------------------------------------
    Index local_n()  const { return local.size(); }
    Index global_size() const { return global_n; }
};

} // namespace hsps
