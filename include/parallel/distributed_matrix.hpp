#pragma once

// =============================================================================
// distributed_matrix.hpp  —  Row-partitioned distributed CSR sparse matrix
//
// Layout
// ------
//   Global matrix A (M×N) is split by rows across MPI ranks.
//   Each rank owns a contiguous block of rows [row_offset, row_offset+local_rows).
//
//   The local CSR block is split into two sub-matrices:
//     diag_block  — columns that map to local rows (square local block)
//     off_block   — columns owned by other ranks (require halo exchange)
//
//   SpMV Algorithm (y = A x):
//     1. Pack and exchange halo values of x (MPI_Isend/Irecv)
//     2. Compute y_local = diag_block * x_local  (no communication)
//     3. Wait for halo exchange to complete
//     4. y_local += off_block * x_halo            (use received halo)
//
//   This overlaps computation with communication for efficiency.
//
// Construction
// ------------
//   // Build from a global matrix on rank 0 (scattered to all ranks):
//   DistributedMatrix dm = DistributedMatrix::scatter(global_A, ctx);
//
//   // Build directly on each rank (assembled in parallel):
//   // Provide local rows of the global matrix.
// =============================================================================

#include "../core/types.hpp"
#include "../core/sparse_matrix.hpp"
#include "distributed_vector.hpp"
#include "parallel_context.hpp"
#include <vector>

namespace hsps {

struct HaloDescriptor {
    // For each neighbour rank: which global columns do we need from them?
    struct NeighbourInfo {
        int               rank;
        std::vector<Index> send_indices;  ///< local indices we send out
        std::vector<Index> recv_indices;  ///< local halo indices we receive into
    };
    std::vector<NeighbourInfo> neighbours;
    std::vector<Real>          halo_buf;  ///< receive buffer (size = total halo)
    Index                      halo_size = 0;
};

class DistributedMatrix {
public:
    // ------------------------------------------------------------------
    // Factories
    // ------------------------------------------------------------------

    /// Distribute a global matrix (on rank 0) to all ranks via even row split.
    static DistributedMatrix scatter(const SparseMatrix& global_A,
                                     const ParallelContext& ctx);

    /// Build a local row block directly (for distributed assembly).
    static DistributedMatrix from_local_block(
            const SparseMatrix& local_block,
            Index global_rows,
            Index global_cols,
            Index row_offset,
            const ParallelContext& ctx);

    // ------------------------------------------------------------------
    // Dimensions
    // ------------------------------------------------------------------
    Index global_rows()  const { return global_rows_; }
    Index global_cols()  const { return global_cols_; }
    Index local_rows()   const { return diag_block_.rows(); }
    Index row_offset()   const { return row_offset_;  }

    // ------------------------------------------------------------------
    // Distributed SpMV:  y = A * x
    //   Uses non-blocking MPI to overlap halo exchange with diagonal SpMV.
    // ------------------------------------------------------------------
    void spmv(const DistributedVector& x,
                    DistributedVector& y,
              const ParallelContext&   ctx) const;

    // ------------------------------------------------------------------
    // Diagonal extraction (local portion only)
    // ------------------------------------------------------------------
    Vector local_diagonal() const;

    // ------------------------------------------------------------------
    // Access to local sub-blocks (for preconditioner construction)
    // ------------------------------------------------------------------
    const SparseMatrix& diag_block() const { return diag_block_; }
    const SparseMatrix& off_block()  const { return off_block_;  }

private:
    SparseMatrix diag_block_;   ///< Local diagonal block (square)
    SparseMatrix off_block_;    ///< Off-diagonal block (local_rows × halo_size)
    HaloDescriptor halo_;       ///< Communication pattern

    Index global_rows_ = 0;
    Index global_cols_ = 0;
    Index row_offset_  = 0;

    /// Build the halo descriptor from sparsity pattern
    static HaloDescriptor build_halo(const SparseMatrix& local_block,
                                     Index row_offset,
                                     Index local_rows,
                                     Index global_cols,
                                     const ParallelContext& ctx);
};

} // namespace hsps
