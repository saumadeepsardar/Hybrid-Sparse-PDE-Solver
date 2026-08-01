#pragma once

// =============================================================================
// sparse_matrix.hpp  —  Compressed Sparse Row (CSR) matrix
//
// Design note: The spec recommends also supporting BSR / SELL-C-σ but notes
// that format switching has high cost.  This prototype uses CSR exclusively
// with OpenMP-parallelised SpMV.  Future work: add a SELL-C-σ path for GPU.
// =============================================================================

#include "types.hpp"
#include "vector.hpp"
#include <vector>
#include <string>
#include <ostream>
#include <utility>

namespace hsps {

// ---------------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------------
class SparseMatrix;
SparseMatrix transpose(const SparseMatrix& A);
SparseMatrix mat_mat_product(const SparseMatrix& A, const SparseMatrix& B);

// ---------------------------------------------------------------------------
// Sparse matrix in CSR (Compressed Sparse Row) format
//
//   row_ptr[i]   .. row_ptr[i+1]-1  : column indices and values of row i
//   col_idx[k]                      : column index of entry k
//   values[k]                       : value of entry k
// ---------------------------------------------------------------------------
class SparseMatrix {
public:
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------
    SparseMatrix() = default;
    SparseMatrix(Index rows, Index cols);

    /// Build from raw CSR arrays (deep copy)
    SparseMatrix(Index rows, Index cols,
                 const std::vector<Index>& row_ptr,
                 const std::vector<Index>& col_idx,
                 const std::vector<Real>&  values);

    SparseMatrix(const SparseMatrix&)            = default;
    SparseMatrix(SparseMatrix&&)                 = default;
    SparseMatrix& operator=(const SparseMatrix&) = default;
    SparseMatrix& operator=(SparseMatrix&&)      = default;
    ~SparseMatrix()                              = default;

    // ------------------------------------------------------------------
    // Dimensions / metadata
    // ------------------------------------------------------------------
    Index rows()  const noexcept { return rows_; }
    Index cols()  const noexcept { return cols_; }
    Index nnz()   const noexcept { return static_cast<Index>(values_.size()); }
    bool  empty() const noexcept { return values_.empty(); }

    /// Non-zero density
    double density() const;

    // ------------------------------------------------------------------
    // Element-level access (expensive — avoid in hot loops)
    // ------------------------------------------------------------------
    Real  get(Index i, Index j)          const;
    void  set(Index i, Index j, Real v);        ///< Modifies existing entry only
    bool  has_entry(Index i, Index j)    const;

    // ------------------------------------------------------------------
    // Raw array access (needed by preconditioners)
    // ------------------------------------------------------------------
    const std::vector<Index>& row_ptr() const noexcept { return row_ptr_; }
    const std::vector<Index>& col_idx() const noexcept { return col_idx_; }
    const std::vector<Real>&  values()  const noexcept { return values_;  }
          std::vector<Real>&  values()        noexcept { return values_;  }

    // ------------------------------------------------------------------
    // Sparse matrix × dense vector  (OpenMP parallelised)
    // ------------------------------------------------------------------
    /// y = A * x
    void spmv(const Vector& x, Vector& y) const;

    /// y = A * x  (returned)
    Vector operator*(const Vector& x) const;

    // ------------------------------------------------------------------
    // Diagonal extraction / scaling
    // ------------------------------------------------------------------
    void   extract_diagonal(Vector& diag) const;
    Vector diagonal()                     const;

    /// Compute spectral radius estimate via power iteration (few steps)
    Real spectral_radius_estimate(int power_iters = 20) const;

    // ------------------------------------------------------------------
    // Matrix arithmetic (used by AMG Galerkin product  R*A*P)
    // ------------------------------------------------------------------
    friend SparseMatrix mat_mat_product(const SparseMatrix& A,
                                        const SparseMatrix& B);
    friend SparseMatrix transpose(const SparseMatrix& A);

    // ------------------------------------------------------------------
    // Builders
    // ------------------------------------------------------------------
    /// Build from coordinate (COO) triples; entries are accumulated (+=)
    static SparseMatrix from_coo(Index rows, Index cols,
                                 const std::vector<Index>& row_vec,
                                 const std::vector<Index>& col_vec,
                                 const std::vector<Real>&  val_vec);

    /// 2-D Poisson on n×n grid (5-point stencil) — convenience factory
    static SparseMatrix poisson_2d(Index n);

    /// 2-D convection-diffusion  -ε∇²u + β·∇u = f
    static SparseMatrix convection_diffusion_2d(Index n,
                                                Real epsilon = 1.0,
                                                Real beta_x  = 0.0,
                                                Real beta_y  = 0.0);

    // ------------------------------------------------------------------
    // Diagnostics
    // ------------------------------------------------------------------
    bool   is_symmetric(Real tol = 1e-10) const;
    Real   frobenius_norm()               const;
    void   print_info(std::ostream& os)   const;

    /// Memory footprint in bytes
    long long mem_bytes() const;

    // ------------------------------------------------------------------
    // Research analysis methods (Thrust 1 — GNN features, Thrust 3 — s-step)
    // ------------------------------------------------------------------

    /// Max |i - j| over all non-zeros — measures stencil width / fill distance
    Index  bandwidth() const;

    /// Variance of per-row non-zero counts — detects irregular sparsity
    double nnz_variance() const;

    /// Fraction of nnz that lie on the main diagonal
    double diagonal_fraction() const;

    /// Ratio nnz(strict lower) / nnz(strict upper) — 1.0 = perfectly symmetric
    double lower_upper_ratio() const;

    /// Estimated ILU(0) fill ratio: (lower_nnz + upper_nnz) / total_nnz
    double estimated_ilu_fill() const;

    /// Estimate one-level AMG coarsening ratio: nodes_coarse / nodes_fine
    double estimated_amg_coarsening_ratio(Real strength_thresh = 0.25) const;

    /// Spectral interval [λ_min, λ_max] via short Lanczos (Thrust 3 Chebyshev basis)
    std::pair<Real, Real> spectral_interval(int lanczos_steps = 20) const;

    /// Count rows with zero diagonal (indicates rank deficiency)
    int    count_zero_diagonal(Real tol = 1e-15) const;

private:
    Index rows_ = 0;
    Index cols_ = 0;
    std::vector<Index> row_ptr_;  ///< length rows_+1
    std::vector<Index> col_idx_;  ///< length nnz
    std::vector<Real>  values_;   ///< length nnz

    // internal helpers
    Index find_entry(Index i, Index j) const;  ///< Returns position or -1
};

} // namespace hsps
