// =============================================================================
// sparse_matrix.cpp  —  CSR sparse matrix implementation
// =============================================================================

#include "../../include/core/sparse_matrix.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cmath>
#include <cassert>
#include <iomanip>
#include <omp.h>

namespace hsps {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
SparseMatrix::SparseMatrix(Index rows, Index cols)
    : rows_(rows), cols_(cols), row_ptr_(rows + 1, 0) {}

SparseMatrix::SparseMatrix(Index rows, Index cols,
                            const std::vector<Index>& row_ptr,
                            const std::vector<Index>& col_idx,
                            const std::vector<Real>&  values)
    : rows_(rows), cols_(cols),
      row_ptr_(row_ptr), col_idx_(col_idx), values_(values) {}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------
double SparseMatrix::density() const {
    if (rows_ == 0 || cols_ == 0) return 0.0;
    return static_cast<double>(nnz()) / (static_cast<double>(rows_) * cols_);
}

// ---------------------------------------------------------------------------
// Element access (O(log(nnz_in_row)) binary search)
// ---------------------------------------------------------------------------
Index SparseMatrix::find_entry(Index i, Index j) const {
    Index start = row_ptr_[i];
    Index end   = row_ptr_[i + 1];
    // Entries are assumed sorted (guaranteed by from_coo)
    auto it = std::lower_bound(col_idx_.begin() + start,
                               col_idx_.begin() + end, j);
    if (it == col_idx_.begin() + end || *it != j) return -1;
    return static_cast<Index>(it - col_idx_.begin());
}

Real SparseMatrix::get(Index i, Index j) const {
    Index k = find_entry(i, j);
    return (k >= 0) ? values_[k] : REAL_ZERO;
}

void SparseMatrix::set(Index i, Index j, Real v) {
    Index k = find_entry(i, j);
    if (k < 0) throw std::runtime_error("SparseMatrix::set — entry does not exist");
    values_[k] = v;
}

bool SparseMatrix::has_entry(Index i, Index j) const {
    return find_entry(i, j) >= 0;
}

// ---------------------------------------------------------------------------
// SpMV  y = A * x   (OpenMP row-parallel)
// ---------------------------------------------------------------------------
void SparseMatrix::spmv(const Vector& x, Vector& y) const {
    if (x.size() != cols_)
        throw std::runtime_error("SpMV: x size != cols");
    if (y.size() != rows_)
        y.resize(rows_, REAL_ZERO);

    const Index* rp  = row_ptr_.data();
    const Index* ci  = col_idx_.data();
    const Real*  val = values_.data();
    const Real*  xd  = x.data();
          Real*  yd  = y.data();

#pragma omp parallel for schedule(dynamic, 64)
    for (Index i = 0; i < rows_; ++i) {
        Real sum = REAL_ZERO;
        for (Index k = rp[i]; k < rp[i + 1]; ++k)
            sum += val[k] * xd[ci[k]];
        yd[i] = sum;
    }
}

Vector SparseMatrix::operator*(const Vector& x) const {
    Vector y(rows_, REAL_ZERO);
    spmv(x, y);
    return y;
}

// ---------------------------------------------------------------------------
// Diagonal
// ---------------------------------------------------------------------------
void SparseMatrix::extract_diagonal(Vector& diag) const {
    diag.resize(rows_, REAL_ZERO);
    for (Index i = 0; i < rows_; ++i) {
        for (Index k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k) {
            if (col_idx_[k] == i) {
                diag[i] = values_[k];
                break;
            }
        }
    }
}

Vector SparseMatrix::diagonal() const {
    Vector d;
    extract_diagonal(d);
    return d;
}

// ---------------------------------------------------------------------------
// Spectral radius estimate via power iteration
// ---------------------------------------------------------------------------
Real SparseMatrix::spectral_radius_estimate(int power_iters) const {
    Vector v(rows_);
    // random-ish init
    for (Index i = 0; i < rows_; ++i) v[i] = (i % 2 == 0) ? 1.0 : -0.5;
    Real lambda = 1.0;
    Vector Av(rows_);
    for (int iter = 0; iter < power_iters; ++iter) {
        spmv(v, Av);
        lambda = Av.norm2();
        if (lambda < REAL_EPS) break;
        Av.scale(1.0 / lambda);
        v = Av;
    }
    return lambda;
}

// ---------------------------------------------------------------------------
// Transpose
// ---------------------------------------------------------------------------
SparseMatrix transpose(const SparseMatrix& A) {
    const Index m = A.rows_;
    const Index n = A.cols_;
    const Index nz = A.nnz();

    std::vector<Index> col_count(n, 0);
    for (Index k = 0; k < nz; ++k)
        col_count[A.col_idx_[k]]++;

    std::vector<Index> row_ptr(n + 1, 0);
    for (Index j = 0; j < n; ++j)
        row_ptr[j + 1] = row_ptr[j] + col_count[j];

    std::vector<Index> col_idx(nz);
    std::vector<Real>  values(nz);
    std::vector<Index> pos(n, 0);
    for (Index j = 0; j < n; ++j) pos[j] = row_ptr[j];

    for (Index i = 0; i < m; ++i) {
        for (Index k = A.row_ptr_[i]; k < A.row_ptr_[i + 1]; ++k) {
            Index j   = A.col_idx_[k];
            Index pos_j = pos[j]++;
            col_idx[pos_j] = i;
            values [pos_j] = A.values_[k];
        }
    }
    return SparseMatrix(n, m, row_ptr, col_idx, values);
}

// ---------------------------------------------------------------------------
// Matrix-matrix product  C = A * B  (CSR × CSR → CSR)
// Uses expand-and-compress (row-by-row with dense accumulator)
// ---------------------------------------------------------------------------
SparseMatrix mat_mat_product(const SparseMatrix& A, const SparseMatrix& B) {
    const Index m  = A.rows_;
    const Index k  = A.cols_;
    const Index n  = B.cols_;
    if (B.rows_ != k)
        throw std::runtime_error("mat_mat_product: inner dimensions mismatch");

    // -----------------------------------------------------------------------
    // Parallel row-by-row SpGEMM.
    // Each thread owns a private dense accumulator (size n) and a "touched"
    // list so no cross-row synchronisation is needed.  The output rows are
    // first collected per-thread, then merged into a global CSR in a serial
    // prefix-sum pass — the only sequential section is O(m), not O(nnz).
    // -----------------------------------------------------------------------
    std::vector<std::vector<std::pair<Index,Real>>> rows_data(m);

#pragma omp parallel
    {
        // Thread-private dense accumulator and flag array
        std::vector<Real>  dense(n, REAL_ZERO);
        std::vector<Index> touched;
        touched.reserve(64);

#pragma omp for schedule(dynamic, 64)
        for (Index i = 0; i < m; ++i) {
            for (Index pa = A.row_ptr_[i]; pa < A.row_ptr_[i + 1]; ++pa) {
                Index j  = A.col_idx_[pa];
                Real  av = A.values_[pa];
                for (Index pb = B.row_ptr_[j]; pb < B.row_ptr_[j + 1]; ++pb) {
                    Index l = B.col_idx_[pb];
                    if (dense[l] == REAL_ZERO) touched.push_back(l);
                    dense[l] += av * B.values_[pb];
                }
            }
            // Sort touched for reproducible column order in CSR output
            std::sort(touched.begin(), touched.end());
            rows_data[i].reserve(touched.size());
            for (Index l : touched) {
                if (std::abs(dense[l]) > REAL_EPS * REAL_EPS)
                    rows_data[i].emplace_back(l, dense[l]);
                dense[l] = REAL_ZERO;
            }
            touched.clear();
        }
    }  // end omp parallel

    // Serial prefix sum for row pointers
    std::vector<Index> c_row_ptr(m + 1, 0);
    for (Index i = 0; i < m; ++i)
        c_row_ptr[i + 1] = c_row_ptr[i] + static_cast<Index>(rows_data[i].size());

    const Index total_nnz = c_row_ptr[m];
    std::vector<Index> col_idx(total_nnz);
    std::vector<Real>  values (total_nnz);

    // Parallel scatter into flat arrays (each row writes to non-overlapping range)
#pragma omp parallel for schedule(dynamic, 64)
    for (Index i = 0; i < m; ++i) {
        Index pos = c_row_ptr[i];
        for (auto& [col, val] : rows_data[i]) {
            col_idx[pos] = col;
            values [pos] = val;
            ++pos;
        }
    }
    return SparseMatrix(m, n, c_row_ptr, col_idx, values);
}

// ---------------------------------------------------------------------------
// Build from COO (coordinate) format
// ---------------------------------------------------------------------------
SparseMatrix SparseMatrix::from_coo(Index rows, Index cols,
                                     const std::vector<Index>& row_vec,
                                     const std::vector<Index>& col_vec,
                                     const std::vector<Real>&  val_vec) {
    const Size nz = row_vec.size();
    // Sort by (row, col)
    std::vector<Size> order(nz);
    std::iota(order.begin(), order.end(), Size(0));
    std::sort(order.begin(), order.end(), [&](Size a, Size b) {
        if (row_vec[a] != row_vec[b]) return row_vec[a] < row_vec[b];
        return col_vec[a] < col_vec[b];
    });

    // Build CSR (accumulate duplicate entries)
    std::vector<Index> row_ptr(rows + 1, 0);
    std::vector<Index> cidx;
    std::vector<Real>  vals;
    cidx.reserve(nz);
    vals.reserve(nz);

    for (Size p = 0; p < nz; ++p) {
        Size o  = order[p];
        Index r = row_vec[o];
        Index c = col_vec[o];
        Real  v = val_vec[o];

        if (!cidx.empty() && cidx.back() == c &&
            row_ptr[r + 1] > 0 &&
            static_cast<Index>(cidx.size()) > row_ptr[r]) {
            vals.back() += v;  // accumulate duplicate
        } else {
            cidx.push_back(c);
            vals.push_back(v);
            row_ptr[r + 1]++;
        }
    }
    for (Index i = 0; i < rows; ++i)
        row_ptr[i + 1] += row_ptr[i];

    return SparseMatrix(rows, cols, row_ptr, cidx, vals);
}

// ---------------------------------------------------------------------------
// 2-D Poisson on n×n grid  (5-point stencil, Dirichlet BC)
// DOF ordering: node (i,j) → index i*n + j
// ---------------------------------------------------------------------------
SparseMatrix SparseMatrix::poisson_2d(Index n) {
    const Index N = n * n;
    std::vector<Index> row_vec, col_vec;
    std::vector<Real>  val_vec;
    row_vec.reserve(5 * N);
    col_vec.reserve(5 * N);
    val_vec.reserve(5 * N);

    auto idx = [&](Index i, Index j) { return i * n + j; };

    for (Index i = 0; i < n; ++i) {
        for (Index j = 0; j < n; ++j) {
            Index row = idx(i, j);
            // Diagonal
            row_vec.push_back(row); col_vec.push_back(row); val_vec.push_back(4.0);
            // Neighbours
            if (i > 0    ) { row_vec.push_back(row); col_vec.push_back(idx(i-1,j)); val_vec.push_back(-1.0); }
            if (i < n-1  ) { row_vec.push_back(row); col_vec.push_back(idx(i+1,j)); val_vec.push_back(-1.0); }
            if (j > 0    ) { row_vec.push_back(row); col_vec.push_back(idx(i,j-1)); val_vec.push_back(-1.0); }
            if (j < n-1  ) { row_vec.push_back(row); col_vec.push_back(idx(i,j+1)); val_vec.push_back(-1.0); }
        }
    }
    return from_coo(N, N, row_vec, col_vec, val_vec);
}

// ---------------------------------------------------------------------------
// 2-D Convection-Diffusion  -ε∇²u + β·∇u = f
// Upwind differencing for convection
// ---------------------------------------------------------------------------
SparseMatrix SparseMatrix::convection_diffusion_2d(Index n, Real eps,
                                                    Real bx, Real by) {
    const Index N  = n * n;
    const Real  h  = 1.0 / (n + 1);
    const Real  eh = eps / (h * h);
    const Real  bxh = bx / (2.0 * h);
    const Real  byh = by / (2.0 * h);

    std::vector<Index> row_vec, col_vec;
    std::vector<Real>  val_vec;
    row_vec.reserve(5 * N);
    col_vec.reserve(5 * N);
    val_vec.reserve(5 * N);

    auto idx = [&](Index i, Index j) { return i * n + j; };
    auto push = [&](Index r, Index c, Real v) {
        row_vec.push_back(r); col_vec.push_back(c); val_vec.push_back(v);
    };

    for (Index i = 0; i < n; ++i) {
        for (Index j = 0; j < n; ++j) {
            Index row = idx(i, j);
            Real diag = 4.0 * eh;
            // x-direction (upwind)
            if (j > 0   ) { push(row, idx(i,j-1), -eh + std::max(bxh, 0.0));  diag += std::max(-bxh, 0.0); }
            if (j < n-1 ) { push(row, idx(i,j+1), -eh - std::min(bxh, 0.0));  diag += std::max( bxh, 0.0); }
            // y-direction (upwind)
            if (i > 0   ) { push(row, idx(i-1,j), -eh + std::max(byh, 0.0));  diag += std::max(-byh, 0.0); }
            if (i < n-1 ) { push(row, idx(i+1,j), -eh - std::min(byh, 0.0));  diag += std::max( byh, 0.0); }
            push(row, row, diag);
        }
    }
    return from_coo(N, N, row_vec, col_vec, val_vec);
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------
bool SparseMatrix::is_symmetric(Real tol) const {
    if (rows_ != cols_) return false;
    for (Index i = 0; i < rows_; ++i)
        for (Index k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k) {
            Index j = col_idx_[k];
            Real  v = values_[k];
            Real  vT = get(j, i);
            if (std::abs(v - vT) > tol * (1.0 + std::abs(v))) return false;
        }
    return true;
}

Real SparseMatrix::frobenius_norm() const {
    Real s = REAL_ZERO;
    for (Real v : values_) s += v * v;
    return std::sqrt(s);
}

void SparseMatrix::print_info(std::ostream& os) const {
    os << "SparseMatrix [" << rows_ << " × " << cols_ << "]"
       << "  nnz=" << nnz()
       << "  density=" << std::scientific << std::setprecision(3) << density()
       << "  mem=" << mem_bytes() / 1024 << " KB\n";
}

long long SparseMatrix::mem_bytes() const {
    return (long long)(row_ptr_.size()) * sizeof(Index)
         + (long long)(col_idx_.size()) * sizeof(Index)
         + (long long)(values_.size() ) * sizeof(Real);
}

} // namespace hsps

// =============================================================================
// Research analysis methods — added for HAMLSS (Thrust 1, Thrust 3)
// =============================================================================

namespace hsps {

Index SparseMatrix::bandwidth() const {
    Index bw = 0;
    for (Index i = 0; i < rows_; ++i)
        for (Index k = row_ptr_[i]; k < row_ptr_[i+1]; ++k) {
            Index diff = std::abs(i - col_idx_[k]);
            if (diff > bw) bw = diff;
        }
    return bw;
}

double SparseMatrix::nnz_variance() const {
    if (rows_ == 0) return 0.0;
    double mean = static_cast<double>(nnz()) / rows_;
    double var  = 0.0;
    for (Index i = 0; i < rows_; ++i) {
        double d = (row_ptr_[i+1] - row_ptr_[i]) - mean;
        var += d * d;
    }
    return var / rows_;
}

double SparseMatrix::diagonal_fraction() const {
    if (nnz() == 0) return 0.0;
    int diag_count = 0;
    for (Index i = 0; i < rows_; ++i)
        for (Index k = row_ptr_[i]; k < row_ptr_[i+1]; ++k)
            if (col_idx_[k] == i) { ++diag_count; break; }
    return static_cast<double>(diag_count) / nnz();
}

double SparseMatrix::lower_upper_ratio() const {
    long long lo = 0, hi = 0;
    for (Index i = 0; i < rows_; ++i)
        for (Index k = row_ptr_[i]; k < row_ptr_[i+1]; ++k) {
            if (col_idx_[k] < i) ++lo;
            if (col_idx_[k] > i) ++hi;
        }
    return (hi > 0) ? static_cast<double>(lo) / hi : 1.0;
}

double SparseMatrix::estimated_ilu_fill() const { return 1.0; }

double SparseMatrix::estimated_amg_coarsening_ratio(Real strength_thresh) const {
    const auto& rp  = row_ptr_; const auto& ci = col_idx_; const auto& val = values_;
    double total_strong = 0.0;
    for (Index i = 0; i < rows_; ++i) {
        Real row_max = 0.0;
        for (Index k = rp[i]; k < rp[i+1]; ++k)
            if (ci[k] != i) row_max = std::max(row_max, std::abs(val[k]));
        for (Index k = rp[i]; k < rp[i+1]; ++k)
            if (ci[k] != i && std::abs(val[k]) >= strength_thresh * row_max)
                total_strong += 1.0;
    }
    double avg_strong = (rows_ > 0) ? total_strong / rows_ : 0.0;
    return 1.0 / (1.0 + avg_strong);
}

std::pair<Real, Real> SparseMatrix::spectral_interval(int lanczos_steps) const {
    const int K = std::min(rows_, lanczos_steps);
    Real inv_sq_n = 1.0 / std::sqrt(static_cast<Real>(rows_));
    Vector q(rows_, inv_sq_n), q_prev(rows_, 0.0);
    std::vector<Real> alpha_v, beta_v;
    Vector Aq(rows_);
    Real beta_j = 0.0;
    for (int j = 0; j < K; ++j) {
        spmv(q, Aq);
        Real alpha_j = q.dot(Aq);
        alpha_v.push_back(alpha_j);
        for (Index i = 0; i < rows_; ++i)
            Aq[i] -= alpha_j * q[i] + beta_j * q_prev[i];
        beta_j = Aq.norm2();
        if (beta_j < REAL_EPS * 1e6) break;
        beta_v.push_back(beta_j);
        q_prev = q; q = Aq; q.scale(1.0 / beta_j);
    }
    int sz = static_cast<int>(alpha_v.size());
    Real lam_min = alpha_v[0], lam_max = alpha_v[0];
    for (int i = 0; i < sz; ++i) {
        Real r = 0.0;
        if (i > 0)   r += std::abs(beta_v[i-1]);
        if (i < sz-1)r += std::abs(beta_v[i]);
        lam_min = std::min(lam_min, alpha_v[i] - r);
        lam_max = std::max(lam_max, alpha_v[i] + r);
    }
    return {lam_min, lam_max};
}

int SparseMatrix::count_zero_diagonal(Real tol) const {
    int count = 0;
    for (Index i = 0; i < rows_; ++i) {
        bool found = false;
        for (Index k = row_ptr_[i]; k < row_ptr_[i+1]; ++k)
            if (col_idx_[k] == i) {
                if (std::abs(values_[k]) <= tol) ++count;
                found = true; break;
            }
        if (!found) ++count;
    }
    return count;
}

} // namespace hsps
