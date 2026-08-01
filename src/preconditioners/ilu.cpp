// =============================================================================
// ilu.cpp  —  ILU(0) and ILUT(τ, p) implementations
//
// ILU(0)  : zero fill, retains sparsity pattern of A.  O(nnz) setup.
// ILUT(τ,p): threshold dropping + bounded fill per row.
//            Each working row is factored with a dense scatter array;
//            small entries (|a_ij| < τ · ‖row‖) are discarded.
//            At most p extra entries per row are kept beyond the original
//            non-zero pattern.  This allows ILUT to approximate ILU(1),
//            ILU(2), … depending on (τ, p) settings.
//
// Recommended starting values:
//   drop_tol = 1e-4   fill_per_row = 10   (good balance for conv-diff)
//   drop_tol = 1e-3   fill_per_row = 5    (more aggressive dropping)
//   drop_tol = 0.0    fill_per_row = 0    (exact ILU(0))
// =============================================================================

#include "../../include/preconditioners/ilu.hpp"
#include "../../include/utils/timer.hpp"
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <vector>
#include <numeric>

namespace hsps {

// ---------------------------------------------------------------------------
// Public dispatch: choose ILU(0) or ILUT based on parameters
// ---------------------------------------------------------------------------
double ILUPreconditioner::setup(const SparseMatrix& A) {
    if (drop_tol_ > 0.0 || fill_p_ > 0)
        return setup_ilut(A);
    else
        return setup_ilu0(A);
}

// ===========================================================================
// ILU(0) — zero-fill, retains sparsity pattern
// ===========================================================================
double ILUPreconditioner::setup_ilu0(const SparseMatrix& A) {
    Timer t; t.start();

    n_       = A.rows();
    row_ptr_ = A.row_ptr();
    col_idx_ = A.col_idx();
    lu_val_  = A.values();

    std::vector<Index> col_to_pos(n_, -1);
    long long flops = 0;

    for (Index i = 1; i < n_; ++i) {
        for (Index p = row_ptr_[i]; p < row_ptr_[i + 1]; ++p)
            col_to_pos[col_idx_[p]] = p;

        for (Index p = row_ptr_[i]; p < row_ptr_[i + 1]; ++p) {
            Index k = col_idx_[p];
            if (k >= i) break;

            // Find diagonal of row k
            Real diag_k = 0.0;
            for (Index q = row_ptr_[k]; q < row_ptr_[k + 1]; ++q)
                if (col_idx_[q] == k) { diag_k = lu_val_[q]; break; }

            if (std::abs(diag_k) < REAL_EPS)
                throw std::runtime_error("ILU(0): near-zero pivot at row "
                                         + std::to_string(k));
            lu_val_[p] /= diag_k;
            Real lik = lu_val_[p];
            ++flops;

            for (Index q = row_ptr_[k]; q < row_ptr_[k + 1]; ++q) {
                Index j = col_idx_[q];
                if (j <= k) continue;
                Index pos_ij = col_to_pos[j];
                if (pos_ij < 0) continue;
                lu_val_[pos_ij] -= lik * lu_val_[q];
                ++flops;
            }
        }
        for (Index p = row_ptr_[i]; p < row_ptr_[i + 1]; ++p)
            col_to_pos[col_idx_[p]] = -1;
    }

    setup_flops_ = flops;
    ready_ = true;
    t.stop();
    return t.elapsed();
}

// ===========================================================================
// ILUT(τ, p) — threshold + bounded fill
// ===========================================================================
double ILUPreconditioner::setup_ilut(const SparseMatrix& A) {
    Timer t; t.start();

    n_ = A.rows();
    const auto& a_rp  = A.row_ptr();
    const auto& a_ci  = A.col_idx();
    const auto& a_val = A.values();

    // Build result in dynamic row-by-row lists, then compress to CSR
    std::vector<std::vector<std::pair<Index,Real>>> L_rows(n_), U_rows(n_);

    // Dense working row and column-to-value map for current row i
    std::vector<Real>  w(n_, 0.0);        // dense working row
    std::vector<bool>  active(n_, false); // which entries are set

    long long flops = 0;

    for (Index i = 0; i < n_; ++i) {
        // ---- scatter row i of A into w ----
        Real row_norm = 0.0;
        for (Index p = a_rp[i]; p < a_rp[i + 1]; ++p) {
            Index j = a_ci[p]; Real v = a_val[p];
            w[j] = v; active[j] = true;
            row_norm += v * v;
        }
        row_norm = std::sqrt(row_norm);
        const Real tau = drop_tol_ * row_norm;

        // ---- eliminate pivot columns k < i ----
        for (Index k = 0; k < i; ++k) {
            if (!active[k]) continue;
            if (std::abs(w[k]) < tau) { w[k] = 0.0; active[k] = false; continue; }

            // Find U[k,k]
            Real ukk = 0.0;
            for (auto& [col, val] : U_rows[k])
                if (col == k) { ukk = val; break; }
            if (std::abs(ukk) < REAL_EPS) { active[k] = false; continue; }

            Real lik = w[k] / ukk;
            w[k] = lik;
            ++flops;

            // w -= lik * U_row_k (columns j > k)
            for (auto& [j, ukj] : U_rows[k]) {
                if (j <= k) continue;
                if (!active[j] && std::abs(lik * ukj) >= tau) active[j] = true;
                w[j] -= lik * ukj;
                ++flops;
            }
        }

        // ---- split into L and U, apply threshold + fill limit ----
        // Collect L entries (j < i)
        std::vector<std::pair<Index,Real>> L_row, U_row;
        for (Index j = 0; j < i; ++j) {
            if (active[j] && std::abs(w[j]) >= tau)
                L_row.emplace_back(j, w[j]);
            w[j] = 0.0; active[j] = false;
        }
        // Diagonal
        Real diag = w[i];
        if (std::abs(diag) < REAL_EPS) {
            // Avoid zero diagonal: use small positive shift
            diag = (diag >= 0 ? 1.0 : -1.0) * REAL_EPS * 1e6;
        }
        U_row.emplace_back(i, diag);
        w[i] = 0.0; active[i] = false;

        // Collect U entries (j > i)
        for (Index j = i + 1; j < n_; ++j) {
            if (active[j] && std::abs(w[j]) >= tau)
                U_row.emplace_back(j, w[j]);
            w[j] = 0.0; active[j] = false;
        }

        // Apply fill limit p: keep largest |value| entries
        auto keep_largest = [&](std::vector<std::pair<Index,Real>>& row,
                                 int max_keep) {
            if (max_keep <= 0 || (int)row.size() <= max_keep) return;
            std::partial_sort(row.begin(),
                              row.begin() + max_keep,
                              row.end(),
                              [](const auto& a, const auto& b){
                                  return std::abs(a.second) > std::abs(b.second);
                              });
            row.resize(max_keep);
            std::sort(row.begin(), row.end(),
                      [](const auto& a, const auto& b){ return a.first < b.first; });
        };

        // Max entries = original nnz_per_row + fill_p_
        int orig_nnz = a_rp[i + 1] - a_rp[i];
        keep_largest(L_row, orig_nnz + fill_p_);
        keep_largest(U_row, orig_nnz + fill_p_);

        L_rows[i] = std::move(L_row);
        U_rows[i] = std::move(U_row);
    }

    // ---- Compress L and U into combined CSR (L\U format) ----
    // Count total entries
    Index total = 0;
    for (Index i = 0; i < n_; ++i)
        total += static_cast<Index>(L_rows[i].size() + U_rows[i].size());

    row_ptr_.resize(n_ + 1);
    col_idx_.resize(total);
    lu_val_ .resize(total);
    row_ptr_[0] = 0;

    Index pos = 0;
    for (Index i = 0; i < n_; ++i) {
        for (auto& [j, v] : L_rows[i]) {
            col_idx_[pos] = j; lu_val_[pos] = v; ++pos;
        }
        for (auto& [j, v] : U_rows[i]) {
            col_idx_[pos] = j; lu_val_[pos] = v; ++pos;
        }
        row_ptr_[i + 1] = pos;
    }

    setup_flops_ = flops;
    ready_ = true;
    t.stop();
    return t.elapsed();
}

// ---------------------------------------------------------------------------
// Forward substitution: L y = r  (unit lower triangular)
// ---------------------------------------------------------------------------
void ILUPreconditioner::forward_sub(const Vector& r, Vector& y) const {
    y.resize(n_);
    const Real* rd = r.data();
          Real* yd = y.data();

    for (Index i = 0; i < n_; ++i) {
        Real s = rd[i];
        for (Index p = row_ptr_[i]; p < row_ptr_[i + 1]; ++p) {
            Index j = col_idx_[p];
            if (j >= i) break;
            s -= lu_val_[p] * yd[j];
        }
        yd[i] = s;
    }
}

// ---------------------------------------------------------------------------
// Backward substitution: U z = y
// ---------------------------------------------------------------------------
void ILUPreconditioner::backward_sub(const Vector& y, Vector& z) const {
    z.resize(n_);
    const Real* yd = y.data();
          Real* zd = z.data();

    for (Index i = n_ - 1; i >= 0; --i) {
        Real s    = yd[i];
        Real diag = REAL_ONE;
        for (Index p = row_ptr_[i]; p < row_ptr_[i + 1]; ++p) {
            Index j = col_idx_[p];
            if (j == i) { diag = lu_val_[p]; continue; }
            if (j > i)  { s   -= lu_val_[p] * zd[j]; }
        }
        if (std::abs(diag) < REAL_EPS)
            throw std::runtime_error("ILU backward sub: near-zero U diagonal at row "
                                     + std::to_string(i));
        zd[i] = s / diag;
    }
}

// ---------------------------------------------------------------------------
// Apply: (L U) z = r
// ---------------------------------------------------------------------------
void ILUPreconditioner::apply(const Vector& r, Vector& z) const {
    Vector y(n_);
    forward_sub(r, y);
    backward_sub(y, z);
}

} // namespace hsps
