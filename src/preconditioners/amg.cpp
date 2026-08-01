// =============================================================================
// amg.cpp  —  Algebraic Multigrid (Smoothed Aggregation) implementation
//
// Setup algorithm
//   1. Compute strength-of-connection graph
//   2. Greedy aggregation  →  aggregate map (fine DOF → aggregate id)
//   3. Piecewise-constant prolongation P
//   4. Restriction  R = P^T
//   5. Galerkin coarse operator  A_c = R * A * P
//   6. Recurse until coarsest level (direct dense solve via Gauss elimination)
//
// V-cycle
//   pre-smooth → restrict residual → coarse solve → prolongate → post-smooth
// =============================================================================

#include "../../include/preconditioners/amg.hpp"
#include "../../include/utils/timer.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cmath>
#include <iostream>

namespace hsps {

// ---------------------------------------------------------------------------
// Greedy aggregation
// Each unaggregated node seeds an aggregate; its strong neighbours join it.
// ---------------------------------------------------------------------------
std::vector<Index> AMGPreconditioner::aggregate(const SparseMatrix& A) const {
    const Index n = A.rows();
    std::vector<Index> agg(n, -1);
    Index nagg = 0;

    const auto& rp  = A.row_ptr();
    const auto& ci  = A.col_idx();
    const auto& val = A.values();

    // Compute row absolute sums for relative strength threshold
    std::vector<Real> row_max(n, REAL_ZERO);
    for (Index i = 0; i < n; ++i) {
        for (Index k = rp[i]; k < rp[i + 1]; ++k) {
            if (ci[k] != i)
                row_max[i] = std::max(row_max[i], std::abs(val[k]));
        }
    }

    for (Index i = 0; i < n; ++i) {
        if (agg[i] >= 0) continue;   // already assigned
        agg[i] = nagg;               // seed new aggregate
        // pull in strong neighbours
        for (Index k = rp[i]; k < rp[i + 1]; ++k) {
            Index j = ci[k];
            if (j == i || agg[j] >= 0) continue;
            // strength criterion: |a_ij| >= theta * max_{l≠i} |a_il|
            if (std::abs(val[k]) >= strength_thresh_ * row_max[i])
                agg[j] = nagg;
        }
        ++nagg;
    }

    // Any unassigned nodes become singleton aggregates
    for (Index i = 0; i < n; ++i)
        if (agg[i] < 0) { agg[i] = nagg++; }

    return agg;
}

// ---------------------------------------------------------------------------
// Build piecewise-constant prolongation matrix P
//   P[i, agg[i]] = 1 for each fine DOF i
// ---------------------------------------------------------------------------
SparseMatrix AMGPreconditioner::build_prolongation(
        Index n_fine,
        const std::vector<Index>& agg_map,
        Index n_coarse) const {

    std::vector<Index> row_ptr(n_fine + 1);
    std::vector<Index> col_idx(n_fine);
    std::vector<Real>  values(n_fine, REAL_ONE);

    for (Index i = 0; i < n_fine; ++i) {
        row_ptr[i]     = i;
        col_idx[i]     = agg_map[i];
    }
    row_ptr[n_fine] = n_fine;

    return SparseMatrix(n_fine, n_coarse, row_ptr, col_idx, values);
}


// ---------------------------------------------------------------------------
// Jacobi-smoothed prolongation  P = (I - ω D^{-1} A)^steps P_raw
//
// Each step:  P_new[i,:] = P_old[i,:] - ω/a_ii * Σ_{j≠i} a_ij * P_old[j,:]
// This is equivalent to one step of damped Jacobi applied to the columns of P.
// The result is stored as a dense intermediate then compressed back to CSR.
// Only called during setup so the O(n * nnz_per_row * n_coarse) cost is
// acceptable for the sizes seen at each AMG level.
// ---------------------------------------------------------------------------
SparseMatrix AMGPreconditioner::smooth_prolongation_op(
        const SparseMatrix& A,
        const SparseMatrix& P_raw) const {

    const Index n_fine   = A.rows();
    const Index n_coarse = P_raw.cols();
    const auto& a_rp  = A.row_ptr();
    const auto& a_ci  = A.col_idx();
    const auto& a_val = A.values();

    // Work with a dense n_fine × n_coarse matrix (acceptable at coarse levels)
    // Represented as row-major std::vector<Real>
    std::vector<Real> P_dense(n_fine * n_coarse, 0.0);

    // Scatter P_raw into dense
    const auto& p_rp  = P_raw.row_ptr();
    const auto& p_ci  = P_raw.col_idx();
    const auto& p_val = P_raw.values();
    for (Index i = 0; i < n_fine; ++i)
        for (Index pk = p_rp[i]; pk < p_rp[i+1]; ++pk)
            P_dense[i * n_coarse + p_ci[pk]] = p_val[pk];

    // Extract diagonal of A for damping
    std::vector<Real> inv_diag(n_fine, 0.0);
    for (Index i = 0; i < n_fine; ++i)
        for (Index ak = a_rp[i]; ak < a_rp[i+1]; ++ak)
            if (a_ci[ak] == i) {
                inv_diag[i] = (std::abs(a_val[ak]) > REAL_EPS)
                              ? omega_ / a_val[ak] : 0.0;
                break;
            }

    std::vector<Real> P_new(n_fine * n_coarse);

    for (int step = 0; step < smooth_p_steps_; ++step) {
        // P_new[i,:] = P_dense[i,:] - inv_diag[i] * (A * P_dense)[i,:]
        //            = P_dense[i,:] - inv_diag[i] * Σ_j a_ij * P_dense[j,:]
#pragma omp parallel for schedule(dynamic, 32)
        for (Index i = 0; i < n_fine; ++i) {
            Real* pnew_i = &P_new[i * n_coarse];
            const Real* pold_i = &P_dense[i * n_coarse];
            // Copy current row
            for (Index c = 0; c < n_coarse; ++c) pnew_i[c] = pold_i[c];
            // Subtract damped A*P contribution
            Real id = inv_diag[i];
            if (std::abs(id) < REAL_EPS) continue;
            for (Index ak = a_rp[i]; ak < a_rp[i+1]; ++ak) {
                Index j = a_ci[ak];
                if (j == i) continue;   // skip diagonal — only off-diagonal part
                Real aij = a_val[ak];
                const Real* pold_j = &P_dense[j * n_coarse];
                for (Index c = 0; c < n_coarse; ++c)
                    pnew_i[c] -= id * aij * pold_j[c];
            }
        }
        P_dense.swap(P_new);
    }

    // Compress back to CSR (drop near-zero entries)
    const Real drop = 1e-14;
    std::vector<Index> row_ptr(n_fine + 1, 0);
    std::vector<Index> col_idx;
    std::vector<Real>  values;
    col_idx.reserve(n_fine * 2);
    values .reserve(n_fine * 2);

    for (Index i = 0; i < n_fine; ++i) {
        for (Index c = 0; c < n_coarse; ++c) {
            Real v = P_dense[i * n_coarse + c];
            if (std::abs(v) > drop) {
                col_idx.push_back(c);
                values .push_back(v);
            }
        }
        row_ptr[i + 1] = static_cast<Index>(col_idx.size());
    }
    return SparseMatrix(n_fine, n_coarse, row_ptr, col_idx, values);
}

// ---------------------------------------------------------------------------
// Setup: build the AMG hierarchy
// ---------------------------------------------------------------------------
double AMGPreconditioner::setup(const SparseMatrix& A) {
    Timer t; t.start();

    hierarchy_.clear();

    // Level 0 (finest)
    AMGLevel lvl0;
    lvl0.A = A;
    lvl0.A.extract_diagonal(lvl0.diag_inv);
    for (Index i = 0; i < lvl0.diag_inv.size(); ++i) {
        Real d = lvl0.diag_inv[i];
        lvl0.diag_inv[i] = (std::abs(d) > REAL_EPS) ? (REAL_ONE / d) : REAL_ZERO;
    }
    hierarchy_.push_back(std::move(lvl0));

    for (int lv = 0; lv < max_levels_ - 1; ++lv) {
        const SparseMatrix& Af = hierarchy_.back().A;
        if (Af.rows() <= coarse_limit_) break;

        // Aggregation
        auto agg_map = aggregate(Af);
        Index n_coarse = *std::max_element(agg_map.begin(), agg_map.end()) + 1;

        if (n_coarse >= Af.rows()) {
            // No meaningful coarsening — stop
            break;
        }

        // Prolongation / Restriction
        SparseMatrix P_raw = build_prolongation(Af.rows(), agg_map, n_coarse);
        // Jacobi-smooth prolongation (SA-AMG) for better convergence
        SparseMatrix P = smooth_p_ ? smooth_prolongation_op(Af, P_raw) : P_raw;
        SparseMatrix R = transpose(P);

        // Galerkin coarse operator: Ac = R * A * P
        SparseMatrix Ac = mat_mat_product(R, mat_mat_product(Af, P));

        AMGLevel lvl;
        lvl.A = Ac;
        lvl.P = P;
        lvl.R = R;
        lvl.A.extract_diagonal(lvl.diag_inv);
        for (Index i = 0; i < lvl.diag_inv.size(); ++i) {
            Real d = lvl.diag_inv[i];
            lvl.diag_inv[i] = (std::abs(d) > REAL_EPS) ? (REAL_ONE / d) : REAL_ZERO;
        }
        hierarchy_.back().P = std::move(P);
        hierarchy_.back().R = std::move(R);
        hierarchy_.push_back(std::move(lvl));
    }

    ready_ = true;
    t.stop();
    return t.elapsed();
}

// ---------------------------------------------------------------------------
// Jacobi smoother: x += ω * D^{-1} * (rhs - A*x)  for n_steps steps
// ---------------------------------------------------------------------------
void AMGPreconditioner::smooth(int level, const Vector& rhs,
                                Vector& x, int n_steps) const {
    const SparseMatrix& A   = hierarchy_[level].A;
    const Vector&       dinv = hierarchy_[level].diag_inv;
    const Index         n    = A.rows();

    Vector res(n);
    for (int s = 0; s < n_steps; ++s) {
        A.spmv(x, res);        // res = A*x
        for (Index i = 0; i < n; ++i)
            x[i] += omega_ * dinv[i] * (rhs[i] - res[i]);
    }
}

// ---------------------------------------------------------------------------
// Coarsest-level direct solve via Gaussian elimination
// (only called for small systems, <= coarse_limit_)
// ---------------------------------------------------------------------------
static void direct_solve(const SparseMatrix& A, const Vector& rhs, Vector& x) {
    const Index n = A.rows();
    // Build dense copy
    std::vector<std::vector<Real>> M(n, std::vector<Real>(n + 1, 0.0));
    for (Index i = 0; i < n; ++i) {
        for (Index k = A.row_ptr()[i]; k < A.row_ptr()[i + 1]; ++k)
            M[i][A.col_idx()[k]] = A.values()[k];
        M[i][n] = rhs[i];
    }
    // Gaussian elimination with partial pivoting
    for (Index col = 0; col < n; ++col) {
        // Find pivot
        Index piv = col;
        for (Index r = col + 1; r < n; ++r)
            if (std::abs(M[r][col]) > std::abs(M[piv][col])) piv = r;
        std::swap(M[col], M[piv]);
        if (std::abs(M[col][col]) < 1e-14) continue;  // singular row — skip
        Real inv_piv = 1.0 / M[col][col];
        for (Index r = col + 1; r < n; ++r) {
            Real factor = M[r][col] * inv_piv;
            for (Index c = col; c <= n; ++c)
                M[r][c] -= factor * M[col][c];
        }
    }
    // Back substitution
    x.resize(n);
    for (Index i = n - 1; i >= 0; --i) {
        Real s = M[i][n];
        for (Index j = i + 1; j < n; ++j) s -= M[i][j] * x[j];
        x[i] = (std::abs(M[i][i]) > 1e-14) ? s / M[i][i] : 0.0;
    }
}

// ---------------------------------------------------------------------------
// V-cycle: recursive
// ---------------------------------------------------------------------------
void AMGPreconditioner::v_cycle(int level, const Vector& rhs, Vector& x) const {
    const int num_levels = static_cast<int>(hierarchy_.size());

    if (level == num_levels - 1) {
        // Coarsest level: direct solve
        direct_solve(hierarchy_[level].A, rhs, x);
        return;
    }

    const SparseMatrix& A = hierarchy_[level].A;
    const SparseMatrix& P = hierarchy_[level].P;
    const SparseMatrix& R = hierarchy_[level].R;

    // 1. Pre-smooth
    smooth(level, rhs, x, smooth_pre_);

    // 2. Compute residual:  r = rhs - A*x
    Vector Ax(A.rows());
    A.spmv(x, Ax);
    Vector res(A.rows());
    for (Index i = 0; i < A.rows(); ++i)
        res[i] = rhs[i] - Ax[i];

    // 3. Restrict residual to coarse grid
    Vector rhs_c;
    R.spmv(res, rhs_c);

    // 4. Coarse-grid correction (initialised to zero)
    Vector e_c(hierarchy_[level + 1].A.rows(), REAL_ZERO);
    v_cycle(level + 1, rhs_c, e_c);

    // 5. Prolongate and correct
    Vector e_fine;
    P.spmv(e_c, e_fine);
    for (Index i = 0; i < A.rows(); ++i)
        x[i] += e_fine[i];

    // 6. Post-smooth
    smooth(level, rhs, x, smooth_post_);
}

// ---------------------------------------------------------------------------
// Apply: one AMG V-cycle (non-stationary → FGMRES-compatible)
// ---------------------------------------------------------------------------
void AMGPreconditioner::apply(const Vector& r, Vector& z) const {
    z.resize(r.size(), REAL_ZERO);
    v_cycle(0, r, z);
}

} // namespace hsps
