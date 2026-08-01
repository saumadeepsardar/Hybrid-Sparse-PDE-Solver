// =============================================================================
// sstep_cg_solver.cpp  —  Communication-Avoiding s-step CG implementation
// =============================================================================

#include "../../include/solvers/sstep_cg_solver.hpp"
#include "../../include/energy/energy_monitor.hpp"
#include "../../include/utils/logger.hpp"
#include "../../include/utils/timer.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#ifdef HSPS_USE_MPI
#  include <mpi.h>
#endif

namespace hsps {

// ---------------------------------------------------------------------------
// Chebyshev nodes on [lam_min, lam_max]
// Used as shifts in Newton-Chebyshev basis to reduce condition number
// ---------------------------------------------------------------------------
std::vector<Real> SStepCGSolver::chebyshev_shifts(
        Real lam_min, Real lam_max, int k) {
    std::vector<Real> shifts(k);
    Real centre = 0.5 * (lam_min + lam_max);
    Real half   = 0.5 * (lam_max - lam_min);
    for (int j = 0; j < k; ++j) {
        // Chebyshev nodes: θ_j = centre + half * cos((2j+1)π / (2k))
        double angle = M_PI * (2.0 * j + 1.0) / (2.0 * k);
        shifts[j] = centre + half * static_cast<Real>(std::cos(angle));
    }
    return shifts;
}

// ---------------------------------------------------------------------------
// Build Newton-Chebyshev Krylov basis B_k
// v_0 = r / ||r||
// v_j = (A - θ_j I) v_{j-1}   (unnormalised — normalise for numerical stability)
// ---------------------------------------------------------------------------
std::vector<Vector> SStepCGSolver::build_basis(
        const SparseMatrix& A,
        const Vector& r,
        const std::vector<Real>& shifts) const {
    const int k = static_cast<int>(shifts.size());
    const Index n = A.rows();
    std::vector<Vector> B(k, Vector(n, REAL_ZERO));

    B[0] = r;
    Real norm0 = B[0].norm2();
    if (norm0 > REAL_EPS) B[0].scale(1.0 / norm0);

    Vector tmp(n);
    for (int j = 1; j < k; ++j) {
        // tmp = A * B[j-1]
        A.spmv(B[j - 1], tmp);
        // B[j] = tmp - θ_j * B[j-1]  (Newton step)
        B[j] = tmp;
        B[j].axpy(-shifts[j - 1], B[j - 1]);
        // Normalise for stability
        Real nj = B[j].norm2();
        if (nj > REAL_EPS) B[j].scale(1.0 / nj);
    }
    return B;
}

// ---------------------------------------------------------------------------
// Gram matrix G[i][j] = B[i] · B[j]  (local, O(k²n) computation)
// ---------------------------------------------------------------------------
void SStepCGSolver::gram_matrix(
        const std::vector<Vector>& B,
        std::vector<std::vector<Real>>& G) const {
    const int k = static_cast<int>(B.size());
    G.assign(k, std::vector<Real>(k, REAL_ZERO));
    for (int i = 0; i < k; ++i)
        for (int j = i; j < k; ++j) {
            G[i][j] = B[i].dot(B[j]);
            G[j][i] = G[i][j];
        }
}

// ---------------------------------------------------------------------------
// Allreduce Gram matrix across MPI ranks (one call for k² doubles)
// ---------------------------------------------------------------------------
void SStepCGSolver::allreduce_gram(
        std::vector<std::vector<Real>>& G) const {
#ifdef HSPS_USE_MPI
    if (!ctx_ || ctx_->nprocs() <= 1) return;
    const int k = static_cast<int>(G.size());
    std::vector<double> flat(k * k);
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j)
            flat[i * k + j] = G[i][j];

    MPI_Allreduce(MPI_IN_PLACE, flat.data(), k * k,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j)
            G[i][j] = static_cast<Real>(flat[i * k + j]);
#else
    (void)G;  // single process: no reduction needed
#endif
}

// ---------------------------------------------------------------------------
// Estimate condition number of Gram matrix via diagonal ratio
// (cheap proxy: max_diag / min_diag — exact for diagonal G)
// ---------------------------------------------------------------------------
double SStepCGSolver::gram_condition_estimate(
        const std::vector<std::vector<Real>>& G) const {
    const int k = static_cast<int>(G.size());
    Real max_d = 0.0, min_d = std::numeric_limits<Real>::infinity();
    for (int i = 0; i < k; ++i) {
        max_d = std::max(max_d, std::abs(G[i][i]));
        if (std::abs(G[i][i]) > REAL_EPS)
            min_d = std::min(min_d, std::abs(G[i][i]));
    }
    if (min_d >= max_d) return 1.0;
    return static_cast<double>(max_d / min_d);
}

// ---------------------------------------------------------------------------
// Solve k×k symmetric positive definite system G * y = rhs
// via Cholesky factorisation (hand-rolled, k is tiny — ≤ 8)
// ---------------------------------------------------------------------------
bool SStepCGSolver::local_lstsq(
        const std::vector<std::vector<Real>>& G,
        const std::vector<Real>& rhs,
        std::vector<Real>& y) const {
    const int k = static_cast<int>(G.size());
    // Copy G for factorisation
    std::vector<std::vector<Real>> L(k, std::vector<Real>(k, 0.0));
    // Cholesky L L^T = G
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j <= i; ++j) {
            Real s = G[i][j];
            for (int l = 0; l < j; ++l) s -= L[i][l] * L[j][l];
            if (i == j) {
                if (s <= 0.0) return false;  // not positive definite
                L[i][j] = std::sqrt(s);
            } else {
                L[i][j] = s / L[j][j];
            }
        }
    }
    // Forward substitution: L * z = rhs
    std::vector<Real> z(k, 0.0);
    for (int i = 0; i < k; ++i) {
        Real s = rhs[i];
        for (int j = 0; j < i; ++j) s -= L[i][j] * z[j];
        z[i] = s / L[i][i];
    }
    // Backward substitution: L^T * y = z
    y.resize(k, 0.0);
    for (int i = k - 1; i >= 0; --i) {
        Real s = z[i];
        for (int j = i + 1; j < k; ++j) s -= L[j][i] * y[j];
        y[i] = s / L[i][i];
    }
    return true;
}

// ---------------------------------------------------------------------------
// Main solve loop
// ---------------------------------------------------------------------------
bool SStepCGSolver::solve(const SparseMatrix& A,
                           const Vector&       b,
                                 Vector&       x,
                           SolverStats&        stats) {
    Timer wall; wall.start();

    const Index n        = A.rows();
    const Index nz       = A.nnz();
    const Real  tol      = params_.tol;
    const int   max_iter = params_.max_iter;
    int         k        = std::min(k_, std::max(1, n / 10));  // safe bound

    if (x.size() != n) x.resize(n, REAL_ZERO);

    // Get spectral interval for Chebyshev shifts
    auto [lam_min_raw, lam_max_raw] = A.spectral_interval(20);
    Real lam_min = std::max(lam_min_raw, REAL_EPS);
    Real lam_max = std::max(lam_max_raw, lam_min * 2.0);

    // Initial residual r = b - A*x
    Vector r(n);
    A.spmv(x, r);
    for (Index i = 0; i < n; ++i) r[i] = b[i] - r[i];

    const Real b_norm = b.norm2();
    Real r_norm = r.norm2();
    stats.initial_residual = r_norm;

    if (b_norm < REAL_EPS) {
        stats.converged = true; stats.final_residual = 0.0; return true;
    }

    Real rel_res = r_norm / b_norm;
    if (rel_res < tol) {
        stats.converged = true; stats.final_residual = rel_res; return true;
    }

    stats.solver_used  = SolverType::SSTEP_CG;
    stats.precond_used = precond_ ? precond_->type() : PrecondType::NONE;
    stats.sstep_k      = k;

    long long total_flops = 0;
    long long total_bytes = 0;
    int       total_iters = 0;

    // CG state (we track r explicitly for convergence checks)
    Vector p(n, REAL_ZERO);  // search direction
    bool first_batch = true;
    Real rho_old = REAL_ONE;

    while (total_iters < max_iter && rel_res > tol) {
        // Adaptive k: reduce if Gram matrix is ill-conditioned
        int k_batch = k;

        // Build Chebyshev basis
        std::vector<Real> shifts = chebyshev_shifts(lam_min, lam_max, k_batch);
        std::vector<Vector> B    = build_basis(A, r, shifts);

        total_flops += (long long)k_batch * flop_model::spmv_flops(nz);
        total_bytes += (long long)k_batch * flop_model::spmv_bytes(n, nz);

        // Gram matrix (local)
        std::vector<std::vector<Real>> G;
        gram_matrix(B, G);

        // Check condition
        double cond_G = gram_condition_estimate(G);
        if (cond_G > params_.sstep_cond_tol && k_batch > 1) {
            // Reduce k and rebuild
            k_batch = std::max(1, k_batch / 2);
            HSPS_LOG_WARN("SStepCG: basis ill-conditioned (κ=", cond_G,
                          "), reducing k to ", k_batch);
            shifts = chebyshev_shifts(lam_min, lam_max, k_batch);
            B      = build_basis(A, r, shifts);
            gram_matrix(B, G);
        }

        // ONE global Allreduce for the entire Gram matrix (k² doubles)
        allreduce_gram(G);
        ++stats.comm_bytes;  // count as 1 communication event

        // RHS for local least-squares: rhs[i] = r · B[i]
        std::vector<Real> rhs_ls(k_batch);
        for (int i = 0; i < k_batch; ++i)
            rhs_ls[i] = r.dot(B[i]);
        total_flops += k_batch * flop_model::dot_flops(n);

        // Solve G * y = rhs_ls  (local Cholesky, tiny k×k system)
        std::vector<Real> y;
        bool solved_ls = local_lstsq(G, rhs_ls, y);
        if (!solved_ls) {
            HSPS_LOG_WARN("SStepCG: local Cholesky failed — falling back to 1 standard CG step");
            // Fallback: one standard CG step
            Vector Ap(n);
            A.spmv(r, Ap);
            Real rr = r.dot(r), rAp = r.dot(Ap);
            if (std::abs(rAp) > REAL_EPS) {
                Real alpha = rr / rAp;
                x.axpy(alpha, r);
                r.axpy(-alpha, Ap);
            }
            ++total_iters; rel_res = r.norm2() / b_norm; continue;
        }

        // Update x += sum_i y[i] * B[i]
        for (int i = 0; i < k_batch; ++i) x.axpy(y[i], B[i]);

        // Recompute residual exactly every k steps (exact restart)
        A.spmv(x, r);
        total_flops += flop_model::spmv_flops(nz);
        total_bytes += flop_model::spmv_bytes(n, nz);
        for (Index i = 0; i < n; ++i) r[i] = b[i] - r[i];

        r_norm  = r.norm2();
        rel_res = r_norm / b_norm;
        total_iters += k_batch;

        if (params_.verbose)
            std::cout << "  s-step CG: iters=" << total_iters
                      << "  k=" << k_batch
                      << "  κ(G)=" << std::scientific << cond_G
                      << "  rel_res=" << rel_res << "\n";
    }

    wall.stop();
    stats.iterations     = total_iters;
    stats.final_residual = rel_res;
    stats.converged      = (rel_res < tol);
    stats.solve_time_s   = wall.elapsed();
    stats.flop_count     = total_flops;
    stats.mem_bytes      = total_bytes;

    constexpr double af = 2e-10, am = 5e-9;
    stats.energy_joules  = af * total_flops + am * total_bytes;
    return stats.converged;
}

} // namespace hsps
