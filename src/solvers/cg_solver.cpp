// =============================================================================
// cg_solver.cpp  —  Preconditioned Conjugate Gradient implementation
//
// Algorithm (Saad, "Iterative Methods for Sparse Linear Systems", Alg 6.18):
//   r₀ = b − A x₀
//   z₀ = M⁻¹ r₀
//   p₀ = z₀
//   ρ₀ = rᵀz
//   for k = 0, 1, 2, …
//       q = A pₖ
//       α = ρₖ / (pₖᵀ q)
//       x_{k+1} = xₖ + α pₖ
//       r_{k+1} = rₖ − α q
//       z_{k+1} = M⁻¹ r_{k+1}
//       ρ_{k+1} = r_{k+1}ᵀ z_{k+1}
//       β = ρ_{k+1} / ρₖ
//       p_{k+1} = z_{k+1} + β pₖ
//
// FLOP accounting per iteration (no preconditioner):
//   SpMV    : 2·nnz
//   3 axpy  : 6n
//   2 dot   : 4n
//   Total   ≈ 2·nnz + 10n
// =============================================================================

#include "../../include/solvers/cg_solver.hpp"
#include "../../include/utils/logger.hpp"
#include "../../include/utils/timer.hpp"
#include "../../include/energy/energy_monitor.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

namespace hsps {

bool CGSolver::solve(const SparseMatrix& A,
                     const Vector&       b,
                           Vector&       x,
                     SolverStats&        stats) {
    Timer wall; wall.start();

    const Index n   = A.rows();
    const Index nz  = A.nnz();
    const Real  tol = params_.tol;
    const int   max_iter = params_.max_iter;

    // Initialise x to zero if wrong size
    if (x.size() != n) { x.resize(n, REAL_ZERO); }

    // r = b - A*x
    Vector r(n);
    A.spmv(x, r);
    for (Index i = 0; i < n; ++i) r[i] = b[i] - r[i];

    const Real b_norm = b.norm2();
    const Real r0_norm = r.norm2();
    stats.initial_residual = r0_norm;

    if (b_norm < REAL_EPS) {
        // trivial RHS
        stats.converged      = true;
        stats.final_residual = r0_norm;
        stats.iterations     = 0;
        return true;
    }

    Real rel_res = r0_norm / b_norm;
    if (rel_res < tol) {
        stats.converged      = true;
        stats.final_residual = rel_res;
        stats.iterations     = 0;
        return true;
    }

    // Precondition initial residual
    Vector z(n), p(n), q(n);
    if (precond_) {
        precond_->apply(r, z);
    } else {
        z = r;
    }
    p = z;

    Real rho     = r.dot(z);
    Real rho_old = rho;

    stats.solver_used  = SolverType::CG;
    stats.precond_used = precond_ ? precond_->type() : PrecondType::NONE;

    long long total_flops = 0;
    long long total_bytes = 0;

    // Stall detection
    std::vector<Real> res_history;
    res_history.reserve(params_.stall_window);

    int iter = 0;
    for (; iter < max_iter; ++iter) {
        // q = A * p
        A.spmv(p, q);
        total_flops += flop_model::spmv_flops(nz);
        total_bytes += flop_model::spmv_bytes(n, nz);

        Real pq = p.dot(q);
        total_flops += flop_model::dot_flops(n);
        total_bytes += flop_model::dot_bytes(n);

        if (std::abs(pq) < REAL_EPS * 1e6 * std::abs(rho)) {
            HSPS_LOG_WARN("CG: near-breakdown (p^T q ≈ 0) at iter ", iter,
                          " — restarting with current residual");
            // Restart: recompute r = b - A x, re-precondition, reset search dir
            A.spmv(x, p);   // reuse p as temp
            for (Index ii = 0; ii < n; ++ii) r[ii] = b[ii] - p[ii];
            if (precond_) precond_->apply(r, z); else z = r;
            p   = z;
            rho = r.dot(z);
            ++stats.restarts;
            continue;
        }
        Real alpha = rho / pq;

        // x += alpha * p
        x.axpy(alpha, p);
        total_flops += flop_model::axpy_flops(n);
        total_bytes += flop_model::axpy_bytes(n);

        // r -= alpha * q
        r.axpy(-alpha, q);
        total_flops += flop_model::axpy_flops(n);
        total_bytes += flop_model::axpy_bytes(n);

        rel_res = r.norm2() / b_norm;
        res_history.push_back(rel_res);

        if (params_.verbose && (iter % params_.print_every == 0)) {
            std::cout << std::fixed << std::setprecision(2)
                      << "  CG iter " << std::setw(5) << iter
                      << "  rel_res = " << std::scientific << std::setprecision(4)
                      << rel_res << "\n";
        }

        if (rel_res < tol) { ++iter; break; }

        // z = M^{-1} r
        if (precond_) {
            precond_->apply(r, z);
        } else {
            z = r;
        }

        rho_old = rho;
        rho     = r.dot(z);
        total_flops += flop_model::dot_flops(n);
        total_bytes += flop_model::dot_bytes(n);

        if (std::abs(rho_old) < REAL_EPS * 1e8 * std::abs(rho)) {
            HSPS_LOG_WARN("CG: rho near-zero at iter ", iter, " — restarting");
            A.spmv(x, z);   // reuse z as temp
            for (Index ii = 0; ii < n; ++ii) r[ii] = b[ii] - z[ii];
            if (precond_) precond_->apply(r, z); else z = r;
            p   = z;
            rho = r.dot(z);
            rho_old = rho;
            ++stats.restarts;
            continue;
        }
        Real beta = rho / rho_old;

        // p = z + beta * p
        p.axpby(REAL_ONE, z, beta);
        total_flops += 2LL * n;
        total_bytes += 3LL * n * sizeof(Real);
    }

    wall.stop();

    stats.iterations     = iter;
    stats.final_residual = rel_res;
    stats.converged      = (rel_res < tol);
    stats.solve_time_s   = wall.elapsed();
    stats.flop_count     = total_flops;
    stats.mem_bytes      = total_bytes;

    // Proxy energy: α·FLOP + β·MEM
    constexpr double alpha_j = 2.0e-10;
    constexpr double beta_j  = 5.0e-9;
    stats.energy_joules = alpha_j * total_flops + beta_j * total_bytes;

    return stats.converged;
}

} // namespace hsps
