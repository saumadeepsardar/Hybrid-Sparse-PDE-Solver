// =============================================================================
// parallel_cg_solver.cpp  —  Backend-dispatched Preconditioned CG
// =============================================================================

#include "../../include/solvers/parallel_cg_solver.hpp"
#include "../../include/energy/energy_monitor.hpp"
#include "../../include/utils/logger.hpp"
#include "../../include/utils/timer.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>

namespace hsps {

bool ParallelCGSolver::solve(const SparseMatrix& A,
                              const Vector&       b,
                                    Vector&       x,
                              SolverStats&        stats) {
    Timer wall; wall.start();

    BackendBase& B        = *backend_;
    const Index  n        = A.rows();
    const Index  nz       = A.nnz();
    const Real   tol      = params_.tol;
    const int    max_iter = params_.max_iter;

    if (x.size() != n) x.resize(n, REAL_ZERO);

    // r = b - A*x
    Vector r(n), z(n), p(n), q(n);
    B.spmv(A, x, r);                     // r = A*x
    B.axpby(-1.0, r, 0.0, r);            // r = -r (will fix)
    for (Index i = 0; i < n; ++i) r[i] = b[i] - r[i];  // r = b - Ax

    const Real b_norm = B.nrm2(b);
    Real r_norm = B.nrm2(r);
    stats.initial_residual = r_norm;

    if (b_norm < REAL_EPS) {
        stats.converged = true; stats.final_residual = 0.0; return true;
    }

    Real rel_res = r_norm / b_norm;
    if (rel_res < tol) {
        stats.converged = true; stats.final_residual = rel_res; return true;
    }

    // z = M^{-1} r
    if (precond_) B.precond_apply(*precond_, r, z);
    else          B.copy(r, z);

    B.copy(z, p);
    Real rho = B.global_dot(B.dot(r, z));   // r^T z with global reduction

    stats.solver_used  = SolverType::CG;
    stats.precond_used = precond_ ? precond_->type() : PrecondType::NONE;

    long long total_flops = 0, total_bytes = 0;
    int iter = 0;

    for (; iter < max_iter; ++iter) {
        B.spmv(A, p, q);
        total_flops += flop_model::spmv_flops(nz);
        total_bytes += flop_model::spmv_bytes(n, nz);

        Real pq = B.global_dot(B.dot(p, q));
        total_flops += flop_model::dot_flops(n);
        total_bytes += flop_model::dot_bytes(n);

        if (std::abs(pq) < REAL_EPS * 1e6 * std::abs(rho)) {
            HSPS_LOG_WARN("ParallelCG: near-breakdown at iter ", iter, " — restarting");
            B.spmv(A, x, q);
            for (Index i = 0; i < n; ++i) r[i] = b[i] - q[i];
            if (precond_) B.precond_apply(*precond_, r, z); else B.copy(r, z);
            B.copy(z, p);
            rho = B.global_dot(B.dot(r, z));
            ++stats.restarts;
            continue;
        }

        Real alpha = rho / pq;
        B.axpy( alpha, p, x);      // x += α p
        B.axpy(-alpha, q, r);      // r -= α q
        total_flops += 2 * flop_model::axpy_flops(n);
        total_bytes += 2 * flop_model::axpy_bytes(n);

        r_norm  = B.nrm2(r);
        rel_res = r_norm / b_norm;

        if (params_.verbose && iter % params_.print_every == 0)
            std::cout << "  ParCG[" << B.name() << "] iter " << std::setw(5) << iter
                      << "  rel_res=" << std::scientific << std::setprecision(4)
                      << rel_res << "\n";

        if (rel_res < tol) { ++iter; break; }

        if (precond_) B.precond_apply(*precond_, r, z); else B.copy(r, z);

        Real rho_old = rho;
        rho = B.global_dot(B.dot(r, z));
        if (std::abs(rho_old) < REAL_EPS * 1e8 * std::abs(rho)) {
            HSPS_LOG_WARN("ParallelCG: rho near-zero at iter ", iter, " — restarting");
            B.spmv(A, x, q);
            for (Index i = 0; i < n; ++i) r[i] = b[i] - q[i];
            if (precond_) B.precond_apply(*precond_, r, z); else B.copy(r, z);
            B.copy(z, p);
            rho = B.global_dot(B.dot(r, z));
            ++stats.restarts; continue;
        }

        Real beta = rho / rho_old;
        B.axpby(1.0, z, beta, p);   // p = z + β p
        total_flops += 2LL * n;
        total_bytes += 3LL * n * sizeof(Real);
    }

    B.sync();
    wall.stop();

    stats.iterations     = iter;
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
