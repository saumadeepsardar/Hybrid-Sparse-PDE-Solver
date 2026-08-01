// =============================================================================
// parallel_fgmres_solver.cpp  —  Backend-dispatched Flexible GMRES(m)
// =============================================================================

#include "../../include/solvers/parallel_fgmres_solver.hpp"
#include "../../include/energy/energy_monitor.hpp"
#include "../../include/utils/logger.hpp"
#include "../../include/utils/timer.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>

namespace hsps {

// ---------------------------------------------------------------------------
void ParallelFGMRESSolver::compute_givens(
        Real f, Real g, Real& cs, Real& sn, Real& r) {
    if (std::abs(g) < REAL_EPS) { cs=1; sn=0; r=f; return; }
    if (std::abs(f) < REAL_EPS) { cs=0; sn=1; r=g; return; }
    Real t = std::sqrt(f*f + g*g);
    cs=f/t; sn=g/t; r=t;
}

void ParallelFGMRESSolver::apply_givens(
        Real& dx, Real& dy, Real cs, Real sn) {
    Real tmp = cs*dx + sn*dy;
    dy = -sn*dx + cs*dy;
    dx = tmp;
}

void ParallelFGMRESSolver::solve_upper_hessenberg(
        const std::vector<std::vector<Real>>& H,
        const std::vector<Real>& g,
        int k, std::vector<Real>& y) {
    y.assign(k, 0.0);
    for (int i=k-1; i>=0; --i) {
        Real s = g[i];
        for (int j=i+1; j<k; ++j) s -= H[i][j]*y[j];
        y[i] = (std::abs(H[i][i]) > REAL_EPS) ? s/H[i][i] : 0.0;
    }
}

// ---------------------------------------------------------------------------
int ParallelFGMRESSolver::inner_cycle(
        const SparseMatrix& A,
        const Vector& b, Vector& x,
        Real tol_abs, int m, Real& res_norm) const {

    BackendBase& B = *backend_;
    const Index n  = A.rows();

    Vector r(n);
    B.spmv(A, x, r);
    for (Index i = 0; i < n; ++i) r[i] = b[i] - r[i];

    Real beta = std::sqrt(B.global_dot(B.dot(r, r)));
    res_norm  = beta;
    if (beta < tol_abs) return 0;

    std::vector<Vector> V(m+1, Vector(n, 0.0));
    std::vector<Vector> Z(m,   Vector(n, 0.0));
    B.copy(r, V[0]);
    B.scale(1.0/beta, V[0]);

    std::vector<std::vector<Real>> H(m+1, std::vector<Real>(m, 0.0));
    std::vector<Real> cs(m,0), sn(m,0), g(m+1,0);
    g[0] = beta;

    int j = 0;
    for (; j < m; ++j) {
        // z_j = M^{-1} v_j
        if (precond_) B.precond_apply(*precond_, V[j], Z[j]);
        else          B.copy(V[j], Z[j]);

        // w = A z_j
        Vector w(n, 0.0);
        B.spmv(A, Z[j], w);

        // Modified Gram-Schmidt
        for (int i = 0; i <= j; ++i) {
            H[i][j] = B.global_dot(B.dot(w, V[i]));
            B.axpy(-H[i][j], V[i], w);
        }
        H[j+1][j] = std::sqrt(B.global_dot(B.dot(w, w)));

        if (H[j+1][j] > REAL_EPS) {
            B.copy(w, V[j+1]);
            B.scale(1.0/H[j+1][j], V[j+1]);
        }

        for (int i = 0; i < j; ++i)
            apply_givens(H[i][j], H[i+1][j], cs[i], sn[i]);

        Real r_val;
        compute_givens(H[j][j], H[j+1][j], cs[j], sn[j], r_val);
        H[j][j]     = r_val;
        H[j+1][j]   = 0.0;
        apply_givens(g[j], g[j+1], cs[j], sn[j]);

        res_norm = std::abs(g[j+1]);
        if (res_norm < tol_abs) { ++j; break; }
    }

    std::vector<Real> y;
    solve_upper_hessenberg(H, g, j, y);
    for (int i = 0; i < j; ++i) B.axpy(y[i], Z[i], x);
    return j;
}

// ---------------------------------------------------------------------------
bool ParallelFGMRESSolver::solve(const SparseMatrix& A,
                                  const Vector& b, Vector& x,
                                  SolverStats& stats) {
    Timer wall; wall.start();

    const Index n        = A.rows();
    const Index nz       = A.nnz();
    const Real  tol      = params_.tol;
    const int   max_iter = params_.max_iter;
    const int   m        = std::min(params_.restart_size, n);

    if (x.size() != n) x.resize(n, REAL_ZERO);

    const Real b_norm  = std::sqrt(backend_->global_dot(backend_->dot(b, b)));
    const Real tol_abs = tol * b_norm;

    stats.solver_used  = SolverType::FGMRES;
    stats.precond_used = precond_ ? precond_->type() : PrecondType::NONE;

    long long total_flops = 0, total_bytes = 0;
    Real res_norm    = b_norm;
    int  total_iters = 0, restarts = 0;

    while (total_iters < max_iter) {
        int cyc_m = std::min(m, max_iter - total_iters);
        int iters = inner_cycle(A, b, x, tol_abs, cyc_m, res_norm);

        total_flops += (long long)iters * (flop_model::spmv_flops(nz) + 4LL*n);
        total_bytes += (long long)iters * (flop_model::spmv_bytes(n,nz) + 4LL*n*sizeof(Real));
        total_iters += iters;
        ++restarts;

        Real rel_res = res_norm / b_norm;

        if (params_.verbose)
            std::cout << "  ParFGMRES[" << backend_->name() << "] restart "
                      << restarts << "  inner=" << iters
                      << "  rel_res=" << std::scientific << rel_res << "\n";

        if (rel_res < tol || iters == 0) {
            stats.converged      = (rel_res < tol);
            stats.final_residual = rel_res;
            break;
        }
    }

    backend_->sync();
    wall.stop();

    stats.iterations   = total_iters;
    stats.restarts     = restarts - 1;
    stats.solve_time_s = wall.elapsed();
    stats.flop_count   = total_flops;
    stats.mem_bytes    = total_bytes;

    constexpr double af=2e-10, am=5e-9;
    stats.energy_joules = af*total_flops + am*total_bytes;
    return stats.converged;
}

} // namespace hsps
