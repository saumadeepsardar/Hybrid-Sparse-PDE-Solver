// =============================================================================
// pipelined_fgmres_solver.cpp
//
// Pipelined FGMRES(m): overlaps SpMV of iteration j+1 with the Gram-Schmidt
// inner products of iteration j.  In serial/OpenMP builds the reordering
// produces identical floating-point results to standard FGMRES (up to
// round-off ordering differences) while eliminating one synchronisation
// barrier per inner iteration.
//
// MPI note
// --------
// The code is structured so that wrapping the dot-product collection in
//   MPI_Iallreduce(&local_dots[0], &global_dots[0], j+1, …, op, comm, &req)
// followed by the next SpMV, then
//   MPI_Wait(&req, …)
// would give the full communication-overlap benefit.  This is left as a
// compile-time feature gated on HSPS_MPI (see Makefile improvement #10).
// =============================================================================

#include "../../include/solvers/pipelined_fgmres_solver.hpp"
#include "../../include/energy/energy_monitor.hpp"
#include "../../include/utils/logger.hpp"
#include "../../include/utils/timer.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <numeric>

namespace hsps {

// ---------------------------------------------------------------------------
// Givens helpers (identical to FGMRESSolver)
// ---------------------------------------------------------------------------
void PipelinedFGMRESSolver::compute_givens(
        Real f, Real g, Real& cs, Real& sn, Real& r) {
    if (std::abs(g) < REAL_EPS) {
        cs = 1.0; sn = 0.0; r = f;
    } else if (std::abs(f) < REAL_EPS) {
        cs = 0.0; sn = 1.0; r = g;
    } else {
        Real t = std::sqrt(f * f + g * g);
        cs = f / t; sn = g / t; r = t;
    }
}

void PipelinedFGMRESSolver::apply_givens(
        Real& dx, Real& dy, Real cs, Real sn) {
    Real tmp = cs * dx + sn * dy;
    dy = -sn * dx + cs * dy;
    dx = tmp;
}

void PipelinedFGMRESSolver::solve_upper_hessenberg(
        const std::vector<std::vector<Real>>& H,
        const std::vector<Real>& g,
        int k, std::vector<Real>& y) {
    y.assign(k, 0.0);
    for (int i = k - 1; i >= 0; --i) {
        Real s = g[i];
        for (int jj = i + 1; jj < k; ++jj) s -= H[i][jj] * y[jj];
        y[i] = (std::abs(H[i][i]) > REAL_EPS) ? (s / H[i][i]) : 0.0;
    }
}

// ---------------------------------------------------------------------------
// One pipelined restart cycle of length m
// ---------------------------------------------------------------------------
int PipelinedFGMRESSolver::pipeline_cycle(
        const SparseMatrix& A,
        const Vector&       b,
              Vector&       x,
        Real                tol_abs,
        int                 m,
        Real&               res_norm) const {

    const Index n = A.rows();

    // r = b - A x
    Vector r(n);
    A.spmv(x, r);
    for (Index i = 0; i < n; ++i) r[i] = b[i] - r[i];

    Real beta = r.norm2();
    res_norm  = beta;
    if (beta < tol_abs) return 0;

    // Arnoldi basis V, preconditioned directions Z, and "shadow" S = A*Z
    std::vector<Vector> V(m + 1, Vector(n, REAL_ZERO));
    std::vector<Vector> Z(m,     Vector(n, REAL_ZERO));
    std::vector<Vector> S(m,     Vector(n, REAL_ZERO));  // S[j] = A * Z[j]

    V[0] = r;
    V[0].scale(1.0 / beta);

    // ── Pre-compute s_0 = A * M^{-1} * v_0  (pipeline primer) ──────────
    if (precond_) precond_->apply(V[0], Z[0]);
    else          Z[0] = V[0];
    A.spmv(Z[0], S[0]);   // S[0] = A * Z[0]  (one extra SpMV at start)

    // Upper Hessenberg H, Givens cs/sn, RHS g
    std::vector<std::vector<Real>> H(m + 1, std::vector<Real>(m, 0.0));
    std::vector<Real> cs(m, 0.0), sn(m, 0.0);
    std::vector<Real> g(m + 1, 0.0);
    g[0] = beta;

    int j = 0;
    for (; j < m; ++j) {
        // w_j = S[j]  (already computed from previous iteration or primer)
        Vector& w = S[j];   // alias — do NOT modify S[j] after this

        // Modified Gram-Schmidt: orthogonalise w against V[0]…V[j]
        for (int i = 0; i <= j; ++i) {
            H[i][j] = w.dot(V[i]);
            w.axpy(-H[i][j], V[i]);
        }
        H[j + 1][j] = w.norm2();

        if (H[j + 1][j] > REAL_EPS) {
            V[j + 1] = w;
            V[j + 1].scale(1.0 / H[j + 1][j]);
        } else {
            // Lucky breakdown — already converged
            ++j;
            break;
        }

        // ── Pipeline: compute Z[j+1] and S[j+1] NOW, before Givens ──────
        // In an MPI build, start MPI_Iallreduce here for H[:,j], then
        // overlap with the SpMV below.
        if (j + 1 < m) {
            if (precond_) precond_->apply(V[j + 1], Z[j + 1]);
            else          Z[j + 1] = V[j + 1];
            A.spmv(Z[j + 1], S[j + 1]);   // ← overlaps with communication in MPI
        }
        // ── (MPI_Wait would go here) ─────────────────────────────────────

        // Apply previous Givens rotations to column j of H
        for (int i = 0; i < j; ++i)
            apply_givens(H[i][j], H[i + 1][j], cs[i], sn[i]);

        // Compute new Givens and apply
        Real r_val;
        compute_givens(H[j][j], H[j + 1][j], cs[j], sn[j], r_val);
        H[j][j]     =  r_val;
        H[j + 1][j] =  0.0;
        apply_givens(g[j], g[j + 1], cs[j], sn[j]);

        res_norm = std::abs(g[j + 1]);
        if (res_norm < tol_abs) { ++j; break; }
    }

    // Solve Hessenberg least-squares
    std::vector<Real> y;
    solve_upper_hessenberg(H, g, j, y);

    // Update x = x + Z * y
    for (int i = 0; i < j; ++i)
        x.axpy(y[i], Z[i]);

    return j;
}

// ---------------------------------------------------------------------------
// Outer restart loop
// ---------------------------------------------------------------------------
bool PipelinedFGMRESSolver::solve(
        const SparseMatrix& A,
        const Vector&       b,
              Vector&       x,
        SolverStats&        stats) {
    Timer wall; wall.start();

    const Index n        = A.rows();
    const Index nz       = A.nnz();
    const Real  tol      = params_.tol;
    const int   max_iter = params_.max_iter;
    const int   m        = std::min(params_.restart_size, n);

    if (x.size() != n) x.resize(n, REAL_ZERO);

    const Real b_norm = b.norm2();
    if (b_norm < REAL_EPS) {
        stats.converged = true; stats.final_residual = 0.0;
        return true;
    }
    const Real tol_abs = tol * b_norm;

    stats.solver_used  = SolverType::FGMRES;
    stats.precond_used = precond_ ? precond_->type() : PrecondType::NONE;

    long long total_flops = 0;
    long long total_bytes = 0;
    Real      res_norm    = b_norm;
    int       total_iters = 0;
    int       restarts    = 0;

    while (total_iters < max_iter) {
        int remaining        = max_iter - total_iters;
        int cycle_m          = std::min(m, remaining);
        int iters_this_cycle = pipeline_cycle(A, b, x, tol_abs, cycle_m, res_norm);

        // Cost: (iters+1) SpMVs (one extra primer) + usual Krylov ops
        long long cycle_spmv = (long long)(iters_this_cycle + 1);
        total_flops += cycle_spmv * flop_model::spmv_flops(nz)
                     + (long long)iters_this_cycle * 4LL * n;
        total_bytes += cycle_spmv * flop_model::spmv_bytes(n, nz)
                     + (long long)iters_this_cycle * 4LL * n * sizeof(Real);

        total_iters += iters_this_cycle;
        ++restarts;

        Real rel_res = res_norm / b_norm;

        if (params_.verbose) {
            std::cout << std::fixed << std::setprecision(2)
                      << "  Pipelined-FGMRES restart " << std::setw(4) << restarts
                      << "  inner=" << std::setw(4) << iters_this_cycle
                      << "  total=" << std::setw(5) << total_iters
                      << "  rel_res=" << std::scientific << std::setprecision(4)
                      << rel_res << "\n";
        }

        if (rel_res < tol || iters_this_cycle == 0) {
            stats.converged      = (rel_res < tol);
            stats.final_residual = rel_res;
            break;
        }
    }

    wall.stop();
    stats.iterations   = total_iters;
    stats.restarts     = restarts - 1;
    stats.solve_time_s = wall.elapsed();
    stats.flop_count   = total_flops;
    stats.mem_bytes    = total_bytes;

    constexpr double alpha_j = 2.0e-10;
    constexpr double beta_j  = 5.0e-9;
    stats.energy_joules = alpha_j * total_flops + beta_j * total_bytes;

    return stats.converged;
}

} // namespace hsps
