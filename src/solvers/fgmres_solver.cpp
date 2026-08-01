// =============================================================================
// fgmres_solver.cpp  —  Flexible GMRES implementation (Saad 1993)
//
// Key property: the preconditioner may change at each inner iteration, making
// FGMRES compatible with AMG V-cycles, inner iteration preconditioners, etc.
//
// Algorithm FGMRES(m):
//   Outer restart loop:
//     r = b − A x;  β = ‖r‖;  v₁ = r / β
//     for j = 1 … m:
//       z_j = M⁻¹ v_j         ← flexible step (M may differ each j)
//       w   = A z_j
//       Modified Gram-Schmidt orthogonalisation against v₁…v_j
//       Apply Givens rotations to maintain upper Hessenberg form
//       Check convergence on ‖g_{j+1}‖ (least-squares residual)
//     Solve  min ‖β e₁ − H̃_m y‖ for y  (already triangulated by Givens)
//     x ← x + Z_m y   where Z_m = [z₁ … z_m]
// =============================================================================

#include "../../include/solvers/fgmres_solver.hpp"
#include "../../include/energy/energy_monitor.hpp"
#include "../../include/utils/logger.hpp"
#include "../../include/utils/timer.hpp"
#include <cmath>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <vector>
#include <algorithm>

namespace hsps {

// ---------------------------------------------------------------------------
// Givens rotation helpers
// ---------------------------------------------------------------------------
void FGMRESSolver::compute_givens(Real f, Real g, Real& cs, Real& sn, Real& r) {
    if (std::abs(g) < REAL_EPS) {
        cs = 1.0; sn = 0.0; r = f;
    } else if (std::abs(f) < REAL_EPS) {
        cs = 0.0; sn = 1.0; r = g;
    } else {
        Real t = std::sqrt(f * f + g * g);
        cs = f / t; sn = g / t; r = t;
    }
}

void FGMRESSolver::apply_givens(Real& dx, Real& dy, Real cs, Real sn) {
    Real tmp = cs * dx + sn * dy;
    dy = -sn * dx + cs * dy;
    dx = tmp;
}

// ---------------------------------------------------------------------------
// Back-substitution for the already-triangularised Hessenberg system
// H(0..k-1, 0..k-1) y = g(0..k-1)
// ---------------------------------------------------------------------------
void FGMRESSolver::solve_hessenberg(const std::vector<std::vector<Real>>& H,
                                     const std::vector<Real>& g,
                                     int k, std::vector<Real>& y) {
    y.resize(k, 0.0);
    for (int i = k - 1; i >= 0; --i) {
        Real s = g[i];
        for (int j = i + 1; j < k; ++j) s -= H[i][j] * y[j];
        y[i] = (std::abs(H[i][i]) > REAL_EPS) ? (s / H[i][i]) : 0.0;
    }
}

// ---------------------------------------------------------------------------
// One FGMRES(m) restart cycle
// Returns number of inner iterations; updates x; writes new ‖res‖ to res_norm.
// ---------------------------------------------------------------------------
int FGMRESSolver::fgmres_cycle(const SparseMatrix& A,
                                const Vector& b,
                                      Vector& x,
                                Real   tol_abs,
                                int    m,
                                Real&  res_norm) const {
    const Index n = A.rows();

    // r = b - A x
    Vector r(n);
    A.spmv(x, r);
    for (Index i = 0; i < n; ++i) r[i] = b[i] - r[i];

    Real beta = r.norm2();
    res_norm  = beta;

    if (beta < tol_abs) return 0;

    // Arnoldi basis V = {v_1, …, v_{m+1}}  and  Z = {z_1, …, z_m}
    std::vector<Vector> V(m + 1, Vector(n, REAL_ZERO));
    std::vector<Vector> Z(m,     Vector(n, REAL_ZERO));

    // Scale first basis vector
    V[0] = r;
    V[0].scale(1.0 / beta);

    // Upper Hessenberg matrix H (stored as (m+1) × m)
    std::vector<std::vector<Real>> H(m + 1, std::vector<Real>(m, 0.0));

    // Givens rotation arrays
    std::vector<Real> cs(m, 0.0), sn(m, 0.0);

    // RHS of least-squares (length m+1)
    std::vector<Real> g(m + 1, 0.0);
    g[0] = beta;

    int j = 0;
    for (; j < m; ++j) {
        // z_j = M^{-1} v_j  (flexible: may vary per j)
        if (precond_) {
            precond_->apply(V[j], Z[j]);
        } else {
            Z[j] = V[j];
        }

        // w = A z_j
        Vector w(n);
        A.spmv(Z[j], w);

        // Modified Gram-Schmidt
        for (int i = 0; i <= j; ++i) {
            H[i][j] = w.dot(V[i]);
            w.axpy(-H[i][j], V[i]);
        }
        H[j + 1][j] = w.norm2();

        if (H[j + 1][j] > REAL_EPS) {
            V[j + 1] = w;
            V[j + 1].scale(1.0 / H[j + 1][j]);
        }

        // Apply previous Givens rotations to column j
        for (int i = 0; i < j; ++i)
            apply_givens(H[i][j], H[i + 1][j], cs[i], sn[i]);

        // Compute new Givens rotation for row j, j+1
        Real r_val;
        compute_givens(H[j][j], H[j + 1][j], cs[j], sn[j], r_val);
        H[j][j]     =  r_val;
        H[j + 1][j] =  0.0;

        // Apply to g
        apply_givens(g[j], g[j + 1], cs[j], sn[j]);

        res_norm = std::abs(g[j + 1]);

        if (res_norm < tol_abs) { ++j; break; }
    }

    // Solve H[0..j-1][0..j-1] y = g[0..j-1]
    std::vector<Real> y;
    solve_hessenberg(H, g, j, y);

    // Update x = x + Z_j * y
    for (int i = 0; i < j; ++i)
        x.axpy(y[i], Z[i]);

    return j;
}

// ---------------------------------------------------------------------------
// Outer restart loop
// ---------------------------------------------------------------------------
bool FGMRESSolver::solve(const SparseMatrix& A,
                          const Vector&       b,
                                Vector&       x,
                          SolverStats&        stats) {
    Timer wall; wall.start();

    const Index n         = A.rows();
    const Index nz        = A.nnz();
    const Real  tol       = params_.tol;
    const int   max_iter  = params_.max_iter;
    const int   m         = std::min(params_.restart_size, n);

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

    Real res_norm       = b_norm;
    int  total_iters    = 0;
    int  restarts       = 0;

    while (total_iters < max_iter) {
        int remaining = max_iter - total_iters;
        int cycle_m   = std::min(m, remaining);

        int iters_this_cycle = fgmres_cycle(A, b, x, tol_abs, cycle_m, res_norm);

        // Accumulate cost estimate (each inner iter ≈ 1 SpMV + 2j dot products)
        total_flops += (long long)iters_this_cycle * (flop_model::spmv_flops(nz) + 4LL * n);
        total_bytes += (long long)iters_this_cycle * (flop_model::spmv_bytes(n, nz) + 4LL * n * sizeof(Real));

        total_iters += iters_this_cycle;
        ++restarts;

        Real rel_res = res_norm / b_norm;

        if (params_.verbose) {
            std::cout << std::fixed << std::setprecision(2)
                      << "  FGMRES restart " << std::setw(4) << restarts
                      << "  inner_iters=" << std::setw(4) << iters_this_cycle
                      << "  total_iters=" << std::setw(5) << total_iters
                      << "  rel_res=" << std::scientific << std::setprecision(4)
                      << rel_res << "\n";
        }

        if (rel_res < tol || iters_this_cycle == 0) {
            stats.converged = (rel_res < tol);
            stats.final_residual = rel_res;
            break;
        }
    }

    wall.stop();

    stats.iterations   = total_iters;
    stats.restarts     = restarts - 1;  // subtract initial
    stats.solve_time_s = wall.elapsed();
    stats.flop_count   = total_flops;
    stats.mem_bytes    = total_bytes;

    constexpr double alpha_j = 2.0e-10;
    constexpr double beta_j  = 5.0e-9;
    stats.energy_joules = alpha_j * total_flops + beta_j * total_bytes;

    return stats.converged;
}

} // namespace hsps
