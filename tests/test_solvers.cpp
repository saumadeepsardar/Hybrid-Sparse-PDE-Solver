// =============================================================================
// test_solvers.cpp  —  Unit tests for CG and FGMRES Krylov solvers
// =============================================================================

#include "test_framework.hpp"
#include "../include/core/sparse_matrix.hpp"
#include "../include/core/vector.hpp"
#include "../include/solvers/cg_solver.hpp"
#include "../include/solvers/fgmres_solver.hpp"
#include "../include/preconditioners/jacobi.hpp"
#include "../include/preconditioners/ilu.hpp"
#include "../include/preconditioners/amg.hpp"
#include <cmath>
#include <numeric>

using namespace hsps;
using namespace hsps_test;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static SolverParams default_params() {
    SolverParams p;
    p.tol         = 1e-9;
    p.max_iter    = 3000;
    p.restart_size= 40;
    p.verbose     = false;
    return p;
}

/// Relative residual  ‖b - Ax‖ / ‖b‖
static double rel_res(const SparseMatrix& A, const Vector& b, const Vector& x) {
    Vector r = A * x;
    Real bnorm = b.norm2();
    for (Index i = 0; i < b.size(); ++i) r[i] = b[i] - r[i];
    return (bnorm > 1e-15) ? r.norm2() / bnorm : r.norm2();
}

// ---------------------------------------------------------------------------
// CG tests
// ---------------------------------------------------------------------------
HSPS_TEST(cg_solves_1x1) {
    // 2x = 4  →  x = 2
    std::vector<Index> r={0}, c={0};
    std::vector<Real>  v={2.0};
    auto A = SparseMatrix::from_coo(1, 1, r, c, v);
    Vector b = {4.0};
    Vector x(1, 0.0);

    auto jac = std::make_shared<JacobiPreconditioner>(); jac->setup(A);
    CGSolver cg; cg.set_params(default_params()); cg.set_preconditioner(jac);
    SolverStats s;
    bool ok = cg.solve(A, b, x, s);

    expect_true(ok, "1x1 CG converged");
    expect_near(x[0], 2.0, 1e-10, "x[0]=2");
}

HSPS_TEST(cg_solves_diagonal) {
    // Diagonal 4x4: d = {2,3,5,7}
    std::vector<Index> r={0,1,2,3}, c={0,1,2,3};
    std::vector<Real>  v={2.0,3.0,5.0,7.0};
    auto D = SparseMatrix::from_coo(4, 4, r, c, v);
    Vector b = {4.0, 9.0, 15.0, 21.0};
    Vector x(4, 0.0);

    auto jac = std::make_shared<JacobiPreconditioner>(); jac->setup(D);
    CGSolver cg; cg.set_params(default_params()); cg.set_preconditioner(jac);
    SolverStats s;
    cg.solve(D, b, x, s);

    expect_near(x[0], 2.0, 1e-10, "x[0]");
    expect_near(x[1], 3.0, 1e-10, "x[1]");
    expect_near(x[2], 3.0, 1e-10, "x[2]");
    expect_near(x[3], 3.0, 1e-10, "x[3]");
}

HSPS_TEST(cg_poisson_jacobi_converges) {
    auto A = SparseMatrix::poisson_2d(16);
    Vector b(256, 1.0);
    Vector x(256, 0.0);

    auto jac = std::make_shared<JacobiPreconditioner>(); jac->setup(A);
    CGSolver cg; cg.set_params(default_params()); cg.set_preconditioner(jac);
    SolverStats s;
    bool ok = cg.solve(A, b, x, s);

    expect_true(ok, "CG+Jacobi Poisson converged");
    expect_less(rel_res(A, b, x), 1e-8, "rel_res < 1e-9");
}

HSPS_TEST(cg_poisson_ilu_converges_faster) {
    // ILU should require fewer iterations than Jacobi on Poisson
    auto A = SparseMatrix::poisson_2d(16);
    Vector b(256, 1.0);
    SolverParams p = default_params();

    auto jac_p = std::make_shared<JacobiPreconditioner>(); jac_p->setup(A);
    auto ilu_p = std::make_shared<ILUPreconditioner>();    ilu_p->setup(A);

    CGSolver cg_jac, cg_ilu;
    cg_jac.set_params(p); cg_jac.set_preconditioner(jac_p);
    cg_ilu.set_params(p); cg_ilu.set_preconditioner(ilu_p);

    Vector x_jac(256, 0.0), x_ilu(256, 0.0);
    SolverStats s_jac, s_ilu;

    cg_jac.solve(A, b, x_jac, s_jac);
    cg_ilu.solve(A, b, x_ilu, s_ilu);

    expect_true(s_jac.converged && s_ilu.converged, "both converged");
    expect_less(static_cast<double>(s_ilu.iterations),
                static_cast<double>(s_jac.iterations),
                "ILU fewer iterations than Jacobi");
}

HSPS_TEST(cg_zero_rhs) {
    auto A = SparseMatrix::poisson_2d(8);
    Vector b(64, 0.0);
    Vector x(64, 0.0);

    auto jac = std::make_shared<JacobiPreconditioner>(); jac->setup(A);
    CGSolver cg; cg.set_params(default_params()); cg.set_preconditioner(jac);
    SolverStats s;
    bool ok = cg.solve(A, b, x, s);

    expect_true(ok, "zero RHS converged immediately");
    expect_true(s.iterations == 0, "zero iterations for zero RHS");
}

HSPS_TEST(cg_stats_populated) {
    auto A = SparseMatrix::poisson_2d(8);
    Vector b(64, 1.0), x(64, 0.0);

    auto jac = std::make_shared<JacobiPreconditioner>(); jac->setup(A);
    CGSolver cg; cg.set_params(default_params()); cg.set_preconditioner(jac);
    SolverStats s;
    cg.solve(A, b, x, s);

    expect_true(s.iterations > 0,        "iterations > 0");
    expect_true(s.flop_count > 0,        "flops counted");
    expect_true(s.mem_bytes  > 0,        "mem_bytes counted");
    expect_true(s.energy_joules > 0.0,   "energy estimated");
    expect_true(s.solve_time_s >= 0.0,   "time recorded");
    expect_true(s.solver_used == SolverType::CG, "solver type set");
}

// ---------------------------------------------------------------------------
// FGMRES tests
// ---------------------------------------------------------------------------
HSPS_TEST(fgmres_solves_diagonal) {
    std::vector<Index> r={0,1,2}, c={0,1,2};
    std::vector<Real>  v={3.0, 6.0, 9.0};
    auto D = SparseMatrix::from_coo(3, 3, r, c, v);
    Vector b = {3.0, 6.0, 9.0};
    Vector x(3, 0.0);

    auto jac = std::make_shared<JacobiPreconditioner>(); jac->setup(D);
    FGMRESSolver fgm; fgm.set_params(default_params()); fgm.set_preconditioner(jac);
    SolverStats s;
    bool ok = fgm.solve(D, b, x, s);

    expect_true(ok, "FGMRES diagonal converged");
    expect_near(x[0], 1.0, 1e-10, "x[0]");
    expect_near(x[1], 1.0, 1e-10, "x[1]");
    expect_near(x[2], 1.0, 1e-10, "x[2]");
}

HSPS_TEST(fgmres_poisson_jacobi) {
    auto A = SparseMatrix::poisson_2d(16);
    Vector b(256, 1.0), x(256, 0.0);

    auto jac = std::make_shared<JacobiPreconditioner>(); jac->setup(A);
    FGMRESSolver fgm; fgm.set_params(default_params()); fgm.set_preconditioner(jac);
    SolverStats s;
    bool ok = fgm.solve(A, b, x, s);

    expect_true(ok, "FGMRES+Jacobi Poisson converged");
    expect_less(rel_res(A, b, x), 1e-8, "rel_res < 1e-9");
}

HSPS_TEST(fgmres_poisson_ilu) {
    auto A = SparseMatrix::poisson_2d(16);
    Vector b(256, 1.0), x(256, 0.0);

    auto ilu = std::make_shared<ILUPreconditioner>(); ilu->setup(A);
    FGMRESSolver fgm; fgm.set_params(default_params()); fgm.set_preconditioner(ilu);
    SolverStats s;
    bool ok = fgm.solve(A, b, x, s);

    expect_true(ok, "FGMRES+ILU Poisson converged");
    expect_less(rel_res(A, b, x), 1e-8, "rel_res < 1e-9");
}

HSPS_TEST(fgmres_poisson_amg) {
    auto A = SparseMatrix::poisson_2d(16);
    Vector b(256, 1.0), x(256, 0.0);

    auto amg = std::make_shared<AMGPreconditioner>();
    amg->set_max_levels(5);
    amg->setup(A);

    FGMRESSolver fgm; fgm.set_params(default_params()); fgm.set_preconditioner(amg);
    SolverStats s;
    bool ok = fgm.solve(A, b, x, s);

    expect_true(ok, "FGMRES+AMG Poisson converged");
    expect_less(rel_res(A, b, x), 1e-8, "rel_res < 1e-9");
}

HSPS_TEST(fgmres_non_symmetric_convdiff) {
    // Non-symmetric convection-diffusion: CG would diverge, FGMRES should work
    auto A = SparseMatrix::convection_diffusion_2d(12, 0.1, 5.0, 0.0);
    const Index N = 12 * 12;
    Vector b(N, 1.0), x(N, 0.0);

    auto ilu = std::make_shared<ILUPreconditioner>(); ilu->setup(A);
    FGMRESSolver fgm; fgm.set_params(default_params()); fgm.set_preconditioner(ilu);
    SolverStats s;
    bool ok = fgm.solve(A, b, x, s);

    expect_true(ok, "FGMRES+ILU conv-diff converged");
    expect_less(rel_res(A, b, x), 1e-8, "rel_res < 1e-9");
}

HSPS_TEST(fgmres_restart_count_tracked) {
    // Use small restart to force multiple restarts
    auto A = SparseMatrix::poisson_2d(16);
    Vector b(256, 1.0), x(256, 0.0);

    SolverParams p = default_params();
    p.restart_size = 5;  // tiny restart → many restarts needed

    auto ilu = std::make_shared<ILUPreconditioner>(); ilu->setup(A);
    FGMRESSolver fgm; fgm.set_params(p); fgm.set_preconditioner(ilu);
    SolverStats s;
    fgm.solve(A, b, x, s);

    expect_true(s.restarts >= 1, "restarts tracked");
}

HSPS_TEST(fgmres_amg_vcycle_flexibility) {
    // AMG is non-stationary → standard GMRES would fail; FGMRES handles it
    auto A = SparseMatrix::poisson_2d(12);
    const Index N = 144;
    Vector b(N, 1.0), x(N, 0.0);

    auto amg = std::make_shared<AMGPreconditioner>();
    amg->setup(A);

    FGMRESSolver fgm;
    SolverParams p = default_params();
    p.restart_size = 20;
    fgm.set_params(p);
    fgm.set_preconditioner(amg);

    SolverStats s;
    bool ok = fgm.solve(A, b, x, s);
    expect_true(ok, "FGMRES flexible with non-stationary AMG");
}

int main() {
    return ::hsps_test::TestRegistry::instance().run_all();
}
