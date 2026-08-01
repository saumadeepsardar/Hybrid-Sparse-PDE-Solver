// =============================================================================
// examples/poisson_2d.cpp
//
// Solves the 2-D Poisson equation
//   -∇²u = f   on [0,1]²,   u = 0 on ∂Ω
//
// with f(x,y) = 2π² sin(πx) sin(πy)   →   u*(x,y) = sin(πx) sin(πy)
//
// Discretised on an n×n interior grid with 5-point stencil.
// h = 1/(n+1),  matrix scaled by h².
//
// Demonstrates:
//   • Direct use of each solver/preconditioner pair
//   • AdaptiveSelector (online adaptive ladder)
//   • EnergyMonitor
//   • L∞ error against exact solution
// =============================================================================

#include "../include/core/types.hpp"
#include "../include/core/sparse_matrix.hpp"
#include "../include/core/vector.hpp"
#include "../include/solvers/cg_solver.hpp"
#include "../include/solvers/fgmres_solver.hpp"
#include "../include/preconditioners/jacobi.hpp"
#include "../include/preconditioners/ilu.hpp"
#include "../include/preconditioners/amg.hpp"
#include "../include/energy/energy_monitor.hpp"
#include "../include/adaptive/adaptive_selector.hpp"
#include "../include/utils/logger.hpp"
#include "../include/utils/timer.hpp"

#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>

using namespace hsps;

// ---------------------------------------------------------------------------
// Build RHS and exact solution vectors for the test problem
// ---------------------------------------------------------------------------
static void build_rhs_exact(Index n, Vector& rhs, Vector& u_exact) {
    const Real h  = 1.0 / (n + 1);
    const Real h2 = h * h;
    const Index N = n * n;
    rhs.resize(N);
    u_exact.resize(N);

    for (Index i = 0; i < n; ++i) {
        Real y = (i + 1) * h;
        for (Index j = 0; j < n; ++j) {
            Real x   = (j + 1) * h;
            Index dof = i * n + j;
            rhs[dof]     = h2 * 2.0 * M_PI * M_PI * std::sin(M_PI * x) * std::sin(M_PI * y);
            u_exact[dof] = std::sin(M_PI * x) * std::sin(M_PI * y);
        }
    }
}

// ---------------------------------------------------------------------------
// Print a result row
// ---------------------------------------------------------------------------
static void print_result(const std::string& label, const SolverStats& s,
                          Real l_inf_err) {
    std::cout << std::left  << std::setw(22) << label
              << std::right
              << "  iters=" << std::setw(5) << s.iterations
              << "  rel_res=" << std::scientific << std::setprecision(3) << s.final_residual
              << "  time=" << std::fixed << std::setprecision(4) << s.solve_time_s << "s"
              << "  energy~" << std::scientific << std::setprecision(2) << s.energy_joules << "J"
              << "  L∞_err=" << std::scientific << std::setprecision(3) << l_inf_err
              << (s.converged ? "  ✓" : "  ✗ FAILED")
              << "\n";
}

// ---------------------------------------------------------------------------
// Compute L∞ error
// ---------------------------------------------------------------------------
static Real l_inf_error(const Vector& u, const Vector& u_exact) {
    Real err = 0.0;
    for (Index i = 0; i < u.size(); ++i)
        err = std::max(err, std::abs(u[i] - u_exact[i]));
    return err;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    // Grid size: pass n as command-line argument (default 32 → 1024 DOFs)
    Index n = 32;
    if (argc > 1) n = std::atoi(argv[1]);

    std::cout << "=== 2-D Poisson  n=" << n << "  DOFs=" << n*n << " ===\n\n";

    Logger::instance().set_level(LogLevel::INFO);

    // Build matrix and vectors
    SparseMatrix A = SparseMatrix::poisson_2d(n);
    Vector rhs, u_exact;
    build_rhs_exact(n, rhs, u_exact);

    A.print_info(std::cout);
    std::cout << "\n";

    SolverParams params;
    params.tol          = 1e-8;
    params.max_iter     = 5000;
    params.restart_size = 50;
    params.verbose      = false;

    // ------------------------------------------------------------------
    // Run 1: CG + Jacobi
    // ------------------------------------------------------------------
    {
        CGSolver cg;
        auto jac = std::make_shared<JacobiPreconditioner>();
        jac->setup(A);
        cg.set_preconditioner(jac);
        cg.set_params(params);

        Vector x(n * n, 0.0);
        SolverStats stats;
        cg.solve(A, rhs, x, stats);
        print_result("CG + Jacobi", stats, l_inf_error(x, u_exact));
    }

    // ------------------------------------------------------------------
    // Run 2: CG + ILU(0)
    // ------------------------------------------------------------------
    {
        CGSolver cg;
        auto ilu = std::make_shared<ILUPreconditioner>();
        ilu->setup(A);
        cg.set_preconditioner(ilu);
        cg.set_params(params);

        Vector x(n * n, 0.0);
        SolverStats stats;
        cg.solve(A, rhs, x, stats);
        print_result("CG + ILU(0)", stats, l_inf_error(x, u_exact));
    }

    // ------------------------------------------------------------------
    // Run 3: FGMRES + Jacobi
    // ------------------------------------------------------------------
    {
        FGMRESSolver fgm;
        auto jac = std::make_shared<JacobiPreconditioner>();
        jac->setup(A);
        fgm.set_preconditioner(jac);
        fgm.set_params(params);

        Vector x(n * n, 0.0);
        SolverStats stats;
        fgm.solve(A, rhs, x, stats);
        print_result("FGMRES + Jacobi", stats, l_inf_error(x, u_exact));
    }

    // ------------------------------------------------------------------
    // Run 4: FGMRES + ILU(0)
    // ------------------------------------------------------------------
    {
        FGMRESSolver fgm;
        auto ilu = std::make_shared<ILUPreconditioner>();
        ilu->setup(A);
        fgm.set_preconditioner(ilu);
        fgm.set_params(params);

        Vector x(n * n, 0.0);
        SolverStats stats;
        fgm.solve(A, rhs, x, stats);
        print_result("FGMRES + ILU(0)", stats, l_inf_error(x, u_exact));
    }

    // ------------------------------------------------------------------
    // Run 5: FGMRES + AMG
    // ------------------------------------------------------------------
    {
        FGMRESSolver fgm;
        auto amg = std::make_shared<AMGPreconditioner>();
        double setup_t = amg->setup(A);
        std::cout << "AMG setup: " << std::fixed << std::setprecision(4)
                  << setup_t << "s  levels=" << amg->levels() << "\n";
        fgm.set_preconditioner(amg);
        fgm.set_params(params);

        Vector x(n * n, 0.0);
        SolverStats stats;
        stats.setup_time_s = setup_t;
        fgm.solve(A, rhs, x, stats);
        print_result("FGMRES + AMG", stats, l_inf_error(x, u_exact));
    }

    // ------------------------------------------------------------------
    // Run 6: Adaptive Selector
    // ------------------------------------------------------------------
    {
        std::cout << "\n--- Adaptive Selector run ---\n";

        auto energy_mon = std::make_shared<EnergyMonitor>();
        energy_mon->start();

        AdaptiveSelector selector(params);
        selector.set_energy_monitor(energy_mon);

        Vector x(n * n, 0.0);
        SolverStats stats;
        selector.solve(A, rhs, x, stats);
        energy_mon->stop();

        print_result("Adaptive", stats, l_inf_error(x, u_exact));
        selector.print_summary(std::cout);
        energy_mon->print_summary(std::cout);
    }

    return 0;
}
