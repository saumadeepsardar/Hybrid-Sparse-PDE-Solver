// =============================================================================
// examples/energy_benchmark.cpp
//
// Isolates the energy cost of each preconditioner on the same matrix so that
// the Adaptive Selector's energy model can be validated and calibrated.
//
// For each (solver, preconditioner) pair it reports:
//   • Setup time
//   • Iterations to convergence
//   • Total proxy energy (J)
//   • Energy per iteration (J/iter)
//   • Memory traffic (MB)
//   • Effective FLOP rate (GFLOP/s)
//
// Intended to guide the ML warm-start model's "minimum-energy configuration"
// objective (design: "ML learning objective: minimum-energy successful config").
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
#include "../include/utils/timer.hpp"

#include <cmath>
#include <iostream>
#include <iomanip>
#include <functional>
#include <string>
#include <vector>

using namespace hsps;

// ---------------------------------------------------------------------------
// One benchmark run
// ---------------------------------------------------------------------------
struct BenchResult {
    std::string label;
    bool        converged;
    int         iterations;
    double      setup_time_s;
    double      solve_time_s;
    double      total_time_s;
    double      energy_proxy_j;
    double      energy_per_iter_j;
    double      mem_mb;
    double      gflops;
};

template <typename SolverT, typename PrecondT>
BenchResult run_bench(const std::string& label,
                      const SparseMatrix& A,
                      const Vector& rhs,
                      const SolverParams& params) {
    BenchResult r;
    r.label = label;

    auto precond = std::make_shared<PrecondT>();
    Timer setup_t; setup_t.start();
    r.setup_time_s = precond->setup(A);
    setup_t.stop();

    SolverT solver;
    solver.set_preconditioner(precond);
    solver.set_params(params);

    Vector x(A.rows(), 0.0);
    SolverStats stats;

    Timer solve_t; solve_t.start();
    r.converged    = solver.solve(A, rhs, x, stats);
    solve_t.stop();

    r.iterations        = stats.iterations;
    r.solve_time_s      = stats.solve_time_s;
    r.total_time_s      = r.setup_time_s + r.solve_time_s;
    r.energy_proxy_j    = stats.energy_joules;
    r.energy_per_iter_j = (r.iterations > 0) ? (r.energy_proxy_j / r.iterations) : 0.0;
    r.mem_mb            = static_cast<double>(stats.mem_bytes) / (1024.0 * 1024.0);
    r.gflops            = (r.solve_time_s > 1e-9)
                          ? (static_cast<double>(stats.flop_count) / r.solve_time_s * 1e-9)
                          : 0.0;
    return r;
}

// ---------------------------------------------------------------------------
// Print results table
// ---------------------------------------------------------------------------
static void print_table(const std::vector<BenchResult>& results) {
    const int w1 = 24, w = 12;
    std::cout << std::left  << std::setw(w1) << "Pair"
              << std::right
              << std::setw(w) << "Conv"
              << std::setw(w) << "Iters"
              << std::setw(w) << "Setup(s)"
              << std::setw(w) << "Solve(s)"
              << std::setw(w) << "E(J)"
              << std::setw(w) << "E/iter(J)"
              << std::setw(w) << "Mem(MB)"
              << std::setw(w) << "GFLOP/s"
              << "\n"
              << std::string(w1 + 8 * w, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::left  << std::setw(w1) << r.label
                  << std::right << std::fixed << std::setprecision(0)
                  << std::setw(w) << (r.converged ? "YES" : "NO")
                  << std::setw(w) << r.iterations
                  << std::scientific << std::setprecision(3)
                  << std::setw(w) << r.setup_time_s
                  << std::setw(w) << r.solve_time_s
                  << std::setw(w) << r.energy_proxy_j
                  << std::setw(w) << r.energy_per_iter_j
                  << std::fixed  << std::setprecision(2)
                  << std::setw(w) << r.mem_mb
                  << std::setw(w) << r.gflops
                  << "\n";
    }
}

int main(int argc, char* argv[]) {
    // Grid sizes to benchmark
    std::vector<Index> grid_sizes = {16, 32, 64};
    if (argc > 1) grid_sizes = { std::atoi(argv[1]) };

    SolverParams params;
    params.tol          = 1e-8;
    params.max_iter     = 3000;
    params.restart_size = 50;
    params.verbose      = false;

    for (Index n : grid_sizes) {
        std::cout << "\n========================================\n"
                  << "  Grid: " << n << "x" << n << "  (N=" << n*n << " DOFs)\n"
                  << "========================================\n";

        SparseMatrix A = SparseMatrix::poisson_2d(n);
        Vector rhs(n * n, 1.0);  // simple unit RHS

        std::vector<BenchResult> results;
        results.push_back(run_bench<CGSolver,     JacobiPreconditioner>("CG + Jacobi",    A, rhs, params));
        results.push_back(run_bench<CGSolver,     ILUPreconditioner   >("CG + ILU(0)",    A, rhs, params));
        results.push_back(run_bench<FGMRESSolver, JacobiPreconditioner>("FGMRES + Jacobi",A, rhs, params));
        results.push_back(run_bench<FGMRESSolver, ILUPreconditioner   >("FGMRES + ILU(0)",A, rhs, params));
        results.push_back(run_bench<FGMRESSolver, AMGPreconditioner   >("FGMRES + AMG",   A, rhs, params));

        print_table(results);

        // Find minimum-energy converged configuration
        const BenchResult* best = nullptr;
        for (const auto& r : results)
            if (r.converged && (!best || r.energy_proxy_j < best->energy_proxy_j))
                best = &r;
        if (best)
            std::cout << "\n→ Minimum-energy converged: " << best->label
                      << "  (" << std::scientific << std::setprecision(3)
                      << best->energy_proxy_j << " J)\n";
    }

    std::cout << "\nNote: Energy values are proxy estimates (α·FLOP + β·MEM).\n"
              << "      See EnergyMonitor::calibrate() for hardware calibration.\n";
    return 0;
}
