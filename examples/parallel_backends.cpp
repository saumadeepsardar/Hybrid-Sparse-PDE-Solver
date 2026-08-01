// =============================================================================
// examples/parallel_backends.cpp
//
// Demonstrates runtime backend selection and benchmarks each backend
// on the same problem so you can compare wall-time and energy across:
//   OMP  — shared-memory (always available)
//   MPI  — distributed  (only if compiled with HSPS_USE_MPI)
//   CUDA — GPU          (only if compiled with HSPS_USE_CUDA)
//
// Usage
//   ./bin/parallel_backends [backend] [grid_n] [solver]
//   backend : OMP | MPI | CUDA          (default: OMP)
//   grid_n  : interior grid points      (default: 64)
//   solver  : CG | FGMRES               (default: CG)
//
// MPI run:
//   mpirun -np 4 ./bin/parallel_backends MPI 64 FGMRES
// =============================================================================

#include "../include/core/types.hpp"
#include "../include/core/sparse_matrix.hpp"
#include "../include/core/vector.hpp"
#include "../include/parallel/parallel_config.hpp"
#include "../include/parallel/parallel_context.hpp"
#include "../include/parallel/backend_factory.hpp"
#include "../include/parallel/omp_backend.hpp"
#include "../include/solvers/parallel_cg_solver.hpp"
#include "../include/solvers/parallel_fgmres_solver.hpp"
#include "../include/preconditioners/jacobi.hpp"
#include "../include/preconditioners/ilu.hpp"
#include "../include/preconditioners/amg.hpp"
#include "../include/energy/energy_monitor.hpp"
#include "../include/adaptive/adaptive_selector.hpp"
#include "../include/utils/logger.hpp"
#include "../include/utils/timer.hpp"

#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <memory>

using namespace hsps;

// ---------------------------------------------------------------------------
static std::string to_upper(const std::string& s) {
    std::string u = s;
    std::transform(u.begin(), u.end(), u.begin(),
                   [](unsigned char c){ return std::toupper(c); });
    return u;
}

static double rel_res(const SparseMatrix& A, const Vector& b, const Vector& x) {
    Vector r = A * x;
    Real bnorm = b.norm2();
    for (Index i = 0; i < b.size(); ++i) r[i] = b[i] - r[i];
    return (bnorm > 1e-15) ? r.norm2() / bnorm : r.norm2();
}

// ---------------------------------------------------------------------------
// Run one backend benchmark
// ---------------------------------------------------------------------------
struct BenchResult {
    std::string backend_name;
    std::string solver_name;
    bool        converged;
    int         iterations;
    double      solve_time_s;
    double      energy_proxy_j;
    double      rel_res_final;
    int         threads;
    int         ranks;
};

BenchResult run_benchmark(const std::string& backend_name,
                           const std::string& solver_name,
                           const SparseMatrix& A,
                           const Vector& rhs,
                           const SolverParams& params,
                           const ParallelContext& ctx) {
    // Build backend
    auto backend = std::shared_ptr<BackendBase>(BackendFactory::create(backend_name, &ctx, 0).release());

    // Build preconditioner (ILU for FGMRES, Jacobi for CG)
    std::shared_ptr<PreconditionerBase> precond;
    if (solver_name == "FGMRES") {
        auto ilu = std::make_shared<ILUPreconditioner>();
        ilu->setup(A);
        precond = ilu;
    } else {
        auto jac = std::make_shared<JacobiPreconditioner>();
        jac->setup(A);
        precond = jac;
    }

    // Build parallel solver
    std::unique_ptr<SolverBase> solver;
    if (solver_name == "FGMRES") {
        auto s = std::make_unique<ParallelFGMRESSolver>(backend);
        s->set_preconditioner(precond);
        s->set_params(params);
        solver = std::move(s);
    } else {
        auto s = std::make_unique<ParallelCGSolver>(backend);
        s->set_preconditioner(precond);
        s->set_params(params);
        solver = std::move(s);
    }

    Vector x(A.rows(), 0.0);
    SolverStats stats;
    solver->solve(A, rhs, x, stats);

    BenchResult r;
    r.backend_name  = backend->name();
    r.solver_name   = solver_name;
    r.converged     = stats.converged;
    r.iterations    = stats.iterations;
    r.solve_time_s  = stats.solve_time_s;
    r.energy_proxy_j= stats.energy_joules;
    r.rel_res_final = rel_res(A, rhs, x);
    r.threads       = ctx.omp_threads();
    r.ranks         = ctx.nprocs();
    return r;
}

// ---------------------------------------------------------------------------
// Print comparison table
// ---------------------------------------------------------------------------
static void print_table(const std::vector<BenchResult>& results) {
    const int w1 = 16, w2 = 8, w = 12;
    std::cout << "\n"
              << std::left  << std::setw(w1) << "Backend"
              << std::left  << std::setw(w2) << "Solver"
              << std::right << std::setw(w)  << "Conv"
              << std::setw(w) << "Iters"
              << std::setw(w) << "Time(s)"
              << std::setw(w) << "Energy(J)"
              << std::setw(w) << "Rel-Res"
              << std::setw(w) << "Ranks×Thrd"
              << "\n"
              << std::string(w1 + w2 + 6*w, '-') << "\n";

    for (const auto& r : results) {
        std::string rt = std::to_string(r.ranks) + "×" + std::to_string(r.threads);
        std::cout
            << std::left  << std::setw(w1) << r.backend_name
            << std::left  << std::setw(w2) << r.solver_name
            << std::right << std::setw(w)  << (r.converged ? "YES" : "NO")
            << std::setw(w) << r.iterations
            << std::fixed     << std::setprecision(4)
            << std::setw(w)   << r.solve_time_s
            << std::scientific<< std::setprecision(3)
            << std::setw(w)   << r.energy_proxy_j
            << std::setw(w)   << r.rel_res_final
            << std::right << std::setw(w)  << rt
            << "\n";
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    // Parse arguments
    std::string backend_arg = (argc > 1) ? to_upper(argv[1]) : "OMP";
    Index n      = (argc > 2) ? std::atoi(argv[2]) : 64;
    std::string solver_arg  = (argc > 3) ? to_upper(argv[3]) : "CG";

    // Initialise parallel context
    ParallelConfig cfg;
    cfg.backends    = ParallelConfig::parse(backend_arg);
    cfg.omp_threads = 0;  // use OMP_NUM_THREADS
    auto ctx = ParallelContext::init(argc, argv, cfg);

    if (ctx.is_root()) {
        ctx.print_info(std::cout);
        std::cout << "Problem: Poisson 2D  n=" << n
                  << "  N=" << n*n << " DOFs\n";
        std::cout << "Selected backend: " << BackendFactory::select_name(cfg) << "\n\n";
    }

    Logger::instance().set_level(LogLevel::WARN);

    // Build problem
    SparseMatrix A = SparseMatrix::poisson_2d(n);
    Vector rhs(n * n, 1.0);
    A.print_info(std::cout);

    SolverParams params;
    params.tol          = 1e-8;
    params.max_iter     = 5000;
    params.restart_size = 50;
    params.verbose      = false;

    std::vector<BenchResult> results;

    // ── Always run OMP baseline ─────────────────────────────────────────────
    if (ctx.is_root())
        std::cout << "\nRunning OMP baseline...\n";
    for (const auto& sv : {"CG", "FGMRES"})
        results.push_back(run_benchmark("OMP", sv, A, rhs, params, ctx));

    // ── Run requested backend (if different from OMP) ────────────────────
    if (backend_arg != "OMP") {
        if (ctx.is_root())
            std::cout << "Running " << backend_arg << " backend...\n";
        results.push_back(
            run_benchmark(backend_arg, solver_arg, A, rhs, params, ctx));
    }

    // ── Adaptive selector with best available backend ────────────────────
    if (ctx.is_root()) {
        std::cout << "\nRunning Adaptive Selector (OMP)...\n";
        AdaptiveSelector sel(params);
        Vector x(n * n, 0.0);
        SolverStats stats;
        sel.solve(A, rhs, x, stats);

        BenchResult ar;
        ar.backend_name   = "Adaptive";
        ar.solver_name    = to_string(stats.solver_used);
        ar.converged      = stats.converged;
        ar.iterations     = stats.iterations;
        ar.solve_time_s   = stats.solve_time_s;
        ar.energy_proxy_j = stats.energy_joules;
        ar.rel_res_final  = rel_res(A, rhs, x);
        ar.threads        = ctx.omp_threads();
        ar.ranks          = 1;
        results.push_back(ar);
    }

    // ── Print results (root only) ─────────────────────────────────────────
    if (ctx.is_root()) {
        print_table(results);

        // Highlight fastest and most energy-efficient
        double min_time = 1e99, min_energy = 1e99;
        const BenchResult *fastest = nullptr, *greenest = nullptr;
        for (const auto& r : results) {
            if (!r.converged) continue;
            if (r.solve_time_s   < min_time)   { min_time   = r.solve_time_s;   fastest = &r; }
            if (r.energy_proxy_j < min_energy) { min_energy = r.energy_proxy_j; greenest= &r; }
        }
        if (fastest)
            std::cout << "\n→ Fastest  : " << fastest->backend_name
                      << "+" << fastest->solver_name
                      << "  " << std::fixed << std::setprecision(4)
                      << fastest->solve_time_s << "s\n";
        if (greenest)
            std::cout << "→ Greenest : " << greenest->backend_name
                      << "+" << greenest->solver_name
                      << "  " << std::scientific << std::setprecision(3)
                      << greenest->energy_proxy_j << "J\n";

        std::cout << "\nBackend capabilities compiled in:\n"
                  << "  OMP  : " << (compiled_with_omp()  ? "YES" : "NO") << "\n"
                  << "  MPI  : " << (compiled_with_mpi()  ? "YES" : "NO") << "\n"
                  << "  CUDA : " << (compiled_with_cuda() ? "YES" : "NO") << "\n";
    }

    return 0;
}
