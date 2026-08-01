// =============================================================================
// examples/convection_diffusion.cpp
//
// Solves the 2-D convection-diffusion equation
//   -ε ∇²u + β·∇u = f   on [0,1]²,   u = 0 on ∂Ω
//
// As ε → 0 the system becomes increasingly convection-dominated and
// non-symmetric, stressing the solver ladder:
//   ε = 1.0  → well-conditioned,  CG+Jacobi works fine
//   ε = 0.1  → moderate,          FGMRES+ILU expected
//   ε = 0.01 → stiff/non-sym,     FGMRES+AMG may be needed
//
// Demonstrates adaptive escalation and energy comparison across regimes.
// =============================================================================

#include "../include/core/types.hpp"
#include "../include/core/sparse_matrix.hpp"
#include "../include/core/vector.hpp"
#include "../include/adaptive/adaptive_selector.hpp"
#include "../include/energy/energy_monitor.hpp"
#include "../include/utils/logger.hpp"

#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>

using namespace hsps;

// ---------------------------------------------------------------------------
// Build uniform RHS vector (f = 1)
// ---------------------------------------------------------------------------
static Vector build_unit_rhs(Index n) {
    const Real h2 = 1.0 / ((n + 1) * (n + 1));
    Vector rhs(n * n);
    for (Index i = 0; i < n * n; ++i) rhs[i] = h2;
    return rhs;
}

// ---------------------------------------------------------------------------
// Print a test case result line
// ---------------------------------------------------------------------------
static void report(const std::string& tag, const SolverStats& s,
                   const EnergySample& energy) {
    std::cout
        << std::left  << std::setw(30) << tag
        << std::right
        << "  state="   << std::left << std::setw(22) << to_string(s.state_used)
        << std::right
        << "  iters="   << std::setw(5)  << s.iterations
        << "  rel_res=" << std::scientific << std::setprecision(2) << s.final_residual
        << "  t_solve=" << std::fixed    << std::setprecision(4) << s.solve_time_s << "s"
        << "  E~"       << std::scientific << std::setprecision(2) << s.energy_joules << "J"
        << (s.converged ? "  ✓" : "  ✗")
        << "\n";
}

int main(int argc, char* argv[]) {
    // Grid size
    Index n = 32;
    if (argc > 1) n = std::atoi(argv[1]);

    std::cout << "=== Convection-Diffusion  n=" << n
              << "  DOFs=" << n * n << " ===\n\n";

    Logger::instance().set_level(LogLevel::WARN);  // quieter for sweep output

    SolverParams params;
    params.tol          = 1e-8;
    params.max_iter     = 5000;
    params.restart_size = 50;
    params.stall_window = 20;
    params.verbose      = false;

    // Sweep over diffusion coefficient and flow velocity
    struct Case {
        Real eps;
        Real bx;
        Real by;
        std::string label;
    };

    std::vector<Case> cases = {
        { 1.0,  0.0,  0.0, "ε=1.0  β=0      (Poisson)" },
        { 0.5,  5.0,  5.0, "ε=0.5  β=(5,5)  (mild conv)" },
        { 0.1,  10.0, 0.0, "ε=0.1  β=(10,0) (moderate)" },
        { 0.05, 20.0, 5.0, "ε=0.05 β=(20,5) (stiff)" },
        { 0.01, 50.0, 0.0, "ε=0.01 β=(50,0) (highly conv)" },
    };

    std::cout << std::left  << std::setw(30) << "Case"
              << std::right
              << "  State                 "
              << "  Iters  Rel-Res     T_solve    Energy\n"
              << std::string(110, '-') << "\n";

    for (const auto& c : cases) {
        SparseMatrix A = SparseMatrix::convection_diffusion_2d(n, c.eps, c.bx, c.by);
        Vector rhs     = build_unit_rhs(n);

        auto energy_mon = std::make_shared<EnergyMonitor>();
        energy_mon->start();

        AdaptiveSelector sel(params);
        sel.set_energy_monitor(energy_mon);

        Vector x(n * n, 0.0);
        SolverStats stats;
        sel.solve(A, rhs, x, stats);

        energy_mon->stop();
        auto esample = energy_mon->sample();

        report(c.label, stats, esample);
    }

    std::cout << "\n(✓ = converged,  ✗ = did not converge within budget)\n";

    // ------------------------------------------------------------------
    // Detailed energy breakdown for the hardest case
    // ------------------------------------------------------------------
    std::cout << "\n=== Detailed Energy Breakdown: ε=0.01, β=(50,0) ===\n";
    {
        SparseMatrix A = SparseMatrix::convection_diffusion_2d(n, 0.01, 50.0, 0.0);
        Vector rhs     = build_unit_rhs(n);

        auto energy_mon = std::make_shared<EnergyMonitor>();
        energy_mon->start();

        AdaptiveSelector sel(params);
        sel.set_energy_monitor(energy_mon);

        Vector x(n * n, 0.0);
        SolverStats stats;
        sel.solve(A, rhs, x, stats);

        energy_mon->stop();
        energy_mon->print_summary(std::cout);
        sel.print_summary(std::cout);

        // Dump per-iteration CSV for post-processing
        energy_mon->dump_csv("conv_diff_energy.csv");
        std::cout << "Per-iteration data written to: conv_diff_energy.csv\n";
    }

    return 0;
}
