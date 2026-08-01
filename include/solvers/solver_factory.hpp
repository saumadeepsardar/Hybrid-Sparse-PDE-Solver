#pragma once

// =============================================================================
// solver_factory.hpp  —  Central creation point for solver + preconditioner
//
// Provides a clean, single-call API for:
//   • Named pair creation  (SolverFactory::make("CG", "Jacobi", params))
//   • Energy-tier creation (SolverFactory::make_for_tier(AdaptiveState, ...))
//   • Registration of custom solver/preconditioner types
//
// All solvers produced here are fully configured (params set, preconditioner
// attached, preconditioner setup() already called against the matrix A).
//
// Usage
// -----
//   auto [solver, precond] = SolverFactory::make(
//       SolverType::FGMRES, PrecondType::ILU, A, params);
//   SolverStats s;
//   solver->solve(A, b, x, s);
// =============================================================================

#include "../core/types.hpp"
#include "../core/sparse_matrix.hpp"
#include "solver_base.hpp"
#include "../preconditioners/preconditioner_base.hpp"
#include <memory>
#include <utility>
#include <string>
#include <stdexcept>

namespace hsps {

// Convenience alias
using SolverPrecondPair = std::pair<std::unique_ptr<SolverBase>,
                                    std::shared_ptr<PreconditionerBase>>;

// ===========================================================================
// SolverFactory
// ===========================================================================
class SolverFactory {
public:
    // ------------------------------------------------------------------
    // Primary factory — enum-based
    // ------------------------------------------------------------------

    /// Build and fully set up a (solver, preconditioner) pair for matrix A.
    /// Preconditioner::setup(A) is called automatically.
    /// Returns elapsed setup time via out-param setup_time_s.
    static SolverPrecondPair make(SolverType   solver_type,
                                  PrecondType  precond_type,
                                  const SparseMatrix& A,
                                  const SolverParams& params,
                                  double* setup_time_s = nullptr);

    /// Build for a given adaptive-ladder state.
    static SolverPrecondPair make_for_state(AdaptiveState       state,
                                            const SparseMatrix& A,
                                            const SolverParams& params,
                                            double* setup_time_s = nullptr);

    // ------------------------------------------------------------------
    // String-based (useful for config-file / CLI driven workflows)
    // ------------------------------------------------------------------
    static SolverPrecondPair make(const std::string& solver_name,
                                  const std::string& precond_name,
                                  const SparseMatrix& A,
                                  const SolverParams& params,
                                  double* setup_time_s = nullptr);

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------
    static SolverType   parse_solver (const std::string& s);
    static PrecondType  parse_precond(const std::string& s);

    /// Create only the preconditioner and run setup().
    static std::shared_ptr<PreconditionerBase>
    make_precond(PrecondType type, const SparseMatrix& A,
                 const SolverParams& params,
                 double* setup_time_s = nullptr);

    /// Create only the solver (no preconditioner attached).
    static std::unique_ptr<SolverBase>
    make_solver(SolverType type, const SolverParams& params);

private:
    SolverFactory() = delete;  // static-only class
};

} // namespace hsps
