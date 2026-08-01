// =============================================================================
// solver_factory.cpp
// =============================================================================

#include "../../include/solvers/solver_factory.hpp"
#include "../../include/solvers/cg_solver.hpp"
#include "../../include/solvers/fgmres_solver.hpp"
#include "../../include/solvers/pipelined_fgmres_solver.hpp"
#include "../../include/solvers/sstep_cg_solver.hpp"
#include "../../include/solvers/async_fgmres_solver.hpp"
#include "../../include/preconditioners/jacobi.hpp"
#include "../../include/preconditioners/ilu.hpp"
#include "../../include/preconditioners/amg.hpp"
#include "../../include/utils/timer.hpp"
#include <stdexcept>
#include <algorithm>
#include <cctype>

namespace hsps {

// ---------------------------------------------------------------------------
// make_solver — solver only (no preconditioner)
// ---------------------------------------------------------------------------
std::unique_ptr<SolverBase>
SolverFactory::make_solver(SolverType type, const SolverParams& params) {
    std::unique_ptr<SolverBase> solver;
    switch (type) {
        case SolverType::CG:
            solver = std::make_unique<CGSolver>();
            break;
        case SolverType::FGMRES:
            solver = std::make_unique<FGMRESSolver>();
            break;
        case SolverType::BICGSTAB:
            // BiCGSTAB not yet implemented — fall back to FGMRES
            solver = std::make_unique<FGMRESSolver>();
            break;
        case SolverType::SSTEP_CG:
            solver = std::make_unique<SStepCGSolver>(params.sstep_k);
            break;
        case SolverType::ASYNC_FGMRES:
            solver = std::make_unique<AsyncFGMRESSolver>();
            break;
        default:
            throw std::invalid_argument("SolverFactory: unknown SolverType");
    }
    solver->set_params(params);
    return solver;
}

// ---------------------------------------------------------------------------
// make_precond — preconditioner only (setup called automatically)
// ---------------------------------------------------------------------------
std::shared_ptr<PreconditionerBase>
SolverFactory::make_precond(PrecondType type, const SparseMatrix& A,
                             const SolverParams& params,
                             double* setup_time_s) {
    std::shared_ptr<PreconditionerBase> precond;
    double t = 0.0;

    switch (type) {
    case PrecondType::NONE:
        // Return nullptr — solver will run un-preconditioned
        if (setup_time_s) *setup_time_s = 0.0;
        return nullptr;

    case PrecondType::JACOBI: {
        auto jac = std::make_shared<JacobiPreconditioner>();
        t = jac->setup(A);
        precond = jac;
        break;
    }
    case PrecondType::ILU: {
        auto ilu = std::make_shared<ILUPreconditioner>();
        ilu->set_drop_tol(params.drop_tol);
        // fill_per_row: expose via a separate param if needed; default 0
        t = ilu->setup(A);
        precond = ilu;
        break;
    }
    case PrecondType::AMG: {
        auto amg = std::make_shared<AMGPreconditioner>();
        amg->set_max_levels    (params.amg_levels);
        amg->set_strength_thresh(params.amg_strong);
        amg->set_smooth_pre    (params.amg_smooth_pre);
        amg->set_smooth_post   (params.amg_smooth_post);
        t = amg->setup(A);
        precond = amg;
        break;
    }
    default:
        throw std::invalid_argument("SolverFactory: unknown PrecondType");
    }

    if (setup_time_s) *setup_time_s = t;
    return precond;
}

// ---------------------------------------------------------------------------
// Primary make (enum-based)
// ---------------------------------------------------------------------------
SolverPrecondPair
SolverFactory::make(SolverType solver_type, PrecondType precond_type,
                    const SparseMatrix& A, const SolverParams& params,
                    double* setup_time_s) {
    double t = 0.0;
    auto precond = make_precond(precond_type, A, params, &t);
    if (setup_time_s) *setup_time_s = t;

    auto solver = make_solver(solver_type, params);
    if (precond) solver->set_preconditioner(precond);
    return {std::move(solver), precond};
}

// ---------------------------------------------------------------------------
// make_for_state — map adaptive ladder state to canonical pair
// ---------------------------------------------------------------------------
SolverPrecondPair
SolverFactory::make_for_state(AdaptiveState state,
                               const SparseMatrix& A,
                               const SolverParams& params,
                               double* setup_time_s) {
    switch (state) {
    case AdaptiveState::EASY:
        return make(SolverType::CG,     PrecondType::JACOBI, A, params, setup_time_s);
    case AdaptiveState::MODERATE:
        return make(SolverType::FGMRES, PrecondType::ILU,    A, params, setup_time_s);
    case AdaptiveState::HARD:
        return make(SolverType::FGMRES, PrecondType::AMG,    A, params, setup_time_s);
    default:
        throw std::invalid_argument("SolverFactory: unknown AdaptiveState");
    }
}

// ---------------------------------------------------------------------------
// parse helpers
// ---------------------------------------------------------------------------
static std::string to_upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::toupper(c); });
    return s;
}

SolverType SolverFactory::parse_solver(const std::string& s) {
    const auto u = to_upper(s);
    if (u == "CG")           return SolverType::CG;
    if (u == "FGMRES")       return SolverType::FGMRES;
    if (u == "BICGSTAB")     return SolverType::BICGSTAB;
    if (u == "SSTEP_CG" || u == "SSTEP" || u == "S-STEP") return SolverType::SSTEP_CG;
    if (u == "ASYNC_FGMRES" || u == "ASYNC") return SolverType::ASYNC_FGMRES;
    throw std::invalid_argument("SolverFactory: unknown solver name '" + s + "'");
}

PrecondType SolverFactory::parse_precond(const std::string& s) {
    const auto u = to_upper(s);
    if (u == "NONE"   || u == "IDENTITY") return PrecondType::NONE;
    if (u == "JACOBI" || u == "DIAG")     return PrecondType::JACOBI;
    if (u == "ILU"    || u == "ILU0")     return PrecondType::ILU;
    if (u == "AMG")                        return PrecondType::AMG;
    throw std::invalid_argument("SolverFactory: unknown precond name '" + s + "'");
}

// ---------------------------------------------------------------------------
// String-based make
// ---------------------------------------------------------------------------
SolverPrecondPair
SolverFactory::make(const std::string& solver_name,
                    const std::string& precond_name,
                    const SparseMatrix& A, const SolverParams& params,
                    double* setup_time_s) {
    return make(parse_solver(solver_name), parse_precond(precond_name),
                A, params, setup_time_s);
}

} // namespace hsps
