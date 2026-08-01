#pragma once

// =============================================================================
// solver_base.hpp  —  Abstract base class for all Krylov solvers
// =============================================================================

#include "../core/types.hpp"
#include "../core/sparse_matrix.hpp"
#include "../core/vector.hpp"
#include "../preconditioners/preconditioner_base.hpp"
#include <memory>
// Forward declaration for EnergyMonitor to avoid circular includes
namespace hsps { class EnergyMonitor; }

namespace hsps {

// ---------------------------------------------------------------------------
// Abstract Krylov solver interface
// ---------------------------------------------------------------------------
class SolverBase {
public:
    explicit SolverBase(SolverType type) : type_(type) {}
    virtual ~SolverBase() = default;

    // ------------------------------------------------------------------
    // Core interface
    // ------------------------------------------------------------------

    /// Solve  A * x = b.
    /// x is used as the initial guess if x.size() == b.size().
    /// Returns true on convergence.
    virtual bool solve(const SparseMatrix& A,
                       const Vector&       b,
                             Vector&       x,
                       SolverStats&        stats) = 0;

    // ------------------------------------------------------------------
    // Configuration
    // ------------------------------------------------------------------
    void set_params(const SolverParams& p) { params_ = p; }
    void set_preconditioner(std::shared_ptr<PreconditionerBase> M) { precond_ = M; }
    void set_verbose(bool v) { params_.verbose = v; }

    /// Attach an EnergyMonitor for per-iteration hardware energy sampling (Thrust 2).
    /// When set and params_.record_hw_energy is true, the solver calls
    /// monitor->snapshot_rapl() at the start and end of each iteration.
    void set_energy_monitor(std::shared_ptr<EnergyMonitor> mon) { energy_mon_ = mon; }

    const SolverParams& params()  const { return params_; }
    SolverType          type()    const { return type_;   }

protected:
    SolverType  type_;
    SolverParams params_;
    std::shared_ptr<PreconditionerBase> precond_    = nullptr;
    std::shared_ptr<EnergyMonitor>      energy_mon_ = nullptr;  ///< Thrust 2

    // Helper: compute relative residual norm
    Real relative_residual(const SparseMatrix& A,
                           const Vector&       b,
                           const Vector&       x) const;
};

} // namespace hsps
