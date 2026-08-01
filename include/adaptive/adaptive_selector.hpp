#pragma once

// =============================================================================
// adaptive_selector.hpp  —  Online Adaptive Engine
//
// Implements the design ladder:
//   EASY     →  CG  + Jacobi   (default start)
//   MODERATE →  FGMRES + ILU   (triggered: CG stalls)
//   HARD     →  FGMRES + AMG   (triggered: system stiff / FGMRES+ILU stalls)
//
// Decision signals
//   • Convergence stall:  residual ratio > stall_threshold for stall_window iters
//   • Stiffness signal:   estimated condition number > stiff_thresh  OR
//                         iter count passes stiff_iter fraction of max_iter
//
// Future hook: replace heuristic decisions with GNN inference.
// =============================================================================

#include "../core/types.hpp"
#include "../core/sparse_matrix.hpp"
#include "../core/vector.hpp"
#include "../energy/energy_monitor.hpp"
// Forward-declare MLAdvisor so the header doesn't pull in ml_advisor.hpp
// unconditionally (avoids circular dependency for users who only need the
// base selector).
namespace hsps { class MLAdvisor; struct SolverAdvice; }
#include "../solvers/solver_base.hpp"
#include "../preconditioners/preconditioner_base.hpp"
#include <memory>
#include <vector>
#include <ostream>

namespace hsps {

// ---------------------------------------------------------------------------
// History entry for one escalation event
// ---------------------------------------------------------------------------
struct EscalationRecord {
    int         at_iteration;
    AdaptiveState from_state;
    AdaptiveState to_state;
    double      residual_at_escalation;
    std::string reason;
};

// ---------------------------------------------------------------------------
// Adaptive selector / controller
// ---------------------------------------------------------------------------
class AdaptiveSelector {
public:
    explicit AdaptiveSelector(const SolverParams& params = {});

    // ------------------------------------------------------------------
    // Main entry point  —  solves A x = b adaptively
    // ------------------------------------------------------------------
    bool solve(const SparseMatrix& A,
               const Vector&       b,
                     Vector&       x,
               SolverStats&        stats);

    // ------------------------------------------------------------------
    // Feature extraction (basic — replaces GNN in prototype)
    // ------------------------------------------------------------------
    struct MatrixFeatures {
        // ── Core fields (original 8) ──────────────────────────────────────
        int    n;
        int    nnz;
        double density;
        double diag_dominance;       ///< min |a_ii| / sum_{j≠i} |a_ij|
        double symmetry_residual;
        double frobenius_norm;
        double estimated_cond;       ///< λ_max / λ_min estimate (Lanczos)
        bool   is_spd;

        // ── Extended fields for GNN (14 new — Thrust 1) ──────────────────
        double bandwidth;            ///< max |i-j| over non-zeros (stencil reach)
        double avg_nnz_per_row;      ///< = nnz / n
        double nnz_variance;         ///< variance of per-row nnz counts
        double diag_sign_fraction;   ///< fraction of rows where a_ii > 0
        double off_diag_symmetry;    ///< max|a_ij - a_ji| / frobenius_norm
        double spectral_radius_est;  ///< λ_max estimate from power iteration
        double lambda_min_est;       ///< λ_min estimate from Lanczos
        double gershgorin_radius;    ///< max row absolute sum (upper bound λ_max)
        double diagonal_fraction;    ///< nnz on diagonal / total nnz
        double lower_upper_ratio;    ///< nnz(L) / nnz(U) — asymmetry indicator
        int    n_zero_diag;          ///< number of zero/missing diagonal entries
        double fill_ratio_ilu0;      ///< estimated ILU(0) fill factor
        double coarsening_ratio;     ///< estimated AMG one-level coarsening ratio
        int    estimated_amg_depth;  ///< floor(log2(n / 50)) — levels needed

        // ── Derived scalar useful for routing (not raw matrix data) ──────
        double energy_proxy_easy;    ///< proxy energy if solved at EASY state
        double energy_proxy_hard;    ///< proxy energy if solved at HARD state

        // Serialise to a flat double array (for GNN feature tensor)
        std::vector<double> to_feature_vector() const;
        static int          feature_dim() { return 24; }  // length of above
    };
    MatrixFeatures extract_features(const SparseMatrix& A) const;

    // ------------------------------------------------------------------
    // Initial state selection (heuristic — later: GNN warm start)
    // ------------------------------------------------------------------
    AdaptiveState select_initial_state(const MatrixFeatures& f) const;

    // ------------------------------------------------------------------
    // Per-iteration callback: return true if escalation required
    // ------------------------------------------------------------------
    bool should_escalate(AdaptiveState current_state,
                         int           iteration,
                         Real          residual_now,
                         const std::vector<Real>& residual_history) const;

    // ------------------------------------------------------------------
    // Getters / reporting
    // ------------------------------------------------------------------
    const std::vector<EscalationRecord>& escalation_log() const { return esc_log_; }
    AdaptiveState current_state() const { return current_state_; }
    void print_summary(std::ostream& os) const;

    // ------------------------------------------------------------------
    // Energy-aware config
    // ------------------------------------------------------------------
    /// Install an ML/heuristic advisor.  When set, it replaces the
    /// built-in select_initial_state() heuristic and also supplies
    /// tuned SolverParams for the chosen state.
    void set_ml_advisor(std::shared_ptr<MLAdvisor> advisor) {
        ml_advisor_ = advisor;
    }

    void set_energy_monitor(std::shared_ptr<EnergyMonitor> mon) { energy_mon_ = mon; }

private:
    // ------------------------------------------------------------------
    // Build solver + preconditioner for a given state
    // ------------------------------------------------------------------
    std::pair<std::unique_ptr<SolverBase>,
              std::shared_ptr<PreconditionerBase>>
    build_for_state(AdaptiveState state,
                    const SparseMatrix& A,
                    double& setup_time_out);

    // ------------------------------------------------------------------
    // Execute one solve attempt with given solver/precond
    // ------------------------------------------------------------------
    bool attempt_solve(SolverBase&          solver,
                       PreconditionerBase&  precond,
                       const SparseMatrix&  A,
                       const Vector&        b,
                             Vector&        x,
                       AdaptiveState        state,
                       SolverStats&         stats);

    AdaptiveState next_state(AdaptiveState s) const;

    SolverParams params_;
    AdaptiveState current_state_ = AdaptiveState::EASY;
    std::vector<EscalationRecord> esc_log_;
    std::shared_ptr<EnergyMonitor> energy_mon_  = nullptr;
    std::shared_ptr<MLAdvisor>     ml_advisor_  = nullptr;
};

} // namespace hsps
