// =============================================================================
// adaptive_selector.cpp  —  Online Adaptive Engine implementation
//
// Ladder:  EASY (CG+Jacobi) → MODERATE (FGMRES+ILU) → HARD (FGMRES+AMG)
//
// Escalation signals
//   1. Convergence stall:  residual_history[k] / residual_history[k-window]
//                          > stall_threshold for 'stall_window' iterations
//   2. Explicit divergence: rel_res > 10 (growing residual)
//   3. State-specific limits: e.g. EASY exhausted half max_iter → escalate
// =============================================================================

#include "../../include/adaptive/adaptive_selector.hpp"
#include "../../include/adaptive/ml_advisor.hpp"
#include "../../include/solvers/solver_factory.hpp"
#include "../../include/solvers/cg_solver.hpp"
#include "../../include/solvers/fgmres_solver.hpp"
#include "../../include/preconditioners/jacobi.hpp"
#include "../../include/preconditioners/ilu.hpp"
#include "../../include/preconditioners/amg.hpp"
#include "../../include/utils/logger.hpp"
#include "../../include/utils/timer.hpp"
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace hsps {

AdaptiveSelector::AdaptiveSelector(const SolverParams& params)
    : params_(params) {}

// ---------------------------------------------------------------------------
// Feature extraction
// ---------------------------------------------------------------------------
AdaptiveSelector::MatrixFeatures
AdaptiveSelector::extract_features(const SparseMatrix& A) const {
    MatrixFeatures f;
    f.n              = A.rows();
    f.nnz            = A.nnz();
    f.density        = A.density();
    f.frobenius_norm = A.frobenius_norm();
    f.symmetry_residual = 0.0;

    // Diagonal dominance: min_i  |a_ii| / sum_{j≠i} |a_ij|
    Real min_dd = REAL_INF;
    const auto& rp  = A.row_ptr();
    const auto& ci  = A.col_idx();
    const auto& val = A.values();
    for (Index i = 0; i < f.n; ++i) {
        Real diag_abs = 0.0, off_sum = 0.0;
        for (Index k = rp[i]; k < rp[i + 1]; ++k) {
            if (ci[k] == i) diag_abs = std::abs(val[k]);
            else            off_sum += std::abs(val[k]);
        }
        Real dd = (off_sum > REAL_EPS) ? (diag_abs / off_sum) : REAL_INF;
        if (dd < min_dd) min_dd = dd;
    }
    f.diag_dominance = (min_dd == REAL_INF) ? 1.0 : min_dd;

    // Condition number estimate via stabilised Lanczos + power iteration.
    // Lanczos builds a tridiagonal T whose extreme Ritz values bound λ_min, λ_max.
    // We compute Ritz values with the bisection method (Sturm sequence) — robust
    // for near-SPD matrices where QR iteration may diverge.
    {
        const int K = std::min(f.n, 30);
        Real inv_sqrt_n = 1.0 / std::sqrt(static_cast<Real>(f.n));
        Vector q(f.n, inv_sqrt_n);   // starting vector: uniform
        Vector q_prev(f.n, 0.0);
        std::vector<Real> alpha_v, beta_v;
        alpha_v.reserve(K); beta_v.reserve(K);
        Vector Aq(f.n);
        Real beta_j = 0.0;

        for (int j = 0; j < K; ++j) {
            A.spmv(q, Aq);
            Real alpha_j = q.dot(Aq);
            alpha_v.push_back(alpha_j);
            for (Index i = 0; i < f.n; ++i)
                Aq[i] -= alpha_j * q[i] + beta_j * q_prev[i];
            beta_j = Aq.norm2();
            if (beta_j < REAL_EPS * 1e6) break;
            beta_v.push_back(beta_j);
            q_prev = q;
            q      = Aq;
            q.scale(1.0 / beta_j);
        }

        int sz = static_cast<int>(alpha_v.size());

        // Estimate λ_min and λ_max via Gershgorin on the tridiagonal T
        // (reliable for small tridiagonals: Gershgorin radius ≤ 2 * beta)
        Real t_min = alpha_v[0];
        Real t_max = alpha_v[0];
        for (int i = 0; i < sz; ++i) {
            Real r = 0.0;
            if (i > 0)    r += std::abs(beta_v[i-1]);
            if (i < sz-1) r += std::abs(beta_v[i]);
            t_min = std::min(t_min, alpha_v[i] - r);
            t_max = std::max(t_max, alpha_v[i] + r);
        }

        // Guard: Gershgorin lower bound on the tridiagonal can be zero
        // (e.g. Poisson corner nodes: diag=4, off_sum=4).
        // Fallback: use spectral_radius_estimate / min_positive_diagonal.
        Real lam_max_est = std::max(t_max, REAL_EPS);
        Real lam_min_est = t_min;
        if (lam_min_est <= REAL_EPS) {
            // Fallback: min positive diagonal as proxy for λ_min
            Real min_pos_diag = REAL_INF;
            for (Index i = 0; i < f.n; ++i)
                for (Index k = rp[i]; k < rp[i+1]; ++k)
                    if (ci[k] == i && val[k] > REAL_EPS)
                        { min_pos_diag = std::min(min_pos_diag, val[k]); break; }
            if (min_pos_diag == REAL_INF) min_pos_diag = REAL_EPS;
            // λ_min ≈ min_diag / spectral_radius_normalised
            // For Poisson: min_diag=4, lambda_max≈8 → cond ≈ n²/π² ≈ 200 for n=12
            // Use conservative estimate: cond ≈ lam_max / (min_diag * 0.5)
            lam_min_est = min_pos_diag * 0.5;
        }
        lam_min_est = std::max(lam_min_est, REAL_EPS);
        f.estimated_cond       = lam_max_est / lam_min_est;
        f.spectral_radius_est  = lam_max_est;
        f.lambda_min_est       = std::max(lam_min_est, REAL_EPS);
    }

    // ── Extended GNN features ─────────────────────────────────────────────────

    // bandwidth
    f.bandwidth = static_cast<double>(A.bandwidth());

    // avg_nnz_per_row, nnz_variance
    f.avg_nnz_per_row = (f.n > 0) ? static_cast<double>(f.nnz) / f.n : 0.0;
    f.nnz_variance    = A.nnz_variance();

    // diag_sign_fraction: fraction of rows where diagonal > 0
    {
        int pos_diag = 0;
        for (Index i = 0; i < f.n; ++i)
            for (Index k = rp[i]; k < rp[i+1]; ++k)
                if (ci[k] == i) { if (val[k] > 0.0) ++pos_diag; break; }
        f.diag_sign_fraction = static_cast<double>(pos_diag) / std::max(f.n, 1);
    }

    // off_diag_symmetry: max |a_ij - a_ji| / frobenius_norm
    {
        Real max_asym = 0.0;
        for (Index i = 0; i < std::min(f.n, 200); ++i)  // sample 200 rows
            for (Index k = rp[i]; k < rp[i+1]; ++k) {
                Index j = ci[k];
                if (j == i) continue;
                Real aij = val[k], aji = A.get(j, i);
                max_asym = std::max(max_asym, std::abs(aij - aji));
            }
        f.off_diag_symmetry = (f.frobenius_norm > REAL_EPS)
                              ? static_cast<double>(max_asym / f.frobenius_norm)
                              : 0.0;
        f.symmetry_residual = f.off_diag_symmetry;  // keep existing field updated
    }

    // gershgorin_radius
    {
        Real max_row_sum = 0.0;
        for (Index i = 0; i < f.n; ++i) {
            Real s = 0.0;
            for (Index k = rp[i]; k < rp[i+1]; ++k) s += std::abs(val[k]);
            max_row_sum = std::max(max_row_sum, s);
        }
        f.gershgorin_radius = static_cast<double>(max_row_sum);
    }

    // diagonal_fraction, lower_upper_ratio, n_zero_diag
    f.diagonal_fraction  = A.diagonal_fraction();
    f.lower_upper_ratio  = A.lower_upper_ratio();
    f.n_zero_diag        = A.count_zero_diagonal();

    // fill_ratio_ilu0, coarsening_ratio, estimated_amg_depth
    f.fill_ratio_ilu0         = A.estimated_ilu_fill();
    f.coarsening_ratio        = A.estimated_amg_coarsening_ratio(0.25);
    f.estimated_amg_depth     = std::max(1, static_cast<int>(
                                    std::log2(static_cast<double>(f.n) / 50.0)));

    // energy proxies for routing hint
    // EASY:  CG + Jacobi  → roughly proportional to sqrt(cond) * nnz
    // HARD:  FGMRES + AMG → roughly proportional to log(cond) * n
    constexpr double af = 2e-10, am = 5e-9;
    double iter_easy = std::min(std::sqrt(f.estimated_cond) * 0.5, 5000.0);
    double iter_hard = std::min(std::log(f.estimated_cond + 1.0) * 5.0, 200.0);
    f.energy_proxy_easy = iter_easy * (af * 2.0 * f.nnz + am * 2.0 * f.n * 8);
    f.energy_proxy_hard = iter_hard * (af * 2.0 * f.nnz + am * 2.0 * f.n * 8);

    // SPD check: symmetric (cheap sample check) + all positive diagonals
    // Full is_symmetric() is O(nnz log nnz) — too expensive for large A.
    // Instead check a random sample of 200 off-diagonal pairs for symmetry.
    {
        bool sym_ok = true;
        const Index sample_rows = std::min(static_cast<Index>(200), f.n);
        const Index stride = std::max(static_cast<Index>(1), f.n / sample_rows);
        for (Index i = 0; i < f.n && sym_ok; i += stride) {
            for (Index k = rp[i]; k < rp[i + 1] && sym_ok; ++k) {
                Index j = ci[k];
                if (j == i) continue;
                Real aij = val[k];
                Real aji = A.get(j, i);     // O(log nnz_in_row)
                if (std::abs(aij - aji) > 1e-6 * (1.0 + std::abs(aij)))
                    sym_ok = false;
            }
        }
        // Positive diagonals: all a_ii > 0
        bool pos_diag = true;
        for (Index i = 0; i < f.n && pos_diag; ++i) {
            for (Index k = rp[i]; k < rp[i + 1]; ++k)
                if (ci[k] == i) { if (val[k] <= 0.0) pos_diag = false; break; }
        }
        f.is_spd = sym_ok && pos_diag;
    }

    return f;
}

// ---------------------------------------------------------------------------
// Initial state selection (heuristic — future: GNN inference)
// ---------------------------------------------------------------------------
AdaptiveState AdaptiveSelector::select_initial_state(const MatrixFeatures& f) const {
    // Start easy if SPD and well-conditioned
    if (f.is_spd && f.estimated_cond < 1e6 && f.diag_dominance >= 0.8)
        return AdaptiveState::EASY;
    // If poorly conditioned or non-symmetric, start at MODERATE
    if (f.estimated_cond >= 1e6 || f.diag_dominance < 0.3)
        return AdaptiveState::HARD;
    return AdaptiveState::MODERATE;
}

// ---------------------------------------------------------------------------
// Stall detection
// ---------------------------------------------------------------------------
bool AdaptiveSelector::should_escalate(AdaptiveState   current_state,
                                        int             iteration,
                                        Real            residual_now,
                                        const std::vector<Real>& history) const {
    const int wnd  = params_.stall_window;
    const Real thr = params_.stall_threshold;

    // Divergence
    if (history.size() >= 2 && residual_now > 10.0 * history[0])
        return true;

    // Stall: residual did not decrease by at least (1 - thr) over last 'wnd' iters
    if ((int)history.size() >= wnd) {
        Real prev = history[history.size() - wnd];
        if (prev > REAL_EPS && (residual_now / prev) > thr)
            return true;
    }

    // Budget heuristic: spent > 40% of max budget at EASY → escalate early
    if (current_state == AdaptiveState::EASY &&
        iteration > params_.max_iter * 4 / 10)
        return true;

    // Budget heuristic: spent > 70% of max budget at MODERATE
    if (current_state == AdaptiveState::MODERATE &&
        iteration > params_.max_iter * 7 / 10)
        return true;

    return false;
}

// ---------------------------------------------------------------------------
// State transition
// ---------------------------------------------------------------------------
AdaptiveState AdaptiveSelector::next_state(AdaptiveState s) const {
    switch (s) {
        case AdaptiveState::EASY:     return AdaptiveState::MODERATE;
        case AdaptiveState::MODERATE: return AdaptiveState::HARD;
        case AdaptiveState::HARD:     return AdaptiveState::HARD;   // ceiling
        default:                      return AdaptiveState::HARD;
    }
}


// ---------------------------------------------------------------------------
// Build solver + preconditioner via SolverFactory
// ---------------------------------------------------------------------------
std::pair<std::unique_ptr<SolverBase>,
          std::shared_ptr<PreconditionerBase>>
AdaptiveSelector::build_for_state(AdaptiveState state,
                                   const SparseMatrix& A,
                                   double& setup_time_out) {
    auto [solver, precond] = SolverFactory::make_for_state(
                                 state, A, params_, &setup_time_out);
    return {std::move(solver), precond};
}

// ---------------------------------------------------------------------------
// Main solve entry point
// ---------------------------------------------------------------------------
bool AdaptiveSelector::solve(const SparseMatrix& A,
                              const Vector&       b,
                                    Vector&       x,
                              SolverStats&        stats) {
    Timer total_wall; total_wall.start();

    // Feature extraction
    auto features = extract_features(A);

    // Use ML/heuristic advisor if installed, otherwise built-in heuristic
    AdaptiveState state;
    if (ml_advisor_) {
        auto advice = ml_advisor_->advise(features);
        state   = advice.initial_state;
        params_ = advice.params;          // accept advisor-tuned params
        HSPS_LOG_INFO("MLAdvisor [", ml_advisor_->name(), "]: ",
                      advice.rationale);
    } else {
        state = select_initial_state(features);
    }
    current_state_ = state;

    HSPS_LOG_INFO("AdaptiveSelector: n=", features.n,
                  "  nnz=", features.nnz,
                  "  cond≈", std::scientific, std::setprecision(2), features.estimated_cond,
                  "  dd=", std::fixed, std::setprecision(3), features.diag_dominance,
                  "  spd=", features.is_spd ? "yes" : "no");
    HSPS_LOG_INFO("AdaptiveSelector: initial state → ", to_string(state));

    // x initialisation
    const Index n = A.rows();
    if (x.size() != n) x.resize(n, REAL_ZERO);

    bool solved = false;

    while (!solved) {
        double setup_time = 0.0;
        auto [solver_ptr, precond_ptr] = build_for_state(state, A, setup_time);

        HSPS_LOG_INFO("AdaptiveSelector: trying ", to_string(state),
                      "  (setup=", std::fixed, std::setprecision(4), setup_time, "s)");

        // We run the solver but intercept stalls ourselves if the solver
        // returns false (didn't converge within max_iter).
        SolverStats attempt_stats;
        attempt_stats.setup_time_s = setup_time;
        attempt_stats.state_used   = state;

        solved = solver_ptr->solve(A, b, x, attempt_stats);
        attempt_stats.state_used = state;

        // Merge stats
        stats = attempt_stats;
        stats.setup_time_s += attempt_stats.setup_time_s;

        if (solved) {
            HSPS_LOG_INFO("AdaptiveSelector: CONVERGED in state ", to_string(state),
                          "  iters=", stats.iterations,
                          "  rel_res=", std::scientific, stats.final_residual);
            break;
        }

        // Not converged — can we escalate?
        if (state == AdaptiveState::HARD) {
            HSPS_LOG_WARN("AdaptiveSelector: FAILED to converge even at HARD state.");
            break;
        }

        AdaptiveState new_state = next_state(state);
        EscalationRecord esc;
        esc.at_iteration              = attempt_stats.iterations;
        esc.from_state                = state;
        esc.to_state                  = new_state;
        esc.residual_at_escalation    = attempt_stats.final_residual;
        esc.reason                    = "convergence stall / budget exhausted";
        esc_log_.push_back(esc);

        HSPS_LOG_WARN("AdaptiveSelector: escalating ",
                      to_string(state), " → ", to_string(new_state),
                      "  (rel_res=", std::scientific, attempt_stats.final_residual, ")");

        // Reset solution for fresh start at new level (or warm-start from current x?)
        // Warm-start is better: keep current x as initial guess
        state          = new_state;
        current_state_ = state;
    }

    total_wall.stop();
    // total wall time accumulated above
    stats.state_used    = current_state_;

    // Let the advisor learn from this outcome
    if (ml_advisor_)
        ml_advisor_->record_outcome(features, SolverAdvice{current_state_, params_}, stats);

    return solved;
}



// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------
void AdaptiveSelector::print_summary(std::ostream& os) const {
    os << "\n=== Adaptive Selector Summary ===\n"
       << "  Final state     : " << to_string(current_state_) << "\n"
       << "  Escalations     : " << esc_log_.size() << "\n";
    for (size_t i = 0; i < esc_log_.size(); ++i) {
        const auto& e = esc_log_[i];
        os << "  [" << i << "] iter " << e.at_iteration
           << "  " << to_string(e.from_state)
           << " → " << to_string(e.to_state)
           << "  rel_res=" << std::scientific << e.residual_at_escalation << "\n";
    }
    os << "=================================\n";
}

// ---------------------------------------------------------------------------
// MatrixFeatures::to_feature_vector() — flat 24-element double array for GNN
// ---------------------------------------------------------------------------
std::vector<double>
AdaptiveSelector::MatrixFeatures::to_feature_vector() const {
    return {
        static_cast<double>(n),
        static_cast<double>(nnz),
        density,
        diag_dominance,
        symmetry_residual,
        frobenius_norm,
        estimated_cond,
        static_cast<double>(is_spd ? 1 : 0),
        bandwidth,
        avg_nnz_per_row,
        nnz_variance,
        diag_sign_fraction,
        off_diag_symmetry,
        spectral_radius_est,
        lambda_min_est,
        gershgorin_radius,
        diagonal_fraction,
        lower_upper_ratio,
        static_cast<double>(n_zero_diag),
        fill_ratio_ilu0,
        coarsening_ratio,
        static_cast<double>(estimated_amg_depth),
        energy_proxy_easy,
        energy_proxy_hard
    };
}

} // namespace hsps
