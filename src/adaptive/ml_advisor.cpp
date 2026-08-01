// =============================================================================
// ml_advisor.cpp  —  Concrete advisor implementations
// =============================================================================

#include "../../include/adaptive/ml_advisor.hpp"
#include <algorithm>
#include <map>
#include <sstream>
#include <iomanip>
#include <iostream>

namespace hsps {

// ===========================================================================
// HeuristicAdvisor — original 3-branch rule
// ===========================================================================
SolverAdvice HeuristicAdvisor::advise(
        const AdaptiveSelector::MatrixFeatures& f) const {
    SolverAdvice adv;
    adv.params = base_;

    if (f.is_spd && f.estimated_cond < 1e6 && f.diag_dominance >= 0.8) {
        adv.initial_state = AdaptiveState::EASY;
        adv.rationale     = "SPD + well-conditioned → CG+Jacobi";
    } else if (f.estimated_cond >= 1e6 || f.diag_dominance < 0.3) {
        adv.initial_state = AdaptiveState::HARD;
        adv.rationale     = "High cond or poor diag-dominance → FGMRES+AMG";
    } else {
        adv.initial_state = AdaptiveState::MODERATE;
        adv.rationale     = "Moderate → FGMRES+ILU";
    }
    return adv;
}

// ===========================================================================
// FeatureAdvisor — richer decision tree
// ===========================================================================
SolverAdvice FeatureAdvisor::advise(
        const AdaptiveSelector::MatrixFeatures& f) const {
    SolverAdvice adv;
    adv.params = base_;

    // ── Symmetry check ──────────────────────────────────────────────────────
    const bool is_sym = (f.symmetry_residual < symm_thresh_);

    // ── Condition-based tier ─────────────────────────────────────────────────
    AdaptiveState tier;
    if (f.estimated_cond < cond_moderate_thresh_ &&
        f.diag_dominance >= dd_good_thresh_ && is_sym && f.is_spd) {
        tier = AdaptiveState::EASY;
    } else if (f.estimated_cond >= cond_hard_thresh_ ||
               f.diag_dominance <  dd_bad_thresh_    ||
               !is_sym) {
        tier = AdaptiveState::HARD;
    } else {
        tier = AdaptiveState::MODERATE;
    }

    // ── Parameter tuning based on problem size ────────────────────────────
    // Restart size: larger for harder problems (more search directions needed)
    if (tier == AdaptiveState::HARD)
        adv.params.restart_size = std::min(adv.params.restart_size * 2, f.n);
    else if (tier == AdaptiveState::MODERATE)
        adv.params.restart_size = std::min(
            static_cast<int>(adv.params.restart_size * 1.5), f.n);

    // ILU drop tolerance: tighter for ill-conditioned systems
    if (f.estimated_cond > 1e5)
        adv.params.drop_tol = 1e-5;
    else if (f.estimated_cond > 1e3)
        adv.params.drop_tol = 1e-4;
    else
        adv.params.drop_tol = 1e-3;

    // AMG strength threshold: looser for nearly-isotropic problems
    if (f.diag_dominance > 0.9)
        adv.params.amg_strong = 0.5;
    else if (f.diag_dominance > 0.5)
        adv.params.amg_strong = 0.25;
    else
        adv.params.amg_strong = 0.1;

    // AMG levels: deeper hierarchy for larger problems
    adv.params.amg_levels = std::max(5, static_cast<int>(
            std::log2(static_cast<double>(f.n) / 50.0)) + 2);

    adv.initial_state = tier;

    // ── Build rationale string ────────────────────────────────────────────
    std::ostringstream oss;
    oss << to_string(tier)
        << "  cond=" << std::scientific << std::setprecision(2) << f.estimated_cond
        << "  dd="   << std::fixed      << std::setprecision(3) << f.diag_dominance
        << "  sym="  << (is_sym ? "yes" : "no")
        << "  spd="  << (f.is_spd ? "yes" : "no")
        << "  restart=" << adv.params.restart_size
        << "  drop_tol=" << std::scientific << adv.params.drop_tol;
    adv.rationale = oss.str();

    return adv;
}

// ===========================================================================
// LoggingAdvisor
// ===========================================================================
LoggingAdvisor::LoggingAdvisor(std::shared_ptr<MLAdvisor> inner,
                                const std::string& csv_path)
    : inner_(std::move(inner)), csv_(csv_path) {
    if (!csv_.good())
        std::cerr << "[LoggingAdvisor] Cannot open " << csv_path << "\n";
}

LoggingAdvisor::~LoggingAdvisor() {
    if (csv_.is_open()) csv_.close();
}

SolverAdvice LoggingAdvisor::advise(
        const AdaptiveSelector::MatrixFeatures& f) const {
    auto adv = inner_->advise(f);

    // Write CSV header on first call
    if (!header_written_) {
        csv_ << "n,nnz,density,diag_dominance,symmetry_residual,"
                "estimated_cond,is_spd,initial_state\n";
        const_cast<LoggingAdvisor*>(this)->header_written_ = true;
    }
    csv_ << f.n << "," << f.nnz << ","
         << std::scientific << std::setprecision(6)
         << f.density << "," << f.diag_dominance << ","
         << f.symmetry_residual << "," << f.estimated_cond << ","
         << (f.is_spd ? 1 : 0) << ","
         << static_cast<int>(adv.initial_state) << "\n";
    csv_.flush();
    return adv;
}

void LoggingAdvisor::record_outcome(
        const AdaptiveSelector::MatrixFeatures& /*f*/,
        const SolverAdvice&                     adv,
        const SolverStats&                      s) {
    // Append outcome columns to the last written row (append-mode CSV).
    // In a production system, match by a row-id instead.
    csv_ << "# outcome: state=" << to_string(adv.initial_state)
         << "  iters="       << s.iterations
         << "  converged="   << s.converged
         << "  energy_J="    << std::scientific << std::setprecision(4)
         << s.energy_joules  << "\n";
    csv_.flush();
}

// ===========================================================================
// EnsembleAdvisor — majority vote over initial_state
// ===========================================================================
SolverAdvice EnsembleAdvisor::advise(
        const AdaptiveSelector::MatrixFeatures& f) const {
    if (members_.empty())
        return SolverAdvice{};   // defaults

    std::map<AdaptiveState, int> votes;
    SolverAdvice last;
    for (const auto& m : members_) {
        last  = m->advise(f);
        votes[last.initial_state]++;
    }

    // Pick majority state
    AdaptiveState best = votes.begin()->first;
    int best_count = 0;
    for (auto& [st, cnt] : votes)
        if (cnt > best_count) { best = st; best_count = cnt; }

    last.initial_state = best;
    last.rationale     = std::string("Ensemble vote (") +
                         std::to_string(best_count) + "/" +
                         std::to_string(static_cast<int>(members_.size())) + ")";
    return last;
}

} // namespace hsps
