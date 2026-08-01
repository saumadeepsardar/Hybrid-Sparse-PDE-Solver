#pragma once

// =============================================================================
// ml_advisor.hpp  —  GNN warm-start hook for the Adaptive Selector
//
// Architecture
// ------------
// MLAdvisor is a pure interface.  The AdaptiveSelector calls it at the start
// of every solve instead of (or alongside) its heuristic rules.
//
// Provided concrete implementations
// ----------------------------------
//   HeuristicAdvisor   — current rule-based logic, extracted here so it can
//                        be compared against future ML models.
//
//   FeatureAdvisor     — richer feature-based heuristic using matrix statistics
//                        not available in the simple 3-branch rule.
//
// Future: GNNAdvisor   — loads a TorchScript / ONNX model.
//   Node features  : [row_nnz_norm, diag_dominance_i, spd_flag, row_norm_i]
//   Edge features  : [abs(a_ij) / row_norm_i, sign(a_ij)]
//   Message-passing : 3 GraphSAGE layers, mean aggregator
//   Output head    : softmax over {EASY, MODERATE, HARD}
//                  + regression for {restart_size, drop_tol, amg_strength}
//   Training signal: (features, best_config) pairs from energy_benchmark runs
//                    with loss = cross_entropy + lambda * energy_per_solve
//
// Data collection
// ---------------
// MLAdvisor::record_outcome() is called after every solve so that the
// (features, chosen_config, achieved_energy) triple can be written to a CSV
// for offline training.
// =============================================================================

#include "adaptive_selector.hpp"   // for MatrixFeatures
#include "../core/types.hpp"
#include <memory>
#include <string>
#include <vector>
#include <fstream>

namespace hsps {

// ===========================================================================
// Advice struct: full solver configuration chosen by the advisor
// ===========================================================================
struct SolverAdvice {
    AdaptiveState  initial_state   = AdaptiveState::EASY;
    SolverParams   params;             ///< Complete parameter block to use
    std::string    rationale;          ///< Human-readable explanation
};

// ===========================================================================
// Abstract ML / heuristic advisor interface
// ===========================================================================
class MLAdvisor {
public:
    virtual ~MLAdvisor() = default;

    /// Given matrix features → return recommended solver configuration.
    virtual SolverAdvice advise(
            const AdaptiveSelector::MatrixFeatures& features) const = 0;

    /// Called after a solve completes so the advisor can log/learn.
    virtual void record_outcome(
            const AdaptiveSelector::MatrixFeatures& features,
            const SolverAdvice&                     advice,
            const SolverStats&                      stats) {
        (void)features; (void)advice; (void)stats;  // default: no-op
    }

    virtual const char* name() const = 0;
};

// ===========================================================================
// HeuristicAdvisor — the original 3-branch rule extracted as an MLAdvisor
// ===========================================================================
class HeuristicAdvisor : public MLAdvisor {
public:
    explicit HeuristicAdvisor(const SolverParams& base_params = {})
        : base_(base_params) {}

    SolverAdvice advise(
            const AdaptiveSelector::MatrixFeatures& f) const override;

    const char* name() const override { return "HeuristicAdvisor"; }

private:
    SolverParams base_;
};

// ===========================================================================
// FeatureAdvisor — richer decision tree using all extracted features
// ===========================================================================
class FeatureAdvisor : public MLAdvisor {
public:
    explicit FeatureAdvisor(const SolverParams& base_params = {})
        : base_(base_params) {}

    SolverAdvice advise(
            const AdaptiveSelector::MatrixFeatures& f) const override;

    const char* name() const override { return "FeatureAdvisor"; }

private:
    SolverParams base_;

    // Threshold knobs (could be learnt by a linear model)
    Real cond_moderate_thresh_  = 1e4;
    Real cond_hard_thresh_      = 1e7;
    Real dd_good_thresh_        = 0.8;
    Real dd_bad_thresh_         = 0.3;
    Real symm_thresh_           = 1e-4;
};

// ===========================================================================
// LoggingAdvisor — wraps another advisor and records outcomes to CSV
//
// CSV columns:
//   n, nnz, density, diag_dominance, symmetry_residual, estimated_cond,
//   is_spd, initial_state, iterations, converged, energy_joules
//
// This dataset feeds offline training of the GNN model.
// ===========================================================================
class LoggingAdvisor : public MLAdvisor {
public:
    LoggingAdvisor(std::shared_ptr<MLAdvisor> inner,
                   const std::string& csv_path);
    ~LoggingAdvisor();

    SolverAdvice advise(
            const AdaptiveSelector::MatrixFeatures& f) const override;

    void record_outcome(
            const AdaptiveSelector::MatrixFeatures& f,
            const SolverAdvice&                     advice,
            const SolverStats&                      stats) override;

    const char* name() const override { return inner_->name(); }

private:
    std::shared_ptr<MLAdvisor> inner_;
    mutable std::ofstream      csv_;
    bool                       header_written_ = false;
};

// ===========================================================================
// EnsembleAdvisor — majority-vote over a list of advisors
//   Useful during A/B testing of different heuristics or partial GNN models.
// ===========================================================================
class EnsembleAdvisor : public MLAdvisor {
public:
    void add(std::shared_ptr<MLAdvisor> a) { members_.push_back(a); }

    SolverAdvice advise(
            const AdaptiveSelector::MatrixFeatures& f) const override;

    const char* name() const override { return "EnsembleAdvisor"; }

private:
    std::vector<std::shared_ptr<MLAdvisor>> members_;
};

} // namespace hsps
