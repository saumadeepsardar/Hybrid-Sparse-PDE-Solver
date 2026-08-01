// =============================================================================
// gnn_advisor.cpp  —  GNNAdvisor and DatasetCollector implementations
//
// Compile without libtorch:   #ifndef HSPS_USE_TORCH → FeatureAdvisor fallback
// Compile with libtorch:      make BACKEND=OMP TORCH=1
// =============================================================================

#include "../../include/adaptive/gnn_advisor.hpp"
#include "../../include/utils/logger.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

#ifdef HSPS_USE_TORCH
#  include <torch/script.h>
#endif

namespace hsps {

// ===========================================================================
// DatasetCollector
// ===========================================================================

DatasetCollector::DatasetCollector(const std::string& output_path)
    : out_(output_path, std::ios::app) {
    if (!out_.good())
        std::cerr << "[DatasetCollector] Cannot open: " << output_path << "\n";
}

DatasetCollector::~DatasetCollector() { flush(); }

void DatasetCollector::flush() {
    if (out_.is_open()) out_.flush();
}

void DatasetCollector::record(
        const AdaptiveSelector::MatrixFeatures& f,
        const SolverAdvice&                     advice,
        const SolverStats&                      stats,
        const std::string& matrix_name) {
    if (!out_.good()) return;

    // Serialise features as JSON array
    auto fvec = f.to_feature_vector();
    out_ << "{";
    if (!matrix_name.empty())
        out_ << "\"name\":\"" << matrix_name << "\",";

    out_ << "\"features\":[";
    for (size_t i = 0; i < fvec.size(); ++i) {
        if (std::isfinite(fvec[i]))
            out_ << std::scientific << std::setprecision(6) << fvec[i];
        else
            out_ << "0.0";
        if (i + 1 < fvec.size()) out_ << ",";
    }
    out_ << "],"
         << "\"label_state\":"   << static_cast<int>(advice.initial_state) << ","
         << "\"label_restart\":" << advice.params.restart_size << ","
         << "\"label_drop_tol\":" << std::scientific << advice.params.drop_tol << ","
         << "\"label_amg_str\":" << advice.params.amg_strong << ","
         << "\"iters\":"         << stats.iterations << ","
         << "\"converged\":"     << (stats.converged ? "true" : "false") << ","
         << "\"energy_j\":"      << stats.energy_joules << ","
         << "\"energy_hw_j\":"   << stats.energy_joules_hw << ","
         << "\"time_s\":"        << stats.solve_time_s
         << "}\n";

    ++n_records_;
    if (n_records_ % 100 == 0) flush();
}

void DatasetCollector::merge(const std::vector<std::string>& inputs,
                              const std::string& output) {
    std::ofstream out(output);
    for (const auto& path : inputs) {
        std::ifstream in(path);
        if (!in.good()) continue;
        out << in.rdbuf();
    }
}

// ===========================================================================
// GNNAdvisor — TorchScript module wrapper
// ===========================================================================

#ifdef HSPS_USE_TORCH
struct GNNAdvisor::TorchImpl {
    torch::jit::script::Module module;
    bool loaded = false;
};
#else
struct GNNAdvisor::TorchImpl {};  // empty stub
#endif

GNNAdvisor::GNNAdvisor(const std::string& model_path,
                         const SolverParams& base_params)
    : base_(base_params), model_path_(model_path),
      fallback_(base_params),
      torch_impl_(std::make_unique<TorchImpl>()) {

#ifdef HSPS_USE_TORCH
    try {
        torch_impl_->module = torch::jit::load(model_path);
        torch_impl_->module.eval();
        torch_impl_->loaded = true;
        model_loaded_ = true;
        HSPS_LOG_INFO("GNNAdvisor: loaded model from ", model_path);
    } catch (const c10::Error& e) {
        std::cerr << "[GNNAdvisor] torch::jit::load failed: " << e.what()
                  << "\nFalling back to FeatureAdvisor.\n";
        model_loaded_ = false;
    } catch (const std::exception& e) {
        std::cerr << "[GNNAdvisor] load error: " << e.what()
                  << "\nFalling back to FeatureAdvisor.\n";
        model_loaded_ = false;
    }
#else
    (void)model_path;
    std::cerr << "[GNNAdvisor] HSPS_USE_TORCH not defined. "
                 "Compile with 'make TORCH=1' to enable GPU inference.\n"
                 "Falling back to FeatureAdvisor.\n";
    model_loaded_ = false;
#endif
}

GNNAdvisor::~GNNAdvisor() = default;

void GNNAdvisor::enable_collection(const std::string& output_path) {
    collector_ = std::make_unique<DatasetCollector>(output_path);
}

SolverAdvice GNNAdvisor::decode_output(const std::vector<float>& state_probs,
                                         float restart_raw,
                                         float log_drop_tol,
                                         float amg_strength_raw) const {
    SolverAdvice adv;
    adv.params = base_;

    // State: argmax of softmax probabilities
    int state_idx = static_cast<int>(
        std::max_element(state_probs.begin(), state_probs.end())
        - state_probs.begin());
    adv.initial_state = static_cast<AdaptiveState>(
        std::clamp(state_idx, 0, 2));
    last_confidence_ = state_probs[state_idx];

    // Restart size: decode from network output (range [10, 300])
    adv.params.restart_size = std::clamp(
        static_cast<int>(restart_raw), 10, 300);

    // Drop tolerance: decode from log-space (range [1e-6, 1e-2])
    double dt = std::exp(static_cast<double>(log_drop_tol));
    adv.params.drop_tol = std::clamp(dt, 1e-6, 1e-2);

    // AMG strength threshold (range [0.05, 0.75])
    adv.params.amg_strong = std::clamp(
        static_cast<double>(amg_strength_raw), 0.05, 0.75);

    std::ostringstream oss;
    oss << "GNN→" << to_string(adv.initial_state)
        << " conf=" << std::fixed << std::setprecision(2) << last_confidence_
        << " restart=" << adv.params.restart_size
        << " drop_tol=" << std::scientific << adv.params.drop_tol;
    adv.rationale = oss.str();
    return adv;
}

SolverAdvice GNNAdvisor::advise(
        const AdaptiveSelector::MatrixFeatures& f) const {

    if (!model_loaded_) return fallback_.advise(f);

#ifdef HSPS_USE_TORCH
    try {
        // Build input tensor from 24-element feature vector
        auto fvec = f.to_feature_vector();
        std::vector<float> fvec_f(fvec.begin(), fvec.end());

        // Replace NaN/Inf with 0 (defensive)
        for (auto& v : fvec_f)
            if (!std::isfinite(v)) v = 0.0f;

        auto input = torch::tensor(fvec_f).unsqueeze(0);  // shape [1, 24]

        torch::NoGradGuard no_grad;
        auto output = torch_impl_->module.forward({input}).toTuple();

        // Decode output tuple (state_logits, restart, log_drop_tol, amg_strength)
        auto state_tensor   = output->elements()[0].toTensor().softmax(1)[0];
        float restart       = output->elements()[1].toTensor().item<float>();
        float log_drop_tol  = output->elements()[2].toTensor().item<float>();
        float amg_strength  = output->elements()[3].toTensor().item<float>();

        std::vector<float> probs(3);
        for (int i = 0; i < 3; ++i)
            probs[i] = state_tensor[i].item<float>();

        return decode_output(probs, restart, log_drop_tol, amg_strength);

    } catch (const std::exception& e) {
        HSPS_LOG_WARN("GNNAdvisor: inference failed (", e.what(),
                      "), falling back to FeatureAdvisor");
        return fallback_.advise(f);
    }
#else
    return fallback_.advise(f);
#endif
}

void GNNAdvisor::record_outcome(
        const AdaptiveSelector::MatrixFeatures& f,
        const SolverAdvice& advice,
        const SolverStats& stats) {
    if (collector_) collector_->record(f, advice, stats);
}

} // namespace hsps
