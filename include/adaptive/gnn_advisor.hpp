#pragma once

// =============================================================================
// gnn_advisor.hpp  —  Graph Neural Network configuration advisor
//
// Thrust 1: replaces the heuristic FeatureAdvisor with a trained GNN model.
//
// Architecture (Python training side)
// ------------------------------------
//   Input:  24-element feature vector (MatrixFeatures::to_feature_vector())
//   Hidden: 3 × GraphSAGE layers (hidden_dim=128)  [for full graph version]
//           OR 3 × MLP layers on feature vector     [feature-only version]
//   Output heads:
//     • softmax(3)  → P(state ∈ {EASY, MODERATE, HARD})
//     • regression  → restart_size ∈ [10, 300]
//     • regression  → log(drop_tol) ∈ [-6, -2]
//     • regression  → amg_strength ∈ [0.05, 0.75]
//
// C++ inference
// -------------
//   The trained model is exported from PyTorch as TorchScript (.pt file) and
//   loaded at runtime via torch::jit::load(). This header is only compiled
//   when HSPS_USE_TORCH is defined.
//
//   When HSPS_USE_TORCH is NOT defined, GNNAdvisor gracefully falls back to
//   FeatureAdvisor so the rest of the codebase compiles without libtorch.
//
// Training pipeline (see scripts/train_gnn.py)
// ----------------------------------------------
//   1. bin/run_dataset --output-jsonl data/training.jsonl --repeat 5
//        writes (features, config, outcome) triples
//   2. python3 scripts/train_gnn.py --input data/training.jsonl \
//        --output models/gnn_v1.pt
//        trains the model and exports TorchScript
//   3. GNNAdvisor gnn("models/gnn_v1.pt");
//      sel.set_ml_advisor(std::make_shared<GNNAdvisor>(gnn));
//
// Model file format
// -----------------
//   The .pt file is a TorchScript module with method signature:
//     forward(features: Tensor[24]) -> Tuple[Tensor[3], Tensor[1], Tensor[1], Tensor[1]]
//   where outputs are: (state_logits, restart, log_drop_tol, amg_strength)
// =============================================================================

#include "adaptive_selector.hpp"
#include "ml_advisor.hpp"
#include "../core/types.hpp"
#include <string>
#include <memory>
#include <vector>

namespace hsps {

// ---------------------------------------------------------------------------
// DatasetCollector  —  writes (matrix, config, outcome) triples for GNN training
// ---------------------------------------------------------------------------
class DatasetCollector {
public:
    /// Open output file. Format: JSON Lines (one JSON object per line).
    explicit DatasetCollector(const std::string& output_path);
    ~DatasetCollector();

    /// Record one training triple after a solve completes.
    void record(const AdaptiveSelector::MatrixFeatures& features,
                const SolverAdvice&                     advice,
                const SolverStats&                      stats,
                const std::string& matrix_name = "");

    /// Force flush to disk.
    void flush();

    int  size()    const { return n_records_; }
    bool is_open() const { return out_.is_open(); }

    /// Merge multiple .jsonl files into one (for distributed collection).
    static void merge(const std::vector<std::string>& input_paths,
                      const std::string& output_path);

private:
    std::ofstream out_;
    int           n_records_ = 0;
    bool          header_written_ = false;
};

// ---------------------------------------------------------------------------
// GNNAdvisor  —  TorchScript inference (Thrust 1 core)
// ---------------------------------------------------------------------------
class GNNAdvisor : public MLAdvisor {
public:
    /// Load a TorchScript .pt model. Falls back to FeatureAdvisor if load fails.
    explicit GNNAdvisor(const std::string& model_path,
                        const SolverParams& base_params = {});
    ~GNNAdvisor() override;

    SolverAdvice advise(
            const AdaptiveSelector::MatrixFeatures& features) const override;

    void record_outcome(
            const AdaptiveSelector::MatrixFeatures& features,
            const SolverAdvice&                     advice,
            const SolverStats&                      stats) override;

    const char* name()      const override { return "GNNAdvisor"; }
    bool        is_loaded() const { return model_loaded_; }

    /// GNN inference confidence (max softmax probability for state prediction).
    float last_confidence() const { return last_confidence_; }

    /// Enable data collection for online learning (appends to output_path).
    void enable_collection(const std::string& output_path);

private:
    SolverParams  base_;
    bool          model_loaded_ = false;
    std::string   model_path_;
    mutable float last_confidence_ = 0.0f;

    // Fallback when model not loaded
    FeatureAdvisor fallback_;

    // Online data collection (optional)
    std::unique_ptr<DatasetCollector> collector_;

    // Opaque TorchScript module (avoids pulling torch headers into this .hpp)
    struct TorchImpl;
    std::unique_ptr<TorchImpl> torch_impl_;

    // Apply output bounds and map to SolverAdvice
    SolverAdvice decode_output(const std::vector<float>& state_probs,
                                float restart_raw,
                                float log_drop_tol,
                                float amg_strength) const;
};

} // namespace hsps
