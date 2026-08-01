# HAMLSS: Complete Code Modification Map
# What changes, what stays, what is new — file by file

Every entry is one of three kinds:
  MODIFY   — existing file needs changes; exact lines/functions listed
  NEW      — file does not exist yet; must be created from scratch
  INTACT   — file is already correct for this research purpose; do not touch

Research thrusts are labelled T1–T4.

================================================================================
THRUST 1 — GNN-Guided Configuration Learning
================================================================================

────────────────────────────────────────────────────────────────────────────────
include/core/types.hpp                                                  MODIFY
────────────────────────────────────────────────────────────────────────────────
WHY: SolverStats needs three new fields for training data collection.
     SolverParams needs two new fields for s-step and async variants.

CHANGE 1 — Add to SolverStats (after energy_joules):
    double  energy_joules_hw  = 0.0;  // RAPL hardware reading (Thrust 2)
    double  rapl_start_j      = 0.0;  // snapshot at solve start
    double  rapl_end_j        = 0.0;  // snapshot at solve end
    int     sstep_k           = 0;    // s-step batch size used (Thrust 3)
    bool    async_mode        = false; // whether async iterations were used

CHANGE 2 — Add to SolverParams (after stall_threshold):
    int   sstep_k         = 1;     // s-step batch size (1 = standard CG)
    Real  sstep_cond_tol  = 1e12;  // max allowed basis condition number
    bool  async_enabled   = false; // enable asynchronous iterations
    int   async_max_stale = 3;     // max staleness τ in async mode
    int   fill_per_row    = 0;     // ILUT fill beyond pattern (Thrust 1 tuning)

CHANGE 3 — Add to SolverType enum:
    BICGSTAB,    // for comparison baseline in Paper 1
    SSTEP_CG,    // Communication-avoiding CG (Thrust 3)
    ASYNC_FGMRES // Asynchronous FGMRES (Thrust 3)

CHANGE 4 — Add to PrecondType enum:
    BLOCK_JACOBI,  // block-diagonal Jacobi (better for GPU, Paper 4)
    CHEBYSHEV      // polynomial preconditioner (s-step compatible, Thrust 3)

────────────────────────────────────────────────────────────────────────────────
include/adaptive/adaptive_selector.hpp                                  MODIFY
────────────────────────────────────────────────────────────────────────────────
WHY: MatrixFeatures struct needs 14 additional fields that the GNN will use
     as node/graph-level features. The current 6 fields are insufficient for
     distinguishing Helmholtz from conv-diff or 2D from 3D problems.

CHANGE — Extend MatrixFeatures struct:
    struct MatrixFeatures {
        // --- existing fields (keep) ---
        int    n;
        int    nnz;
        double density;
        double diag_dominance;
        double symmetry_residual;
        double frobenius_norm;
        double estimated_cond;
        bool   is_spd;

        // --- new fields for GNN (add) ---
        double bandwidth;           // max |i-j| over non-zeros (stencil width)
        double avg_nnz_per_row;     // = nnz/n
        double nnz_variance;        // variance of row nnz counts
        double diag_sign_fraction;  // fraction of rows where a_ii > 0
        double off_diag_symmetry;   // max(|a_ij - a_ji|) / frobenius_norm
        double spectral_radius_est; // from power iteration (reuse Lanczos)
        double gershgorin_radius;   // max row absolute sum
        double diagonal_fraction;   // nnz on diagonal / total nnz
        double lower_upper_ratio;   // nnz(L) / nnz(U) — asymmetry indicator
        int    n_zero_diag;         // number of zero diagonal entries
        double algebraic_smoothness;// estimate: min eigenvalue gap
        double fill_ratio_ilu0;     // nnz(ILU) / nnz(A) — estimated
        double coarsening_ratio;    // estimated AMG coarsening rate
        int    estimated_depth;     // floor(log2(n / 50)) — AMG levels needed
    };

CHANGE — Add method to extract_features():
    The body of extract_features() in adaptive_selector.cpp needs ~60 new
    lines computing bandwidth, nnz_variance, fill_ratio_ilu0, etc. from the
    CSR arrays. No signature change required — just a richer body.

────────────────────────────────────────────────────────────────────────────────
include/adaptive/ml_advisor.hpp                                         MODIFY
────────────────────────────────────────────────────────────────────────────────
WHY: The GNNAdvisor class needs to be declared here. The LoggingAdvisor CSV
     needs more columns (all 22 MatrixFeatures + full SolverStats).

CHANGE 1 — Add GNNAdvisor class declaration:
    class GNNAdvisor : public MLAdvisor {
    public:
        /// Load a TorchScript .pt model exported from PyTorch
        explicit GNNAdvisor(const std::string& model_path,
                            const SolverParams& base_params = {});

        SolverAdvice advise(
            const AdaptiveSelector::MatrixFeatures& f) const override;

        const char* name() const override { return "GNNAdvisor"; }

        bool is_loaded() const { return model_loaded_; }

    private:
        SolverParams   base_;
        bool           model_loaded_ = false;
        std::string    model_path_;

        // Opaque pointer to TorchScript module
        // Use void* to avoid pulling <torch/script.h> into the header
        struct Impl; std::unique_ptr<Impl> impl_;
    };

CHANGE 2 — Expand LoggingAdvisor CSV header:
    Current CSV has 8 columns. Research needs:
    n, nnz, density, bandwidth, avg_nnz_per_row, nnz_variance,
    diag_dominance, symmetry_residual, estimated_cond, spectral_radius,
    gershgorin_radius, diagonal_fraction, lower_upper_ratio, n_zero_diag,
    fill_ratio_ilu0, coarsening_ratio, estimated_depth, is_spd,
    chosen_state, chosen_restart, chosen_drop_tol, chosen_amg_strength,
    iterations, converged, energy_joules, energy_joules_hw, solve_time_s
    (27 columns total)
    Change is purely in the csv_ << ... lines in ml_advisor.cpp.

CHANGE 3 — Add DatasetCollector helper class:
    A new class (can be in this same header) that batches (matrix, config,
    outcome) triples and writes them as JSON Lines format for PyTorch ingestion:
    class DatasetCollector {
    public:
        explicit DatasetCollector(const std::string& output_path);
        void record(const AdaptiveSelector::MatrixFeatures& f,
                    const SolverAdvice& advice,
                    const SolverStats& stats,
                    const std::string& matrix_name = "");
        void flush();
        int  size() const { return n_records_; }
    private:
        std::ofstream out_;
        int n_records_ = 0;
    };

────────────────────────────────────────────────────────────────────────────────
src/adaptive/adaptive_selector.cpp                                      MODIFY
────────────────────────────────────────────────────────────────────────────────
WHY: extract_features() computes only 8 fields. Needs 14 more.

CHANGE — In extract_features(), after the existing Lanczos block, add:

    // Bandwidth: scan all non-zeros for max |row - col|
    f.bandwidth = 0;
    for (Index i = 0; i < f.n; ++i)
        for (Index k = rp[i]; k < rp[i+1]; ++k)
            f.bandwidth = std::max(f.bandwidth, (double)std::abs(i - ci[k]));

    // nnz_variance: variance of per-row non-zero counts
    { double mean = (double)f.nnz / f.n; double var = 0;
      for (Index i = 0; i < f.n; ++i) {
          double d = (rp[i+1]-rp[i]) - mean; var += d*d; }
      f.nnz_variance = var / f.n; }

    // diagonal_fraction, lower_upper_ratio, n_zero_diag
    // fill_ratio_ilu0 (estimated as: nnz/n * avg_row_nnz / n)
    // ... (all computable from CSR arrays in O(nnz))

    // coarsening_ratio: ratio of nnz(A^coarse) / nnz(A) after 1 level AMG
    // estimated_depth: floor(log2(n / 50))
    // algebraic_smoothness: (lambda_max - lambda_min) / lambda_max from Lanczos

NO signature changes. Only the body of extract_features() grows.

────────────────────────────────────────────────────────────────────────────────
src/adaptive/ml_advisor.cpp                                             MODIFY
────────────────────────────────────────────────────────────────────────────────
WHY: Implement GNNAdvisor::advise() and DatasetCollector.

CHANGE 1 — Implement GNNAdvisor using TorchScript C++ API:
    #ifdef HSPS_USE_TORCH
    #include <torch/script.h>
    struct GNNAdvisor::Impl { torch::jit::script::Module module; };
    GNNAdvisor::GNNAdvisor(const std::string& path, const SolverParams& p)
        : base_(p) {
        try { impl_ = std::make_unique<Impl>();
              impl_->module = torch::jit::load(path);
              model_loaded_ = true; }
        catch (...) { std::cerr << "[GNNAdvisor] load failed: " << path << "\n"; }
    }
    SolverAdvice GNNAdvisor::advise(const MatrixFeatures& f) const {
        if (!model_loaded_) return HeuristicAdvisor(base_).advise(f);
        // Build 22-element feature tensor
        auto t = torch::tensor({ f.n, f.nnz, f.density, ... });
        auto out = impl_->module.forward({t}).toTuple();
        int   state_idx = out->elements()[0].toTensor().argmax().item<int>();
        float restart   = out->elements()[1].toTensor().item<float>();
        float drop_tol  = std::exp(out->elements()[2].toTensor().item<float>());
        SolverAdvice adv; adv.params = base_;
        adv.initial_state = static_cast<AdaptiveState>(state_idx);
        adv.params.restart_size = std::clamp((int)restart, 10, 300);
        adv.params.drop_tol     = std::clamp((double)drop_tol, 1e-6, 1e-2);
        return adv;
    }
    #else
    // Fallback: GNNAdvisor delegates to FeatureAdvisor
    #endif

CHANGE 2 — Expand LoggingAdvisor::advise() CSV output to 27 columns.
    Only the csv_ << ... lines change; no interface modification.

CHANGE 3 — Implement DatasetCollector::record() writing JSON Lines:
    out_ << "{\"features\":[" << f.n << "," << f.nnz << ",...],\"label\":"
         << static_cast<int>(advice.initial_state) << ","
         << "\"energy\":" << stats.energy_joules << "}\n";

────────────────────────────────────────────────────────────────────────────────
NEW: scripts/train_gnn.py
────────────────────────────────────────────────────────────────────────────────
Python training script (outside the C++ build). Reads the JSON Lines output
of DatasetCollector, builds torch_geometric.data.Data objects, trains the
GraphSAGE model, and exports TorchScript (.pt) for C++ inference.

Dependencies: torch, torch_geometric, numpy, pandas, scikit-learn, tqdm
Estimated: ~400 lines Python.

────────────────────────────────────────────────────────────────────────────────
NEW: scripts/collect_dataset.sh
────────────────────────────────────────────────────────────────────────────────
Shell script that calls bin/run_dataset on every matrix in data/, cycling
through all (state, restart, drop_tol) combinations and writing a unified
training_data.jsonl file. Estimated: ~50 lines bash + Python post-processing.

────────────────────────────────────────────────────────────────────────────────
NEW: include/adaptive/gnn_advisor.hpp (separate from ml_advisor.hpp)    NEW
────────────────────────────────────────────────────────────────────────────────
Move GNNAdvisor declaration here (cleaner separation, makes #ifdef HSPS_USE_TORCH
conditional compile isolated to one header, so rest of codebase compiles
without libtorch installed).

================================================================================
THRUST 2 — Mathematically Rigorous Energy Modelling
================================================================================

────────────────────────────────────────────────────────────────────────────────
include/energy/energy_monitor.hpp                                       MODIFY
────────────────────────────────────────────────────────────────────────────────
WHY: (a) calibrate() is a stub that returns defaults.
     (b) Per-kernel energy breakdown (not just total) needed for roofline model.
     (c) RAPL sampling happens only at solve boundaries, not per-iteration.

CHANGE 1 — Add per-kernel energy breakdown to IterRecord:
    struct IterRecord {
        int       iteration;
        double    residual;
        long long flops_spmv;      // was: flops (now split)
        long long flops_blas1;
        long long bytes_spmv;
        long long bytes_blas1;
        double    energy_spmv_j;   // roofline estimate for SpMV only
        double    energy_blas1_j;  // roofline estimate for BLAS-1 only
        double    energy_precond_j;// roofline estimate for precond apply
        double    rapl_j_delta;    // hardware measurement for this iteration
        double    elapsed_s;
    };

CHANGE 2 — Add per-kernel roofline coefficients to EnergyModelCoeffs:
    struct EnergyModelCoeffs {
        double alpha_flop_spmv  = 2.0e-10; // J/FLOP for SpMV
        double alpha_flop_blas1 = 1.5e-10; // J/FLOP for dot/axpy
        double alpha_mem_dram   = 5.0e-9;  // J/byte (DRAM miss)
        double alpha_mem_l3     = 5.0e-10; // J/byte (L3 cache hit)
        double alpha_mem_l2     = 5.0e-11; // J/byte (L2 hit)
        double l3_threshold_bytes = 40e6;  // L3 size; above → DRAM model
        double idle_watts       = 50.0;
        double comm_joules_per_byte = 1e-8;// MPI communication energy
        bool   calibrated       = false;   // set true after calibrate()
    };

CHANGE 3 — Add per-iteration RAPL snapshot:
    // New method: call at the START and END of each solver iteration
    double snapshot_rapl() const;   // returns current RAPL joules reading

    // New method: record with hardware measurement
    void record_with_hw(int iteration, double residual,
                        long long flops_spmv, long long bytes_spmv,
                        long long flops_blas1, long long bytes_blas1,
                        double rapl_start_j, double rapl_end_j);

CHANGE 4 — Implement calibrate():
    static EnergyModelCoeffs calibrate();
    // Implementation: run STREAM Triad for bandwidth, DGEMM for FLOP/s,
    // read RAPL during each, fit coefficients. ~100 lines in .cpp.
    // Currently this function exists but returns hardcoded defaults.

────────────────────────────────────────────────────────────────────────────────
src/energy/energy_monitor.cpp                                           MODIFY
────────────────────────────────────────────────────────────────────────────────
WHY: Implement the three changes above.

CHANGE 1 — Implement snapshot_rapl():
    double EnergyMonitor::snapshot_rapl() const {
    #ifdef __linux__
        return read_rapl_joules();  // already implemented; just expose it
    #else
        return 0.0;
    #endif
    }
    (This is literally one line change — the private read_rapl_joules()
     exists; just add a public snapshot_rapl() that calls it.)

CHANGE 2 — Implement calibrate() properly:
    ~100 new lines: STREAM-style memory bandwidth test, FLOP rate test,
    RAPL measurement during each, least-squares fit via Eigen or hand-rolled.

CHANGE 3 — Implement record_with_hw() that stores per-kernel breakdown
    in the new IterRecord fields.

────────────────────────────────────────────────────────────────────────────────
src/solvers/cg_solver.cpp                                               MODIFY
────────────────────────────────────────────────────────────────────────────────
WHY: Needs to call energy_monitor.snapshot_rapl() at start and end of each
     iteration to get per-iteration hardware energy readings.

CURRENT STATE: Accumulates flop/byte counts but only computes one proxy
     energy at the end. No per-iteration RAPL reads.

CHANGE — Add optional EnergyMonitor* field to SolverBase (or pass via params).
    In the CG loop body, before and after each iteration:
        double rapl_before = energy_mon_ ? energy_mon_->snapshot_rapl() : 0.0;
        // ... iteration body ...
        double rapl_after  = energy_mon_ ? energy_mon_->snapshot_rapl() : 0.0;
        if (energy_mon_)
            energy_mon_->record_with_hw(iter, rel_res, spmv_flops, spmv_bytes,
                                         blas1_flops, blas1_bytes,
                                         rapl_before, rapl_after);
    Same change needed in fgmres_solver.cpp, pipelined_fgmres_solver.cpp,
    parallel_cg_solver.cpp, parallel_fgmres_solver.cpp.
    Total: 5 files, ~8 lines each.

────────────────────────────────────────────────────────────────────────────────
include/solvers/solver_base.hpp                                         MODIFY
────────────────────────────────────────────────────────────────────────────────
WHY: SolverBase needs an optional EnergyMonitor pointer so callers can attach
     hardware energy monitoring without changing solver call signatures.

CHANGE — Add one field and one setter:
    void set_energy_monitor(std::shared_ptr<EnergyMonitor> mon) {
        energy_mon_ = mon;
    }
protected:
    std::shared_ptr<EnergyMonitor> energy_mon_ = nullptr;  // NEW

────────────────────────────────────────────────────────────────────────────────
NEW: include/energy/roofline_model.hpp                                  NEW
────────────────────────────────────────────────────────────────────────────────
Standalone roofline model class usable without EnergyMonitor. Takes platform
bandwidth and FLOP rate, returns energy estimate for any (kernel, n, nnz).
Used by Paper 2 experiments and by the theoretical E_min calculation.

NEW: src/energy/roofline_model.cpp                                      NEW
NEW: scripts/calibrate_energy.py                                        NEW
    Python script: runs STREAM + DGEMM, reads RAPL/AMD uProf/NVML,
    writes energy_coeffs.json. C++ reads this JSON at startup.

================================================================================
THRUST 3 — Communication-Optimal Distributed Krylov
================================================================================

────────────────────────────────────────────────────────────────────────────────
src/parallel/distributed_matrix.cpp                                     MODIFY
────────────────────────────────────────────────────────────────────────────────
WHY: This is the most critical fix in the entire codebase for research
     purposes. The current spmv() uses MPI_Allgatherv (O(N) communication)
     and has an acknowledged bug: the off_block_ contribution is silently
     skipped. This means multi-rank results are WRONG for any matrix where
     off-diagonal entries exist (i.e., all real matrices).

CURRENT STATE (bug):
    Lines 305-322: off_block SpMV skipped with comment "TODO: store reverse map"

CHANGE 1 — Add reverse map to HaloDescriptor (in distributed_matrix.hpp):
    struct HaloDescriptor {
        // existing fields ...
        std::vector<Index> halo_global_cols;  // ADD: global col for each halo slot
        // ... rest unchanged
    };
    Then in from_local_block(): store halo_global_cols alongside halo_map.

CHANGE 2 — Replace MPI_Allgatherv SpMV with nearest-neighbour exchange:
    The new spmv() body:

    Phase 1: post non-blocking receives for each neighbour rank
        for each neighbour r:
            MPI_Irecv(recv_buf_r, recv_count_r, MPI_DOUBLE, r, tag, comm, &reqs[r])

    Phase 2: pack and post non-blocking sends
        for each neighbour r:
            pack local values they need (send_indices[r] → send_buf_r)
            MPI_Isend(send_buf_r, send_count_r, MPI_DOUBLE, r, tag, comm, &reqs[r+nprocs])

    Phase 3: diagonal block SpMV (no communication needed)
        diag_block_.spmv(x.local, y.local)

    Phase 4: wait for all receives to complete
        MPI_Waitall(2*n_neighbours, reqs.data(), MPI_STATUSES_IGNORE)

    Phase 5: off-diagonal block SpMV using received halo values
        Vector halo_vec(halo_size);
        for (int slot = 0; slot < halo_size; ++slot)
            halo_vec[slot] = halo_buf_[slot];  // filled by receives
        Vector off_contrib(local_rows, 0.0);
        off_block_.spmv(halo_vec, off_contrib);
        y.local.axpy(1.0, off_contrib);

    This fixes the correctness bug AND reduces communication from O(N) to
    O(boundary_nnz) — a factor of √N for 2D problems.

CHANGE 3 — Populate send_indices in build_halo():
    Current build_halo() has comment "send_indices populated via Alltoall —
    left as placeholder". Replace with:
        MPI_Alltoall(how_many_I_need_from_each, 1, MPI_INT,
                     how_many_each_needs_from_me, 1, MPI_INT, comm)
        MPI_Alltoallv(my_global_col_requests, ...,   // tell each rank which cols I need
                      send_indices_packed, ...)        // learn which cols I must send
    ~40 lines, standard sparse distributed matrix setup pattern.

────────────────────────────────────────────────────────────────────────────────
NEW: include/solvers/sstep_cg_solver.hpp                                NEW
NEW: src/solvers/sstep_cg_solver.cpp                                    NEW
────────────────────────────────────────────────────────────────────────────────
WHY: Thrust 3 core contribution. s-step CG with Chebyshev basis.
     Reduces MPI_Allreduce calls from k per k-iterations to 1 per k-iterations.

Key interfaces that USE existing code (no changes to those files):
    - Uses SparseMatrix::spmv() directly  (INTACT)
    - Uses Vector BLAS-1 operations       (INTACT)
    - Extends SolverBase                  (INTACT, just add new SolverType::SSTEP_CG)
    - Uses EnergyMonitor::snapshot_rapl() (needs MODIFY above)
    - Uses ParallelContext::allreduce_sum() with one bulk reduce (INTACT)

New field needed in SolverParams (already added above): sstep_k, sstep_cond_tol

────────────────────────────────────────────────────────────────────────────────
NEW: include/solvers/async_fgmres_solver.hpp                            NEW
NEW: src/solvers/async_fgmres_solver.cpp                                NEW
────────────────────────────────────────────────────────────────────────────────
WHY: Thrust 3 secondary contribution. Uses MPI_Iallreduce (non-blocking)
     to overlap dot products with the next SpMV computation.

Difference from PipelinedFGMRESSolver:
    PipelinedFGMRESSolver: restructures loop order (compute next SpMV before
      applying Givens) — purely serial/OMP benefit, no MPI change.
    AsyncFGMRESSolver: issues MPI_Iallreduce BEFORE the next SpMV, then
      MPI_Wait AFTER — actual communication overlap. New file needed.

Existing pipelined_fgmres_solver.cpp has the right loop structure; AsyncFGMRES
wraps it with MPI non-blocking calls. About 60% code reuse from pipelined.

────────────────────────────────────────────────────────────────────────────────
src/solvers/solver_factory.cpp                                          MODIFY
────────────────────────────────────────────────────────────────────────────────
WHY: Factory must know about new SolverTypes (SSTEP_CG, ASYNC_FGMRES, BICGSTAB)
     and new PrecondTypes (BLOCK_JACOBI, CHEBYSHEV).

CHANGE — Add cases to make_solver() and make_precond() switch statements.
    case SolverType::SSTEP_CG:
        return std::make_unique<SStepCGSolver>();
    case SolverType::ASYNC_FGMRES:
        return std::make_unique<AsyncFGMRESSolver>();
    ~10 lines per new type.

────────────────────────────────────────────────────────────────────────────────
include/parallel/distributed_matrix.hpp                                 MODIFY
────────────────────────────────────────────────────────────────────────────────
WHY: HaloDescriptor needs halo_global_cols and send buffers for the new spmv().

CHANGE — Extend HaloDescriptor:
    struct HaloDescriptor {
        struct NeighbourInfo {
            int                rank;
            std::vector<Index> send_indices;   // local indices to send
            std::vector<Index> recv_indices;   // local halo slots to fill
            // ADD:
            std::vector<Index> send_global;    // global cols being sent
        };
        std::vector<NeighbourInfo> neighbours;
        std::vector<Real>          halo_buf;
        Index                      halo_size = 0;
        // ADD:
        std::vector<Index>         halo_global_cols; // global col for slot i
        std::vector<Real>          send_buf;         // packed send buffer
        std::vector<Real>          recv_buf;         // packed recv buffer
    };

================================================================================
THRUST 4 — Heterogeneous CPU-GPU Execution
================================================================================

────────────────────────────────────────────────────────────────────────────────
src/parallel/cuda_backend.cpp                                           MODIFY
────────────────────────────────────────────────────────────────────────────────
WHY: (a) cusparseConstDnVecDescr_t used in spmv_device() was introduced in
     CUDA 11.4. Needs a compile-guard for older CUDA SDKs.
     (b) NVML integration for real-time GPU power measurement.
     (c) Unified Memory path (cudaMallocManaged) for CPU-GPU data sharing.

CHANGE 1 — Add CUDA SDK version guard around spmv_device():
    #if CUDART_VERSION >= 11040
        // use cusparseConstDnVecDescr_t  (current code)
    #else
        // use cusparseCreateDnVec (non-const variant)
    #endif

CHANGE 2 — Add NVML power monitoring:
    #ifdef HSPS_USE_NVML
    #include <nvml.h>
    unsigned int CUDABackend::gpu_power_mw() const {
        nvmlDevice_t dev;
        nvmlDeviceGetHandleByIndex(ctx_.cuda_device(), &dev);
        unsigned int power_mw;
        nvmlDeviceGetPowerUsage(dev, &power_mw);
        return power_mw;
    }
    #endif
    Add declaration to cuda_backend.hpp:
        unsigned int gpu_power_mw() const;  // milliwatts, 0 if NVML unavailable

CHANGE 3 — Add Unified Memory allocation option:
    void alloc_unified(Index n, Real val, DeviceVector& dv) const;
    // Uses cudaMallocManaged instead of cudaMalloc.
    // CPU can then access dv.d_data directly without explicit copy.

────────────────────────────────────────────────────────────────────────────────
NEW: include/parallel/heterogeneous_context.hpp                         NEW
NEW: src/parallel/heterogeneous_context.cpp                             NEW
────────────────────────────────────────────────────────────────────────────────
WHY: Research Thrust 4's core: routes tasks between CPU and GPU based on
     real-time power measurements.

USES (INTACT, no changes to these):
    - CUDABackend::gpu_power_mw()    (new method above)
    - OMPBackend (all existing methods)
    - BackendBase interface

KEY new logic:
    BackendBase& HeterogeneousContext::route(TaskType t, Index problem_size) {
        // If GPU power near TDP limit → use CPU
        if (use_cuda_ && gpu_.gpu_power_mw() > gpu_power_limit_mw_)
            return cpu_;
        // If problem too small for GPU → use CPU
        if (t == TaskType::SPMV && problem_size < gpu_min_n_) return cpu_;
        // If AMG setup (irregular) → always CPU
        if (t == TaskType::AMG_SETUP)  return cpu_;
        // If ILU triangular solve → always CPU (sequential by nature)
        if (t == TaskType::PRECOND_ILU) return cpu_;
        // Default GPU for BLAS-1 and SpMV above threshold
        if (use_cuda_) return gpu_;
        return cpu_;
    }

────────────────────────────────────────────────────────────────────────────────
NEW: include/parallel/nvml_monitor.hpp                                  NEW
NEW: src/parallel/nvml_monitor.cpp                                      NEW
────────────────────────────────────────────────────────────────────────────────
Wraps NVML into the EnergyHardwareInterface so GPU power readings can be
aggregated alongside CPU RAPL readings for total node energy.
Gracefully falls back (returns 0) when NVML is not available.

================================================================================
INFRASTRUCTURE — affects all research thrusts
================================================================================

────────────────────────────────────────────────────────────────────────────────
Makefile                                                                MODIFY
────────────────────────────────────────────────────────────────────────────────
CHANGE 1 — Add HSPS_USE_TORCH flag for GNN:
    ifeq ($(TORCH),1)
        TORCH_HOME ?= /usr/local/libtorch
        BACKEND_DEFS += -DHSPS_USE_TORCH
        INCS         += -I$(TORCH_HOME)/include -I$(TORCH_HOME)/include/torch/csrc/api/include
        LDFLAGS      += -L$(TORCH_HOME)/lib -ltorch -ltorch_cpu -lc10 -Wl,-rpath,$(TORCH_HOME)/lib
    endif
    Usage: make BACKEND=OMP TORCH=1

CHANGE 2 — Add HSPS_USE_NVML flag:
    ifeq ($(NVML),1)
        BACKEND_DEFS += -DHSPS_USE_NVML
        LDFLAGS      += -lnvidia-ml
    endif

CHANGE 3 — Add new source files to LIB_SRCS:
    $(SRC_DIR)/solvers/sstep_cg_solver.cpp
    $(SRC_DIR)/solvers/async_fgmres_solver.cpp
    $(SRC_DIR)/parallel/heterogeneous_context.cpp
    $(SRC_DIR)/parallel/nvml_monitor.cpp
    $(SRC_DIR)/energy/roofline_model.cpp
    $(SRC_DIR)/adaptive/gnn_advisor.cpp          (conditionally if TORCH=1)

CHANGE 4 — Add reproduce targets:
    reproduce_paper1: download_matrices run_paper1_experiments plot_paper1
    reproduce_paper2: run_scaling_experiments plot_scaling

────────────────────────────────────────────────────────────────────────────────
examples/run_dataset.cpp                                                MODIFY
────────────────────────────────────────────────────────────────────────────────
WHY: Needs --force-state and --restart CLI flags so collect_dataset.sh can
     enumerate all (state, restart_size) combinations for training data.

CHANGE — Parse additional argv:
    --force-state EASY|MODERATE|HARD   (bypass adaptive selector)
    --restart N                         (override restart_size)
    --output-jsonl path                 (write DatasetCollector JSON Lines)
    --repeat N                          (run N times, collect N training triples)
    ~30 lines of argument parsing + DatasetCollector integration.

────────────────────────────────────────────────────────────────────────────────
include/core/sparse_matrix.hpp                                          MODIFY
────────────────────────────────────────────────────────────────────────────────
WHY: Research needs 3 new matrix analysis functions used by extract_features()
     and the s-step Chebyshev basis construction.

CHANGE — Add method declarations:
    /// Compute bandwidth (max |i-j| over non-zeros)
    Index bandwidth() const;

    /// Compute variance of per-row nnz counts
    double nnz_variance() const;

    /// Estimate fill ratio for ILU(0): nnz(L+U) / nnz(A)
    double estimated_ilu_fill() const;

    /// Estimate one-level AMG coarsening ratio (for feature extraction)
    double estimated_amg_coarsening_ratio(Real strength = 0.25) const;

    /// Chebyshev spectral interval [lambda_min, lambda_max] via Lanczos
    std::pair<Real, Real> spectral_interval(int lanczos_steps = 20) const;

src/core/sparse_matrix.cpp needs the implementations (~80 new lines total).

================================================================================
FILES COMPLETELY INTACT — zero changes needed
================================================================================

These files are already correctly designed for the research work.
Do not modify them:

  include/core/vector.hpp              — BLAS-1 interface is sufficient
  src/core/vector.cpp                  — correct and parallelised
  include/preconditioners/jacobi.hpp   — INTACT
  src/preconditioners/jacobi.cpp       — INTACT
  include/preconditioners/ilu.hpp      — already has drop_tol/fill_per_row
  src/preconditioners/ilu.cpp          — ILUT already implemented
  include/preconditioners/amg.hpp      — already has smoothed prolongation
  src/preconditioners/amg.cpp          — SA-AMG already implemented
  include/solvers/fgmres_solver.hpp    — INTACT
  src/solvers/fgmres_solver.cpp        — INTACT
  src/solvers/pipelined_fgmres_solver.cpp — INTACT (async extends it)
  include/parallel/omp_backend.hpp     — INTACT
  src/parallel/omp_backend.cpp         — INTACT
  include/parallel/backend_base.hpp    — INTACT
  src/parallel/backend_base.cpp        — INTACT
  include/parallel/backend_factory.hpp — INTACT (add TORCH/NVML cases)
  include/parallel/parallel_config.hpp — INTACT
  src/parallel/parallel_config.cpp     — INTACT
  include/parallel/parallel_context.hpp — INTACT
  include/parallel/distributed_vector.hpp — INTACT
  src/parallel/distributed_vector.cpp  — INTACT
  include/utils/matrix_market_io.hpp   — INTACT
  src/utils/matrix_market_io.cpp       — INTACT
  include/utils/logger.hpp             — INTACT
  include/utils/timer.hpp              — INTACT
  data/*.mtx                           — INTACT (expand with SuiteSparse downloads)

================================================================================
SUMMARY TABLE — 18-month modification schedule
================================================================================

Month   Files Modified/Created                            Thrust  Risk
──────────────────────────────────────────────────────────────────────────────
1       types.hpp (new enum values + SolverStats fields)  T1,T3   LOW
1       sparse_matrix.hpp + .cpp (5 new analysis methods) T1      LOW
1       adaptive_selector.cpp (extend extract_features)   T1      LOW
2       ml_advisor.hpp + .cpp (GNNAdvisor + DataCollector)T1      MEDIUM
2       scripts/train_gnn.py                              T1      MEDIUM
2       run_dataset.cpp (--force-state, --output-jsonl)   T1      LOW
3       energy_monitor.hpp + .cpp (per-kernel + calibrate)T2      MEDIUM
3       roofline_model.hpp + .cpp (NEW)                   T2      LOW
3       solver_base.hpp (add energy_mon_ field)            T2      LOW
3       cg_solver.cpp, fgmres_solver.cpp (RAPL snapshots) T2      LOW
4       scripts/train_gnn.py (full training loop)         T1      MEDIUM
4       gnn_advisor.hpp + .cpp (TorchScript inference)    T1      HIGH
4       Makefile (TORCH=1 flag)                           T1      LOW
5       scripts/calibrate_energy.py                       T2      LOW
5       E_min theorem (informal draft, not code)          T2      HIGH
6       distributed_matrix.hpp (extend HaloDescriptor)    T3      MEDIUM
6       distributed_matrix.cpp (halo exchange rewrite)    T3      HIGH
7       sstep_cg_solver.hpp + .cpp (NEW)                  T3      HIGH
7       solver_factory.cpp (new cases)                    T3      LOW
8       types.hpp (SSTEP_CG, ASYNC_FGMRES in enum)        T3      LOW
9       async_fgmres_solver.hpp + .cpp (NEW)              T3      HIGH
10      cuda_backend.cpp (SDK guard + NVML + UnifiedMem)  T4      MEDIUM
10      nvml_monitor.hpp + .cpp (NEW)                     T4      MEDIUM
11      heterogeneous_context.hpp + .cpp (NEW)            T4      HIGH
11      Makefile (NVML=1 flag, new sources)               T4      LOW
12      backend_factory.cpp (TORCH/NVML cases)            T1,T4   LOW
──────────────────────────────────────────────────────────────────────────────
Total:  14 files MODIFIED, 12 files NEW, 25 files INTACT
        Estimated new C++ lines: ~3,500
        Estimated new Python lines: ~600
