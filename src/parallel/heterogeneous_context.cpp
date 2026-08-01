// =============================================================================
// heterogeneous_context.cpp  —  Dynamic CPU-GPU task routing
// =============================================================================

#include "../../include/parallel/heterogeneous_context.hpp"
#include "../../include/utils/logger.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

namespace hsps {

HeterogeneousContext::HeterogeneousContext(const ParallelContext& ctx,
                                             RoutingPolicy policy)
    : ctx_(ctx)
    , policy_(policy)
    , cuda_available_(ctx.config().use_cuda() && compiled_with_cuda())
    , cpu_(ctx.config().omp_threads)
    , gpu_(ctx)
    , nvml_(ctx.config().cuda_device) {
    if (cuda_available_) nvml_.start();
}

// ---------------------------------------------------------------------------
// Core routing decision
// ---------------------------------------------------------------------------
BackendBase& HeterogeneousContext::route(TaskType t, Index n) const {
    bool use_gpu = false;

    switch (t) {
    // AMG setup and ILU: always CPU (irregular access or sequential)
    case TaskType::AMG_SETUP:
    case TaskType::PRECOND_ILU:
    case TaskType::COARSE_SOLVE:
    case TaskType::MPI_EXCHANGE:
        use_gpu = false;
        break;

    // Jacobi and BLAS-1: GPU if above threshold
    case TaskType::PRECOND_JACOBI:
    case TaskType::DOT:
    case TaskType::AXPY:
    case TaskType::SCALE:
        use_gpu = cuda_available_ && (n >= policy_.gpu_blas1_threshold);
        break;

    // SpMV: GPU if above threshold
    case TaskType::SPMV:
        use_gpu = cuda_available_ && (n >= policy_.gpu_spmv_threshold);
        break;

    // AMG V-cycle: GPU for coarse levels only (heuristic: n > 1000)
    case TaskType::PRECOND_AMG:
        use_gpu = cuda_available_ && (n >= policy_.gpu_spmv_threshold);
        break;

    default:
        use_gpu = false;
    }

    // Power safety override: redirect to CPU if GPU is near TDP
    if (use_gpu && cuda_available_ && gpu_near_tdp()) {
        use_gpu = false;
        ++stats_.gpu_redirects;
        if (policy_.verbose)
            HSPS_LOG_WARN("HeterogeneousContext: GPU near TDP — routing to CPU");
    }

    if (policy_.verbose) log_route(t, n, use_gpu);

    if (use_gpu) {
        ++stats_.gpu_tasks;
        return const_cast<CUDABackend&>(gpu_);
    } else {
        ++stats_.cpu_tasks;
        return const_cast<OMPBackend&>(cpu_);
    }
}

BackendBase& HeterogeneousContext::route_spmv(Index n) const {
    return route(TaskType::SPMV, n);
}

BackendBase& HeterogeneousContext::route_blas1(Index n) const {
    return route(TaskType::AXPY, n);
}

// ---------------------------------------------------------------------------
// Power queries
// ---------------------------------------------------------------------------
double HeterogeneousContext::gpu_power_w() const {
    return cuda_available_ ? nvml_.current_power_w() : 0.0;
}

double HeterogeneousContext::gpu_power_limit_w() const {
    return cuda_available_ ? nvml_.power_limit_w() : 0.0;
}

double HeterogeneousContext::gpu_headroom_w() const {
    return cuda_available_ ? nvml_.power_headroom_w() : 0.0;
}

bool HeterogeneousContext::gpu_near_tdp() const {
    if (!cuda_available_) return false;
    double limit = gpu_power_limit_w();
    if (limit < 1.0) return false;
    return gpu_power_w() > limit * policy_.power_safety_factor;
}

// ---------------------------------------------------------------------------
// Unified Memory allocation
// ---------------------------------------------------------------------------
void HeterogeneousContext::alloc_unified(Index n, Real val, Vector& v) const {
#ifdef HSPS_USE_CUDA
    if (cuda_available_ && policy_.unified_memory) {
        // Allocate managed memory accessible from both CPU and GPU
        double* ptr = nullptr;
        cudaMallocManaged(&ptr, n * sizeof(double));
        // Fill on CPU
        for (Index i = 0; i < n; ++i) ptr[i] = val;
        // Wrap in Vector (the Vector class uses std::vector internally;
        // for true unified memory we need a custom allocator — stub here)
        v.resize(n, val);  // fallback: standard allocation
        return;
    }
#endif
    v.resize(n, val);
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------
void HeterogeneousContext::log_route(TaskType t, Index n, bool to_gpu) const {
    const char* task_names[] = {
        "SpMV", "dot", "axpy", "scale",
        "Jacobi", "ILU", "AMG V-cycle", "AMG setup", "MPI", "coarse"
    };
    const char* tname = task_names[static_cast<int>(t) < 10
                                   ? static_cast<int>(t) : 0];
    HSPS_LOG_DEBUG("Route: ", tname, "(n=", n, ") → ",
                   (to_gpu ? "GPU" : "CPU"),
                   (gpu_near_tdp() ? " [power-limited]" : ""));
}

void HeterogeneousContext::print_routing_report(std::ostream& os) const {
    os << "\n=== Heterogeneous Routing Report ===\n"
       << "  GPU tasks        : " << stats_.gpu_tasks    << "\n"
       << "  CPU tasks        : " << stats_.cpu_tasks    << "\n"
       << "  Power redirects  : " << stats_.gpu_redirects << "\n"
       << "  GPU available    : " << (cuda_available_ ? "YES" : "NO") << "\n";
    if (cuda_available_) {
        os << "  GPU device       : " << nvml_.device_name() << "\n"
           << "  Current power    : " << std::fixed << std::setprecision(1)
           << gpu_power_w() << " W\n"
           << "  Power limit      : " << gpu_power_limit_w() << " W\n"
           << "  GPU utilisation  : " << nvml_.gpu_utilisation_pct() << " %\n";
    }
    os << "====================================\n";
}

} // namespace hsps
