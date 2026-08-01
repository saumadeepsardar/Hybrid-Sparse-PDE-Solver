// =============================================================================
// backend_factory.cpp
// =============================================================================

#include "../../include/parallel/backend_factory.hpp"
#include "../../include/parallel/omp_backend.hpp"
#include "../../include/parallel/mpi_backend.hpp"
#include "../../include/parallel/cuda_backend.hpp"
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <iostream>

namespace hsps {

std::unique_ptr<BackendBase>
BackendFactory::create(const ParallelConfig& cfg,
                       const ParallelContext* ctx) {
    // Priority order: CUDA > MPI > OMP
    if (cfg.use_cuda()) {
#ifdef HSPS_USE_CUDA
        if (ctx) return std::make_unique<CUDABackend>(*ctx);
        std::cerr << "[BackendFactory] CUDA requested but no context provided; "
                     "falling back to OMP.\n";
#else
        std::cerr << "[BackendFactory] CUDA requested but not compiled in; "
                     "falling back to OMP.\n";
#endif
    }

    if (cfg.use_mpi()) {
#ifdef HSPS_USE_MPI
        if (ctx) return std::make_unique<MPIBackend>(*ctx, cfg.omp_threads);
        std::cerr << "[BackendFactory] MPI requested but no context provided; "
                     "falling back to OMP.\n";
#else
        std::cerr << "[BackendFactory] MPI requested but not compiled in; "
                     "falling back to OMP.\n";
#endif
    }

    // Default: OMP
    return std::make_unique<OMPBackend>(cfg.omp_threads);
}

std::unique_ptr<BackendBase>
BackendFactory::create(const std::string& name,
                       const ParallelContext* ctx,
                       int threads) {
    std::string u = name;
    std::transform(u.begin(), u.end(), u.begin(),
                   [](unsigned char c){ return std::toupper(c); });

    if (u == "OMP" || u == "OPENMP") {
        return std::make_unique<OMPBackend>(threads);
    }
    if (u == "MPI" || u == "MPI+OMP") {
#ifdef HSPS_USE_MPI
        if (ctx) return std::make_unique<MPIBackend>(*ctx, threads);
        throw std::runtime_error("BackendFactory: MPI backend requires a ParallelContext");
#else
        std::cerr << "[BackendFactory] MPI not compiled in; returning OMP.\n";
        return std::make_unique<OMPBackend>(threads);
#endif
    }
    if (u == "CUDA") {
#ifdef HSPS_USE_CUDA
        if (ctx) return std::make_unique<CUDABackend>(*ctx);
        throw std::runtime_error("BackendFactory: CUDA backend requires a ParallelContext");
#else
        std::cerr << "[BackendFactory] CUDA not compiled in; returning OMP.\n";
        return std::make_unique<OMPBackend>(threads);
#endif
    }
    if (u == "HETERO" || u == "HETEROGENEOUS") {
#ifdef HSPS_USE_CUDA
        if (ctx)
            return std::make_unique<OMPBackend>(threads);  // TODO: return HeterogeneousContext
        std::cerr << "[BackendFactory] HETERO backend needs a context; returning OMP.\n";
#else
        std::cerr << "[BackendFactory] HETERO requires CUDA; returning OMP.\n";
#endif
        return std::make_unique<OMPBackend>(threads);
    }
    throw std::invalid_argument("BackendFactory: unknown backend name '" + name + "'");
}

std::string BackendFactory::select_name(const ParallelConfig& cfg) {
    if (cfg.use_cuda() && compiled_with_cuda()) return "CUDA";
    if (cfg.use_mpi()  && compiled_with_mpi())  return "MPI+OpenMP";
    if (cfg.use_omp()  && compiled_with_omp())  return "OpenMP";
    return "Serial";
}

} // namespace hsps
