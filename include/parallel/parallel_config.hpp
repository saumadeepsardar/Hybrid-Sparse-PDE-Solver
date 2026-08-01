#pragma once

// =============================================================================
// parallel_config.hpp  —  Compile-time and run-time parallel backend selection
//
// Three orthogonal backends, any combination enabled at once:
//
//   HSPS_USE_OMP   — shared-memory multi-threading via OpenMP
//   HSPS_USE_MPI   — distributed memory via MPI
//   HSPS_USE_CUDA  — GPU acceleration via CUDA + cuBLAS + cuSPARSE
//
// These macros are set by the Makefile:
//   make BACKEND=OMP          → HSPS_USE_OMP
//   make BACKEND=MPI          → HSPS_USE_OMP + HSPS_USE_MPI
//   make BACKEND=CUDA         → HSPS_USE_OMP + HSPS_USE_CUDA
//   make BACKEND=MPI_CUDA     → all three
//
// Runtime selection uses ParallelConfig (see below).
// =============================================================================

#include <string>
#include <stdexcept>

namespace hsps {

// ---------------------------------------------------------------------------
// Backend flags (bitmask)
// ---------------------------------------------------------------------------
enum class BackendFlags : unsigned {
    NONE     = 0u,
    OMP      = 1u << 0,    ///< OpenMP shared-memory threading
    MPI      = 1u << 1,    ///< MPI distributed memory
    CUDA     = 1u << 2,    ///< NVIDIA CUDA GPU
};

inline BackendFlags operator|(BackendFlags a, BackendFlags b) {
    return static_cast<BackendFlags>(
        static_cast<unsigned>(a) | static_cast<unsigned>(b));
}
inline BackendFlags operator&(BackendFlags a, BackendFlags b) {
    return static_cast<BackendFlags>(
        static_cast<unsigned>(a) & static_cast<unsigned>(b));
}
inline BackendFlags operator~(BackendFlags a) {
    return static_cast<BackendFlags>(~static_cast<unsigned>(a));
}
inline bool has_flag(BackendFlags flags, BackendFlags test) {
    return (static_cast<unsigned>(flags) & static_cast<unsigned>(test)) != 0u;
}

// ---------------------------------------------------------------------------
// Runtime parallel configuration  —  filled once, queried everywhere
// ---------------------------------------------------------------------------
struct ParallelConfig {
    BackendFlags backends = BackendFlags::OMP;  ///< Active backend set

    // OpenMP settings
    int  omp_threads = 0;       ///< 0 = use OMP_NUM_THREADS / hardware default

    // MPI settings
    int  mpi_rank    = 0;
    int  mpi_size    = 1;
    bool mpi_inited  = false;

    // CUDA settings
    int  cuda_device = 0;       ///< GPU device index
    bool cuda_inited = false;

    // Convenience queries
    bool use_omp()  const { return has_flag(backends, BackendFlags::OMP);  }
    bool use_mpi()  const { return has_flag(backends, BackendFlags::MPI);  }
    bool use_cuda() const { return has_flag(backends, BackendFlags::CUDA); }
    bool is_root()  const { return mpi_rank == 0; }

    /// Parse a string like "OMP", "MPI", "CUDA", "MPI_CUDA"
    static BackendFlags parse(const std::string& s);

    /// Human-readable description
    std::string describe() const;
};

// ---------------------------------------------------------------------------
// Feature-detection helpers (safe to call without runtime init)
// ---------------------------------------------------------------------------
inline bool compiled_with_omp() {
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}

inline bool compiled_with_mpi() {
#ifdef HSPS_USE_MPI
    return true;
#else
    return false;
#endif
}

inline bool compiled_with_cuda() {
#ifdef HSPS_USE_CUDA
    return true;
#else
    return false;
#endif
}

} // namespace hsps
