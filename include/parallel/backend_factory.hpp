#pragma once

// =============================================================================
// backend_factory.hpp  —  Runtime backend selection
//
// Usage
// -----
//   auto backend = BackendFactory::create(ctx.config());
//   // backend is now OMP / MPI+OMP / CUDA depending on compile flags and config
//
//   backend->spmv(A, x, y);   // dispatches to the right implementation
// =============================================================================

#include "backend_base.hpp"
#include "parallel_config.hpp"
#include "parallel_context.hpp"
#include <memory>
#include <string>

namespace hsps {

class BackendFactory {
public:
    /// Create a backend from a ParallelConfig.
    /// Priority: CUDA > MPI > OMP (picks the most capable active backend).
    static std::unique_ptr<BackendBase>
    create(const ParallelConfig& cfg, const ParallelContext* ctx = nullptr);

    /// Create a named backend (ignores config flags).
    static std::unique_ptr<BackendBase>
    create(const std::string& name,
           const ParallelContext* ctx = nullptr,
           int threads = 0);

    /// Query which backend would be selected for a given config
    static std::string select_name(const ParallelConfig& cfg);

private:
    BackendFactory() = delete;
};

} // namespace hsps
