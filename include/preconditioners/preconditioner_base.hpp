#pragma once

// =============================================================================
// preconditioner_base.hpp  —  Abstract interface for all preconditioners
// =============================================================================

#include "../core/types.hpp"
#include "../core/sparse_matrix.hpp"
#include "../core/vector.hpp"

namespace hsps {

// ---------------------------------------------------------------------------
// Abstract preconditioner M ≈ A^{-1}
// ---------------------------------------------------------------------------
class PreconditionerBase {
public:
    explicit PreconditionerBase(PrecondType type) : type_(type) {}
    virtual ~PreconditionerBase() = default;

    // ------------------------------------------------------------------
    // Setup: build internal data structures from A.
    // Must be called before apply().
    // Returns setup wall-clock time in seconds.
    // ------------------------------------------------------------------
    virtual double setup(const SparseMatrix& A) = 0;

    // ------------------------------------------------------------------
    // Apply: solve  M * z = r  (approximately)
    // For FGMRES flexibility, this may vary each call (e.g., AMG V-cycle).
    // ------------------------------------------------------------------
    virtual void apply(const Vector& r, Vector& z) const = 0;

    // ------------------------------------------------------------------
    // Diagnostics
    // ------------------------------------------------------------------
    PrecondType type()         const { return type_; }
    bool        is_ready()     const { return ready_; }
    long long   setup_flops()  const { return setup_flops_; }

    /// Energy tier: CHEAP / MODERATE / EXPENSIVE
    virtual const char* energy_tier() const = 0;

protected:
    PrecondType type_;
    bool        ready_       = false;
    long long   setup_flops_ = 0;
};

} // namespace hsps
