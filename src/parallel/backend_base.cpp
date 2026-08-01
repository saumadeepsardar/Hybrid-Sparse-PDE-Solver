// =============================================================================
// backend_base.cpp
// =============================================================================

#include "../../include/parallel/backend_base.hpp"
#include "../../include/preconditioners/preconditioner_base.hpp"
#include "../../include/core/vector.hpp"

namespace hsps {

void BackendBase::precond_apply(const PreconditionerBase& M,
                                const Vector& r, Vector& z) const {
    M.apply(r, z);
}

} // namespace hsps
