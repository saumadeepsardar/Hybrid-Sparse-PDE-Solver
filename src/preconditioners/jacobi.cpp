// =============================================================================
// jacobi.cpp  —  Diagonal (Jacobi) preconditioner implementation
// =============================================================================

#include "../../include/preconditioners/jacobi.hpp"
#include "../../include/utils/timer.hpp"
#include <stdexcept>
#include <cmath>
#include <omp.h>

namespace hsps {

double JacobiPreconditioner::setup(const SparseMatrix& A) {
    Timer t; t.start();

    const Index n = A.rows();
    inv_diag_.resize(n);

    const auto& rp  = A.row_ptr();
    const auto& ci  = A.col_idx();
    const auto& val = A.values();

    for (Index i = 0; i < n; ++i) {
        Real diag = REAL_ZERO;
        for (Index k = rp[i]; k < rp[i + 1]; ++k) {
            if (ci[k] == i) { diag = val[k]; break; }
        }
        if (std::abs(diag) < REAL_EPS)
            throw std::runtime_error("JacobiPreconditioner: zero diagonal at row "
                                     + std::to_string(i));
        inv_diag_[i] = omega_ / diag;
    }

    setup_flops_ = n;   // n divisions
    ready_ = true;
    t.stop();
    return t.elapsed();
}

void JacobiPreconditioner::apply(const Vector& r, Vector& z) const {
    const Index n = r.size();
    if (z.size() != n) z.resize(n);

    const Real* rd  = r.data();
    const Real* id  = inv_diag_.data();
          Real* zd  = z.data();

#pragma omp parallel for schedule(static)
    for (Index i = 0; i < n; ++i)
        zd[i] = id[i] * rd[i];
}

} // namespace hsps
