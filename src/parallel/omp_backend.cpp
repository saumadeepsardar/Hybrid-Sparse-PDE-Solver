// =============================================================================
// omp_backend.cpp
// =============================================================================

#include "../../include/parallel/omp_backend.hpp"
#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace hsps {

OMPBackend::OMPBackend(int num_threads) : threads_(num_threads) {
    set_threads(num_threads);
}

void OMPBackend::set_threads(int n) {
    threads_ = n;
#ifdef _OPENMP
    if (n > 0) omp_set_num_threads(n);
    else        threads_ = omp_get_max_threads();
#else
    threads_ = 1;
#endif
}

// ---------------------------------------------------------------------------
void OMPBackend::alloc(Index n, Real val, Vector& out) const {
    out.resize(n, val);
}

// ---------------------------------------------------------------------------
Real OMPBackend::dot(const Vector& x, const Vector& y) const {
    const Index n  = x.size();
    const Real* xd = x.data();
    const Real* yd = y.data();
    Real s = REAL_ZERO;
#pragma omp parallel for reduction(+:s) schedule(static)
    for (Index i = 0; i < n; ++i) s += xd[i] * yd[i];
    return s;
}

Real OMPBackend::nrm2(const Vector& x) const {
    return std::sqrt(dot(x, x));
}

void OMPBackend::axpy(Real a, const Vector& x, Vector& y) const {
    const Index n  = x.size();
    const Real* xd = x.data();
          Real* yd = y.data();
#pragma omp parallel for schedule(static)
    for (Index i = 0; i < n; ++i) yd[i] += a * xd[i];
}

void OMPBackend::axpby(Real a, const Vector& x, Real b, Vector& y) const {
    const Index n  = x.size();
    const Real* xd = x.data();
          Real* yd = y.data();
#pragma omp parallel for schedule(static)
    for (Index i = 0; i < n; ++i) yd[i] = a * xd[i] + b * yd[i];
}

void OMPBackend::copy(const Vector& src, Vector& dst) const {
    dst.resize(src.size());
    const Real* sd = src.data();
          Real* dd = dst.data();
    const Index n  = src.size();
#pragma omp parallel for schedule(static)
    for (Index i = 0; i < n; ++i) dd[i] = sd[i];
}

void OMPBackend::scale(Real a, Vector& x) const {
    const Index n = x.size();
          Real* d = x.data();
#pragma omp parallel for schedule(static)
    for (Index i = 0; i < n; ++i) d[i] *= a;
}

void OMPBackend::fill(Vector& x, Real val) const {
    const Index n = x.size();
          Real* d = x.data();
#pragma omp parallel for schedule(static)
    for (Index i = 0; i < n; ++i) d[i] = val;
}

// ---------------------------------------------------------------------------
// SpMV: y = A * x  (CSR, row-parallel, dynamic scheduling for load balance)
// ---------------------------------------------------------------------------
void OMPBackend::spmv(const SparseMatrix& A,
                       const Vector& x, Vector& y) const {
    const Index  m   = A.rows();
    const Index* rp  = A.row_ptr().data();
    const Index* ci  = A.col_idx().data();
    const Real*  val = A.values().data();
    const Real*  xd  = x.data();

    if (y.size() != m) y.resize(m);
    Real* yd = y.data();

#pragma omp parallel for schedule(dynamic, 128)
    for (Index i = 0; i < m; ++i) {
        Real s = REAL_ZERO;
        for (Index k = rp[i]; k < rp[i + 1]; ++k)
            s += val[k] * xd[ci[k]];
        yd[i] = s;
    }
}

} // namespace hsps
