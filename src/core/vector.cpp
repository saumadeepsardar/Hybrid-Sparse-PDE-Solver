// =============================================================================
// vector.cpp  —  Dense vector implementation
// =============================================================================

#include "../../include/core/vector.hpp"
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <omp.h>

namespace hsps {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
Vector::Vector(Index n, Real val)
    : data_(static_cast<Size>(n), val) {}

Vector::Vector(std::initializer_list<Real> il)
    : data_(il) {}

Vector::Vector(const std::vector<Real>& data)
    : data_(data) {}

void Vector::resize(Index n, Real val) {
    data_.assign(static_cast<Size>(n), val);
}

void Vector::fill(Real val) {
    std::fill(data_.begin(), data_.end(), val);
}

void Vector::zero() { fill(REAL_ZERO); }

// ---------------------------------------------------------------------------
// BLAS-like operations
// ---------------------------------------------------------------------------
Real Vector::dot(const Vector& other) const {
    const Index n = size();
    if (other.size() != n)
        throw std::runtime_error("Vector::dot — size mismatch");

    Real result = REAL_ZERO;
#pragma omp parallel for reduction(+:result) schedule(static)
    for (Index i = 0; i < n; ++i)
        result += data_[i] * other.data_[i];
    return result;
}

Real Vector::norm2() const {
    return std::sqrt(dot(*this));
}

Real Vector::norm_inf() const {
    const Index n = size();
    Real mx = REAL_ZERO;
#pragma omp parallel for reduction(max:mx) schedule(static)
    for (Index i = 0; i < n; ++i) {
        Real v = std::abs(data_[i]);
        if (v > mx) mx = v;
    }
    return mx;
}

void Vector::scale(Real alpha) {
    const Index n = size();
#pragma omp parallel for schedule(static)
    for (Index i = 0; i < n; ++i)
        data_[i] *= alpha;
}

void Vector::axpy(Real alpha, const Vector& x) {
    const Index n = size();
    if (x.size() != n)
        throw std::runtime_error("Vector::axpy — size mismatch");
#pragma omp parallel for schedule(static)
    for (Index i = 0; i < n; ++i)
        data_[i] += alpha * x.data_[i];
}

void Vector::axpby(Real a, const Vector& x, Real b) {
    const Index n = size();
    if (x.size() != n)
        throw std::runtime_error("Vector::axpby — size mismatch");
#pragma omp parallel for schedule(static)
    for (Index i = 0; i < n; ++i)
        data_[i] = a * x.data_[i] + b * data_[i];
}

void Vector::copy_from(const Vector& src) {
    data_ = src.data_;
}

Vector Vector::operator+(const Vector& rhs) const {
    const Index n = size();
    Vector result(n);
#pragma omp parallel for schedule(static)
    for (Index i = 0; i < n; ++i)
        result.data_[i] = data_[i] + rhs.data_[i];
    return result;
}

Vector Vector::operator-(const Vector& rhs) const {
    const Index n = size();
    Vector result(n);
#pragma omp parallel for schedule(static)
    for (Index i = 0; i < n; ++i)
        result.data_[i] = data_[i] - rhs.data_[i];
    return result;
}

Vector Vector::operator*(Real scalar) const {
    Vector result(*this);
    result.scale(scalar);
    return result;
}

// ---------------------------------------------------------------------------
// Stream output
// ---------------------------------------------------------------------------
std::ostream& operator<<(std::ostream& os, const Vector& v) {
    os << "[";
    for (Index i = 0; i < v.size(); ++i) {
        os << v[i];
        if (i + 1 < v.size()) os << ", ";
    }
    os << "]";
    return os;
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------
Real dot(const Vector& a, const Vector& b) { return a.dot(b); }
Real norm2(const Vector& v) { return v.norm2(); }

Vector axpy(Real alpha, const Vector& x, const Vector& y) {
    Vector result(y);
    result.axpy(alpha, x);
    return result;
}

} // namespace hsps
