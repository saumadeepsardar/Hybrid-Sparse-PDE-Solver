#pragma once

// =============================================================================
// vector.hpp  —  Dense vector with BLAS-like operations and OpenMP support
// =============================================================================

#include "types.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <ostream>
#include <initializer_list>

namespace hsps {

class Vector {
public:
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------
    Vector() = default;
    explicit Vector(Index n, Real val = REAL_ZERO);
    Vector(std::initializer_list<Real> il);
    Vector(const std::vector<Real>& data);
    Vector(const Vector&)            = default;
    Vector(Vector&&)                 = default;
    Vector& operator=(const Vector&) = default;
    Vector& operator=(Vector&&)      = default;
    ~Vector()                        = default;

    // ------------------------------------------------------------------
    // Basic accessors
    // ------------------------------------------------------------------
    Index        size()              const noexcept { return static_cast<Index>(data_.size()); }
    bool         empty()             const noexcept { return data_.empty(); }
    Real&        operator[](Index i)       noexcept { return data_[i]; }
    const Real&  operator[](Index i) const noexcept { return data_[i]; }
    Real*        data()                    noexcept { return data_.data(); }
    const Real*  data()              const noexcept { return data_.data(); }

    void resize(Index n, Real val = REAL_ZERO);
    void fill(Real val);
    void zero();

    // ------------------------------------------------------------------
    // Level-1 BLAS equivalents (parallelised with OpenMP)
    // ------------------------------------------------------------------
    Real   dot(const Vector& other)    const;   ///< this · other
    Real   norm2()                     const;   ///< sqrt(this · this)
    Real   norm_inf()                  const;   ///< max|xi|
    void   scale(Real alpha);                   ///< this *= alpha
    void   axpy(Real alpha, const Vector& x);   ///< this += alpha * x
    void   axpby(Real a, const Vector& x,
                 Real b);                       ///< this = a*x + b*this
    void   copy_from(const Vector& src);        ///< this = src
    Vector operator+(const Vector& rhs) const;
    Vector operator-(const Vector& rhs) const;
    Vector operator*(Real scalar)       const;

    // ------------------------------------------------------------------
    // Memory / diagnostics
    // ------------------------------------------------------------------
    long long mem_bytes() const { return static_cast<long long>(data_.size()) * sizeof(Real); }

    friend std::ostream& operator<<(std::ostream& os, const Vector& v);

private:
    std::vector<Real> data_;
};

// ---------------------------------------------------------------------------
// Free-function helpers
// ---------------------------------------------------------------------------
Real   dot(const Vector& a, const Vector& b);
Real   norm2(const Vector& v);
Vector axpy(Real alpha, const Vector& x, const Vector& y); ///< returns alpha*x + y

} // namespace hsps
