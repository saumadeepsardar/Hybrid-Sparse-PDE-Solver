// =============================================================================
// test_preconditioners.cpp
// =============================================================================

#include "test_framework.hpp"
#include "../include/core/sparse_matrix.hpp"
#include "../include/core/vector.hpp"
#include "../include/preconditioners/jacobi.hpp"
#include "../include/preconditioners/ilu.hpp"
#include "../include/preconditioners/amg.hpp"
#include <cmath>

using namespace hsps;
using namespace hsps_test;

// ---------------------------------------------------------------------------
// Helper: build a 2-D Poisson system
// ---------------------------------------------------------------------------
static SparseMatrix make_poisson(Index n) {
    return SparseMatrix::poisson_2d(n);
}

// ---------------------------------------------------------------------------
// Jacobi
// ---------------------------------------------------------------------------
HSPS_TEST(jacobi_setup_apply_diagonal) {
    // For a diagonal matrix D, Jacobi * D = I  →  J(r) = D⁻¹ r
    std::vector<Index> r={0,1,2}, c={0,1,2};
    std::vector<Real>  v={2.0, 4.0, 8.0};
    auto D = SparseMatrix::from_coo(3, 3, r, c, v);

    JacobiPreconditioner jac;
    jac.setup(D);

    Vector rhs = {4.0, 8.0, 16.0};
    Vector z(3);
    jac.apply(rhs, z);

    expect_near(z[0], 2.0, 1e-14, "z[0]");
    expect_near(z[1], 2.0, 1e-14, "z[1]");
    expect_near(z[2], 2.0, 1e-14, "z[2]");
}

HSPS_TEST(jacobi_reduces_residual_poisson) {
    // One Jacobi application should reduce ‖r - D z‖ compared to ‖r‖
    auto A = make_poisson(8);
    Vector rhs(64, 1.0);

    JacobiPreconditioner jac;
    jac.setup(A);

    Vector z(64);
    jac.apply(rhs, z);

    // z should be close to D⁻¹ rhs  (rhs / 4 for Poisson)
    for (Index i = 0; i < 64; ++i)
        expect_near(z[i], 0.25, 1e-14, "Jacobi on Poisson: z[" + std::to_string(i) + "]");
}

// ---------------------------------------------------------------------------
// ILU
// ---------------------------------------------------------------------------
HSPS_TEST(ilu_setup_no_throw) {
    auto A = make_poisson(10);
    ILUPreconditioner ilu;
    double t = ilu.setup(A);
    expect_true(ilu.is_ready(), "ILU ready after setup");
    expect_true(t >= 0.0, "setup time non-negative");
}

HSPS_TEST(ilu_apply_diagonal_system) {
    // For a diagonal system, ILU(0) = Jacobi: L=I, U=D
    std::vector<Index> r={0,1,2,3}, c={0,1,2,3};
    std::vector<Real>  v={2.0, 3.0, 5.0, 7.0};
    auto D = SparseMatrix::from_coo(4, 4, r, c, v);

    ILUPreconditioner ilu;
    ilu.setup(D);

    Vector rhs = {6.0, 9.0, 15.0, 21.0};
    Vector z(4);
    ilu.apply(rhs, z);

    expect_near(z[0], 3.0, 1e-12, "ILU diagonal z[0]");
    expect_near(z[1], 3.0, 1e-12, "ILU diagonal z[1]");
    expect_near(z[2], 3.0, 1e-12, "ILU diagonal z[2]");
    expect_near(z[3], 3.0, 1e-12, "ILU diagonal z[3]");
}

HSPS_TEST(ilu_approximate_inverse) {
    // z = ILU⁻¹ r  →  ‖r - A z‖ should be smaller than ‖r‖  for Poisson
    auto A   = make_poisson(8);
    Vector r(64, 1.0);

    ILUPreconditioner ilu;
    ilu.setup(A);

    Vector z(64);
    ilu.apply(r, z);

    // Compute residual A*z - r
    Vector Az = A * z;
    Real res = 0.0;
    for (Index i = 0; i < 64; ++i) res = std::max(res, std::abs(Az[i] - r[i]));
    Real r_norm = r.norm_inf();

    expect_less(res, r_norm, "ILU reduces residual");
}

// ---------------------------------------------------------------------------
// AMG
// ---------------------------------------------------------------------------
HSPS_TEST(amg_setup_builds_hierarchy) {
    auto A = make_poisson(16);
    AMGPreconditioner amg;
    amg.set_max_levels(5);
    double t = amg.setup(A);

    expect_true(amg.is_ready(), "AMG ready after setup");
    expect_true(amg.levels() >= 2, "AMG built at least 2 levels");
    expect_true(t >= 0.0, "setup time non-negative");
}

HSPS_TEST(amg_vcycle_reduces_residual) {
    // One V-cycle should reduce the error on Poisson
    auto A = make_poisson(16);
    const Index N = 256;
    Vector rhs(N, 1.0);

    AMGPreconditioner amg;
    amg.setup(A);

    Vector z(N, 0.0);
    amg.apply(rhs, z);

    // Residual after one V-cycle: ‖rhs - A*z‖ < ‖rhs‖
    Vector Az = A * z;
    Real res2 = 0.0;
    for (Index i = 0; i < N; ++i) res2 += (rhs[i] - Az[i]) * (rhs[i] - Az[i]);
    Real res_norm = std::sqrt(res2);
    Real rhs_norm = rhs.norm2();

    expect_less(res_norm, rhs_norm, "AMG V-cycle reduces residual");
}

HSPS_TEST(amg_apply_size_correct) {
    auto A = make_poisson(8);
    Vector r(64, 1.0);
    AMGPreconditioner amg;
    amg.setup(A);

    Vector z;
    amg.apply(r, z);
    expect_true(z.size() == 64, "AMG output size matches input");
}

int main() {
    return ::hsps_test::TestRegistry::instance().run_all();
}
