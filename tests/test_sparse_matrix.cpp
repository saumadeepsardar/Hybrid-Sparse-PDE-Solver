// =============================================================================
// test_sparse_matrix.cpp
// =============================================================================

#include "test_framework.hpp"
#include "../include/core/sparse_matrix.hpp"
#include "../include/core/vector.hpp"
#include <cmath>

using namespace hsps;
using namespace hsps_test;

// ---------------------------------------------------------------------------
HSPS_TEST(sparse_matrix_from_coo_basic) {
    // 3x3 identity via COO
    std::vector<Index> row = {0,1,2};
    std::vector<Index> col = {0,1,2};
    std::vector<Real>  val = {1.0, 1.0, 1.0};
    auto I = SparseMatrix::from_coo(3, 3, row, col, val);

    expect_true(I.rows() == 3 && I.cols() == 3, "dimensions");
    expect_true(I.nnz()  == 3,                  "nnz");
    expect_near(I.get(0,0), 1.0, 1e-15, "I[0,0]");
    expect_near(I.get(1,1), 1.0, 1e-15, "I[1,1]");
    expect_near(I.get(0,1), 0.0, 1e-15, "I[0,1] zero");
}

HSPS_TEST(sparse_matrix_spmv_identity) {
    // I * x = x
    std::vector<Index> row = {0,1,2,3};
    std::vector<Index> col = {0,1,2,3};
    std::vector<Real>  val = {1,1,1,1};
    auto I = SparseMatrix::from_coo(4, 4, row, col, val);

    Vector x = {3.0, -1.0, 2.5, 0.0};
    Vector y = I * x;

    for (Index i = 0; i < 4; ++i)
        expect_near(y[i], x[i], 1e-14, "I*x component " + std::to_string(i));
}

HSPS_TEST(sparse_matrix_spmv_tridiag) {
    // Tridiagonal [-1, 2, -1]  size 4
    // Row 0: 2*x0 - x1
    // Row 1: -x0 + 2*x1 - x2
    // Row 2: -x1 + 2*x2 - x3
    // Row 3: -x2 + 2*x3
    std::vector<Index> row_v, col_v;
    std::vector<Real>  val_v;
    for (Index i = 0; i < 4; ++i) {
        row_v.push_back(i); col_v.push_back(i); val_v.push_back(2.0);
        if (i > 0) { row_v.push_back(i); col_v.push_back(i-1); val_v.push_back(-1.0); }
        if (i < 3) { row_v.push_back(i); col_v.push_back(i+1); val_v.push_back(-1.0); }
    }
    auto A = SparseMatrix::from_coo(4, 4, row_v, col_v, val_v);

    Vector x = {1.0, 1.0, 1.0, 1.0};
    Vector y = A * x;

    expect_near(y[0], 1.0, 1e-14, "y[0]");
    expect_near(y[1], 0.0, 1e-14, "y[1]");
    expect_near(y[2], 0.0, 1e-14, "y[2]");
    expect_near(y[3], 1.0, 1e-14, "y[3]");
}

HSPS_TEST(sparse_matrix_poisson_2d_symmetry) {
    auto A = SparseMatrix::poisson_2d(8);
    expect_true(A.is_symmetric(1e-12), "Poisson 2D must be symmetric");
}

HSPS_TEST(sparse_matrix_poisson_2d_diagonal) {
    // All diagonal entries of Poisson 2D should be 4
    Index n = 6;
    auto A = SparseMatrix::poisson_2d(n);
    Vector diag = A.diagonal();
    for (Index i = 0; i < diag.size(); ++i)
        expect_near(diag[i], 4.0, 1e-14, "diag[" + std::to_string(i) + "]");
}

HSPS_TEST(sparse_matrix_transpose) {
    std::vector<Index> row_v = {0,0,1,2};
    std::vector<Index> col_v = {0,1,0,2};
    std::vector<Real>  val_v = {1.0, 2.0, 3.0, 4.0};
    auto A  = SparseMatrix::from_coo(3, 3, row_v, col_v, val_v);
    auto AT = transpose(A);

    expect_near(AT.get(0,0), 1.0, 1e-14, "AT[0,0]");
    expect_near(AT.get(1,0), 2.0, 1e-14, "AT[1,0]");
    expect_near(AT.get(0,1), 3.0, 1e-14, "AT[0,1]");
    expect_near(AT.get(2,2), 4.0, 1e-14, "AT[2,2]");
}

HSPS_TEST(sparse_matrix_mat_mat_identity) {
    // I * I = I
    std::vector<Index> r={0,1,2}, c={0,1,2};
    std::vector<Real>  v={1,1,1};
    auto I  = SparseMatrix::from_coo(3, 3, r, c, v);
    auto I2 = mat_mat_product(I, I);
    for (Index i = 0; i < 3; ++i)
        for (Index j = 0; j < 3; ++j)
            expect_near(I2.get(i,j), (i==j) ? 1.0 : 0.0, 1e-14,
                        "I2[" + std::to_string(i) + "," + std::to_string(j) + "]");
}

HSPS_TEST(sparse_matrix_coo_duplicate_accumulation) {
    // Two entries at (0,0): should accumulate to 3.0
    std::vector<Index> r={0,0}, c={0,0};
    std::vector<Real>  v={1.0, 2.0};
    auto A = SparseMatrix::from_coo(2, 2, r, c, v);
    expect_near(A.get(0,0), 3.0, 1e-14, "accumulated (0,0)");
}

HSPS_TEST(sparse_matrix_convdiff_non_symmetric) {
    // Convection-diffusion with non-zero convection must be non-symmetric
    auto A = SparseMatrix::convection_diffusion_2d(8, 0.1, 10.0, 0.0);
    expect_true(!A.is_symmetric(1e-6), "conv-diff non-symmetric");
}

int main() {
    return ::hsps_test::TestRegistry::instance().run_all();
}
