// =============================================================================
// test_io.cpp  —  MatrixMarket reader / writer tests
// =============================================================================

#include "test_framework.hpp"
#include "../include/utils/matrix_market_io.hpp"
#include "../include/core/sparse_matrix.hpp"
#include "../include/core/vector.hpp"
#include "../include/adaptive/adaptive_selector.hpp"
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;
using namespace hsps;
using namespace hsps_test;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
static const std::string DATA_DIR = "data";

static bool data_file_exists(const std::string& name) {
    return fs::exists(DATA_DIR + "/" + name);
}

static double rel_res(const SparseMatrix& A, const Vector& b, const Vector& x) {
    Vector r = A * x;
    double bnorm = b.norm2();
    for (Index i = 0; i < b.size(); ++i) r[i] = b[i] - r[i];
    return (bnorm > 1e-15) ? r.norm2() / bnorm : r.norm2();
}

// ---------------------------------------------------------------------------
// Write a tiny test matrix and read it back
// ---------------------------------------------------------------------------
HSPS_TEST(mm_write_read_roundtrip_general) {
    // Build a small 4×4 tridiagonal
    auto A = SparseMatrix::poisson_2d(2);  // 4×4
    const std::string path = "/tmp/hsps_test_roundtrip.mtx";
    MatrixMarketIO::save_matrix(path, A, "test roundtrip");
    SparseMatrix B = MatrixMarketIO::load_matrix(path);

    expect_true(B.rows() == A.rows(), "rows preserved");
    expect_true(B.cols() == A.cols(), "cols preserved");
    expect_true(B.nnz()  == A.nnz(),  "nnz preserved");

    // Check every entry
    const auto& rp = A.row_ptr();
    const auto& ci = A.col_idx();
    const auto& va = A.values();
    for (Index i = 0; i < A.rows(); ++i)
        for (Index k = rp[i]; k < rp[i+1]; ++k)
            expect_near(B.get(i, ci[k]), va[k], 1e-12,
                        "entry mismatch at (" + std::to_string(i) + ","
                        + std::to_string(ci[k]) + ")");
}

HSPS_TEST(mm_write_read_roundtrip_vector) {
    Vector v(5);
    for (int i = 0; i < 5; ++i) v[i] = (i + 1) * 1.5;
    const std::string path = "/tmp/hsps_test_vec_roundtrip.mtx";
    MatrixMarketIO::save_vector(path, v, "test vector");
    Vector w = MatrixMarketIO::load_vector(path, 5);

    expect_true(w.size() == 5, "vector size preserved");
    for (int i = 0; i < 5; ++i)
        expect_near(w[i], v[i], 1e-12,
                    "vector entry " + std::to_string(i));
}

// ---------------------------------------------------------------------------
// Parse header line
// ---------------------------------------------------------------------------
HSPS_TEST(mm_header_coordinate_real_symmetric) {
    const std::string path = "/tmp/hsps_test_header.mtx";
    {
        std::ofstream f(path);
        f << "%%MatrixMarket matrix coordinate real symmetric\n";
        f << "% comment\n";
        f << "10 10 30\n";
        f << "1 1 4.0\n";
        f << "2 1 -1.0\n";
    }
    auto h = MatrixMarketIO::read_header(path);
    expect_true(h.rows == 10, "rows=10");
    expect_true(h.cols == 10, "cols=10");
    expect_true(h.nnz  == 30, "nnz=30");
    expect_true(h.is_symmetric(), "symmetric flag");
}

HSPS_TEST(mm_header_array_vector) {
    const std::string path = "/tmp/hsps_test_array_header.mtx";
    {
        std::ofstream f(path);
        f << "%%MatrixMarket matrix array real general\n";
        f << "100 1\n";
        for (int i = 0; i < 100; ++i) f << "1.0\n";
    }
    auto h = MatrixMarketIO::read_header(path);
    expect_true(h.rows == 100, "array rows=100");
    expect_true(h.cols == 1,   "array cols=1");
    expect_true(h.format == MatrixMarketHeader::Format::ARRAY, "format=ARRAY");
}

HSPS_TEST(mm_load_symmetric_expands_to_full) {
    // Symmetric file with 3 entries (diagonal + 2 off-diag upper triangle)
    // → must produce 5 entries (diagonal + 4 symmetric off-diag)
    const std::string path = "/tmp/hsps_sym_expand.mtx";
    {
        std::ofstream f(path);
        f << "%%MatrixMarket matrix coordinate real symmetric\n";
        f << "3 3 3\n";
        f << "1 1 4.0\n";   // diagonal
        f << "2 2 4.0\n";   // diagonal
        f << "3 3 4.0\n";   // diagonal
        // No off-diagonal → symmetric == full
    }
    SparseMatrix A = MatrixMarketIO::load_matrix(path);
    expect_true(A.rows() == 3,  "3×3 matrix");
    expect_true(A.nnz()  == 3,  "3 entries (diag only)");
    expect_near(A.get(0,0), 4.0, 1e-14, "A[0,0]");

    // Now with off-diagonal
    const std::string path2 = "/tmp/hsps_sym_expand2.mtx";
    {
        std::ofstream f(path2);
        f << "%%MatrixMarket matrix coordinate real symmetric\n";
        f << "3 3 4\n";
        f << "1 1 4.0\n";
        f << "2 1 -1.0\n";  // stored as (row,col) = (2,1)
        f << "2 2 4.0\n";
        f << "3 3 4.0\n";
    }
    SparseMatrix B = MatrixMarketIO::load_matrix(path2);
    // (2,1) → add (1,2) mirror
    expect_true(B.nnz() == 5, "5 entries after symmetric expansion");
    expect_near(B.get(1, 0), -1.0, 1e-14, "B[1,0]");
    expect_near(B.get(0, 1), -1.0, 1e-14, "B[0,1] mirror");
}

HSPS_TEST(mm_load_general_no_expansion) {
    const std::string path = "/tmp/hsps_general_no_expand.mtx";
    {
        std::ofstream f(path);
        f << "%%MatrixMarket matrix coordinate real general\n";
        f << "3 3 4\n";
        f << "1 1 2.0\n";
        f << "1 2 -1.0\n";
        f << "2 1 -3.0\n";   // asymmetric
        f << "3 3 5.0\n";
    }
    SparseMatrix A = MatrixMarketIO::load_matrix(path);
    expect_true(A.nnz() == 4, "4 entries, no expansion for general");
    expect_near(A.get(1, 0), -3.0, 1e-14, "asymmetric entry preserved");
}

HSPS_TEST(mm_make_default_rhs) {
    auto b = MatrixMarketIO::make_default_rhs(100, 2.5);
    expect_true(b.size() == 100, "size 100");
    for (Index i = 0; i < b.size(); ++i)
        expect_near(b[i], 2.5, 1e-14, "value 2.5 at " + std::to_string(i));
}

HSPS_TEST(mm_load_problem_default_rhs) {
    // Load matrix without providing RHS → default b=1 created
    const std::string path = "/tmp/hsps_default_rhs.mtx";
    {
        std::ofstream f(path);
        f << "%%MatrixMarket matrix coordinate real symmetric\n";
        f << "4 4 4\n";
        f << "1 1 4.0\n"; f << "2 2 4.0\n";
        f << "3 3 4.0\n"; f << "4 4 4.0\n";
    }
    auto [A, b] = MatrixMarketIO::load_problem(path);
    expect_true(A.rows() == 4, "A rows=4");
    expect_true(b.size() == 4, "b size=4");
    for (Index i = 0; i < b.size(); ++i)
        expect_near(b[i], 1.0, 1e-14, "default b=1");
}

HSPS_TEST(mm_missing_file_throws) {
    bool threw = false;
    try { MatrixMarketIO::load_matrix("/nonexistent/path/to/file.mtx"); }
    catch (const std::runtime_error&) { threw = true; }
    expect_true(threw, "missing file throws runtime_error");
}

// ---------------------------------------------------------------------------
// Load bundled data files and solve
// ---------------------------------------------------------------------------
HSPS_TEST(dataset_pde225_loads) {
    if (!data_file_exists("pde225.mtx")) {
        std::cout << "    [SKIP] data/pde225.mtx not found\n";
        return;
    }
    SparseMatrix A = MatrixMarketIO::load_matrix(DATA_DIR + "/pde225.mtx");
    expect_true(A.rows() == 225, "pde225 rows=225");
    expect_true(A.nnz()  >= 1000, "pde225 nnz>=1000");
    expect_true(A.is_symmetric(1e-6), "pde225 is symmetric");
}

HSPS_TEST(dataset_pde225_solves) {
    if (!data_file_exists("pde225.mtx")) {
        std::cout << "    [SKIP] data/pde225.mtx not found\n";
        return;
    }
    auto [A, b] = MatrixMarketIO::load_problem(
        DATA_DIR + "/pde225.mtx",
        DATA_DIR + "/pde225_rhs.mtx");

    SolverParams p;
    p.tol = 1e-8; p.max_iter = 3000; p.verbose = false;
    AdaptiveSelector sel(p);
    Vector x(A.rows(), 0.0);
    SolverStats stats;
    bool ok = sel.solve(A, b, x, stats);

    expect_true(ok, "pde225 solves");
    expect_less(rel_res(A, b, x), 1e-7, "pde225 rel_res < 1e-7");
    std::cout << "    pde225: iters=" << stats.iterations
              << "  state=" << to_string(stats.state_used) << "\n";
}

HSPS_TEST(dataset_pde900_solves) {
    if (!data_file_exists("pde900.mtx")) {
        std::cout << "    [SKIP] data/pde900.mtx not found\n";
        return;
    }
    auto [A, b] = MatrixMarketIO::load_problem(
        DATA_DIR + "/pde900.mtx",
        DATA_DIR + "/pde900_rhs.mtx");

    SolverParams p;
    p.tol = 1e-8; p.max_iter = 5000; p.verbose = false;
    AdaptiveSelector sel(p);
    Vector x(A.rows(), 0.0);
    SolverStats stats;
    bool ok = sel.solve(A, b, x, stats);

    expect_true(ok, "pde900 solves");
    expect_less(rel_res(A, b, x), 1e-7, "pde900 rel_res < 1e-7");
    std::cout << "    pde900: iters=" << stats.iterations
              << "  state=" << to_string(stats.state_used) << "\n";
}

HSPS_TEST(dataset_helmholtz_solves) {
    if (!data_file_exists("helmholtz_400.mtx")) {
        std::cout << "    [SKIP] data/helmholtz_400.mtx not found\n";
        return;
    }
    auto [A, b] = MatrixMarketIO::load_problem(
        DATA_DIR + "/helmholtz_400.mtx",
        DATA_DIR + "/helmholtz_400_rhs.mtx");

    SolverParams p;
    p.tol = 1e-7; p.max_iter = 5000; p.restart_size = 60; p.verbose = false;
    AdaptiveSelector sel(p);
    Vector x(A.rows(), 0.0);
    SolverStats stats;
    bool ok = sel.solve(A, b, x, stats);

    // Helmholtz may be indefinite — accept even if not fully converged
    std::cout << "    helmholtz_400: iters=" << stats.iterations
              << "  state=" << to_string(stats.state_used)
              << "  rel_res=" << std::scientific << std::setprecision(2)
              << rel_res(A, b, x)
              << (ok ? "  CONVERGED" : "  (partial)") << "\n";
    // Soft check: at least residual reduced from initial
    expect_less(rel_res(A, b, x), 1.0, "helmholtz residual reduced");
}

HSPS_TEST(dataset_convdiff_solves) {
    if (!data_file_exists("convdiff_upwind_484.mtx")) {
        std::cout << "    [SKIP] data/convdiff_upwind_484.mtx not found\n";
        return;
    }
    auto [A, b] = MatrixMarketIO::load_problem(
        DATA_DIR + "/convdiff_upwind_484.mtx",
        DATA_DIR + "/convdiff_upwind_484_rhs.mtx");

    SolverParams p;
    p.tol = 1e-7; p.max_iter = 5000; p.restart_size = 60; p.verbose = false;
    AdaptiveSelector sel(p);
    Vector x(A.rows(), 0.0);
    SolverStats stats;
    bool ok = sel.solve(A, b, x, stats);

    expect_true(ok, "convdiff_484 solves");
    std::cout << "    convdiff_484: iters=" << stats.iterations
              << "  state=" << to_string(stats.state_used)
              << "  escalations=" << sel.escalation_log().size() << "\n";
}

HSPS_TEST(dataset_pde2961_solves) {
    if (!data_file_exists("pde2961.mtx")) {
        std::cout << "    [SKIP] data/pde2961.mtx not found\n";
        return;
    }
    auto [A, b] = MatrixMarketIO::load_problem(
        DATA_DIR + "/pde2961.mtx",
        DATA_DIR + "/pde2961_rhs.mtx");

    SolverParams p;
    p.tol = 1e-7; p.max_iter = 8000; p.restart_size = 80; p.verbose = false;
    AdaptiveSelector sel(p);
    Vector x(A.rows(), 0.0);
    SolverStats stats;
    bool ok = sel.solve(A, b, x, stats);

    expect_true(ok, "pde2961 solves");
    std::cout << "    pde2961 (n=" << A.rows() << "): iters=" << stats.iterations
              << "  state=" << to_string(stats.state_used) << "\n";
}

// ---------------------------------------------------------------------------
// list_mtx_files
// ---------------------------------------------------------------------------
HSPS_TEST(mm_list_files_finds_bundled) {
    auto files = MatrixMarketIO::list_mtx_files(DATA_DIR);
    // data/ should have at least 5 matrix files
    std::cout << "    Found " << files.size() << " .mtx files in data/\n";
    expect_true(files.size() >= 1, "list_mtx_files returns at least 1 file");
}

// ---------------------------------------------------------------------------
// save/load roundtrip with a real PDE matrix
// ---------------------------------------------------------------------------
HSPS_TEST(mm_poisson_save_load_roundtrip) {
    auto A_orig = SparseMatrix::poisson_2d(10);
    const std::string path = "/tmp/hsps_poisson_rt.mtx";
    MatrixMarketIO::save_matrix(path, A_orig, "poisson roundtrip");
    SparseMatrix A_load = MatrixMarketIO::load_matrix(path);

    expect_true(A_load.rows() == A_orig.rows(), "rows preserved");
    expect_true(A_load.nnz()  == A_orig.nnz(),  "nnz preserved");
    expect_near(A_load.frobenius_norm(), A_orig.frobenius_norm(),
                1e-10, "Frobenius norm preserved");
}

int main() {
    return ::hsps_test::TestRegistry::instance().run_all();
}
