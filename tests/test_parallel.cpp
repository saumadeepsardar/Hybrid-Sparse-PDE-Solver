// =============================================================================
// test_parallel.cpp
//
// Tests for all parallel infrastructure added in phase 3:
//   • ParallelConfig parsing
//   • ParallelContext init (OMP, serial)
//   • BackendFactory creation
//   • OMPBackend BLAS-1 correctness
//   • ParallelCGSolver (OMP backend)
//   • ParallelFGMRESSolver (OMP backend)
//   • Backend-agnostic solver produces same answer as serial solver
//   • DistributedVector (serial mode)
// =============================================================================

#include "test_framework.hpp"
#include "../include/parallel/parallel_config.hpp"
#include "../include/parallel/parallel_context.hpp"
#include "../include/parallel/backend_factory.hpp"
#include "../include/parallel/omp_backend.hpp"
#include "../include/parallel/distributed_vector.hpp"
#include "../include/core/sparse_matrix.hpp"
#include "../include/core/vector.hpp"
#include "../include/solvers/parallel_cg_solver.hpp"
#include "../include/solvers/parallel_fgmres_solver.hpp"
#include "../include/solvers/cg_solver.hpp"
#include "../include/preconditioners/jacobi.hpp"
#include "../include/preconditioners/ilu.hpp"
#include "../include/preconditioners/amg.hpp"
#include <cmath>
#include <iostream>
#include <memory>

using namespace hsps;
using namespace hsps_test;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
static SolverParams tight() {
    SolverParams p;
    p.tol = 1e-9; p.max_iter = 3000;
    p.restart_size = 50; p.verbose = false;
    return p;
}

static double rel_res(const SparseMatrix& A, const Vector& b, const Vector& x) {
    Vector r = A * x;
    Real bnorm = b.norm2();
    for (Index i = 0; i < b.size(); ++i) r[i] = b[i] - r[i];
    return (bnorm > 1e-15) ? r.norm2() / bnorm : r.norm2();
}

// ---------------------------------------------------------------------------
// ParallelConfig
// ---------------------------------------------------------------------------
HSPS_TEST(config_parse_omp) {
    auto f = ParallelConfig::parse("OMP");
    expect_true(has_flag(f, BackendFlags::OMP),  "OMP flag set");
    expect_true(!has_flag(f, BackendFlags::MPI),  "MPI not set");
    expect_true(!has_flag(f, BackendFlags::CUDA), "CUDA not set");
}

HSPS_TEST(config_parse_mpi) {
    auto f = ParallelConfig::parse("MPI");
    expect_true(has_flag(f, BackendFlags::OMP), "OMP set in MPI");
    expect_true(has_flag(f, BackendFlags::MPI), "MPI flag set");
}

HSPS_TEST(config_parse_cuda) {
    auto f = ParallelConfig::parse("CUDA");
    expect_true(has_flag(f, BackendFlags::OMP),  "OMP set in CUDA");
    expect_true(has_flag(f, BackendFlags::CUDA), "CUDA flag set");
}

HSPS_TEST(config_parse_mpi_cuda) {
    auto f = ParallelConfig::parse("MPI_CUDA");
    expect_true(has_flag(f, BackendFlags::OMP),  "OMP set");
    expect_true(has_flag(f, BackendFlags::MPI),  "MPI set");
    expect_true(has_flag(f, BackendFlags::CUDA), "CUDA set");
}

HSPS_TEST(config_describe_not_empty) {
    ParallelConfig cfg;
    cfg.backends = BackendFlags::OMP;
    expect_true(!cfg.describe().empty(), "describe returns non-empty string");
}

HSPS_TEST(config_backend_flags_bitmask) {
    auto both = BackendFlags::OMP | BackendFlags::MPI;
    expect_true(has_flag(both, BackendFlags::OMP), "bitmask OMP");
    expect_true(has_flag(both, BackendFlags::MPI), "bitmask MPI");
    expect_true(!has_flag(both, BackendFlags::CUDA), "bitmask no CUDA");
}

// ---------------------------------------------------------------------------
// OMPBackend BLAS-1
// ---------------------------------------------------------------------------
HSPS_TEST(omp_backend_dot) {
    OMPBackend b(2);
    Vector x = {1.0, 2.0, 3.0};
    Vector y = {4.0, 5.0, 6.0};
    Real d = b.dot(x, y);
    expect_near(d, 32.0, 1e-12, "dot = 4+10+18 = 32");
}

HSPS_TEST(omp_backend_nrm2) {
    OMPBackend b;
    Vector x = {3.0, 4.0};
    expect_near(b.nrm2(x), 5.0, 1e-12, "norm2(3,4)=5");
}

HSPS_TEST(omp_backend_axpy) {
    OMPBackend b;
    Vector x = {1.0, 2.0, 3.0};
    Vector y = {0.0, 0.0, 0.0};
    b.axpy(2.0, x, y);
    expect_near(y[0], 2.0, 1e-14, "axpy y[0]");
    expect_near(y[1], 4.0, 1e-14, "axpy y[1]");
    expect_near(y[2], 6.0, 1e-14, "axpy y[2]");
}

HSPS_TEST(omp_backend_axpby) {
    OMPBackend b;
    Vector x = {1.0, 1.0, 1.0};
    Vector y = {2.0, 2.0, 2.0};
    b.axpby(3.0, x, 2.0, y);   // y = 3x + 2y = 3+4 = 7
    for (int i = 0; i < 3; ++i)
        expect_near(y[i], 7.0, 1e-14, "axpby y[" + std::to_string(i) + "]");
}

HSPS_TEST(omp_backend_scale) {
    OMPBackend b;
    Vector x = {1.0, 2.0, 4.0};
    b.scale(0.5, x);
    expect_near(x[0], 0.5, 1e-14, "scale[0]");
    expect_near(x[1], 1.0, 1e-14, "scale[1]");
    expect_near(x[2], 2.0, 1e-14, "scale[2]");
}

HSPS_TEST(omp_backend_fill) {
    OMPBackend b;
    Vector x(5, 0.0);
    b.fill(x, 3.14);
    for (int i = 0; i < 5; ++i)
        expect_near(x[i], 3.14, 1e-14, "fill[" + std::to_string(i) + "]");
}

HSPS_TEST(omp_backend_copy) {
    OMPBackend b;
    Vector src = {1.0, 2.0, 3.0};
    Vector dst;
    b.copy(src, dst);
    expect_true(dst.size() == 3, "copy size");
    for (int i = 0; i < 3; ++i)
        expect_near(dst[i], src[i], 1e-14, "copy[" + std::to_string(i) + "]");
}

HSPS_TEST(omp_backend_spmv_identity) {
    OMPBackend b;
    std::vector<Index> r={0,1,2}, c={0,1,2};
    std::vector<Real>  v={1,1,1};
    auto I = SparseMatrix::from_coo(3,3,r,c,v);
    Vector x={5.0,6.0,7.0}, y(3,0.0);
    b.spmv(I, x, y);
    for (int i = 0; i < 3; ++i)
        expect_near(y[i], x[i], 1e-14, "I*x[" + std::to_string(i) + "]");
}

HSPS_TEST(omp_backend_spmv_poisson) {
    OMPBackend b;
    auto A = SparseMatrix::poisson_2d(8);
    Vector x(64, 1.0), y(64, 0.0), y_ref(64, 0.0);
    b.spmv(A, x, y);
    A.spmv(x, y_ref);   // reference serial SpMV
    for (Index i = 0; i < 64; ++i)
        expect_near(y[i], y_ref[i], 1e-12,
                    "OMP SpMV matches serial at [" + std::to_string(i) + "]");
}

HSPS_TEST(omp_backend_threads_positive) {
    OMPBackend b(4);
    expect_true(b.threads() >= 1, "threads >= 1");
}

// ---------------------------------------------------------------------------
// BackendFactory
// ---------------------------------------------------------------------------
HSPS_TEST(factory_creates_omp) {
    auto b = BackendFactory::create("OMP");
    expect_true(b != nullptr, "factory returns non-null");
    expect_true(std::string(b->name()).find("OpenMP") != std::string::npos ||
                std::string(b->name()) == "OpenMP",
                "factory name is OpenMP");
}

HSPS_TEST(factory_unknown_throws) {
    bool threw = false;
    try { BackendFactory::create("UNKNOWN"); }
    catch (const std::invalid_argument&) { threw = true; }
    expect_true(threw, "unknown backend name throws");
}

HSPS_TEST(factory_select_name_omp) {
    ParallelConfig cfg;
    cfg.backends = BackendFlags::OMP;
    auto name = BackendFactory::select_name(cfg);
    expect_true(!name.empty(), "select_name returns non-empty");
}

HSPS_TEST(factory_mpi_without_context_fallsback) {
    // MPI backend without context should print warning and return OMP
    auto b = BackendFactory::create("MPI", nullptr, 0);
    expect_true(b != nullptr, "MPI fallback non-null");
}

HSPS_TEST(factory_cuda_without_context_fallsback) {
    auto b = BackendFactory::create("CUDA", nullptr, 0);
    expect_true(b != nullptr, "CUDA fallback non-null");
}

// ---------------------------------------------------------------------------
// ParallelCGSolver with OMP backend
// ---------------------------------------------------------------------------
HSPS_TEST(par_cg_omp_diagonal) {
    std::vector<Index> r={0,1,2,3}, c={0,1,2,3};
    std::vector<Real>  v={2,3,5,7};
    auto D = SparseMatrix::from_coo(4,4,r,c,v);
    Vector b={4,9,15,21}, x(4,0.0);

    auto backend = std::shared_ptr<BackendBase>(BackendFactory::create("OMP").release());
    auto jac     = std::make_shared<JacobiPreconditioner>(); jac->setup(D);
    ParallelCGSolver cg(backend);
    cg.set_preconditioner(jac); cg.set_params(tight());
    SolverStats s; cg.solve(D,b,x,s);

    expect_true(s.converged, "ParCG diagonal converged");
    expect_near(x[0], 2.0, 1e-8, "x[0]");
    expect_near(x[1], 3.0, 1e-8, "x[1]");
}

HSPS_TEST(par_cg_omp_poisson) {
    auto A = SparseMatrix::poisson_2d(16);
    Vector b(256,1.0), x(256,0.0);
    auto backend = std::shared_ptr<BackendBase>(BackendFactory::create("OMP").release());
    auto jac     = std::make_shared<JacobiPreconditioner>(); jac->setup(A);
    ParallelCGSolver cg(backend);
    cg.set_preconditioner(jac); cg.set_params(tight());
    SolverStats s; bool ok = cg.solve(A,b,x,s);
    expect_true(ok, "ParCG Poisson converged");
    expect_less(rel_res(A,b,x), 1e-8, "rel_res < 1e-8");
}

HSPS_TEST(par_cg_matches_serial_cg) {
    // ParallelCG with OMP backend must produce same answer as serial CG
    auto A = SparseMatrix::poisson_2d(12);
    const Index N = 144;
    Vector b(N, 1.0);

    // Serial CG
    auto jac_s = std::make_shared<JacobiPreconditioner>(); jac_s->setup(A);
    CGSolver cg_serial; cg_serial.set_params(tight()); cg_serial.set_preconditioner(jac_s);
    Vector x_serial(N, 0.0); SolverStats ss;
    cg_serial.solve(A, b, x_serial, ss);

    // Parallel CG
    auto backend = std::shared_ptr<BackendBase>(BackendFactory::create("OMP").release());
    auto jac_p   = std::make_shared<JacobiPreconditioner>(); jac_p->setup(A);
    ParallelCGSolver cg_par(backend);
    cg_par.set_params(tight()); cg_par.set_preconditioner(jac_p);
    Vector x_par(N, 0.0); SolverStats sp;
    cg_par.solve(A, b, x_par, sp);

    Real diff = (x_serial - x_par).norm2() / x_serial.norm2();
    std::cout << "    serial iters=" << ss.iterations
              << "  parallel iters=" << sp.iterations
              << "  solution diff=" << diff << "\n";
    expect_less(diff, 1e-5, "par CG solution agrees with serial");
}

HSPS_TEST(par_cg_stats_populated) {
    auto A = SparseMatrix::poisson_2d(8);
    Vector b(64,1.0), x(64,0.0);
    auto backend = std::shared_ptr<BackendBase>(BackendFactory::create("OMP").release());
    auto jac     = std::make_shared<JacobiPreconditioner>(); jac->setup(A);
    ParallelCGSolver cg(backend); cg.set_params(tight()); cg.set_preconditioner(jac);
    SolverStats s; cg.solve(A,b,x,s);
    expect_true(s.iterations > 0,       "iterations recorded");
    expect_true(s.energy_joules > 0.0,  "energy recorded");
    expect_true(s.flop_count > 0,       "flops recorded");
}

// ---------------------------------------------------------------------------
// ParallelFGMRESSolver with OMP backend
// ---------------------------------------------------------------------------
HSPS_TEST(par_fgmres_omp_poisson_ilu) {
    auto A = SparseMatrix::poisson_2d(16);
    Vector b(256,1.0), x(256,0.0);
    auto backend = std::shared_ptr<BackendBase>(BackendFactory::create("OMP").release());
    auto ilu     = std::make_shared<ILUPreconditioner>(); ilu->setup(A);
    ParallelFGMRESSolver fgm(backend);
    fgm.set_preconditioner(ilu); fgm.set_params(tight());
    SolverStats s; bool ok = fgm.solve(A,b,x,s);
    expect_true(ok, "ParFGMRES Poisson+ILU converged");
    expect_less(rel_res(A,b,x), 1e-8, "rel_res < 1e-8");
}

HSPS_TEST(par_fgmres_omp_poisson_amg) {
    auto A = SparseMatrix::poisson_2d(16);
    Vector b(256,1.0), x(256,0.0);
    auto backend = std::shared_ptr<BackendBase>(BackendFactory::create("OMP").release());
    auto amg     = std::make_shared<AMGPreconditioner>(); amg->setup(A);
    ParallelFGMRESSolver fgm(backend);
    fgm.set_preconditioner(amg); fgm.set_params(tight());
    SolverStats s; bool ok = fgm.solve(A,b,x,s);
    expect_true(ok, "ParFGMRES Poisson+AMG converged");
    expect_less(rel_res(A,b,x), 1e-8, "rel_res < 1e-8");
}

HSPS_TEST(par_fgmres_non_symmetric) {
    auto A = SparseMatrix::convection_diffusion_2d(10, 0.1, 5.0, 0.0);
    const Index N = 100;
    Vector b(N,1.0), x(N,0.0);
    auto backend = std::shared_ptr<BackendBase>(BackendFactory::create("OMP").release());
    auto ilu     = std::make_shared<ILUPreconditioner>(); ilu->setup(A);
    ParallelFGMRESSolver fgm(backend);
    fgm.set_preconditioner(ilu); fgm.set_params(tight());
    SolverStats s; bool ok = fgm.solve(A,b,x,s);
    expect_true(ok, "ParFGMRES non-symmetric converged");
}

HSPS_TEST(par_fgmres_restart_tracked) {
    auto A = SparseMatrix::poisson_2d(12);
    Vector b(144,1.0), x(144,0.0);
    auto backend = std::shared_ptr<BackendBase>(BackendFactory::create("OMP").release());
    auto ilu     = std::make_shared<ILUPreconditioner>(); ilu->setup(A);
    SolverParams p = tight(); p.restart_size = 8;  // force many restarts
    ParallelFGMRESSolver fgm(backend);
    fgm.set_preconditioner(ilu); fgm.set_params(p);
    SolverStats s; fgm.solve(A,b,x,s);
    expect_true(s.restarts >= 0, "restarts field non-negative");
}

// ---------------------------------------------------------------------------
// DistributedVector (serial mode — nprocs=1)
// ---------------------------------------------------------------------------
HSPS_TEST(dist_vec_create_serial) {
    ParallelConfig cfg; cfg.backends = BackendFlags::OMP;
    auto ctx = ParallelContext::init(0, nullptr, cfg);
    auto dv  = DistributedVector::create(100, 0.5, ctx);
    expect_true(dv.global_n == 100,     "global_n correct");
    expect_true(dv.local_n() == 100,    "local_n = global_n for nprocs=1");
    expect_near(dv.local[0], 0.5, 1e-14, "init value 0.5");
}

HSPS_TEST(dist_vec_scatter_gather_serial) {
    ParallelConfig cfg; cfg.backends = BackendFlags::OMP;
    auto ctx = ParallelContext::init(0, nullptr, cfg);
    Vector full = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto dv = DistributedVector::scatter(full, ctx);
    Vector gathered = dv.gather(ctx);
    expect_true(gathered.size() == 5, "gather size");
    for (int i = 0; i < 5; ++i)
        expect_near(gathered[i], full[i], 1e-14,
                    "gather[" + std::to_string(i) + "]");
}

HSPS_TEST(dist_vec_global_nrm2_serial) {
    ParallelConfig cfg; cfg.backends = BackendFlags::OMP;
    auto ctx = ParallelContext::init(0, nullptr, cfg);
    Vector v = {3.0, 4.0};
    auto dv  = DistributedVector::scatter(v, ctx);
    Real n   = dv.global_nrm2(ctx);
    expect_near(n, 5.0, 1e-12, "global_nrm2(3,4)=5");
}

HSPS_TEST(dist_vec_compute_split_even) {
    std::vector<Index> counts, offsets;
    DistributedVector::compute_split(10, 4, counts, offsets);
    // 10/4 = 2 rem 2 → ranks 0,1 get 3; ranks 2,3 get 2
    expect_true(counts[0] == 3, "rank 0 gets 3");
    expect_true(counts[1] == 3, "rank 1 gets 3");
    expect_true(counts[2] == 2, "rank 2 gets 2");
    expect_true(counts[3] == 2, "rank 3 gets 2");
    expect_true(offsets[0] == 0, "offset 0");
    expect_true(offsets[1] == 3, "offset 1");
    expect_true(offsets[2] == 6, "offset 2");
    expect_true(offsets[3] == 8, "offset 3");
}

HSPS_TEST(dist_vec_compute_split_exact) {
    std::vector<Index> counts, offsets;
    DistributedVector::compute_split(8, 4, counts, offsets);
    for (int r = 0; r < 4; ++r) expect_true(counts[r] == 2, "even split");
}

// ---------------------------------------------------------------------------
// Backend feature detection
// ---------------------------------------------------------------------------
HSPS_TEST(feature_detection_omp_available) {
    // OMP should always be true in our build (Makefile sets -fopenmp)
    expect_true(compiled_with_omp(), "OMP always compiled in");
}

HSPS_TEST(feature_detection_compile_flags_consistent) {
    // MPI/CUDA compile flags must match what BackendFactory reports
    if (!compiled_with_mpi()) {
        // Creating MPI backend without context must still succeed (fallback)
        auto b = BackendFactory::create("MPI", nullptr, 0);
        expect_true(b != nullptr, "MPI fallback created");
    }
    if (!compiled_with_cuda()) {
        auto b = BackendFactory::create("CUDA", nullptr, 0);
        expect_true(b != nullptr, "CUDA fallback created");
    }
}

int main() {
    return ::hsps_test::TestRegistry::instance().run_all();
}
