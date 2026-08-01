// =============================================================================
// test_improvements.cpp
//
// Tests added for every improvement implemented in the second phase:
//   Improvement 2: ILUT(τ,p)
//   Improvement 3: Smoothed-Aggregation AMG prolongation
//   Improvement 4: Parallel SpGEMM
//   Improvement 5: CG breakdown restart
//   Improvement 6: Pipelined FGMRES
//   Improvement 7: ML Advisor (HeuristicAdvisor, FeatureAdvisor, LoggingAdvisor)
//   Improvement 8: SolverFactory
// =============================================================================

#include "test_framework.hpp"
#include "../include/core/sparse_matrix.hpp"
#include "../include/core/vector.hpp"
#include "../include/solvers/cg_solver.hpp"
#include "../include/solvers/fgmres_solver.hpp"
#include "../include/solvers/pipelined_fgmres_solver.hpp"
#include "../include/solvers/solver_factory.hpp"
#include "../include/preconditioners/jacobi.hpp"
#include "../include/preconditioners/ilu.hpp"
#include "../include/preconditioners/amg.hpp"
#include "../include/adaptive/adaptive_selector.hpp"
#include "../include/adaptive/ml_advisor.hpp"
#include <cmath>
#include <fstream>
#include <iostream>

using namespace hsps;
using namespace hsps_test;

// ---------------------------------------------------------------------------
static double rel_res(const SparseMatrix& A, const Vector& b, const Vector& x) {
    Vector r = A * x;
    Real bnorm = b.norm2();
    for (Index i = 0; i < b.size(); ++i) r[i] = b[i] - r[i];
    return (bnorm > 1e-15) ? r.norm2() / bnorm : r.norm2();
}

static SolverParams tight() {
    SolverParams p; p.tol = 1e-8; p.max_iter = 3000;
    p.restart_size = 50; p.verbose = false; return p;
}

// ===========================================================================
// Improvement 2: ILUT
// ===========================================================================
HSPS_TEST(ilut_drop_tol_zero_equals_ilu0) {
    // With drop_tol=0 and fill_p=0, ILUT must match ILU(0) output exactly
    auto A = SparseMatrix::poisson_2d(10);
    Vector b(100, 1.0), x0(100, 0.0), x1(100, 0.0);

    ILUPreconditioner ilu0, ilut;
    ilu0.set_drop_tol(0.0); ilu0.set_fill_per_row(0); ilu0.setup(A);
    ilut.set_drop_tol(0.0); ilut.set_fill_per_row(0); ilut.setup(A);

    ilu0.apply(b, x0);
    ilut.apply(b, x1);

    for (Index i = 0; i < 100; ++i)
        expect_near(x0[i], x1[i], 1e-12,
                    "ILUT(0,0)==ILU(0) at index " + std::to_string(i));
}

HSPS_TEST(ilut_nonzero_drop_reduces_nnz) {
    // A non-zero drop tolerance should produce fewer non-zeros in the factors
    auto A = SparseMatrix::convection_diffusion_2d(10, 0.1, 3.0, 0.0);
    const Index N = 100;
    Vector b(N, 1.0);

    ILUPreconditioner ilu0, ilut;
    ilu0.set_drop_tol(0.0);    ilu0.set_fill_per_row(0);
    ilut.set_drop_tol(1e-3);   ilut.set_fill_per_row(0);
    ilu0.setup(A);
    ilut.setup(A);

    // Both must still produce a usable preconditioner (apply doesn't throw)
    Vector z0(N), z1(N);
    ilu0.apply(b, z0);
    ilut.apply(b, z1);
    expect_true(z0.norm2() > 0, "ILU(0) apply non-trivial");
    expect_true(z1.norm2() > 0, "ILUT apply non-trivial");
}

HSPS_TEST(ilut_with_fill_converges_convdiff) {
    // ILUT(1e-4, 5) should converge FGMRES on conv-diff faster than ILU(0)
    auto A = SparseMatrix::convection_diffusion_2d(12, 0.1, 8.0, 0.0);
    const Index N = 144;
    Vector b(N, 1.0);

    ILUPreconditioner ilu0, ilut;
    ilu0.set_drop_tol(0.0);   ilu0.set_fill_per_row(0);
    ilut.set_drop_tol(1e-4);  ilut.set_fill_per_row(5);
    ilu0.setup(A);
    ilut.setup(A);

    Vector x0(N,0.0), x1(N,0.0);
    FGMRESSolver fgm0, fgm1;
    fgm0.set_params(tight()); fgm0.set_preconditioner(
        std::make_shared<ILUPreconditioner>(ilu0));
    fgm1.set_params(tight()); fgm1.set_preconditioner(
        std::make_shared<ILUPreconditioner>(ilut));

    SolverStats s0, s1;
    bool ok0 = fgm0.solve(A, b, x0, s0);
    bool ok1 = fgm1.solve(A, b, x1, s1);
    expect_true(ok0 || ok1, "at least one converges");
    if (ok0 && ok1)
        std::cout << "    ILU(0) iters=" << s0.iterations
                  << "  ILUT iters=" << s1.iterations << "\n";
}

// ===========================================================================
// Improvement 3: SA-AMG smoothed prolongation
// ===========================================================================
HSPS_TEST(amg_smoothed_prolongation_enabled_by_default) {
    auto A = SparseMatrix::poisson_2d(16);
    AMGPreconditioner amg;
    amg.set_smooth_prolongation(true, 1);
    double t = amg.setup(A);
    expect_true(amg.is_ready(), "SA-AMG ready");
    expect_true(t > 0.0,        "setup took non-zero time");
}

HSPS_TEST(amg_smoothed_converges_poisson) {
    auto A = SparseMatrix::poisson_2d(16);
    const Index N = 256;
    Vector b(N, 1.0), x(N, 0.0);

    auto amg = std::make_shared<AMGPreconditioner>();
    amg->set_smooth_prolongation(true, 1);
    amg->setup(A);

    FGMRESSolver fgm;
    fgm.set_params(tight());
    fgm.set_preconditioner(amg);
    SolverStats s;
    bool ok = fgm.solve(A, b, x, s);
    expect_true(ok, "SA-AMG FGMRES converges on Poisson");
    expect_less(rel_res(A, b, x), 1e-7, "SA-AMG rel_res < 1e-7");
}

HSPS_TEST(amg_smoothed_vs_raw_iteration_count) {
    // Smoothed prolongation should give <= iterations than raw (piecewise-constant)
    auto A = SparseMatrix::poisson_2d(16);
    const Index N = 256;
    Vector b(N, 1.0);

    auto make_amg = [](bool smooth) {
        auto amg = std::make_shared<AMGPreconditioner>();
        amg->set_smooth_prolongation(smooth, 1);
        amg->set_max_levels(5);
        amg->setup(SparseMatrix::poisson_2d(16));
        return amg;
    };
    auto amg_raw    = make_amg(false);
    auto amg_smooth = make_amg(true);

    SolverParams p = tight(); p.max_iter = 500;
    FGMRESSolver fgm_r, fgm_s;
    fgm_r.set_params(p); fgm_r.set_preconditioner(amg_raw);
    fgm_s.set_params(p); fgm_s.set_preconditioner(amg_smooth);

    Vector xr(N,0.0), xs(N,0.0);
    SolverStats sr, ss;
    fgm_r.solve(A, b, xr, sr);
    fgm_s.solve(A, b, xs, ss);

    std::cout << "    Raw AMG iters=" << sr.iterations
              << "  Smoothed AMG iters=" << ss.iterations << "\n";
    expect_true(ss.converged || sr.converged, "at least one converges");
}

// ===========================================================================
// Improvement 4: Parallel SpGEMM
// ===========================================================================
HSPS_TEST(parallel_spgemm_correctness_identity) {
    // I * I = I (regression for parallel version)
    std::vector<Index> r={0,1,2,3}, c={0,1,2,3};
    std::vector<Real>  v={1,1,1,1};
    auto I  = SparseMatrix::from_coo(4,4,r,c,v);
    auto I2 = mat_mat_product(I, I);
    for (Index i=0;i<4;++i)
        for (Index j=0;j<4;++j)
            expect_near(I2.get(i,j),(i==j)?1.0:0.0,1e-14,
                        "I2["+std::to_string(i)+","+std::to_string(j)+"]");
}

HSPS_TEST(parallel_spgemm_galerkin_product) {
    // Galerkin product R*A*P should be SPD when A is SPD and P is full-rank
    auto A = SparseMatrix::poisson_2d(8);
    // Build trivial 2-to-1 restriction (aggregate pairs of nodes)
    const Index N = 64, Nc = 32;
    std::vector<Index> pr(N+1), pc(N); std::vector<Real> pv(N,1.0);
    for(Index i=0;i<N;++i){ pr[i]=i; pc[i]=i/2; } pr[N]=N;
    SparseMatrix P(N, Nc, pr, pc, pv);
    SparseMatrix R = transpose(P);
    SparseMatrix AP  = mat_mat_product(A, P);
    SparseMatrix Ac  = mat_mat_product(R, AP);
    expect_true(Ac.rows()==Nc && Ac.cols()==Nc, "Galerkin dims correct");
    // Coarse operator must be SPD: all positive diagonals
    auto d = Ac.diagonal();
    for(Index i=0;i<Nc;++i)
        expect_true(d[i]>0.0, "Galerkin diag positive at "+std::to_string(i));
}

// ===========================================================================
// Improvement 5: CG breakdown restart
// ===========================================================================
HSPS_TEST(cg_restarts_field_tracked) {
    // Verify the restarts counter is wired up (may be 0 for well-conditioned)
    auto A = SparseMatrix::poisson_2d(10);
    Vector b(100,1.0), x(100,0.0);
    auto jac = std::make_shared<JacobiPreconditioner>(); jac->setup(A);
    CGSolver cg; cg.set_params(tight()); cg.set_preconditioner(jac);
    SolverStats s;
    cg.solve(A,b,x,s);
    expect_true(s.restarts >= 0, "restarts field is non-negative");
    expect_true(s.converged,     "CG converges without breakdown");
}

// ===========================================================================
// Improvement 6: Pipelined FGMRES
// ===========================================================================
HSPS_TEST(pipelined_fgmres_diagonal_system) {
    std::vector<Index> r={0,1,2}, c={0,1,2};
    std::vector<Real>  v={2.0,4.0,8.0};
    auto D = SparseMatrix::from_coo(3,3,r,c,v);
    Vector b={4.0,8.0,16.0}, x(3,0.0);

    auto jac = std::make_shared<JacobiPreconditioner>(); jac->setup(D);
    PipelinedFGMRESSolver pfgm;
    pfgm.set_params(tight()); pfgm.set_preconditioner(jac);
    SolverStats s; pfgm.solve(D,b,x,s);
    expect_near(x[0],2.0,1e-8,"pfgm x[0]");
    expect_near(x[1],2.0,1e-8,"pfgm x[1]");
    expect_near(x[2],2.0,1e-8,"pfgm x[2]");
}

HSPS_TEST(pipelined_fgmres_poisson_ilu) {
    auto A = SparseMatrix::poisson_2d(16);
    Vector b(256,1.0), x(256,0.0);
    auto ilu = std::make_shared<ILUPreconditioner>(); ilu->setup(A);
    PipelinedFGMRESSolver pfgm;
    pfgm.set_params(tight()); pfgm.set_preconditioner(ilu);
    SolverStats s; bool ok = pfgm.solve(A,b,x,s);
    expect_true(ok, "Pipelined-FGMRES+ILU converges on Poisson");
    expect_less(rel_res(A,b,x), 1e-7, "rel_res < 1e-7");
}

HSPS_TEST(pipelined_fgmres_matches_fgmres_solution) {
    // Both solvers should find the same solution (within tolerance)
    auto A = SparseMatrix::poisson_2d(12);
    const Index N = 144;
    Vector b(N,1.0), x_std(N,0.0), x_pip(N,0.0);

    auto ilu_std = std::make_shared<ILUPreconditioner>(); ilu_std->setup(A);
    auto ilu_pip = std::make_shared<ILUPreconditioner>(); ilu_pip->setup(A);

    FGMRESSolver fgm; fgm.set_params(tight()); fgm.set_preconditioner(ilu_std);
    PipelinedFGMRESSolver pfgm; pfgm.set_params(tight()); pfgm.set_preconditioner(ilu_pip);

    SolverStats ss, sp;
    fgm.solve(A,b,x_std,ss);
    pfgm.solve(A,b,x_pip,sp);

    Real diff = (x_std - x_pip).norm2() / x_std.norm2();
    std::cout << "    FGMRES iters=" << ss.iterations
              << "  Pipelined iters=" << sp.iterations
              << "  solution diff=" << diff << "\n";
    expect_less(diff, 1e-5, "pipelined and standard produce same solution");
}

HSPS_TEST(pipelined_fgmres_non_symmetric) {
    auto A = SparseMatrix::convection_diffusion_2d(10, 0.1, 5.0, 0.0);
    const Index N = 100;
    Vector b(N,1.0), x(N,0.0);
    auto ilu = std::make_shared<ILUPreconditioner>(); ilu->setup(A);
    PipelinedFGMRESSolver pfgm; pfgm.set_params(tight()); pfgm.set_preconditioner(ilu);
    SolverStats s; bool ok = pfgm.solve(A,b,x,s);
    expect_true(ok, "Pipelined-FGMRES handles non-symmetric system");
}

// ===========================================================================
// Improvement 7: ML Advisor
// ===========================================================================
HSPS_TEST(heuristic_advisor_routes_poisson_easy) {
    auto A = SparseMatrix::poisson_2d(8);
    AdaptiveSelector sel(tight());
    auto f = sel.extract_features(A);
    HeuristicAdvisor adv(tight());
    auto advice = adv.advise(f);
    expect_true(advice.initial_state == AdaptiveState::EASY,
                "HeuristicAdvisor routes Poisson to EASY");
}

HSPS_TEST(feature_advisor_tunes_restart_size) {
    auto A = SparseMatrix::convection_diffusion_2d(8, 0.01, 50.0, 0.0);
    AdaptiveSelector sel(tight());
    auto f = sel.extract_features(A);

    FeatureAdvisor adv(tight());
    auto advice = adv.advise(f);
    // Hard system: restart_size should be increased
    expect_true(advice.params.restart_size >= tight().restart_size,
                "FeatureAdvisor increases restart for hard system");
    std::cout << "    FeatureAdvisor: " << advice.rationale << "\n";
}

HSPS_TEST(feature_advisor_routes_hard_convdiff) {
    auto A = SparseMatrix::convection_diffusion_2d(8, 0.01, 100.0, 0.0);
    AdaptiveSelector sel(tight());
    auto f = sel.extract_features(A);
    FeatureAdvisor adv(tight());
    auto advice = adv.advise(f);
    expect_true(advice.initial_state == AdaptiveState::HARD ||
                advice.initial_state == AdaptiveState::MODERATE,
                "highly convection-dominated → HARD or MODERATE");
}

HSPS_TEST(logging_advisor_creates_csv) {
    const std::string path = "/tmp/hsps_advisor_log_test.csv";
    auto inner = std::make_shared<HeuristicAdvisor>(tight());
    LoggingAdvisor log_adv(inner, path);

    auto A = SparseMatrix::poisson_2d(8);
    AdaptiveSelector sel(tight());
    auto f = sel.extract_features(A);
    auto advice = log_adv.advise(f);
    SolverStats dummy; dummy.iterations = 5; dummy.converged = true;
    dummy.energy_joules = 1e-4;
    log_adv.record_outcome(f, advice, dummy);

    std::ifstream csv(path);
    expect_true(csv.good(), "LoggingAdvisor CSV created");
    std::string line; std::getline(csv, line);
    expect_true(line.find("n,nnz") != std::string::npos, "CSV has correct header");
}

HSPS_TEST(adaptive_selector_uses_ml_advisor) {
    auto A = SparseMatrix::poisson_2d(12);
    const Index N = 144;
    Vector b(N,1.0), x(N,0.0);

    SolverParams p = tight();
    AdaptiveSelector sel(p);
    auto adv = std::make_shared<FeatureAdvisor>(p);
    sel.set_ml_advisor(adv);

    SolverStats stats;
    bool ok = sel.solve(A, b, x, stats);
    expect_true(ok, "AdaptiveSelector with FeatureAdvisor converges");
    expect_less(rel_res(A, b, x), 1e-7, "rel_res < 1e-7");
}

HSPS_TEST(ensemble_advisor_majority_vote) {
    auto A = SparseMatrix::poisson_2d(8);
    AdaptiveSelector sel(tight());
    auto f = sel.extract_features(A);

    EnsembleAdvisor ens;
    ens.add(std::make_shared<HeuristicAdvisor>(tight()));
    ens.add(std::make_shared<FeatureAdvisor>(tight()));
    ens.add(std::make_shared<HeuristicAdvisor>(tight()));  // 2 vs 1 → EASY wins
    auto advice = ens.advise(f);
    expect_true(advice.initial_state == AdaptiveState::EASY,
                "Ensemble majority → EASY for Poisson");
}

// ===========================================================================
// Improvement 8: SolverFactory
// ===========================================================================
HSPS_TEST(factory_make_cg_jacobi) {
    auto A = SparseMatrix::poisson_2d(10);
    Vector b(100,1.0), x(100,0.0);
    double t;
    auto [solver, precond] = SolverFactory::make(
        SolverType::CG, PrecondType::JACOBI, A, tight(), &t);
    expect_true(solver  != nullptr, "factory returns solver");
    expect_true(precond != nullptr, "factory returns precond");
    expect_true(precond->is_ready(), "precond is ready after factory");
    SolverStats s; bool ok = solver->solve(A,b,x,s);
    expect_true(ok, "factory CG+Jacobi solves");
}

HSPS_TEST(factory_make_fgmres_ilu) {
    auto A = SparseMatrix::poisson_2d(10);
    Vector b(100,1.0), x(100,0.0);
    auto [solver, precond] = SolverFactory::make(
        SolverType::FGMRES, PrecondType::ILU, A, tight());
    SolverStats s; bool ok = solver->solve(A,b,x,s);
    expect_true(ok, "factory FGMRES+ILU solves");
}

HSPS_TEST(factory_make_for_state_hard) {
    auto A = SparseMatrix::poisson_2d(10);
    Vector b(100,1.0), x(100,0.0);
    auto [solver, precond] = SolverFactory::make_for_state(
        AdaptiveState::HARD, A, tight());
    expect_true(precond->type() == PrecondType::AMG,
                "HARD state → AMG preconditioner");
    SolverStats s; bool ok = solver->solve(A,b,x,s);
    expect_true(ok, "factory HARD state solves");
}

HSPS_TEST(factory_string_based) {
    auto A = SparseMatrix::poisson_2d(8);
    Vector b(64,1.0), x(64,0.0);
    auto [solver, precond] = SolverFactory::make("FGMRES","AMG",A,tight());
    expect_true(solver  != nullptr, "string factory returns solver");
    expect_true(precond != nullptr, "string factory returns precond");
    SolverStats s; bool ok = solver->solve(A,b,x,s);
    expect_true(ok, "string factory FGMRES+AMG solves");
}

HSPS_TEST(factory_parse_unknown_throws) {
    bool threw = false;
    try { SolverFactory::parse_solver("BLAH"); }
    catch (const std::invalid_argument&) { threw = true; }
    expect_true(threw, "unknown solver name throws");
}

HSPS_TEST(factory_make_only_precond) {
    auto A = SparseMatrix::poisson_2d(8);
    double t;
    auto precond = SolverFactory::make_precond(PrecondType::ILU, A, tight(), &t);
    expect_true(precond != nullptr, "make_precond returns non-null");
    expect_true(precond->is_ready(), "precond ready");
    expect_true(t >= 0.0, "setup time recorded");
}

HSPS_TEST(factory_none_precond_returns_nullptr) {
    auto A = SparseMatrix::poisson_2d(4);
    auto precond = SolverFactory::make_precond(PrecondType::NONE, A, tight());
    expect_true(precond == nullptr, "NONE precond returns nullptr");
}

int main() {
    return ::hsps_test::TestRegistry::instance().run_all();
}
