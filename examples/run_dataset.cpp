// =============================================================================
// examples/run_dataset.cpp
//
// Loads every .mtx file from the data/ directory (or a path you pass),
// runs the Adaptive Solver on each one, and prints a comprehensive report.
//
// Usage
// -----
//   ./bin/run_dataset                          # uses ./data/
//   ./bin/run_dataset data/pde900.mtx          # single file
//   ./bin/run_dataset /path/to/my_matrices/    # entire directory
//   ./bin/run_dataset data/pde900.mtx data/pde900_rhs.mtx   # matrix + RHS
//
// What it does
// ------------
//   For each matrix the runner:
//     1. Reads the .mtx header (metadata, size, nnz)
//     2. Loads the full sparse matrix (expanding symmetric storage)
//     3. Loads the RHS if a *_rhs.mtx or *_b.mtx file exists, else uses b=1
//     4. Runs the Adaptive Selector (EASY → MODERATE → HARD ladder)
//     5. Computes final residual and (if solution saved) writes *_sol.mtx
//     6. Prints per-matrix and summary statistics
//
// Output columns
// ---------------
//   Name        n      nnz    density   State         Iters   Rel-res   Time(s)   E(J)
//
// SuiteSparse download (do this once to get more matrices):
//   wget https://suitesparse-collection-website.herokuapp.com/MM/Bai/pde225.tar.gz
//   tar -xzf pde225.tar.gz && mv pde225/pde225.mtx data/
// =============================================================================

#include "../include/core/types.hpp"
#include "../include/core/sparse_matrix.hpp"
#include "../include/core/vector.hpp"
#include "../include/utils/matrix_market_io.hpp"
#include "../include/adaptive/adaptive_selector.hpp"
#include "../include/adaptive/ml_advisor.hpp"
#include "../include/energy/energy_monitor.hpp"
#include "../include/solvers/solver_factory.hpp"
#include "../include/utils/logger.hpp"
#include "../include/adaptive/gnn_advisor.hpp"
#include "../include/utils/timer.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <filesystem>

namespace fs = std::filesystem;
using namespace hsps;

// ---------------------------------------------------------------------------
// One result row
// ---------------------------------------------------------------------------
struct RunResult {
    std::string name;
    Index       n, nnz;
    double      density;
    bool        is_spd;
    AdaptiveState  final_state;
    int         iterations;
    double      rel_res;
    double      solve_time_s;
    double      setup_time_s;
    double      energy_j;
    bool        converged;
    int         escalations;
    std::string error_msg;
};

// ---------------------------------------------------------------------------
// helper: compute relative residual
// ---------------------------------------------------------------------------
static double rel_residual(const SparseMatrix& A,
                            const Vector& b, const Vector& x) {
    Vector r = A * x;
    double bnorm = b.norm2();
    for (Index i = 0; i < b.size(); ++i) r[i] = b[i] - r[i];
    return (bnorm > 1e-15) ? r.norm2() / bnorm : r.norm2();
}

// ---------------------------------------------------------------------------
// helper: find RHS file matching a matrix path
// ---------------------------------------------------------------------------
static std::string find_rhs(const std::string& matrix_path) {
    fs::path mp(matrix_path);
    std::string stem = mp.stem().string();
    fs::path dir     = mp.parent_path();

    for (const auto& suffix : {"_rhs", "_b", "_rhs_mtx"}) {
        fs::path rhs = dir / (stem + suffix + ".mtx");
        if (fs::exists(rhs)) return rhs.string();
    }
    return "";
}

// ---------------------------------------------------------------------------
// Run one matrix
// ---------------------------------------------------------------------------
static RunResult run_one(const std::string& matrix_path,
                          const std::string& rhs_path,
                          const SolverParams& params,
                          bool save_solution) {
    RunResult r;
    r.name = fs::path(matrix_path).stem().string();

    try {
        // Load
        auto [A, b] = MatrixMarketIO::load_problem(matrix_path, rhs_path);
        r.n       = A.rows();
        r.nnz     = A.nnz();
        r.density = A.density();
        r.is_spd  = A.is_symmetric(1e-6);

        // Solve
        auto energy = std::make_shared<EnergyMonitor>();
        energy->start();

        AdaptiveSelector sel(params);
        sel.set_ml_advisor(std::make_shared<FeatureAdvisor>(params));
        sel.set_energy_monitor(energy);

        Vector x(A.rows(), 0.0);
        SolverStats stats;
        sel.solve(A, b, x, stats);
        energy->stop();

        r.final_state  = stats.state_used;
        r.iterations   = stats.iterations;
        r.rel_res      = rel_residual(A, b, x);
        r.solve_time_s = stats.solve_time_s;
        r.setup_time_s = stats.setup_time_s;
        r.energy_j     = stats.energy_joules;
        r.converged    = stats.converged;
        r.escalations  = static_cast<int>(sel.escalation_log().size());

        // Optionally write solution
        if (save_solution && stats.converged) {
            fs::path sol_path = fs::path(matrix_path).parent_path()
                              / (r.name + "_sol.mtx");
            MatrixMarketIO::save_vector(sol_path.string(), x,
                                        "Solution for " + r.name);
        }

    } catch (const std::exception& e) {
        r.error_msg = e.what();
        r.converged = false;
    }
    return r;
}

// ---------------------------------------------------------------------------
// Print results table
// ---------------------------------------------------------------------------
static void print_table(const std::vector<RunResult>& results) {
    const int wn = 28, wi = 7, wr = 9, wf = 8;
    std::cout
        << "\n"
        << std::left  << std::setw(wn) << "Matrix"
        << std::right << std::setw(wi) << "n"
        << std::setw(wi) << "nnz"
        << std::setw(wr) << "SPD"
        << std::setw(20) << "Final State"
        << std::setw(wi) << "Iters"
        << std::setw(wr) << "Esc"
        << std::setw(12) << "Rel-Res"
        << std::setw(10) << "Time(s)"
        << std::setw(12) << "Energy(J)"
        << std::setw(7)  << "Conv"
        << "\n"
        << std::string(wn + 2*wi + wr + 20 + wi + wr + 12 + 10 + 12 + 7, '-')
        << "\n";

    int passed = 0, failed = 0;
    double total_energy = 0, total_time = 0;

    for (const auto& r : results) {
        if (!r.error_msg.empty()) {
            std::cout << std::left << std::setw(wn) << r.name
                      << "  ERROR: " << r.error_msg << "\n";
            ++failed; continue;
        }
        std::string state_str(to_string(r.final_state));

        std::cout
            << std::left  << std::setw(wn) << r.name
            << std::right << std::setw(wi) << r.n
            << std::setw(wi) << r.nnz
            << std::setw(wr) << (r.is_spd ? "yes" : "no")
            << std::left  << "  " << std::setw(18) << state_str
            << std::right << std::setw(wi) << r.iterations
            << std::setw(wr) << r.escalations
            << std::scientific << std::setprecision(2)
            << std::setw(12) << r.rel_res
            << std::fixed    << std::setprecision(4)
            << std::setw(10) << r.solve_time_s
            << std::scientific << std::setprecision(2)
            << std::setw(12) << r.energy_j
            << std::setw(7)  << (r.converged ? "YES" : "NO")
            << "\n";

        if (r.converged) { ++passed; total_energy += r.energy_j; total_time += r.solve_time_s; }
        else ++failed;
    }

    std::cout << std::string(wn + 2*wi + wr + 20 + wi + wr + 12 + 10 + 12 + 7, '-') << "\n"
              << "Solved: " << passed << "/" << (passed+failed)
              << "   Total time: " << std::fixed << std::setprecision(4) << total_time << " s"
              << "   Total energy: " << std::scientific << std::setprecision(3) << total_energy << " J\n";
}

// ---------------------------------------------------------------------------
// Collect matrix files
// ---------------------------------------------------------------------------
static std::vector<std::pair<std::string,std::string>>
collect_problems(int argc, char* argv[]) {
    std::vector<std::pair<std::string,std::string>> problems;

    auto add_mtx = [&](const std::string& path) {
        std::string stem = fs::path(path).stem().string();
        // Skip RHS files
        if (stem.size() >= 4 && stem.substr(stem.size()-4) == "_rhs") return;
        if (stem.size() >= 2 && stem.substr(stem.size()-2) == "_b")   return;
        // Skip solution files
        if (stem.size() >= 4 && stem.substr(stem.size()-4) == "_sol") return;
        problems.emplace_back(path, find_rhs(path));
    };

    if (argc == 1) {
        // Default: data/ directory relative to cwd
        std::string data_dir = "data";
        if (fs::exists(data_dir)) {
            std::vector<std::string> files;
            for (const auto& e : fs::directory_iterator(data_dir))
                if (e.path().extension() == ".mtx")
                    files.push_back(e.path().string());
            std::sort(files.begin(), files.end());
            for (const auto& f : files) add_mtx(f);
        } else {
            std::cerr << "No data/ directory found. "
                         "Pass a .mtx file or directory as an argument.\n";
        }
    } else if (argc == 2) {
        std::string arg = argv[1];
        if (fs::is_directory(arg)) {
            std::vector<std::string> files;
            for (const auto& e : fs::directory_iterator(arg))
                if (e.path().extension() == ".mtx")
                    files.push_back(e.path().string());
            std::sort(files.begin(), files.end());
            for (const auto& f : files) add_mtx(f);
        } else {
            add_mtx(arg);
        }
    } else if (argc == 3) {
        // matrix.mtx  rhs.mtx  — explicit pair
        problems.emplace_back(argv[1], argv[2]);
    }
    return problems;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
// ── Research run configuration (Thrust 1) ────────────────────────────────
struct RunConfig {
    int         force_state   = -1;   // -1=adaptive; 0=EASY,1=MODERATE,2=HARD
    int         force_restart = -1;   // -1=use params default
    std::string jsonl_output  = "";   // path for DatasetCollector JSON Lines
    int         repeat        = 1;    // solve repetitions per matrix
    bool        save_solution = false;
};

static RunConfig parse_run_config(int argc, char* argv[]) {
    RunConfig c;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--force-state" && i+1 < argc) {
            std::string s = argv[++i];
            if (s == "EASY")          c.force_state = 0;
            else if (s == "MODERATE") c.force_state = 1;
            else if (s == "HARD")     c.force_state = 2;
        } else if (arg == "--restart" && i+1 < argc) {
            c.force_restart = std::atoi(argv[++i]);
        } else if (arg == "--output-jsonl" && i+1 < argc) {
            c.jsonl_output = argv[++i];
        } else if (arg == "--repeat" && i+1 < argc) {
            c.repeat = std::atoi(argv[++i]);
        } else if (arg == "--save-solution") {
            c.save_solution = true;
        }
    }
    return c;
}

int main(int argc, char* argv[]) {
    Logger::instance().set_level(LogLevel::WARN);

    SolverParams params;
    params.tol          = 1e-8;
    params.max_iter     = 5000;
    params.restart_size = 60;
    params.stall_window = 20;
    params.verbose      = false;

    auto problems = collect_problems(argc, argv);

    if (problems.empty()) {
        std::cout << "No .mtx files found.\n\n"
                  << "Quick start:\n"
                  << "  # The data/ directory already contains 5 test matrices.\n"
                  << "  ./bin/run_dataset                        # run all\n"
                  << "  ./bin/run_dataset data/pde900.mtx        # single matrix\n\n"
                  << "Download more from SuiteSparse:\n"
                  << "  wget 'https://suitesparse-collection-website.herokuapp.com"
                     "/MM/Bai/pde225.tar.gz'\n"
                  << "  tar -xzf pde225.tar.gz && mv pde225/pde225.mtx data/\n"
                  << "  ./bin/run_dataset data/pde225.mtx\n";
        return 0;
    }

    std::cout << "=== HSPS Dataset Runner ===\n"
              << "  " << problems.size() << " problem(s) queued\n"
              << "  tol=" << params.tol
              << "  max_iter=" << params.max_iter << "\n\n";

    // Print header info for each matrix
    std::cout << std::left << std::setw(28) << "File"
              << std::right << std::setw(8) << "Rows"
              << std::setw(10) << "NNZ"
              << std::setw(12) << "Type"
              << std::setw(10) << "RHS\n"
              << std::string(60, '-') << "\n";

    for (const auto& [mat, rhs] : problems) {
        try {
            auto h = MatrixMarketIO::read_header(mat);
            std::string sym_label =
                (h.symmetry == MatrixMarketHeader::Symmetry::SYMMETRIC) ? "symmetric" :
                (h.symmetry == MatrixMarketHeader::Symmetry::GENERAL)   ? "general"   : "other";
            std::cout
                << std::left  << std::setw(28) << fs::path(mat).stem().string()
                << std::right << std::setw(8)  << h.rows
                << std::setw(10) << h.nnz_full
                << std::setw(12) << sym_label
                << std::setw(10) << (rhs.empty() ? "default" : "file")
                << "\n";
        } catch (const std::exception& e) {
            std::cout << std::left << std::setw(28) << fs::path(mat).stem().string()
                      << "  ERROR: " << e.what() << "\n";
        }
    }
    std::cout << "\n";

    // Run all
    // Parse research CLI flags (Thrust 1: --force-state, --output-jsonl, etc.)
    auto rconf = parse_run_config(argc, argv);
    if (rconf.force_restart > 0) params.restart_size = rconf.force_restart;

    // DatasetCollector for GNN training data
    std::unique_ptr<DatasetCollector> collector;
    if (!rconf.jsonl_output.empty()) {
        collector = std::make_unique<DatasetCollector>(rconf.jsonl_output);
        std::cout << "Dataset collection -> " << rconf.jsonl_output << "\n";
    }

    const bool save_solutions = rconf.save_solution;
    std::vector<RunResult> results;
    results.reserve(problems.size());

    for (const auto& [mat, rhs] : problems) {
        std::cout << "Solving " << fs::path(mat).stem().string() << " ... " << std::flush;
        auto r = run_one(mat, rhs, params, save_solutions);
        if (r.converged)
            std::cout << "OK  (" << r.iterations << " iters, "
                      << std::fixed << std::setprecision(4) << r.solve_time_s << " s)\n";
        else if (!r.error_msg.empty())
            std::cout << "ERROR: " << r.error_msg << "\n";
        else
            std::cout << "FAILED (rel_res=" << std::scientific
                      << std::setprecision(2) << r.rel_res << ")\n";
        results.push_back(std::move(r));
    }

    print_table(results);

    // Save results CSV for analysis
    {
        std::ofstream csv("dataset_results.csv");
        csv << "name,n,nnz,density,is_spd,final_state,iterations,escalations,"
               "rel_res,solve_time_s,setup_time_s,energy_j,converged\n";
        for (const auto& r : results) {
            if (!r.error_msg.empty()) continue;
            csv << r.name << ","
                << r.n << "," << r.nnz << ","
                << std::scientific << std::setprecision(4) << r.density << ","
                << r.is_spd << ","
                << static_cast<int>(r.final_state) << ","
                << r.iterations << "," << r.escalations << ","
                << r.rel_res << ","
                << std::fixed << r.solve_time_s << "," << r.setup_time_s << ","
                << r.energy_j << "," << r.converged << "\n";
        }
        std::cout << "\nResults written to: dataset_results.csv\n";
    }

    return 0;
}
