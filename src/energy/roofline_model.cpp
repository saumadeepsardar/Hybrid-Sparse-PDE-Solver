// =============================================================================
// roofline_model.cpp  —  RooflineModel and RooflineCoeffs implementations
// =============================================================================

#include "../../include/energy/roofline_model.hpp"
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <iomanip>

namespace hsps {

// ---------------------------------------------------------------------------
// RooflineCoeffs platform defaults
// ---------------------------------------------------------------------------
RooflineCoeffs RooflineCoeffs::server_defaults() {
    RooflineCoeffs c;
    c.platform_name    = "Intel Xeon Platinum (estimated)";
    c.alpha_flop_spmv  = 2.5e-10;
    c.alpha_flop_blas1 = 1.8e-10;
    c.alpha_mem_dram   = 6.0e-9;
    c.alpha_mem_l3     = 6.0e-10;
    c.peak_gflops_s    = 200.0;
    c.peak_gb_s_dram   = 100.0;
    c.l3_size_bytes    = 80e6;   // typical 2-socket: 2 × 40 MB
    c.idle_watts       = 80.0;
    c.calibrated       = false;
    return c;
}

RooflineCoeffs RooflineCoeffs::desktop_defaults() {
    RooflineCoeffs c;
    c.platform_name    = "Intel Core i9 (estimated)";
    c.alpha_flop_spmv  = 2.0e-10;
    c.alpha_flop_blas1 = 1.5e-10;
    c.alpha_mem_dram   = 5.0e-9;
    c.alpha_mem_l3     = 5.0e-10;
    c.peak_gflops_s    = 100.0;
    c.peak_gb_s_dram   = 50.0;
    c.l3_size_bytes    = 40e6;
    c.idle_watts       = 50.0;
    c.calibrated       = false;
    return c;
}

// ---------------------------------------------------------------------------
// JSON load/save (minimal hand-rolled parser — no external dependency)
// ---------------------------------------------------------------------------
RooflineCoeffs RooflineCoeffs::from_json(const std::string& path) {
    RooflineCoeffs c = desktop_defaults();
    std::ifstream f(path);
    if (!f.good()) {
        std::cerr << "[RooflineCoeffs] Cannot open '" << path
                  << "', using desktop defaults.\n";
        return c;
    }
    // Very simple key-value parsing (expects: "key": value per line)
    std::string line;
    auto parse_double = [](const std::string& s) {
        size_t p = s.find(':');
        if (p == std::string::npos) return 0.0;
        return std::stod(s.substr(p + 1));
    };
    while (std::getline(f, line)) {
        if (line.find("alpha_flop_spmv")  != std::string::npos) c.alpha_flop_spmv  = parse_double(line);
        if (line.find("alpha_flop_blas1") != std::string::npos) c.alpha_flop_blas1 = parse_double(line);
        if (line.find("alpha_mem_dram")   != std::string::npos) c.alpha_mem_dram   = parse_double(line);
        if (line.find("alpha_mem_l3")     != std::string::npos) c.alpha_mem_l3     = parse_double(line);
        if (line.find("peak_gflops_s")    != std::string::npos) c.peak_gflops_s    = parse_double(line);
        if (line.find("peak_gb_s_dram")   != std::string::npos) c.peak_gb_s_dram   = parse_double(line);
        if (line.find("l3_size_bytes")    != std::string::npos) c.l3_size_bytes    = parse_double(line);
        if (line.find("idle_watts")       != std::string::npos) c.idle_watts       = parse_double(line);
        if (line.find("alpha_comm")       != std::string::npos) c.alpha_comm       = parse_double(line);
    }
    c.calibrated = true;
    return c;
}

void RooflineCoeffs::to_json(const std::string& path) const {
    std::ofstream f(path);
    if (!f.good()) { std::cerr << "[RooflineCoeffs] Cannot write to '" << path << "'\n"; return; }
    f << std::scientific << std::setprecision(6);
    f << "{\n"
      << "  \"platform_name\":    \"" << platform_name << "\",\n"
      << "  \"calibrated\":       " << (calibrated ? "true" : "false") << ",\n"
      << "  \"alpha_flop_spmv\":  " << alpha_flop_spmv  << ",\n"
      << "  \"alpha_flop_blas1\": " << alpha_flop_blas1 << ",\n"
      << "  \"alpha_flop_amg\":   " << alpha_flop_amg   << ",\n"
      << "  \"alpha_mem_l1\":     " << alpha_mem_l1     << ",\n"
      << "  \"alpha_mem_l2\":     " << alpha_mem_l2     << ",\n"
      << "  \"alpha_mem_l3\":     " << alpha_mem_l3     << ",\n"
      << "  \"alpha_mem_dram\":   " << alpha_mem_dram   << ",\n"
      << "  \"alpha_comm\":       " << alpha_comm       << ",\n"
      << "  \"idle_watts\":       " << idle_watts       << ",\n"
      << "  \"peak_gflops_s\":    " << peak_gflops_s    << ",\n"
      << "  \"peak_gb_s_dram\":   " << peak_gb_s_dram   << ",\n"
      << "  \"peak_gb_s_l3\":     " << peak_gb_s_l3     << ",\n"
      << "  \"l3_size_bytes\":    " << l3_size_bytes    << ",\n"
      << "  \"l2_size_bytes\":    " << l2_size_bytes    << "\n"
      << "}\n";
}

// ---------------------------------------------------------------------------
// Memory level selection
// ---------------------------------------------------------------------------
double RooflineModel::alpha_mem(long long data_bytes) const {
    if (data_bytes <= static_cast<long long>(c_.l2_size_bytes)) return c_.alpha_mem_l2;
    if (data_bytes <= static_cast<long long>(c_.l3_size_bytes)) return c_.alpha_mem_l3;
    return c_.alpha_mem_dram;
}

// ---------------------------------------------------------------------------
// Arithmetic intensity of CSR SpMV
// ---------------------------------------------------------------------------
double RooflineModel::spmv_arithmetic_intensity(long long nnz, Index n) const {
    // FLOPs = 2*nnz (one multiply, one add per non-zero)
    // Bytes = (n+1 + nnz) * 4  (row_ptr int32 + col_idx int32)
    //       + nnz * 8            (values double)
    //       + n * 8              (x vector read)
    //       + n * 8              (y vector write)
    long long bytes = static_cast<long long>(n + 1 + nnz) * 4
                    + nnz * 8 + 2LL * n * 8;
    return (bytes > 0) ? (2.0 * nnz / bytes) : 0.0;
}

// ---------------------------------------------------------------------------
// Per-kernel energy estimates
// ---------------------------------------------------------------------------
double RooflineModel::energy_spmv(long long nnz, Index n) const {
    // SpMV is almost always memory-bandwidth bound (I < ridge_point)
    long long bytes_idx = static_cast<long long>(n + 1 + nnz) * sizeof(int);
    long long bytes_val = nnz * sizeof(double);
    long long bytes_vec = 2LL * n * sizeof(double);  // read x, write y
    long long total_bytes = bytes_idx + bytes_val + bytes_vec;

    double am = alpha_mem(total_bytes);
    double e_mem  = am * static_cast<double>(total_bytes);
    double e_flop = c_.alpha_flop_spmv * 2.0 * nnz;
    return std::max(e_mem, e_flop);
}

double RooflineModel::energy_dot(Index n) const {
    long long bytes = 2LL * n * sizeof(double);  // read x and y
    return std::max(alpha_mem(bytes) * bytes,
                    c_.alpha_flop_blas1 * 2.0 * n);
}

double RooflineModel::energy_axpy(Index n) const {
    long long bytes = 3LL * n * sizeof(double);  // read x, read y, write y
    return std::max(alpha_mem(bytes) * bytes,
                    c_.alpha_flop_blas1 * 2.0 * n);
}

double RooflineModel::energy_precond_apply(int precond_type, long long nnz,
                                             Index n, int amg_levels) const {
    switch (precond_type) {
    case 0: // Jacobi: z = D^{-1} r  — one vector scale
        return energy_axpy(n);
    case 1: // ILU(0): forward + backward substitution
        // Each triangular solve is ~nnz/2 multiply-adds, fully sequential
        // Cost ≈ 2 × (nnz/2 FLOPs + nnz*8 bytes)
        return c_.alpha_flop_blas1 * nnz
             + alpha_mem(nnz * sizeof(double)) * nnz * sizeof(double);
    case 2: // AMG: amg_levels V-cycles
        // Each level i: SpMV(nnz_i ≈ nnz * coarsen^i) + axpy
        // coarsen ≈ 4 for 2D problems
        {
            double e = 0.0;
            double nnz_i = static_cast<double>(nnz);
            double n_i   = static_cast<double>(n);
            for (int lv = 0; lv < amg_levels && n_i > 50; ++lv) {
                e     += energy_spmv(static_cast<long long>(nnz_i),
                                     static_cast<Index>(n_i));
                e     += energy_axpy(static_cast<Index>(n_i)) * 4; // smoothing
                nnz_i *= 0.25;
                n_i   *= 0.25;
            }
            return e * 2.0;  // V-cycle = down + up
        }
    default:
        return energy_axpy(n);
    }
}

double RooflineModel::energy_cg_iter(long long nnz, Index n,
                                      int precond_type) const {
    return energy_spmv(nnz, n)
         + energy_dot(n)          // p^T q
         + energy_axpy(n)         // x += alpha*p
         + energy_axpy(n)         // r -= alpha*q
         + energy_dot(n)          // r^T z (for rho update)
         + energy_precond_apply(precond_type, nnz, n);
}

double RooflineModel::energy_fgmres_iter(long long nnz, Index n, int j,
                                          int precond_type) const {
    return energy_spmv(nnz, n)
         + static_cast<double>(j + 1) * energy_dot(n)  // Arnoldi dots
         + static_cast<double>(j + 2) * energy_axpy(n) // MGS + update
         + energy_precond_apply(precond_type, nnz, n);
}

// ---------------------------------------------------------------------------
// Theoretical minimum CG iterations (Voevodin 1983 / Shewchuk 1994)
// ---------------------------------------------------------------------------
int RooflineModel::min_cg_iterations(double kappa, double epsilon) {
    if (kappa <= 1.0 + 1e-10) return 1;
    double sqrt_kappa = std::sqrt(kappa);
    double ratio = (sqrt_kappa + 1.0) / (sqrt_kappa - 1.0);
    if (ratio <= 1.0 + 1e-10) return 1;
    double K = 0.5 * std::log(2.0 / epsilon) / std::log(ratio);
    return std::max(1, static_cast<int>(std::ceil(K)));
}

// ---------------------------------------------------------------------------
// E_min: theoretical minimum energy for Krylov solve
//
// Theorem (T1 of research proposal):
//   For SPD A with condition κ, tolerance ε, n DOFs, nnz non-zeros:
//
//   E_min ≥ K_min(κ,ε) × E_spmv(nnz,n)   + K_min(κ,ε) × E_blas1(n)
//
//   where K_min = ⌈ (1/2) log(2/ε) / log((√κ+1)/(√κ−1)) ⌉
//
//   Proof sketch:
//   (1) No Krylov method solves in fewer than K_min iterations on SPD systems
//       (Voevodin 1983: CG is optimal over Krylov subspaces for SPD).
//   (2) Each Krylov iteration requires at least 1 SpMV (otherwise the
//       generated subspace has dimension < iteration count, contradiction).
//   (3) Each SpMV requires at least E_spmv(nnz,n) energy (reading nnz
//       matrix entries is necessary; FLOP lower bound = 2*nnz).
//   ∴ E_total ≥ K_min × E_spmv(nnz,n) + K_min × O(n) for vector operations.
// ---------------------------------------------------------------------------
double RooflineModel::energy_min_cg(double kappa, double epsilon,
                                     long long nnz, Index n) const {
    int K_min = min_cg_iterations(kappa, epsilon);

    // Per-iteration minimum: 1 SpMV + minimum BLAS-1 (dot + axpy)
    double e_spmv  = energy_spmv(nnz, n);
    double e_blas1 = energy_dot(n) + energy_axpy(n);  // minimum overhead

    return static_cast<double>(K_min) * (e_spmv + e_blas1);
}

} // namespace hsps
