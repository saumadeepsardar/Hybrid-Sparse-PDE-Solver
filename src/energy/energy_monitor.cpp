// =============================================================================
// energy_monitor.cpp  —  Energy tracking and RAPL interface
// =============================================================================

#include "../../include/energy/energy_monitor.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>

#ifdef __linux__
#  include <glob.h>
#endif

namespace hsps {

EnergyMonitor::EnergyMonitor(const EnergyModelCoeffs& coeffs)
    : coeffs_(coeffs) {
    rapl_available_ = init_rapl();
}

// ---------------------------------------------------------------------------
// RAPL initialisation (Linux only)
// ---------------------------------------------------------------------------
bool EnergyMonitor::init_rapl() {
#ifdef __linux__
    // Try to find Intel RAPL package-0 energy file
    const std::vector<std::string> candidates = {
        "/sys/class/powercap/intel-rapl:0/energy_uj",
        "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
    };
    for (const auto& path : candidates) {
        std::ifstream f(path);
        if (f.good()) {
            rapl_pkg_path_ = path;
            return true;
        }
    }
#endif
    return false;
}

double EnergyMonitor::read_rapl_joules() const {
#ifdef __linux__
    if (rapl_pkg_path_.empty()) return 0.0;
    std::ifstream f(rapl_pkg_path_);
    if (!f.good()) return 0.0;
    unsigned long long uj = 0;
    f >> uj;
    return static_cast<double>(uj) * 1e-6;  // micro-joules → joules
#else
    return 0.0;
#endif
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------
void EnergyMonitor::start() {
    start_time_      = Clock::now();
    running_         = true;
    total_flops_     = 0;
    total_mem_bytes_ = 0;
    history_.clear();
    if (rapl_available_) rapl_start_j_ = read_rapl_joules();
}

void EnergyMonitor::stop() {
    running_ = false;
}

void EnergyMonitor::reset() {
    running_         = false;
    total_flops_     = 0;
    total_mem_bytes_ = 0;
    history_.clear();
    rapl_start_j_    = 0.0;
}

// ---------------------------------------------------------------------------
// Per-iteration recording
// ---------------------------------------------------------------------------
void EnergyMonitor::record(int iteration, double residual,
                            long long flops, long long mem_bytes) {
    total_flops_     += flops;
    total_mem_bytes_ += mem_bytes;

    double elapsed = elapsed_seconds();
    double eproxy  = coeffs_.alpha_flop * flops + coeffs_.alpha_mem * mem_bytes;

    history_.push_back({iteration, residual, flops, mem_bytes, eproxy, elapsed});
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------
double EnergyMonitor::elapsed_seconds() const {
    auto now = Clock::now();
    return std::chrono::duration<double>(now - start_time_).count();
}

double EnergyMonitor::total_energy_proxy_j() const {
    return coeffs_.alpha_flop * static_cast<double>(total_flops_)
         + coeffs_.alpha_mem  * static_cast<double>(total_mem_bytes_);
}

double EnergyMonitor::total_energy_j() const {
    if (rapl_available_) {
        double rapl_now = read_rapl_joules();
        double delta    = rapl_now - rapl_start_j_;
        // Handle RAPL counter wraparound (max ~262 kJ for Intel)
        if (delta < 0) delta += 262144.0;
        return delta;
    }
    return total_energy_proxy_j();
}

EnergySample EnergyMonitor::sample() const {
    EnergySample s;
    s.elapsed_s    = elapsed_seconds();
    s.flops        = total_flops_;
    s.mem_bytes    = total_mem_bytes_;
    s.hw_available = rapl_available_;
    s.energy_joules= total_energy_j();
    s.avg_watts    = (s.elapsed_s > 1e-9) ? (s.energy_joules / s.elapsed_s) : 0.0;
    return s;
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------
void EnergyMonitor::print_summary(std::ostream& os) const {
    auto s = sample();
    os << "\n=== Energy Monitor Summary ===\n"
       << "  Elapsed time    : " << std::fixed << std::setprecision(4)
                                 << s.elapsed_s << " s\n"
       << "  Total FLOPs     : " << s.flops      << "\n"
       << "  Memory traffic  : " << s.mem_bytes / (1024*1024) << " MB\n"
       << "  Energy (proxy)  : " << std::setprecision(6)
                                 << total_energy_proxy_j() << " J\n";
    if (rapl_available_)
        os << "  Energy (RAPL)   : " << s.energy_joules << " J\n";
    os << "  Avg power       : " << std::setprecision(3)
                                 << s.avg_watts << " W\n"
       << "  HW energy avail : " << (rapl_available_ ? "YES" : "NO (proxy used)") << "\n"
       << "==============================\n";
}

void EnergyMonitor::dump_csv(const std::string& path) const {
    std::ofstream f(path);
    if (!f.good()) {
        std::cerr << "[EnergyMonitor] Cannot open " << path << "\n";
        return;
    }
    f << "iteration,residual,flops,mem_bytes,energy_proxy_j,elapsed_s\n";
    for (const auto& r : history_) {
        f << r.iteration << ","
          << std::scientific << std::setprecision(8) << r.residual << ","
          << r.flops << "," << r.mem_bytes << ","
          << r.energy_proxy_j << "," << r.elapsed_s << "\n";
    }
}

// calibrate() implemented in Thrust 2 section below

// ---------------------------------------------------------------------------
// Singleton logger helper (in logger.cpp)
// ---------------------------------------------------------------------------

} // namespace hsps

// =============================================================================
// Thrust 2 additions: snapshot_rapl(), record_with_hw(), calibrate()
// =============================================================================

namespace hsps {

double EnergyMonitor::snapshot_rapl() const {
    if (!rapl_available_) return 0.0;
    return read_rapl_joules();
}

void EnergyMonitor::record_with_hw(
        int iteration, double residual,
        long long flops_spmv,  long long bytes_spmv,
        long long flops_blas1, long long bytes_blas1,
        double rapl_start_j, double rapl_end_j) {

    long long total_flops = flops_spmv + flops_blas1;
    long long total_bytes = bytes_spmv + bytes_blas1;
    total_flops_     += total_flops;
    total_mem_bytes_ += total_bytes;

    double elapsed = elapsed_seconds();

    double e_spmv  = coeffs_.alpha_flop_spmv  * flops_spmv
                   + coeffs_.alpha_mem_dram    * bytes_spmv;
    double e_blas1 = coeffs_.alpha_flop_blas1 * flops_blas1
                   + coeffs_.alpha_mem_l3      * bytes_blas1;
    double e_total = e_spmv + e_blas1;
    double rapl_delta = (rapl_end_j > rapl_start_j) ? (rapl_end_j - rapl_start_j) : 0.0;

    IterRecord r;
    r.iteration       = iteration;
    r.residual        = residual;
    r.flops           = total_flops;
    r.mem_bytes       = total_bytes;
    r.energy_proxy_j  = e_total;
    r.elapsed_s       = elapsed;
    r.flops_spmv      = flops_spmv;
    r.flops_blas1     = flops_blas1;
    r.bytes_spmv      = bytes_spmv;
    r.bytes_blas1     = bytes_blas1;
    r.energy_spmv_j   = e_spmv;
    r.energy_blas1_j  = e_blas1;
    r.rapl_j_delta    = rapl_delta;
    history_.push_back(r);
}

EnergyModelCoeffs EnergyMonitor::calibrate() {
    EnergyModelCoeffs c;
    const size_t N = 8 * 1024 * 1024;
    std::vector<double> A(N, 1.0), B(N, 2.0), C(N, 0.5);
    const double scalar = 3.14159;

    EnergyMonitor mon;
    mon.init_rapl();

    auto t0 = Clock::now();
    double rapl0 = mon.rapl_available_ ? mon.read_rapl_joules() : 0.0;
    for (size_t i = 0; i < N; ++i) C[i] = A[i] + scalar * B[i];
    volatile double sink = C[N/2]; (void)sink;
    auto t1 = Clock::now();
    double rapl1 = mon.rapl_available_ ? mon.read_rapl_joules() : 0.0;

    double dt_s   = std::chrono::duration<double>(t1 - t0).count();
    double bytes  = 3.0 * N * sizeof(double);
    double gbs    = bytes / dt_s / 1e9;
    double e_hw   = (rapl1 > rapl0) ? (rapl1 - rapl0) : 0.0;
    double alpha_mem = (bytes > 0 && e_hw > 0) ? (e_hw / bytes) : 5.0e-9;

    c.alpha_mem       = alpha_mem;
    c.alpha_mem_dram  = alpha_mem;
    c.alpha_mem_l3    = alpha_mem * 0.1;

    const size_t Nd = 4 * 1024 * 1024;
    std::vector<double> X(Nd, 1.0/Nd), Y(Nd, 1.0/Nd);
    double dot_sum = 0.0;
    auto t2 = Clock::now();
    for (size_t i = 0; i < Nd; ++i) dot_sum += X[i] * Y[i];
    volatile double sink2 = dot_sum; (void)sink2;
    auto t3 = Clock::now();
    double dt_dot     = std::chrono::duration<double>(t3 - t2).count();
    double flops_dot  = 2.0 * Nd;
    double bytes_dot  = 2.0 * Nd * sizeof(double);
    c.alpha_flop      = alpha_mem * bytes_dot / flops_dot;
    c.alpha_flop_spmv = c.alpha_flop * 1.3;
    c.alpha_flop_blas1= c.alpha_flop;
    c.calibrated      = true;

    std::cout << "[EnergyMonitor::calibrate()] STREAM: " << gbs << " GB/s"
              << "  alpha_mem=" << std::scientific << alpha_mem << " J/byte"
              << "  dot: " << (flops_dot/dt_dot/1e9) << " GFLOP/s"
              << "  alpha_flop=" << c.alpha_flop << " J/FLOP\n";
    return c;
}

} // namespace hsps
