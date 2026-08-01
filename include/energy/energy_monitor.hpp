#pragma once

// =============================================================================
// energy_monitor.hpp  —  Energy consumption proxy model + RAPL interface
//
// Two-level design
// ----------------
//   1. Proxy model: estimate energy from FLOP count + memory traffic
//        E = α·FLOP + β·MEM_BYTES
//      Coefficients α, β calibrated for a typical server CPU.
//      This works on any platform (no root required).
//
//   2. RAPL (Linux/Intel): if available, read hardware energy counters.
//      Falls back gracefully to proxy model when unavailable.
//
// Usage
// -----
//   EnergyMonitor mon;
//   mon.start();
//   ... do work ...
//   auto sample = mon.sample();       // returns joules, watts, elapsed_s
//   mon.record_flops(count, bytes);   // called by solver after each iteration
// =============================================================================

#include "../core/types.hpp"
#include <string>
#include <vector>
#include <chrono>
#include <fstream>

namespace hsps {

struct EnergySample {
    double elapsed_s     = 0.0;
    double energy_joules = 0.0;  ///< hardware (RAPL) or proxy
    double avg_watts     = 0.0;
    long long flops      = 0;
    long long mem_bytes  = 0;
    bool   hw_available  = false;
};

// ---------------------------------------------------------------------------
// Energy model coefficients (tuneable — see calibrate())
// ---------------------------------------------------------------------------
struct EnergyModelCoeffs {
    // Per-kernel energy coefficients (Thrust 2 — roofline-calibrated)
    double alpha_flop       = 2.0e-10;   ///< ~200 pJ/FLOP (SpMV, generic)
    double alpha_flop_spmv  = 2.0e-10;   ///< SpMV-specific FLOP coefficient
    double alpha_flop_blas1 = 1.5e-10;   ///< BLAS-1 FLOP coefficient
    double alpha_mem        = 5.0e-9;    ///< ~5 nJ/byte (DRAM, generic)
    double alpha_mem_l3     = 5.0e-10;   ///< L3 cache coefficient
    double alpha_mem_dram   = 5.0e-9;    ///< DRAM coefficient
    double alpha_comm       = 1.0e-8;    ///< MPI communication J/byte
    double idle_watts       = 50.0;      ///< Idle platform power
    bool   calibrated       = false;     ///< true after calibrate() is run
};

// ---------------------------------------------------------------------------
// Lightweight per-iteration statistics accumulated inside the monitor
// ---------------------------------------------------------------------------
struct IterRecord {
    int       iteration;
    double    residual;
    // Original aggregate fields
    long long flops;
    long long mem_bytes;
    double    energy_proxy_j;
    double    elapsed_s;
    // Thrust 2 — per-kernel breakdown
    long long flops_spmv      = 0;
    long long flops_blas1     = 0;
    long long bytes_spmv      = 0;
    long long bytes_blas1     = 0;
    double    energy_spmv_j   = 0.0;
    double    energy_blas1_j  = 0.0;
    double    energy_precond_j= 0.0;
    double    rapl_j_delta    = 0.0;  ///< Hardware RAPL reading for this iteration
};

// ---------------------------------------------------------------------------
// EnergyMonitor
// ---------------------------------------------------------------------------
class EnergyMonitor {
public:
    explicit EnergyMonitor(const EnergyModelCoeffs& coeffs = {});

    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------
    void start();
    void stop();
    void reset();

    // ------------------------------------------------------------------
    // Per-iteration recording (called by solvers)
    // ------------------------------------------------------------------
    void record(int iteration, double residual,
                long long flops, long long mem_bytes);

    /// Thrust 2: record with per-kernel breakdown and hardware RAPL delta.
    /// Call snapshot_rapl() before and after each iteration body.
    void record_with_hw(int iteration, double residual,
                        long long flops_spmv, long long bytes_spmv,
                        long long flops_blas1, long long bytes_blas1,
                        double rapl_start_j, double rapl_end_j);

    /// Return current RAPL reading in joules (0.0 if RAPL unavailable).
    /// Lightweight: reads a memory-mapped register, ~10 ns on Intel.
    double snapshot_rapl() const;

    // ------------------------------------------------------------------
    // Query
    // ------------------------------------------------------------------
    EnergySample sample() const;
    double elapsed_seconds()  const;
    double total_energy_j()   const;   ///< RAPL if available, else proxy
    double total_energy_proxy_j() const;

    // Accumulated totals since start()
    long long total_flops()    const { return total_flops_; }
    long long total_mem_bytes()const { return total_mem_bytes_; }

    bool hw_energy_available() const { return rapl_available_; }

    const std::vector<IterRecord>& history() const { return history_; }

    // ------------------------------------------------------------------
    // Reporting
    // ------------------------------------------------------------------
    void print_summary(std::ostream& os) const;
    void dump_csv(const std::string& path) const;

    // ------------------------------------------------------------------
    // Calibration helper  (populates coeffs from a micro-benchmark)
    // ------------------------------------------------------------------
    static EnergyModelCoeffs calibrate();

private:
    // RAPL helpers (Linux only)
    bool        init_rapl();
    double      read_rapl_joules() const;

    EnergyModelCoeffs coeffs_;
    bool              rapl_available_ = false;
    std::string       rapl_pkg_path_;   ///< /sys/class/powercap/…/energy_uj
    mutable double    rapl_start_j_ = 0.0;

    using Clock    = std::chrono::high_resolution_clock;
    using TimePoint= std::chrono::time_point<Clock>;

    TimePoint   start_time_;
    bool        running_ = false;

    long long   total_flops_     = 0;
    long long   total_mem_bytes_ = 0;

    std::vector<IterRecord> history_;
};

// ---------------------------------------------------------------------------
// SpMV FLOP / memory cost calculators
// ---------------------------------------------------------------------------
namespace flop_model {
    /// SpMV on A: 2*nnz FLOPs, (rows+1+nnz)*sizeof(Index) + nnz*sizeof(Real) bytes
    inline long long spmv_flops(int nnz)      { return 2LL * nnz; }
    inline long long spmv_bytes(int rows, int nnz) {
        return (long long)(rows + 1 + nnz) * sizeof(Index)
             + (long long)nnz              * sizeof(Real)
             + (long long)rows             * sizeof(Real) * 2; // read x, write y
    }
    /// Dot product: 2*n FLOPs, 2*n*sizeof(Real) bytes
    inline long long dot_flops(int n)  { return 2LL * n; }
    inline long long dot_bytes(int n)  { return 2LL * n * sizeof(Real); }
    /// axpy: 2*n FLOPs, 3*n*sizeof(Real) bytes
    inline long long axpy_flops(int n) { return 2LL * n; }
    inline long long axpy_bytes(int n) { return 3LL * n * sizeof(Real); }
}

} // namespace hsps
