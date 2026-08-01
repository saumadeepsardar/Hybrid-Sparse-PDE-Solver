#pragma once

// =============================================================================
// roofline_model.hpp  —  Hardware-calibrated roofline energy model (Thrust 2)
//
// The roofline model (Williams, Waterman & Patterson, 2009) characterises
// each kernel by its arithmetic intensity I = FLOPs / bytes:
//
//   E_kernel = max( FLOPs / peak_GFLOPS,  bytes / peak_GB_s ) × P_idle
//            + FLOPs × α_flop  +  bytes × α_mem(cache_level)
//
// For sparse kernels:
//   SpMV arithmetic intensity: ~0.125 FLOP/byte for CSR double
//     → always memory-bandwidth bound on modern CPUs
//   BLAS-1 (dot, axpy): ~0.25–0.5 FLOP/byte → also memory-bound
//
// Per-kernel coefficients are calibrated by micro-benchmark (scripts/calibrate_energy.py)
// and stored in energy_coeffs.json. The RooflineModel loads this JSON at startup.
//
// Theoretical minimum energy (E_min) for a linear system
// -------------------------------------------------------
// For SPD A with condition number κ, tolerance ε, matrix of size n×n with nnz
// non-zeros, the CG iteration lower bound (Voevodin 1983, Shewchuk 1994) gives:
//
//   K_min = ceil( 0.5 * log(2/ε) / log( (√κ+1)/(√κ-1) ) )
//
//   E_min = K_min × E_spmv(nnz) + K_min × E_blas1(n)
//
// where E_spmv and E_blas1 are computed from the roofline model.
// =============================================================================

#include "../core/types.hpp"
#include <string>
#include <cmath>
#include <stdexcept>

namespace hsps {

// ---------------------------------------------------------------------------
// Per-platform calibrated coefficients
// ---------------------------------------------------------------------------
struct RooflineCoeffs {
    // Energy per operation
    double alpha_flop_spmv  = 2.0e-10;  ///< J/FLOP for SpMV kernel
    double alpha_flop_blas1 = 1.5e-10;  ///< J/FLOP for dot/axpy
    double alpha_flop_amg   = 2.5e-10;  ///< J/FLOP for AMG setup/apply

    // Energy per byte at each cache level
    double alpha_mem_l1     = 5.0e-12;  ///< J/byte (L1 cache hit)
    double alpha_mem_l2     = 5.0e-11;  ///< J/byte (L2 hit)
    double alpha_mem_l3     = 5.0e-10;  ///< J/byte (L3 hit)
    double alpha_mem_dram   = 5.0e-9;   ///< J/byte (DRAM access)

    // MPI communication energy
    double alpha_comm       = 1.0e-8;   ///< J/byte (network communication)

    // Platform idle power (watts)
    double idle_watts       = 50.0;

    // Peak throughputs (used for roofline ceiling)
    double peak_gflops_s    = 100.0;    ///< GFLOPs/s (single socket)
    double peak_gb_s_dram   = 50.0;     ///< GB/s DRAM bandwidth
    double peak_gb_s_l3     = 200.0;    ///< GB/s L3 bandwidth

    // Effective cache sizes (bytes)
    double l3_size_bytes    = 40e6;     ///< 40 MB L3 (typical)
    double l2_size_bytes    = 2e6;      ///< 2 MB L2

    bool calibrated = false;
    std::string platform_name = "default";

    /// Load from JSON file produced by scripts/calibrate_energy.py
    static RooflineCoeffs from_json(const std::string& path);

    /// Save to JSON file
    void to_json(const std::string& path) const;

    /// Returns hardcoded defaults for a modern server CPU (Intel Xeon Platinum)
    static RooflineCoeffs server_defaults();

    /// Returns hardcoded defaults for a consumer CPU (Intel Core i9)
    static RooflineCoeffs desktop_defaults();
};

// ---------------------------------------------------------------------------
// Roofline energy model — one instance per platform
// ---------------------------------------------------------------------------
class RooflineModel {
public:
    explicit RooflineModel(const RooflineCoeffs& coeffs = {})
        : c_(coeffs) {}

    // ------------------------------------------------------------------
    // Per-kernel energy estimates
    // ------------------------------------------------------------------

    /// Energy for one SpMV (y = A x), CSR format, nnz non-zeros, n rows
    double energy_spmv(long long nnz, Index n) const;

    /// Energy for one dot product: x · y of length n
    double energy_dot(Index n) const;

    /// Energy for one axpy: y += a*x, length n
    double energy_axpy(Index n) const;

    /// Energy for one preconditioner apply: M^{-1} r
    /// precond_type: 0=Jacobi, 1=ILU, 2=AMG (per-v-cycle estimate)
    double energy_precond_apply(int precond_type, long long nnz, Index n,
                                 int amg_levels = 3) const;

    // ------------------------------------------------------------------
    // Full Krylov iteration energy estimates
    // ------------------------------------------------------------------

    /// One CG iteration: SpMV + 3 BLAS-1 ops + precond apply
    double energy_cg_iter(long long nnz, Index n, int precond_type) const;

    /// One FGMRES inner iteration (Arnoldi step j): SpMV + j dot products + axpy
    double energy_fgmres_iter(long long nnz, Index n, int j,
                               int precond_type) const;

    // ------------------------------------------------------------------
    // Theoretical minimum energy — T1 of the research proposal
    // ------------------------------------------------------------------

    /// Minimum energy for CG to solve Ax=b with condition κ to tolerance ε.
    /// Implements the Voevodin/Shewchuk iteration lower bound.
    double energy_min_cg(double kappa, double epsilon,
                          long long nnz, Index n) const;

    /// Minimum CG iteration count (from the same bound)
    static int min_cg_iterations(double kappa, double epsilon);

    // ------------------------------------------------------------------
    // Memory model helpers
    // ------------------------------------------------------------------

    /// Select appropriate α_mem based on data size
    double alpha_mem(long long data_bytes) const;

    /// Arithmetic intensity of SpMV on CSR: ~2*nnz / (rows*8 + nnz*12) bytes
    double spmv_arithmetic_intensity(long long nnz, Index n) const;

    const RooflineCoeffs& coeffs() const { return c_; }

private:
    RooflineCoeffs c_;
};

} // namespace hsps
