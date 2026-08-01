#pragma once
#include <memory>

// =============================================================================
// nvml_monitor.hpp  —  NVIDIA Management Library (NVML) power monitor
//
// Provides real-time GPU power measurement for the heterogeneous execution
// router (Thrust 4). Falls back to zero-readings when NVML is unavailable
// so the rest of the codebase compiles on non-NVIDIA platforms.
//
// Compile with:  make BACKEND=CUDA NVML=1
//
// Typical usage
// -------------
//   NVMLMonitor mon(device_index);
//   mon.start();
//   // ... GPU work ...
//   double power_w  = mon.current_power_w();   // instantaneous watts
//   double energy_j = mon.elapsed_energy_j();  // joules since start()
// =============================================================================

#include <string>
#include <vector>
#include <cstdint>

namespace hsps {

struct GPUPowerSample {
    double elapsed_s;
    double power_w;
    double energy_j_cumulative;
};

class NVMLMonitor {
public:
    explicit NVMLMonitor(int device_index = 0);
    ~NVMLMonitor();

    // Non-copyable
    NVMLMonitor(const NVMLMonitor&)            = delete;
    NVMLMonitor& operator=(const NVMLMonitor&) = delete;

    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------
    bool start();   ///< Initialise NVML and begin sampling. Returns false if NVML unavailable.
    void stop();
    void reset();

    // ------------------------------------------------------------------
    // Readings
    // ------------------------------------------------------------------
    double current_power_w()   const;  ///< Instantaneous power in watts
    double elapsed_energy_j()  const;  ///< Total energy since start(), in joules
    double gpu_utilisation_pct() const;///< GPU utilisation (0–100 %)
    double memory_used_mb()    const;  ///< Used GPU memory in MB
    double temperature_c()     const;  ///< GPU temperature in Celsius
    double power_limit_w()     const;  ///< TDP / power limit in watts
    double power_headroom_w()  const;  ///< power_limit - current_power

    bool   available()         const { return nvml_available_; }
    int    device_index()      const { return device_index_;   }
    std::string device_name()  const { return device_name_;    }

    // ------------------------------------------------------------------
    // Sampling history
    // ------------------------------------------------------------------
    void                           take_sample();
    const std::vector<GPUPowerSample>& history() const { return history_; }
    void                           dump_csv(const std::string& path) const;

private:
    int         device_index_;
    bool        nvml_available_ = false;
    std::string device_name_;
    double      start_energy_j_ = 0.0;

    std::vector<GPUPowerSample> history_;

    // Opaque NVML handle (avoids nvml.h in header)
    struct NVMLImpl;
    std::unique_ptr<NVMLImpl> impl_;

    double read_energy_j() const;
};

} // namespace hsps
