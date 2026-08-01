// =============================================================================
// nvml_monitor.cpp
// =============================================================================

#include "../../include/parallel/nvml_monitor.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <cmath>
#include <memory>

#ifdef HSPS_USE_NVML
#  include <nvml.h>
#endif

namespace hsps {

// ---------------------------------------------------------------------------
// Opaque NVML state
// ---------------------------------------------------------------------------
struct NVMLMonitor::NVMLImpl {
#ifdef HSPS_USE_NVML
    nvmlDevice_t      device_handle = nullptr;
    bool              inited        = false;
#endif
    using Clock     = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    TimePoint start_time;
};

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
NVMLMonitor::NVMLMonitor(int device_index)
    : device_index_(device_index)
    , impl_(std::make_unique<NVMLImpl>()) {

#ifdef HSPS_USE_NVML
    nvmlReturn_t ret = nvmlInit();
    if (ret != NVML_SUCCESS) {
        std::cerr << "[NVMLMonitor] nvmlInit failed: "
                  << nvmlErrorString(ret) << "\n";
        return;
    }
    ret = nvmlDeviceGetHandleByIndex(device_index, &impl_->device_handle);
    if (ret != NVML_SUCCESS) {
        std::cerr << "[NVMLMonitor] nvmlDeviceGetHandleByIndex(" << device_index
                  << ") failed: " << nvmlErrorString(ret) << "\n";
        return;
    }
    char name[NVML_DEVICE_NAME_BUFFER_SIZE];
    nvmlDeviceGetName(impl_->device_handle, name, sizeof(name));
    device_name_    = std::string(name);
    nvml_available_ = true;
    impl_->inited   = true;
#else
    (void)device_index;
#endif
}

NVMLMonitor::~NVMLMonitor() {
#ifdef HSPS_USE_NVML
    if (impl_ && impl_->inited) nvmlShutdown();
#endif
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------
bool NVMLMonitor::start() {
    if (!nvml_available_) return false;
    impl_->start_time = NVMLImpl::Clock::now();
    start_energy_j_   = read_energy_j();
    history_.clear();
    return true;
}

void NVMLMonitor::stop()  { take_sample(); }
void NVMLMonitor::reset() { history_.clear(); start_energy_j_ = read_energy_j(); }

// ---------------------------------------------------------------------------
// Energy reading (milli-joules from NVML)
// ---------------------------------------------------------------------------
double NVMLMonitor::read_energy_j() const {
#ifdef HSPS_USE_NVML
    if (!nvml_available_ || !impl_->device_handle) return 0.0;
    unsigned long long energy_mj = 0;
    nvmlReturn_t ret = nvmlDeviceGetTotalEnergyConsumption(
        impl_->device_handle, &energy_mj);
    if (ret == NVML_SUCCESS)
        return static_cast<double>(energy_mj) * 1e-3;  // mJ → J
    // Fallback: integrate power readings
    unsigned int power_mw = 0;
    nvmlDeviceGetPowerUsage(impl_->device_handle, &power_mw);
    return 0.0;  // can't integrate without time reference here
#else
    return 0.0;
#endif
}

// ---------------------------------------------------------------------------
// Readings
// ---------------------------------------------------------------------------
double NVMLMonitor::current_power_w() const {
#ifdef HSPS_USE_NVML
    if (!nvml_available_) return 0.0;
    unsigned int power_mw = 0;
    nvmlDeviceGetPowerUsage(impl_->device_handle, &power_mw);
    return static_cast<double>(power_mw) * 1e-3;  // mW → W
#else
    return 0.0;
#endif
}

double NVMLMonitor::elapsed_energy_j() const {
    double now_j = read_energy_j();
    return std::max(0.0, now_j - start_energy_j_);
}

double NVMLMonitor::gpu_utilisation_pct() const {
#ifdef HSPS_USE_NVML
    if (!nvml_available_) return 0.0;
    nvmlUtilization_t util;
    nvmlDeviceGetUtilizationRates(impl_->device_handle, &util);
    return static_cast<double>(util.gpu);
#else
    return 0.0;
#endif
}

double NVMLMonitor::memory_used_mb() const {
#ifdef HSPS_USE_NVML
    if (!nvml_available_) return 0.0;
    nvmlMemory_t mem;
    nvmlDeviceGetMemoryInfo(impl_->device_handle, &mem);
    return static_cast<double>(mem.used) / (1024.0 * 1024.0);
#else
    return 0.0;
#endif
}

double NVMLMonitor::temperature_c() const {
#ifdef HSPS_USE_NVML
    if (!nvml_available_) return 0.0;
    unsigned int temp = 0;
    nvmlDeviceGetTemperature(impl_->device_handle, NVML_TEMPERATURE_GPU, &temp);
    return static_cast<double>(temp);
#else
    return 0.0;
#endif
}

double NVMLMonitor::power_limit_w() const {
#ifdef HSPS_USE_NVML
    if (!nvml_available_) return 0.0;
    unsigned int limit_mw = 0;
    nvmlDeviceGetEnforcedPowerLimit(impl_->device_handle, &limit_mw);
    return static_cast<double>(limit_mw) * 1e-3;
#else
    return 0.0;
#endif
}

double NVMLMonitor::power_headroom_w() const {
    return power_limit_w() - current_power_w();
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------
void NVMLMonitor::take_sample() {
    auto now = NVMLImpl::Clock::now();
    double elapsed = std::chrono::duration<double>(
        now - impl_->start_time).count();
    GPUPowerSample s;
    s.elapsed_s          = elapsed;
    s.power_w            = current_power_w();
    s.energy_j_cumulative= elapsed_energy_j();
    history_.push_back(s);
}

void NVMLMonitor::dump_csv(const std::string& path) const {
    std::ofstream f(path);
    if (!f.good()) return;
    f << "elapsed_s,power_w,energy_j_cumulative\n";
    for (const auto& s : history_)
        f << std::fixed << std::setprecision(6) << s.elapsed_s << ","
          << std::setprecision(3) << s.power_w << ","
          << std::setprecision(6) << s.energy_j_cumulative << "\n";
}

} // namespace hsps
