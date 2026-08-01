// =============================================================================
// test_energy.cpp  —  Unit tests for EnergyMonitor and flop model
// =============================================================================

#include "test_framework.hpp"
#include "../include/energy/energy_monitor.hpp"
#include "../include/core/sparse_matrix.hpp"
#include "../include/core/vector.hpp"
#include <cmath>
#include <thread>
#include <chrono>
#include <fstream>
#include <sstream>

using namespace hsps;
using namespace hsps_test;

// ---------------------------------------------------------------------------
HSPS_TEST(energy_monitor_elapsed_increases) {
    EnergyMonitor mon;
    mon.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    double t = mon.elapsed_seconds();
    mon.stop();
    expect_true(t >= 0.01, "elapsed time >= 10 ms");
}

HSPS_TEST(energy_monitor_proxy_accumulates) {
    EnergyMonitor mon;
    mon.start();
    mon.record(1, 1.0, 1000, 4000);
    mon.record(2, 0.5, 2000, 8000);
    mon.stop();

    expect_true(mon.total_flops()     == 3000, "flops accumulated");
    expect_true(mon.total_mem_bytes() == 12000, "mem_bytes accumulated");
    expect_true(mon.total_energy_proxy_j() > 0.0, "proxy energy positive");
}

HSPS_TEST(energy_monitor_history_entries) {
    EnergyMonitor mon;
    mon.start();
    for (int i = 0; i < 5; ++i)
        mon.record(i, 1.0 / (i + 1), 1000, 4000);
    mon.stop();

    expect_true(mon.history().size() == 5, "5 history entries");
    expect_true(mon.history()[0].iteration == 0, "first entry iter=0");
    expect_true(mon.history()[4].iteration == 4, "last entry iter=4");
}

HSPS_TEST(energy_monitor_reset_clears) {
    EnergyMonitor mon;
    mon.start();
    mon.record(1, 0.5, 1000, 4000);
    mon.stop();
    mon.reset();

    expect_true(mon.total_flops()     == 0, "flops cleared");
    expect_true(mon.total_mem_bytes() == 0, "mem_bytes cleared");
    expect_true(mon.history().empty(),       "history cleared");
}

HSPS_TEST(energy_monitor_sample_fields) {
    EnergyMonitor mon;
    mon.start();
    mon.record(1, 1.0, 1'000'000, 4'000'000);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    auto s = mon.sample();
    mon.stop();

    expect_true(s.flops    == 1'000'000,  "sample flops");
    expect_true(s.mem_bytes == 4'000'000, "sample mem_bytes");
    expect_true(s.elapsed_s >= 0.0,       "sample elapsed_s >= 0");
    expect_true(s.energy_joules > 0.0,    "sample energy > 0");
}

HSPS_TEST(energy_monitor_csv_dump) {
    EnergyMonitor mon;
    mon.start();
    mon.record(0, 1.0,   1000, 4000);
    mon.record(1, 0.5,   2000, 8000);
    mon.record(2, 0.25,  3000, 12000);
    mon.stop();

    const std::string path = "/tmp/hsps_energy_test.csv";
    mon.dump_csv(path);

    std::ifstream f(path);
    expect_true(f.good(), "CSV file created");

    std::string header;
    std::getline(f, header);
    expect_true(header.find("iteration") != std::string::npos, "CSV has header");

    int lines = 0;
    std::string line;
    while (std::getline(f, line)) ++lines;
    expect_true(lines == 3, "CSV has 3 data rows");
}

HSPS_TEST(flop_model_spmv) {
    int nnz = 1000, rows = 200;
    long long flops = flop_model::spmv_flops(nnz);
    long long bytes = flop_model::spmv_bytes(rows, nnz);
    expect_true(flops == 2000LL, "SpMV flops = 2*nnz");
    expect_true(bytes > 0,       "SpMV bytes > 0");
}

HSPS_TEST(flop_model_dot) {
    int n = 500;
    expect_true(flop_model::dot_flops(n) == 1000LL, "dot flops = 2*n");
    expect_true(flop_model::dot_bytes(n) == 2LL * n * sizeof(Real), "dot bytes");
}

HSPS_TEST(flop_model_axpy) {
    int n = 300;
    expect_true(flop_model::axpy_flops(n) == 600LL, "axpy flops = 2*n");
    expect_true(flop_model::axpy_bytes(n) == 3LL * n * sizeof(Real), "axpy bytes");
}

HSPS_TEST(energy_model_coeffs_positive) {
    EnergyModelCoeffs c;
    expect_true(c.alpha_flop > 0.0, "alpha_flop > 0");
    expect_true(c.alpha_mem  > 0.0, "alpha_mem > 0");
    expect_true(c.idle_watts > 0.0, "idle_watts > 0");
}

HSPS_TEST(energy_proxy_scales_with_problem_size) {
    // Larger problem → more FLOPs → higher estimated energy
    EnergyMonitor mon_small, mon_large;

    mon_small.start();
    mon_small.record(1, 0.1, 10'000, 40'000);
    mon_small.stop();

    mon_large.start();
    mon_large.record(1, 0.1, 1'000'000, 4'000'000);
    mon_large.stop();

    expect_less(mon_small.total_energy_proxy_j(),
                mon_large.total_energy_proxy_j(),
                "larger problem → higher proxy energy");
}

int main() {
    return ::hsps_test::TestRegistry::instance().run_all();
}
