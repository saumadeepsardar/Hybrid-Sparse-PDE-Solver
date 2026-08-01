#pragma once

// =============================================================================
// test_framework.hpp  —  Minimal single-header unit test harness
// =============================================================================

#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <cmath>
#include <stdexcept>

namespace hsps_test {

struct TestCase {
    std::string              name;
    std::function<void()>    fn;
};

struct TestRegistry {
    static TestRegistry& instance() {
        static TestRegistry r;
        return r;
    }
    void add(const std::string& name, std::function<void()> fn) {
        cases_.push_back({name, fn});
    }
    int run_all() {
        int passed = 0, failed = 0;
        std::cout << "\n=== Running " << cases_.size() << " test(s) ===\n";
        for (const auto& tc : cases_) {
            try {
                tc.fn();
                std::cout << "  [PASS] " << tc.name << "\n";
                ++passed;
            } catch (const std::exception& e) {
                std::cout << "  [FAIL] " << tc.name << "\n         " << e.what() << "\n";
                ++failed;
            } catch (...) {
                std::cout << "  [FAIL] " << tc.name << "\n         (unknown exception)\n";
                ++failed;
            }
        }
        std::cout << "=== " << passed << " passed, " << failed << " failed ===\n\n";
        return failed;
    }
    std::vector<TestCase> cases_;
};

// ---------------------------------------------------------------------------
// Registration macro
// ---------------------------------------------------------------------------
#define HSPS_TEST(name) \
    static void _test_fn_##name(); \
    static const bool _reg_##name = (::hsps_test::TestRegistry::instance().add(#name, _test_fn_##name), true); \
    static void _test_fn_##name()

// ---------------------------------------------------------------------------
// Assertion helpers
// ---------------------------------------------------------------------------
inline void expect_true(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error("expect_true failed: " + msg);
}

inline void expect_near(double a, double b, double tol, const std::string& msg) {
    if (std::abs(a - b) > tol)
        throw std::runtime_error("expect_near failed (" + std::to_string(a)
                                 + " vs " + std::to_string(b)
                                 + ", tol=" + std::to_string(tol)
                                 + "): " + msg);
}

inline void expect_less(double a, double b, const std::string& msg) {
    if (!(a < b))
        throw std::runtime_error("expect_less failed (" + std::to_string(a)
                                 + " >= " + std::to_string(b)
                                 + "): " + msg);
}

} // namespace hsps_test
