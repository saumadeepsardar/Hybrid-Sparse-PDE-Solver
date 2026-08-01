#pragma once

// =============================================================================
// timer.hpp  —  RAII high-resolution wall-clock timer
// =============================================================================

#include <chrono>
#include <string>
#include <iostream>

namespace hsps {

class Timer {
public:
    using Clock     = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    using Duration  = std::chrono::duration<double>;

    Timer() { reset(); }

    void start()  { start_ = Clock::now(); running_ = true; }
    void stop()   { end_   = Clock::now(); running_ = false; accum_ += elapsed_now(); }
    void reset()  { running_ = false; accum_ = Duration::zero(); }

    /// Elapsed seconds since start() (running or stopped)
    double elapsed() const {
        if (running_) return (accum_ + elapsed_now()).count();
        return accum_.count();
    }

    /// Restart: stop + reset + start
    void restart() { reset(); start(); }

private:
    Duration elapsed_now() const {
        return std::chrono::duration_cast<Duration>(Clock::now() - start_);
    }

    TimePoint start_;
    TimePoint end_;
    Duration  accum_  = Duration::zero();
    bool      running_ = false;
};

// ---------------------------------------------------------------------------
// ScopedTimer: prints elapsed time on destruction
// ---------------------------------------------------------------------------
class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& label, std::ostream& os = std::cout)
        : label_(label), os_(os) { t_.start(); }

    ~ScopedTimer() {
        t_.stop();
        os_ << "[Timer] " << label_ << " : "
            << t_.elapsed() << " s\n";
    }

private:
    std::string   label_;
    std::ostream& os_;
    Timer         t_;
};

} // namespace hsps
