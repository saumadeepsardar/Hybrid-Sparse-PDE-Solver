#pragma once

// =============================================================================
// logger.hpp  —  Structured logging with severity levels
// =============================================================================

#include <iostream>
#include <sstream>
#include <string>
#include <mutex>
#include <chrono>

namespace hsps {

enum class LogLevel { DEBUG, INFO, WARN, ERROR };

class Logger {
public:
    static Logger& instance();

    void set_level(LogLevel l) { level_ = l; }
    void set_stream(std::ostream& os) { out_ = &os; }
    void set_prefix(const std::string& p) { prefix_ = p; }

    template <typename... Args>
    void log(LogLevel l, Args&&... args) {
        if (l < level_) return;
        std::lock_guard<std::mutex> lock(mtx_);
        *out_ << "[" << level_str(l) << "] ";
        if (!prefix_.empty()) *out_ << "[" << prefix_ << "] ";
        print_all(*out_, std::forward<Args>(args)...);
        *out_ << "\n";
    }

    template <typename... Args> void debug(Args&&... a) { log(LogLevel::DEBUG, std::forward<Args>(a)...); }
    template <typename... Args> void info (Args&&... a) { log(LogLevel::INFO,  std::forward<Args>(a)...); }
    template <typename... Args> void warn (Args&&... a) { log(LogLevel::WARN,  std::forward<Args>(a)...); }
    template <typename... Args> void error(Args&&... a) { log(LogLevel::ERROR, std::forward<Args>(a)...); }

private:
    Logger() = default;
    static const char* level_str(LogLevel l) {
        switch (l) {
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO:  return " INFO";
            case LogLevel::WARN:  return " WARN";
            case LogLevel::ERROR: return "ERROR";
            default:              return "?????";
        }
    }

    template <typename T>
    void print_all(std::ostream& os, T&& t) { os << t; }
    template <typename T, typename... Rest>
    void print_all(std::ostream& os, T&& t, Rest&&... rest) {
        os << t;
        print_all(os, std::forward<Rest>(rest)...);
    }

    std::ostream* out_   = &std::cout;
    LogLevel      level_ = LogLevel::INFO;
    std::string   prefix_;
    std::mutex    mtx_;
};

#define HSPS_LOG_DEBUG(...) ::hsps::Logger::instance().debug(__VA_ARGS__)
#define HSPS_LOG_INFO(...)  ::hsps::Logger::instance().info (__VA_ARGS__)
#define HSPS_LOG_WARN(...)  ::hsps::Logger::instance().warn (__VA_ARGS__)
#define HSPS_LOG_ERROR(...) ::hsps::Logger::instance().error(__VA_ARGS__)

} // namespace hsps
