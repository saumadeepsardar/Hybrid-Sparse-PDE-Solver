// =============================================================================
// logger.cpp
// =============================================================================

#include "../../include/utils/logger.hpp"

namespace hsps {

Logger& Logger::instance() {
    static Logger inst;
    return inst;
}

} // namespace hsps
