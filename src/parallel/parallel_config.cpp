// =============================================================================
// parallel_config.cpp
// =============================================================================

#include "../../include/parallel/parallel_config.hpp"
#include <algorithm>
#include <sstream>
#include <cctype>

namespace hsps {

BackendFlags ParallelConfig::parse(const std::string& s) {
    std::string u = s;
    std::transform(u.begin(), u.end(), u.begin(),
                   [](unsigned char c){ return std::toupper(c); });

    if (u == "NONE")       return BackendFlags::NONE;
    if (u == "OMP")        return BackendFlags::OMP;
    if (u == "MPI")        return BackendFlags::OMP | BackendFlags::MPI;
    if (u == "CUDA")       return BackendFlags::OMP | BackendFlags::CUDA;
    if (u == "MPI_CUDA" || u == "CUDA_MPI")
                           return BackendFlags::OMP | BackendFlags::MPI | BackendFlags::CUDA;
    // Allow raw combinations
    BackendFlags result = BackendFlags::NONE;
    if (u.find("OMP")  != std::string::npos) result = result | BackendFlags::OMP;
    if (u.find("MPI")  != std::string::npos) result = result | BackendFlags::MPI;
    if (u.find("CUDA") != std::string::npos) result = result | BackendFlags::CUDA;
    return result;
}

std::string ParallelConfig::describe() const {
    std::ostringstream oss;
    oss << "ParallelConfig {";
    if (use_omp())  oss << " OMP(threads=" << omp_threads << ")";
    if (use_mpi())  oss << " MPI(rank=" << mpi_rank << "/" << mpi_size << ")";
    if (use_cuda()) oss << " CUDA(device=" << cuda_device << ")";
    if (!use_omp() && !use_mpi() && !use_cuda()) oss << " SERIAL";
    oss << " }";
    return oss.str();
}

} // namespace hsps
