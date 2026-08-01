// =============================================================================
// parallel_context.cpp
// =============================================================================

#include "../../include/parallel/parallel_context.hpp"
#include <iostream>
#include <stdexcept>
#include <iomanip>

#ifdef _OPENMP
#  include <omp.h>
#endif

#ifdef HSPS_USE_MPI
#  include <mpi.h>
#endif

#ifdef HSPS_USE_CUDA
#  include <cuda_runtime.h>
#  include <cublas_v2.h>
#  include <cusparse.h>
#endif

namespace hsps {

ParallelContext* ParallelContext::instance_ = nullptr;

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------
ParallelContext ParallelContext::init(int argc, char** argv,
                                      ParallelConfig config) {
    ParallelContext ctx;
    ctx.cfg_ = config;

    if (config.use_omp())  ctx.init_omp();
    if (config.use_mpi())  ctx.init_mpi(argc, argv);
    if (config.use_cuda()) ctx.init_cuda();

    // Store as global singleton
    static ParallelContext global_ctx = std::move(ctx);
    instance_ = &global_ctx;
    return std::move(*instance_);   // caller gets a reference-equivalent copy
}

ParallelContext& ParallelContext::get() {
    if (!instance_)
        throw std::runtime_error("ParallelContext::get() called before init()");
    return *instance_;
}

// ---------------------------------------------------------------------------
// OMP init
// ---------------------------------------------------------------------------
void ParallelContext::init_omp() {
#ifdef _OPENMP
    if (cfg_.omp_threads > 0)
        omp_set_num_threads(cfg_.omp_threads);
    cfg_.omp_threads = omp_get_max_threads();
#else
    cfg_.omp_threads = 1;
#endif
}

// ---------------------------------------------------------------------------
// MPI init
// ---------------------------------------------------------------------------
void ParallelContext::init_mpi(int argc, char** argv) {
#ifdef HSPS_USE_MPI
    int already = 0;
    MPI_Initialized(&already);
    if (!already) {
        // Request thread-multiple for hybrid MPI+OMP
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        owns_mpi_ = true;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &cfg_.mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &cfg_.mpi_size);
    cfg_.mpi_inited = true;
#else
    (void)argc; (void)argv;
    cfg_.mpi_rank   = 0;
    cfg_.mpi_size   = 1;
    cfg_.mpi_inited = false;
    if (cfg_.is_root())
        std::cerr << "[ParallelContext] WARNING: MPI requested but not compiled in "
                     "(HSPS_USE_MPI not set). Running as single process.\n";
#endif
}

// ---------------------------------------------------------------------------
// CUDA init
// ---------------------------------------------------------------------------
void ParallelContext::init_cuda() {
#ifdef HSPS_USE_CUDA
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count == 0)
        throw std::runtime_error("ParallelContext: no CUDA devices found");

    // For MPI+CUDA: assign device by rank (round-robin over available GPUs)
    cfg_.cuda_device = cfg_.mpi_rank % dev_count;
    cudaSetDevice(cfg_.cuda_device);

    cublasCreate  (&cublas_handle_);
    cusparseCreate(&cusparse_handle_);

    cfg_.cuda_inited = true;
    owns_cuda_       = true;

    if (cfg_.is_root()) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, cfg_.cuda_device);
        std::cout << "[ParallelContext] CUDA device " << cfg_.cuda_device
                  << ": " << prop.name
                  << "  " << (prop.totalGlobalMem >> 20) << " MB\n";
    }
#else
    cfg_.cuda_inited = false;
    std::cerr << "[ParallelContext] WARNING: CUDA requested but not compiled in "
                 "(HSPS_USE_CUDA not set). Falling back to OMP.\n";
    cfg_.backends = cfg_.backends & ~BackendFlags::CUDA | BackendFlags::OMP;
#endif
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
ParallelContext::~ParallelContext() {
    if (owns_cuda_)  finalize_cuda();
    if (owns_mpi_)   finalize_mpi();
}

void ParallelContext::finalize_mpi() {
#ifdef HSPS_USE_MPI
    int fin = 0;
    MPI_Finalized(&fin);
    if (!fin) MPI_Finalize();
#endif
}

void ParallelContext::finalize_cuda() {
#ifdef HSPS_USE_CUDA
    if (cublas_handle_)   cublasDestroy(cublas_handle_);
    if (cusparse_handle_) cusparseDestroy(cusparse_handle_);
    cudaDeviceReset();
#endif
}

// ---------------------------------------------------------------------------
// Move
// ---------------------------------------------------------------------------
ParallelContext::ParallelContext(ParallelContext&& o) noexcept
    : cfg_(o.cfg_), owns_mpi_(o.owns_mpi_), owns_cuda_(o.owns_cuda_)
#ifdef HSPS_USE_CUDA
    , cublas_handle_(o.cublas_handle_)
    , cusparse_handle_(o.cusparse_handle_)
#endif
{
    o.owns_mpi_  = false;
    o.owns_cuda_ = false;
#ifdef HSPS_USE_CUDA
    o.cublas_handle_   = nullptr;
    o.cusparse_handle_ = nullptr;
#endif
}

ParallelContext& ParallelContext::operator=(ParallelContext&& o) noexcept {
    if (this != &o) {
        if (owns_cuda_) finalize_cuda();
        if (owns_mpi_)  finalize_mpi();
        cfg_       = o.cfg_;
        owns_mpi_  = o.owns_mpi_;
        owns_cuda_ = o.owns_cuda_;
        o.owns_mpi_  = false;
        o.owns_cuda_ = false;
#ifdef HSPS_USE_CUDA
        cublas_handle_   = o.cublas_handle_;
        cusparse_handle_ = o.cusparse_handle_;
        o.cublas_handle_   = nullptr;
        o.cusparse_handle_ = nullptr;
#endif
    }
    return *this;
}

// ---------------------------------------------------------------------------
// Collective operations
// ---------------------------------------------------------------------------
void ParallelContext::barrier() const {
#ifdef HSPS_USE_MPI
    if (cfg_.mpi_inited) MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef _OPENMP
    // OMP barrier only valid inside parallel region
#endif
}

double ParallelContext::allreduce_sum(double local_val) const {
#ifdef HSPS_USE_MPI
    if (cfg_.mpi_inited && cfg_.mpi_size > 1) {
        double result = 0.0;
        MPI_Allreduce(&local_val, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return result;
    }
#endif
    return local_val;
}

void ParallelContext::allreduce_sum(double* data, int count) const {
#ifdef HSPS_USE_MPI
    if (cfg_.mpi_inited && cfg_.mpi_size > 1)
        MPI_Allreduce(MPI_IN_PLACE, data, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
    (void)data; (void)count;
#endif
}

// ---------------------------------------------------------------------------
// OMP thread count
// ---------------------------------------------------------------------------
int ParallelContext::omp_threads() const {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// ---------------------------------------------------------------------------
// print_info
// ---------------------------------------------------------------------------
void ParallelContext::print_info(std::ostream& os) const {
    os << "=== ParallelContext ===\n"
       << "  OpenMP : " << (compiled_with_omp()  ? "YES" : "NO")
       << "  threads=" << cfg_.omp_threads << "\n"
       << "  MPI    : " << (compiled_with_mpi()  ? "YES" : "NO")
       << "  rank="     << cfg_.mpi_rank
       << "/"           << cfg_.mpi_size << "\n"
       << "  CUDA   : " << (compiled_with_cuda() ? "YES" : "NO")
       << "  device="   << cfg_.cuda_device << "\n"
       << "======================\n";
}

} // namespace hsps
