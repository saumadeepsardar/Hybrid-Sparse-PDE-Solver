// =============================================================================
// cuda_backend.cpp
//
// When HSPS_USE_CUDA is defined: real cuBLAS/cuSPARSE implementation.
// When not defined: stub that prints a warning and delegates to OMPBackend.
// =============================================================================

#include "../../include/parallel/cuda_backend.hpp"
#include "../../include/parallel/omp_backend.hpp"
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <cmath>

#ifdef HSPS_USE_CUDA
#  include <cuda_runtime.h>
#  include <cublas_v2.h>
#  include <cusparse.h>
#endif

namespace hsps {

// ===========================================================================
// DeviceVector
// ===========================================================================
DeviceVector::DeviceVector(Index n_, Real init_val) : n(n_) {
#ifdef HSPS_USE_CUDA
    cudaMalloc(&d_data, n * sizeof(Real));
    // Fill with init_val via thrust or a simple kernel
    // For now use cudaMemset for zero, or upload a host vector
    if (init_val == 0.0) {
        cudaMemset(d_data, 0, n * sizeof(Real));
    } else {
        std::vector<Real> tmp(n, init_val);
        cudaMemcpy(d_data, tmp.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
    }
#else
    (void)n_; (void)init_val;
    throw std::runtime_error("DeviceVector: CUDA not compiled in");
#endif
}

DeviceVector::~DeviceVector() {
#ifdef HSPS_USE_CUDA
    if (d_data) cudaFree(d_data);
#endif
}

DeviceVector::DeviceVector(DeviceVector&& o) noexcept
    : d_data(o.d_data), n(o.n) { o.d_data = nullptr; o.n = 0; }

DeviceVector& DeviceVector::operator=(DeviceVector&& o) noexcept {
    if (this != &o) {
#ifdef HSPS_USE_CUDA
        if (d_data) cudaFree(d_data);
#endif
        d_data = o.d_data; n = o.n;
        o.d_data = nullptr; o.n = 0;
    }
    return *this;
}

void DeviceVector::upload(const Vector& host_vec) {
#ifdef HSPS_USE_CUDA
    if (n != host_vec.size()) {
        if (d_data) cudaFree(d_data);
        n = host_vec.size();
        cudaMalloc(&d_data, n * sizeof(Real));
    }
    cudaMemcpy(d_data, host_vec.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
#else
    (void)host_vec;
#endif
}

void DeviceVector::download(Vector& host_vec) const {
#ifdef HSPS_USE_CUDA
    host_vec.resize(n);
    cudaMemcpy(host_vec.data(), d_data, n * sizeof(Real), cudaMemcpyDeviceToHost);
#else
    (void)host_vec;
#endif
}

void DeviceVector::fill(Real val) {
#ifdef HSPS_USE_CUDA
    if (val == 0.0) { cudaMemset(d_data, 0, n * sizeof(Real)); return; }
    std::vector<Real> tmp(n, val);
    cudaMemcpy(d_data, tmp.data(), n * sizeof(Real), cudaMemcpyHostToDevice);
#else
    (void)val;
#endif
}

// ===========================================================================
// DeviceSparseMatrix
// ===========================================================================
DeviceSparseMatrix::~DeviceSparseMatrix() {
#ifdef HSPS_USE_CUDA
    if (d_row_ptr) cudaFree(d_row_ptr);
    if (d_col_idx) cudaFree(d_col_idx);
    if (d_values)  cudaFree(d_values);
    if (spmv_buffer) cudaFree(spmv_buffer);
    if (mat_descr)   cusparseDestroySpMat(mat_descr);
    if (vec_x_descr) cusparseDestroyDnVec(vec_x_descr);
    if (vec_y_descr) cusparseDestroyDnVec(vec_y_descr);
#endif
}

void DeviceSparseMatrix::upload(const SparseMatrix& A,
                                 const ParallelContext& ctx) {
#ifdef HSPS_USE_CUDA
    rows = A.rows(); cols = A.cols(); nnz = A.nnz();

    // Upload CSR arrays
    cudaMalloc(&d_row_ptr, (rows + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, nnz        * sizeof(int));
    cudaMalloc(&d_values,  nnz        * sizeof(double));

    cudaMemcpy(d_row_ptr, A.row_ptr().data(),
               (rows+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, A.col_idx().data(),
               nnz*sizeof(int),      cudaMemcpyHostToDevice);
    cudaMemcpy(d_values,  A.values().data(),
               nnz*sizeof(double),   cudaMemcpyHostToDevice);

    // Create cuSPARSE sparse matrix descriptor
    cusparseCreateCsr(&mat_descr,
                      rows, cols, nnz,
                      d_row_ptr, d_col_idx, d_values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    // Allocate dummy x/y descriptors for buffer size query
    double* tmp_x; double* tmp_y;
    cudaMalloc(&tmp_x, cols * sizeof(double));
    cudaMalloc(&tmp_y, rows * sizeof(double));
    cusparseCreateDnVec(&vec_x_descr, cols, tmp_x, CUDA_R_64F);
    cusparseCreateDnVec(&vec_y_descr, rows, tmp_y, CUDA_R_64F);

    double alpha = 1.0, beta = 0.0;
    cusparseSpMV_bufferSize(
        ctx.cusparse_handle(),
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat_descr, vec_x_descr,
        &beta,  vec_y_descr,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
        &spmv_buffer_size);

    if (spmv_buffer_size > 0)
        cudaMalloc(&spmv_buffer, spmv_buffer_size);

    cusparseDestroyDnVec(vec_x_descr); vec_x_descr = nullptr;
    cusparseDestroyDnVec(vec_y_descr); vec_y_descr = nullptr;
    cudaFree(tmp_x); cudaFree(tmp_y);
#else
    (void)A; (void)ctx;
#endif
}

// ===========================================================================
// CUDABackend
// ===========================================================================
CUDABackend::CUDABackend(const ParallelContext& ctx) : ctx_(ctx) {
#ifndef HSPS_USE_CUDA
    std::cerr << "[CUDABackend] WARNING: compiled without CUDA. "
                 "All operations will fall back to OMP.\n";
#endif
}

void CUDABackend::sync() const {
#ifdef HSPS_USE_CUDA
    cudaDeviceSynchronize();
#endif
}

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------
void CUDABackend::cuda_check(int code, const char* file, int line) {
#ifdef HSPS_USE_CUDA
    if (code != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error at ") + file + ":" +
                                 std::to_string(line) + " — " +
                                 cudaGetErrorString(static_cast<cudaError_t>(code)));
#else
    (void)code; (void)file; (void)line;
#endif
}

void CUDABackend::cublas_check(int code, const char* file, int line) {
#ifdef HSPS_USE_CUDA
    if (code != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cuBLAS error at ") + file + ":" +
                                 std::to_string(line) + " code=" + std::to_string(code));
#else
    (void)code; (void)file; (void)line;
#endif
}

void CUDABackend::cusparse_check(int code, const char* file, int line) {
#ifdef HSPS_USE_CUDA
    if (code != CUSPARSE_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cuSPARSE error at ") + file + ":" +
                                 std::to_string(line) + " code=" + std::to_string(code));
#else
    (void)code; (void)file; (void)line;
#endif
}

// ---------------------------------------------------------------------------
// Alloc
// ---------------------------------------------------------------------------
void CUDABackend::alloc(Index n, Real val, Vector& out) const {
    out.resize(n, val);  // Host allocation; device allocation happens on upload
}

// ---------------------------------------------------------------------------
// Host-interface BLAS-1  (upload→operate→download)
// For use in setup code; hot-path code should use device-native overloads.
// ---------------------------------------------------------------------------
Real CUDABackend::dot(const Vector& x, const Vector& y) const {
#ifdef HSPS_USE_CUDA
    DeviceVector dx, dy;
    dx.upload(x); dy.upload(y);
    return dot_device(dx, dy);
#else
    OMPBackend omp;
    return omp.dot(x, y);
#endif
}

Real CUDABackend::nrm2(const Vector& x) const {
#ifdef HSPS_USE_CUDA
    DeviceVector dx;
    dx.upload(x);
    return nrm2_device(dx);
#else
    OMPBackend omp;
    return omp.nrm2(x);
#endif
}

void CUDABackend::axpy(Real a, const Vector& x, Vector& y) const {
#ifdef HSPS_USE_CUDA
    DeviceVector dx, dy;
    dx.upload(x); dy.upload(y);
    axpy_device(a, dx, dy);
    dy.download(y);
#else
    OMPBackend omp;
    omp.axpy(a, x, y);
#endif
}

void CUDABackend::axpby(Real a, const Vector& x, Real b, Vector& y) const {
#ifdef HSPS_USE_CUDA
    DeviceVector dx, dy;
    dx.upload(x); dy.upload(y);
    axpby_device(a, dx, b, dy);
    dy.download(y);
#else
    OMPBackend omp;
    omp.axpby(a, x, b, y);
#endif
}

void CUDABackend::copy(const Vector& src, Vector& dst) const {
#ifdef HSPS_USE_CUDA
    dst.resize(src.size());
    cudaMemcpy(const_cast<Real*>(dst.data()), src.data(),
               src.size() * sizeof(Real), cudaMemcpyHostToHost);
#else
    dst = src;
#endif
}

void CUDABackend::scale(Real a, Vector& x) const {
#ifdef HSPS_USE_CUDA
    DeviceVector dx;
    dx.upload(x);
    scale_device(a, dx);
    dx.download(x);
#else
    OMPBackend omp;
    omp.scale(a, x);
#endif
}

void CUDABackend::fill(Vector& x, Real val) const {
    x.fill(val);  // Host fill is sufficient here
}

// ---------------------------------------------------------------------------
// Device-native BLAS-1
// ---------------------------------------------------------------------------
Real CUDABackend::dot_device(const DeviceVector& x,
                              const DeviceVector& y) const {
#ifdef HSPS_USE_CUDA
    double result = 0.0;
    CUBLAS_CHECK(cublasDdot(ctx_.cublas_handle(),
                            x.n, x.d_data, 1, y.d_data, 1, &result));
    return static_cast<Real>(result);
#else
    (void)x; (void)y; return 0.0;
#endif
}

Real CUDABackend::nrm2_device(const DeviceVector& x) const {
#ifdef HSPS_USE_CUDA
    double result = 0.0;
    CUBLAS_CHECK(cublasDnrm2(ctx_.cublas_handle(), x.n, x.d_data, 1, &result));
    return static_cast<Real>(result);
#else
    (void)x; return 0.0;
#endif
}

void CUDABackend::axpy_device(Real a, const DeviceVector& x,
                               DeviceVector& y) const {
#ifdef HSPS_USE_CUDA
    double alpha = static_cast<double>(a);
    CUBLAS_CHECK(cublasDaxpy(ctx_.cublas_handle(),
                             x.n, &alpha, x.d_data, 1, y.d_data, 1));
#else
    (void)a; (void)x; (void)y;
#endif
}

void CUDABackend::axpby_device(Real a, const DeviceVector& x,
                                Real b, DeviceVector& y) const {
#ifdef HSPS_USE_CUDA
    // y = a*x + b*y  ==>  scale y by b, then axpy
    double bd = static_cast<double>(b);
    CUBLAS_CHECK(cublasDscal(ctx_.cublas_handle(), y.n, &bd, y.d_data, 1));
    double ad = static_cast<double>(a);
    CUBLAS_CHECK(cublasDaxpy(ctx_.cublas_handle(), x.n, &ad, x.d_data, 1, y.d_data, 1));
#else
    (void)a; (void)x; (void)b; (void)y;
#endif
}

void CUDABackend::copy_device(const DeviceVector& src, DeviceVector& dst) const {
#ifdef HSPS_USE_CUDA
    if (dst.n != src.n) {
        if (dst.d_data) cudaFree(dst.d_data);
        dst.n = src.n;
        cudaMalloc(&dst.d_data, dst.n * sizeof(Real));
    }
    cudaMemcpy(dst.d_data, src.d_data, src.n * sizeof(Real),
               cudaMemcpyDeviceToDevice);
#else
    (void)src; (void)dst;
#endif
}

void CUDABackend::scale_device(Real a, DeviceVector& x) const {
#ifdef HSPS_USE_CUDA
    double ad = static_cast<double>(a);
    CUBLAS_CHECK(cublasDscal(ctx_.cublas_handle(), x.n, &ad, x.d_data, 1));
#else
    (void)a; (void)x;
#endif
}

void CUDABackend::fill_device(DeviceVector& x, Real val) const {
#ifdef HSPS_USE_CUDA
    if (val == 0.0) { cudaMemset(x.d_data, 0, x.n * sizeof(Real)); return; }
    std::vector<Real> tmp(x.n, val);
    cudaMemcpy(x.d_data, tmp.data(), x.n * sizeof(Real), cudaMemcpyHostToDevice);
#else
    (void)x; (void)val;
#endif
}

// ---------------------------------------------------------------------------
// SpMV (host API: upload→spmv→download)
// ---------------------------------------------------------------------------
void CUDABackend::spmv(const SparseMatrix& A,
                        const Vector& x, Vector& y) const {
#ifdef HSPS_USE_CUDA
    DeviceSparseMatrix dA;
    DeviceVector dx, dy_d;
    dA.upload(A, ctx_);
    dx.upload(x);
    dy_d = DeviceVector(A.rows(), 0.0);
    spmv_device(dA, dx, dy_d);
    dy_d.download(y);
#else
    OMPBackend omp;
    omp.spmv(A, x, y);
#endif
}

void CUDABackend::spmv_device(const DeviceSparseMatrix& dA,
                               const DeviceVector& x,
                               DeviceVector& y) const {
#ifdef HSPS_USE_CUDA
    double alpha = 1.0, beta = 0.0;
    cusparseConstDnVecDescr_t vx;
    cusparseDnVecDescr_t      vy;
    cusparseCreateConstDnVec(&vx, x.n, x.d_data, CUDA_R_64F);
    cusparseCreateDnVec     (&vy, y.n, y.d_data, CUDA_R_64F);

    CUSPARSE_CHECK(cusparseSpMV(
        ctx_.cusparse_handle(),
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, dA.mat_descr, vx,
        &beta,  vy,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
        dA.spmv_buffer));

    cusparseDestroyConstDnVec(vx);
    cusparseDestroyDnVec(vy);
#else
    (void)dA; (void)x; (void)y;
#endif
}

void CUDABackend::upload_matrix(const SparseMatrix& A,
                                 DeviceSparseMatrix& dA) const {
    dA.upload(A, ctx_);
}

void CUDABackend::upload_vector(const Vector& v, DeviceVector& dv) const {
    dv.upload(v);
}

void CUDABackend::download_vector(const DeviceVector& dv, Vector& v) const {
    dv.download(v);
}

} // namespace hsps
