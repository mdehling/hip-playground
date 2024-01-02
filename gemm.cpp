#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <hip/hip_runtime_api.h>

#define HIP_CHECK_ERROR(error)                                              \
    if (error != hipSuccess) {                                              \
        std::cerr << __FILE__ << "(" << __LINE__ << ") hip error: "         \
            << hipGetErrorString(error) << std::endl;                       \
        exit(EXIT_FAILURE);                                                 \
    }

#include "blas.h"


//
// compute alpha * A[m,k] * B[k,n] + beta * C[m,n]
//
template <typename T>
void host_gemm(int m, int n, int k, T alpha, const T *A, int lda, const T *B, int ldb, T beta, T* C, int ldc) {
    for (int mi = 0; mi < m; mi++) {
        for (int ni = 0; ni < n; ni++) {
            C[mi+ni*ldc] *= beta;
            for (int ki = 0; ki < k; ki++) {
                C[mi+ni*ldc] += alpha * A[mi+ki*lda] * B[ki+ni*ldb];
            };
        };
    };
}


template <typename T>
T host_max_squared_error(int m, int n, const T *A, int lda, const T *B, int ldb) {
    T mse{0};
    for (int mi = 0; mi < m; mi++)
        for (int ni = 0; ni < n; ni++)
            mse = std::max<T>(mse, std::pow(A[mi+ni*lda] - B[mi+ni*ldb], 2));
    return mse;
}


template <typename T>
__global__
void dev_gemm_v1(int m, int n, int k, T alpha, const T *A, int lda, const T *B, int ldb, T beta, T *C, int ldc) {
    int tidx = threadIdx.x, ntidx = blockDim.x;
    int tidy = threadIdx.y, ntidy = blockDim.y;
    int ctaidx = blockIdx.x, nctaidx = gridDim.x;
    int ctaidy = blockIdx.y, nctaidy = gridDim.y;
    for (int mi = ctaidy*ntidy+tidy; mi < m; mi += ntidy*nctaidy) {
        for (int ni = ctaidx*ntidx+tidx; ni < n; ni += ntidx*nctaidx) {
            C[mi+ni*ldc] *= beta;
            for (int ki = 0; ki < k; ki++) {
                C[mi+ni*ldc] += alpha * A[mi+ki*lda] * B[ki+ni*ldb];
            };
        };
    };
}


template <int N = 3, typename T>
std::ostream &operator<<(std::ostream &os, std::vector<T> &v) {
    int i{0};
    os << "[";
    for (auto vi : v) {
        if (++i > N) {
            os << " ...";
            break;
        };
        os << " " << vi;
    };
    os << " ]";
    return os;
}


int main() {
    typedef std::chrono::high_resolution_clock clock;
    clock::time_point t0, t1;
    auto ms_cast = std::chrono::duration_cast<std::chrono::milliseconds,clock::rep,clock::period>;

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<float> dist{};
    auto std_normal = [&dist, &gen]() { return dist(gen); };

    std::cout << "Generating random matrices A, B, C on host..." << std::endl;
    int m = 1<<10, n = 1<<10, k = 1<<10;
    float alpha = std_normal(), beta = std_normal();
    std::vector<float> A(m*k), B(k*n), C(m*n);
    t0 = clock::now();
    std::generate(std::begin(A), std::end(A), std_normal);
    std::generate(std::begin(B), std::end(B), std_normal);
    std::generate(std::begin(C), std::end(C), std_normal);
    t1 = clock::now();
    std::cout << "alpha = " << alpha << ", beta = " << beta << std::endl;
    std::cout << "A = " << A << std::endl;
    std::cout << "B = " << B << std::endl;
    std::cout << "C = " << C << std::endl;
    std::cout << "dt = " << ms_cast(t1-t0).count() << "ms" << std::endl;
    std::cout << std::endl;

    std::cout << "Performing SGEMM (host)..." << std::endl;
    std::vector<float> D1(C);
    t0 = clock::now();
    host_gemm(m, n, k, alpha, A.data(), m, B.data(), k, beta, D1.data(), m);
    t1 = clock::now();
    std::cout << "alpha*A*B + beta*C = " << D1 << std::endl;
    std::cout << "dt = " << ms_cast(t1-t0).count() << "ms" << std::endl;
    std::cout << std::endl;

    std::cout << "Allocating device memory and copying vectors..." << std::endl;
    float *dA, *dB, *dC, *dD;
    t0 = clock::now();
    HIP_CHECK_ERROR(hipMalloc(&dA, m*k * sizeof *dA));
    HIP_CHECK_ERROR(hipMalloc(&dB, k*n * sizeof *dB));
    HIP_CHECK_ERROR(hipMalloc(&dC, m*n * sizeof *dC));
    HIP_CHECK_ERROR(hipMalloc(&dD, m*n * sizeof *dD));
    HIP_CHECK_ERROR(hipMemcpy(dA, A.data(), m*k * sizeof *dA, hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemcpy(dB, B.data(), k*n * sizeof *dB, hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemcpy(dC, C.data(), m*n * sizeof *dC, hipMemcpyHostToDevice));
    t1 = clock::now();
    std::cout << "dt = " << ms_cast(t1-t0).count() << "ms" << std::endl;
    std::cout << std::endl;

    std::cout << "Performing SGEMM (blas on device)..." << std::endl;
    std::vector<float> D2(m*n);
    HIP_CHECK_ERROR(hipMemcpy(dD, dC, m*n * sizeof *dD, hipMemcpyDeviceToDevice));
    blasHandle_t handle;
    BLAS_CHECK_ERROR(blasCreate(&handle));
    t0 = clock::now();
    BLAS_CHECK_ERROR(blasSgemm(handle, BLAS_OP_N, BLAS_OP_N, m, n, k, &alpha, dA, m, dB, k, &beta, dD, m));
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    t1 = clock::now();
    BLAS_CHECK_ERROR(blasDestroy(handle));
    HIP_CHECK_ERROR(hipMemcpy(D2.data(), dD, m*n * sizeof(decltype(D2)::value_type), hipMemcpyDeviceToHost));
    std::cout << "alpha*A*B + beta*C = " << D2 << std::endl;
    std::cout << "max_sqerr = " << host_max_squared_error(m, n, D1.data(), m, D2.data(), m) << std::endl;
    std::cout << "dt = " << ms_cast(t1-t0).count() << "ms" << std::endl;
    std::cout << std::endl;

    std::cout << "Performing SGEMM (device v1)..." << std::endl;
    std::vector<float> D3(m*n);
    HIP_CHECK_ERROR(hipMemcpy(dD, dC, m*n * sizeof *dD, hipMemcpyDeviceToDevice));
    dim3 group_dim{32,32};
    dim3 grid_dim{64,64};
    t0 = clock::now();
    dev_gemm_v1<<<grid_dim,group_dim>>>(m, n, k, alpha, dA, m, dB, k, beta, dD, m);
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    t1 = clock::now();
    HIP_CHECK_ERROR(hipMemcpy(D3.data(), dD, m*n * sizeof(decltype(D3)::value_type), hipMemcpyDeviceToHost));
    std::cout << "alpha*A*B + beta*C = " << D3 << std::endl;
    std::cout << "max_sqerr = " << host_max_squared_error(m, n, D1.data(), m, D3.data(), m) << std::endl;
    std::cout << "dt = " << ms_cast(t1-t0).count() << "ms" << std::endl;
    std::cout << std::endl;

    std::cout << "Freeing device memory..." << std::endl;
    HIP_CHECK_ERROR(hipFree(dA));
    HIP_CHECK_ERROR(hipFree(dB));
    HIP_CHECK_ERROR(hipFree(dC));
    HIP_CHECK_ERROR(hipFree(dD));
}

