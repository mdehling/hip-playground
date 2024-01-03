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
    int tidm = threadIdx.y, ntidm = blockDim.y;
    int tidn = threadIdx.x, ntidn = blockDim.x;
    int ctaidm = blockIdx.y, nctaidm = gridDim.y;
    int ctaidn = blockIdx.x, nctaidn = gridDim.x;

    for (int mi = ctaidm*ntidm+tidm; mi < m; mi += ntidm*nctaidm) {
        for (int ni = ctaidn*ntidn+tidn; ni < n; ni += ntidn*nctaidn) {
            C[mi+ni*ldc] *= beta;
            for (int ki = 0; ki < k; ki++) {
                C[mi+ni*ldc] += alpha * A[mi+ki*lda] * B[ki+ni*ldb];
            };
        };
    };
}


template <int BLK_M=64, int BLK_N=64, int BLK_K=64, typename T>
__global__
void dev_gemm_v2(int m, int n, int k, T alpha, const T *A, int lda, const T *B, int ldb, T beta, T *C, int ldc) {
    int tidm = threadIdx.x, ntidm = blockDim.x;
    int tidn = threadIdx.y, ntidn = blockDim.y;
    int ctaidm = blockIdx.x, nctaidm = gridDim.x;
    int ctaidn = blockIdx.y, nctaidn = gridDim.y;

    //
    // NOTE: Keep in mind that 48kb is the maximum amount of static shared
    // memory. The default template parameters BLK_M = BLK_N = BLK_K = 64 work
    // for T = float: 3*(64*64*4b) = 48kb but need to be adjust for, e.g.,
    // matrices of doubles.
    //
    T __shared__ sA[BLK_M*BLK_K];
    T __shared__ sB[BLK_K*BLK_N];
    T __shared__ sC[BLK_M*BLK_N];

    // for output blocks [mi0:mi0+BLK_M,ni0:ni0+BLK_N]
    for (int mi0 = ctaidm*BLK_M; mi0 < m; mi0 += nctaidm*BLK_M) {
        for (int ni0 = ctaidn*BLK_N; ni0 < n; ni0 += nctaidn*BLK_N) {

            // colab init sC = beta*C[mi0:mi0+BLK_M,ni0:ni0+BLK_N]
            for (int mi_ = tidm; mi_ < BLK_M; mi_ += ntidm) {
                for (int ni_ = tidn; ni_ < BLK_N; ni_ += ntidn) {
                    int mi = mi0+mi_, ni = ni0+ni_;
                    if (mi < m && ni < n) {
                        sC[mi_+ni_*BLK_M] = beta * C[mi+ni*ldc];
                    } else {
                        sC[mi_+ni_*BLK_M] = 0;
                    };
                };
            };

            // for input blocks [mi0:mi0+BLK_M,ki0:ki0+BLK_K], [ki0:ki0+BLK_K,ni0:ni0+BLK_N]
            for (int ki0 = 0; ki0 < k; ki0 += BLK_K) {

                // colab copy sA = A[mi0:mi0+BLK_M,ki0:ki0+BLK_K]
                for (int mi_ = tidm; mi_ < BLK_M; mi_ += ntidm) {
                    for (int ki_ = tidn; ki_ < BLK_K; ki_ += ntidn) {
                        int mi = mi0+mi_, ki = ki0+ki_;
                        if (mi < m && ki < k) {
                            sA[mi_+ki_*BLK_M] = A[mi+ki*lda];
                        } else {
                            sA[mi_+ki_*BLK_M] = 0;
                        };
                    };
                };
                // colab copy sB = B[ki0:ki0+BLK_K,ni0:ni0+BLK_N]
                for (int ki_ = tidm; ki_ < BLK_K; ki_ += ntidm) {
                    for (int ni_ = tidn; ni_ < BLK_N; ni_ += ntidn) {
                        int ki = ki0+ki_, ni = ni0+ni_;
                        if (ki < k && ni < n) {
                            sB[ki_+ni_*BLK_K] = B[ki+ni*ldb];
                        } else {
                            sB[ki_+ni_*BLK_K] = 0;
                        };
                    };
                };

                __syncthreads();

                // colab compute sC += alpha * sA * sB
                for (int mi_ = tidm; mi_ < BLK_M; mi_ += ntidm) {
                    for (int ni_ = tidn; ni_ < BLK_N; ni_ += ntidn) {
                        T AB = 0;
                        for (int ki_ = 0; ki_ < BLK_K; ki_++) {
                            AB += sA[mi_+ki_*BLK_M] * sB[ki_+ni_*BLK_K];
                        };
                        sC[mi_+ni_*BLK_M] += alpha * AB;
                    };
                };

                __syncthreads();
            };

            // colab copy C[mi0:mi0+BLK_M,ni0:ni0+BLK_N] = sC
            for (int mi_ = tidm; mi_ < BLK_M; mi_ += ntidm) {
                for (int ni_ = tidn; ni_ < BLK_N; ni_ += ntidn) {
                    int mi = mi0+mi_, ni = ni0+ni_;
                    if (mi < m && ni < n) {
                        C[mi+ni*ldc] = sC[mi_+ni_*BLK_M];
                    };
                };
            };

            __syncthreads();
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

    std::cout << "Generating random matrices A(m,k), B(k,n), C(m,n) on host..." << std::endl;
    int m = 1<<10, n = 1<<10, k = 1<<10;
    std::cout << "(m = " << m << ", n = " << n << ", k = " << k << ")" << std::endl;
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
    dim3 group_dim_v1{32,32};
    dim3 grid_dim_v1{64,64};
    t0 = clock::now();
    dev_gemm_v1<<<grid_dim_v1,group_dim_v1>>>(m, n, k, alpha, dA, m, dB, k, beta, dD, m);
    HIP_CHECK_ERROR(hipGetLastError());
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    t1 = clock::now();
    HIP_CHECK_ERROR(hipMemcpy(D3.data(), dD, m*n * sizeof(decltype(D3)::value_type), hipMemcpyDeviceToHost));
    std::cout << "alpha*A*B + beta*C = " << D3 << std::endl;
    std::cout << "max_sqerr = " << host_max_squared_error(m, n, D1.data(), m, D3.data(), m) << std::endl;
    std::cout << "dt = " << ms_cast(t1-t0).count() << "ms" << std::endl;
    std::cout << std::endl;

    std::cout << "Performing SGEMM (device v2)..." << std::endl;
    std::vector<float> D4(m*n);
    HIP_CHECK_ERROR(hipMemcpy(dD, dC, m*n * sizeof *dD, hipMemcpyDeviceToDevice));
    // FIXME: It should be possible to run groups of 32*32 threads, but trying
    // to do so results in a non-desript error below ("hip error: no error").
    dim3 group_dim_v2{16,16};
    dim3 grid_dim_v2{64,64};
    t0 = clock::now();
    dev_gemm_v2<<<grid_dim_v2,group_dim_v2>>>(m, n, k, alpha, dA, m, dB, k, beta, dD, m);
    HIP_CHECK_ERROR(hipGetLastError());
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    t1 = clock::now();
    HIP_CHECK_ERROR(hipMemcpy(D4.data(), dD, m*n * sizeof(decltype(D4)::value_type), hipMemcpyDeviceToHost));
    std::cout << "alpha*A*B + beta*C = " << D4 << std::endl;
    std::cout << "max_sqerr = " << host_max_squared_error(m, n, D1.data(), m, D4.data(), m) << std::endl;
    std::cout << "dt = " << ms_cast(t1-t0).count() << "ms" << std::endl;
    std::cout << std::endl;

    std::cout << "Freeing device memory..." << std::endl;
    HIP_CHECK_ERROR(hipFree(dA));
    HIP_CHECK_ERROR(hipFree(dB));
    HIP_CHECK_ERROR(hipFree(dC));
    HIP_CHECK_ERROR(hipFree(dD));
}

