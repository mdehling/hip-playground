#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>


#define HIP_CHECK_ERROR(error)                                              \
    if (error != hipSuccess) {                                              \
        std::cerr << __FILE__ << "(" << __LINE__ << ") hip error: "         \
            << hipGetErrorString(error) << std::endl;                       \
        exit(EXIT_FAILURE);                                                 \
    }

#define HIPBLAS_CHECK_ERROR(status)                                         \
    if (status != HIPBLAS_STATUS_SUCCESS) {                                 \
        std::cerr << __FILE__ << "(" << __LINE__ << ") hipblas error: "     \
            << hipblasStatusToString(status) << std::endl;                  \
        exit(EXIT_FAILURE);                                                 \
    }


template <typename T>
int axpy(int n, const T *alpha, const T *x, int incx, T *y, int incy) {
    for (int i = 0; i < n; i++)
        y[i*incy] += *alpha * x[i*incx];
    return 0;
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

    std::cout << "Generating random vectors x, y on host..." << std::endl;
    int size = 1<<24;
    float alpha = std_normal();
    std::vector<float> x(size), y(size);
    t0 = clock::now();
    std::generate(std::begin(x), std::end(x), std_normal);
    std::generate(std::begin(y), std::end(y), std_normal);
    t1 = clock::now();
    std::cout << "alpha = " << alpha << std::endl;
    std::cout << "x = " << x << std::endl;
    std::cout << "y = " << y << std::endl;
    std::cout << "dt = " << ms_cast(t1-t0).count() << "ms" << std::endl;
    std::cout << std::endl;

    std::cout << "Performing SAXPY (host)..." << std::endl;
    std::vector<float> z1(y);
    t0 = clock::now();
    axpy(size, &alpha, x.data(), 1, z1.data(), 1);
    t1 = clock::now();
    std::cout << "alpha*x + y = " << z1 << std::endl;
    std::cout << "dt = " << ms_cast(t1-t0).count() << "ms" << std::endl;
    std::cout << std::endl;

    std::cout << "Allocating device memory and copying vectors..." << std::endl;
    float *dx, *dy, *dz;
    t0 = clock::now();
    HIP_CHECK_ERROR(hipMalloc(&dx, size * sizeof *dx));
    HIP_CHECK_ERROR(hipMalloc(&dy, size * sizeof *dy));
    HIP_CHECK_ERROR(hipMalloc(&dz, size * sizeof *dz));
    HIP_CHECK_ERROR(hipMemcpy(dx, x.data(), size * sizeof *dx, hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemcpy(dy, y.data(), size * sizeof *dy, hipMemcpyHostToDevice));
    t1 = clock::now();
    std::cout << "dt = " << ms_cast(t1-t0).count() << "ms" << std::endl;
    std::cout << std::endl;

    std::cout << "Performing SAXPY (hipblas on device)..." << std::endl;
    std::vector<float> z2(size);
    HIP_CHECK_ERROR(hipMemcpy(dz, dy, size * sizeof *dz, hipMemcpyDeviceToDevice));
    hipblasHandle_t handle;
    HIPBLAS_CHECK_ERROR(hipblasCreate(&handle));
    t0 = clock::now();
    HIPBLAS_CHECK_ERROR(hipblasSaxpy(&handle, size, &alpha, dx, 1, dz, 1));
    t1 = clock::now();
    HIPBLAS_CHECK_ERROR(hipblasDestroy(handle));
    HIP_CHECK_ERROR(hipMemcpy(z2.data(), dz, size * sizeof *dz, hipMemcpyDeviceToHost));
    std::cout << "alpha*x + y = " << z2 << std::endl;
    std::cout << "dt = " << ms_cast(t1-t0).count() << "ms" << std::endl;
    std::cout << std::endl;

    HIP_CHECK_ERROR(hipFree(dx));
    HIP_CHECK_ERROR(hipFree(dy));
    HIP_CHECK_ERROR(hipFree(dz));
}

