#ifdef USE_HIPBLAS


#include <hipblas/hipblas.h>

#define blasHandle_t        hipblasHandle_t

#define blasCreate          hipblasCreate
#define blasDestroy         hipblasDestroy

#define blasSaxpy           hipblasSaxpy
#define blasSgemm           hipblasSgemm

#define BLAS_CHECK_ERROR(status)                                            \
    if (status != HIPBLAS_STATUS_SUCCESS) {                                 \
        std::cerr << __FILE__ << "(" << __LINE__ << ") hipblas error: "     \
            << hipblasStatusToString(status) << std::endl;                  \
        exit(EXIT_FAILURE);                                                 \
    }


#else


#include <cublas_v2.h>

#define blasHandle_t        cublasHandle_t

#define blasCreate          cublasCreate
#define blasDestroy         cublasDestroy

#define blasSaxpy           cublasSaxpy
#define blasSgemm           cublasSgemm

#define BLAS_CHECK_ERROR(status)                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                  \
        std::cerr << __FILE__ << "(" << __LINE__ << ") cublas error: "      \
            << cublasGetStatusString(status) << std::endl;                  \
        exit(EXIT_FAILURE);                                                 \
    }


#endif

