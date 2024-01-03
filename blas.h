#include <iostream>


#if defined(USE_HIPBLAS)

#include <hipblas/hipblas.h>

#define blasHandle_t        hipblasHandle_t
#define blasCreate          hipblasCreate
#define blasDestroy         hipblasDestroy

#define blasSaxpy           hipblasSaxpy
#define blasSgemm           hipblasSgemm

#define BLAS_OP_N           HIPBLAS_OP_N
#define BLAS_OP_T           HIPBLAS_OP_T

#define blasStatusToString  hipblasStatusToString
#define BLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS

#elif defined(USE_CUBLAS)

#include <cublas_v2.h>

#define blasHandle_t        cublasHandle_t

#define blasCreate          cublasCreate
#define blasDestroy         cublasDestroy

#define blasSaxpy           cublasSaxpy
#define blasSgemm           cublasSgemm

#define BLAS_OP_N           CUBLAS_OP_N
#define BLAS_OP_T           CUBLAS_OP_T

#define blasStatusToString  cublasGetStatusString
#define BLAS_STATUS_SUCCESS CUBLAS_STATUS_SUCCESS

#else

#error "Must define one of USE_HIPBLAS or USE_CUBLAS."

#endif


#define BLAS_CHECK_ERROR(status)                                            \
    if (status != BLAS_STATUS_SUCCESS) {                                    \
        std::cerr << __FILE__ << "(" << __LINE__ << ") blas error: "        \
            << blasStatusToString(status) << std::endl;                     \
        exit(EXIT_FAILURE);                                                 \
    }

