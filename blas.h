#include <iostream>


#ifdef USE_HIPBLAS

#include <hipblas/hipblas.h>

#define blasHandle_t        hipblasHandle_t
#define blasCreate          hipblasCreate
#define blasDestroy         hipblasDestroy

#define blasSaxpy           hipblasSaxpy
#define blasSgemm           hipblasSgemm

#define blasStatusToString  hipblasStatusToString
#define BLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS

#else

#include <cublas_v2.h>

#define blasHandle_t        cublasHandle_t

#define blasCreate          cublasCreate
#define blasDestroy         cublasDestroy

#define blasSaxpy           cublasSaxpy
#define blasSgemm           cublasSgemm

#define blasStatusToString  cublasGetStatusString
#define BLAS_STATUS_SUCCESS CUBLAS_STATUS_SUCCESS

#endif


#define BLAS_CHECK_ERROR(status)                                            \
    if (status != BLAS_STATUS_SUCCESS) {                                    \
        std::cerr << __FILE__ << "(" << __LINE__ << ") blas error: "        \
            << blasStatusToString(status) << std::endl;                     \
        exit(EXIT_FAILURE);                                                 \
    }

