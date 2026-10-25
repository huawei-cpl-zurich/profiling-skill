/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief aclblas status codes definition */
typedef enum aclblasStatus {
    ACLBLAS_STATUS_SUCCESS = 0,           /**< Function succeeds */
    ACLBLAS_STATUS_NOT_INITIALIZED = 1,   /**< aclblas library not initialized */
    ACLBLAS_STATUS_ALLOC_FAILED = 2,      /**< resource allocation failed */
    ACLBLAS_STATUS_INVALID_VALUE = 3,     /**< unsupported numerical value was passed to function */
    ACLBLAS_STATUS_MAPPING_ERROR = 4,     /**< access to memory space failed */
    ACLBLAS_STATUS_EXECUTION_FAILED = 5,  /**< program failed to execute */
    ACLBLAS_STATUS_INTERNAL_ERROR = 6,    /**< an internal aclblas operation failed */
    ACLBLAS_STATUS_NOT_SUPPORTED = 7,     /**< function not implemented */
    ACLBLAS_STATUS_ARCH_MISMATCH = 8,     /**< architecture mismatch */
    ACLBLAS_STATUS_HANDLE_IS_NULLPTR = 9, /**< aclblas handle is null pointer */
    ACLBLAS_STATUS_INVALID_ENUM = 10,     /**< unsupported enum value was passed to function */
    ACLBLAS_STATUS_UNKNOWN = 11           /**< back-end returned an unsupported status code */
} aclblasStatus_t;
typedef aclblasStatus_t aclblasLtStatus;

/*! \brief Specifies whether the upper or lower triangular part of a matrix is used. */
typedef enum aclblasFillMode {
    ACLBLAS_UPPER = 121, /**< Upper triangular */
    ACLBLAS_LOWER = 122  /**< Lower triangular */
} aclblasFillMode_t;

/*! \brief Specifies whether the diagonal elements are assumed to be unit or non-unit. */
typedef enum aclblasDiagType {
    ACLBLAS_NON_UNIT = 131, /**< Non-unit diagonal */
    ACLBLAS_UNIT = 132      /**< Unit diagonal */
} aclblasDiagType_t;

/*! \brief Specifies whether the triangular matrix is on the left or right side of the multiplication. */
typedef enum aclblasSideMode {
    ACLBLAS_SIDE_LEFT = 141, /**< Triangular matrix on the left side */
    ACLBLAS_SIDE_RIGHT = 142 /**< Triangular matrix on the right side */
} aclblasSideMode_t;

#ifndef ACLBLAS_OPERATION_DECLARED
#define ACLBLAS_OPERATION_DECLARED
/*! \brief Used to specify whether the matrix is to be transposed or not (Fortran BLAS style). */
typedef enum aclblasOperation {
    ACLBLAS_OP_N = 111, /**< Operate with the matrix */
    ACLBLAS_OP_T = 112, /**< Operate with the transpose of the matrix */
    ACLBLAS_OP_C = 113  /**< Operate with the conjugate transpose of the matrix */
} aclblasOperation_t;

#elif __cplusplus >= 201103L
static_assert(ACLBLAS_OP_N == 111, "Inconsistent declaration of ACLBLAS_OP_N");
static_assert(ACLBLAS_OP_T == 112, "Inconsistent declaration of ACLBLAS_OP_T");
static_assert(ACLBLAS_OP_C == 113, "Inconsistent declaration of ACLBLAS_OP_C");
#endif // ACLBLAS_OPERATION_DECLARED

/*! \brief Single-precision complex type. */
typedef struct aclblasComplex {
    float real;
    float imag;
} aclblasComplex;

/*! \brief Double-precision complex type. */
typedef struct aclblasDoubleComplex {
    double real;
    double imag;
} aclblasDoubleComplex;

/*! \brief The compute type to be used. Currently only used with GemmEx.
 *  Note that support for compute types is largely dependent on backend. */
typedef enum aclblasComputeType {
    ACLBLAS_COMPUTE_16F = 0,           /**< compute will be at least 16-bit precision */
    ACLBLAS_COMPUTE_16F_PEDANTIC = 1,  /**< compute will be exactly 16-bit precision */
    ACLBLAS_COMPUTE_32F = 2,           /**< compute will be at least 32-bit precision */
    ACLBLAS_COMPUTE_32F_PEDANTIC = 3,  /**< compute will be exactly 32-bit precision */
    ACLBLAS_COMPUTE_32F_FAST_16F = 4,  /**< 32-bit input can use 16-bit compute */
    ACLBLAS_COMPUTE_32F_FAST_16BF = 5, /**< 32-bit input is bf16 compute */
    ACLBLAS_COMPUTE_32F_FAST_TF32 = 6, /**< 32-bit input can use tensor cores w/ TF32 compute */
    ACLBLAS_COMPUTE_64F = 7,           /**< compute will be at least 64-bit precision */
    ACLBLAS_COMPUTE_64F_PEDANTIC = 8,  /**< compute will be exactly 64-bit precision */
    ACLBLAS_COMPUTE_32I = 9,           /**< compute will be at least 32-bit integer precision */
    ACLBLAS_COMPUTE_32I_PEDANTIC = 10, /**< compute will be exactly 32-bit integer precision */
} aclblasComputeType_t;

/*! \brief The algorithm to be used for GemmEx operations.
 *  Note that support for algorithms is largely dependent on backend. */
typedef enum aclblasGemmAlgo {
    ACLBLAS_GEMM_DEFAULT = 0, /**< Default algorithm (auto-selected by backend) */
    ACLBLAS_GEMM_ALGO0 = 1,   /**< Algorithm 0 (reserved for future use) */
    ACLBLAS_GEMM_ALGO1 = 2,   /**< Algorithm 1 (reserved for future use) */
    ACLBLAS_GEMM_ALGO2 = 3,   /**< Algorithm 2 (reserved for future use) */
    ACLBLAS_GEMM_ALGO3 = 4,   /**< Algorithm 3 (reserved for future use) */
    ACLBLAS_GEMM_ALGO4 = 5,   /**< Algorithm 4 (reserved for future use) */
    ACLBLAS_GEMM_ALGO5 = 6,   /**< Algorithm 5 (reserved for future use) */
    ACLBLAS_GEMM_ALGO6 = 7,   /**< Algorithm 6 (reserved for future use) */
    ACLBLAS_GEMM_ALGO7 = 8,   /**< Algorithm 7 (reserved for future use) */
} aclblasGemmAlgo_t;

/*! \brief Standard LAPACK argument reporting values for functions using `int* info`.
 *
 *  Convention (matches LAPACK xINFO semantics):
 *    = 0 : successful exit
 *    < 0 : if info = -i, the i-th argument had an illegal value
 *
 *  Argument numbering follows standard LAPACK convention: `info` itself is
 *  excluded from counting.  For ACLBLAS wrappers, the `aclblas handle` is
 *  also excluded (reported separately via ACLBLAS_STATUS_HANDLE_IS_NULLPTR).
 *  For example, for `aclblasSgeqrfBatched(handle, m, n, Aarray, lda, TauArray, info, batchSize)`,
 *  argument numbering (excluding handle and info) is:
 *    m=1, n=2, Aarray=3, lda=4, TauArray=5, batchSize=6.
 */
typedef enum aclblasLapackInfo {
    ACLBLAS_LAPACK_INFO_OK = 0,     /**< successful exit */
    ACLBLAS_LAPACK_INFO_ARG_1 = -1, /**< 1st LAPACK-style argument is invalid */
    ACLBLAS_LAPACK_INFO_ARG_2 = -2, /**< 2nd LAPACK-style argument is invalid */
    ACLBLAS_LAPACK_INFO_ARG_3 = -3, /**< 3rd LAPACK-style argument is invalid */
    ACLBLAS_LAPACK_INFO_ARG_4 = -4, /**< 4th LAPACK-style argument is invalid */
    ACLBLAS_LAPACK_INFO_ARG_5 = -5, /**< 5th LAPACK-style argument is invalid */
    ACLBLAS_LAPACK_INFO_ARG_6 = -6, /**< 6th LAPACK-style argument is invalid */
    ACLBLAS_LAPACK_INFO_ARG_7 = -7, /**< 7th LAPACK-style argument is invalid */
    ACLBLAS_LAPACK_INFO_ARG_8 = -8  /**< 8th LAPACK-style argument is invalid */
} aclblasLapackInfo_t;

#ifdef __cplusplus
}
#endif
