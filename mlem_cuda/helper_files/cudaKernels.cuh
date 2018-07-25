#ifndef __CUDAKERNELS_H__
#define __CUDAKERNELS_H__

#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <stdint.h>

using namespace std;

#define WARP_SIZE 32
#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
void cuda_check(std::string file, int line);

__global__ void d_calcCorrel( const float* fwproj_d, const float* lmsino_d, float* correlation_d, const uint32_t nRows);

__global__ void d_update(const float* update_d, const float* norm_d,  float* image_d, const int cols);

__global__ void d_set_value(float* rowVector_d , const float value, const size_t num_elements);

__global__ void d_mat_vec_mul(float* result, const float* vec , const float* csrVal_d, const int* csrRowInd_d, const int* csrColInd_d, const uint32_t rows);

__global__ void d_tran_mat_vec_mul(float* result, const float* vec , const float* csrVal_d, const int* csrRowInd_d, const int* csrColInd_d, const uint32_t rows);

__global__ void trans_mat_vec_mul_warp( const uint32_t nnz_to_skip, const uint32_t num_rows_this_rank, const int* csrRowInd_d, const int* csrColInd_d, const float* csrVal_d, const float* vector, float* result);

__global__ void trans_mat_unit_vec_mul_warp( const uint32_t nnz_to_skip, const uint32_t num_rows_this_part, const int* csrRowInd_d, const int* csrColInd_d , const float* csrVal_d, float* result);

__global__ void mat_vec_mul_warp ( const uint32_t nnz_to_skip, const int num_rows_this_part , const int* csrRowInd_d , const int* csrColInd_d , const float* csrVal_d , const float* vector , float* result);

__global__ void move_csrrow_ind(int* csrRowInd_d, size_t num_rows, size_t how_much_to_move);

#endif
