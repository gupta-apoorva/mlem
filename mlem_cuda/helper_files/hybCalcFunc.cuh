#ifndef _HYBCALCFUNC_CUH_
#define _HYBCALCFUNC_CUH_

#define IMG_CHECKPOINT_NAME "img.chkpt"
#define ITER_CHECKPOINT_NAME "iter.chkpt"

#include "cudaKernels.cuh"
#include "helper.cuh"
#include "../../helper_files_common/structures.hpp"
#include "../../helper_files_common/csr4matrix.hpp"
#include "../../helper_files_common/vector.hpp"

#include <omp.h>
#include <mpi.h>
#include <iostream>
#include <fstream> 
#include <string>
#include <cstdlib>
#include <stdexcept>
#include <cassert>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sys/stat.h>
#include <stdio.h>
#include <sys/time.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cuda_runtime.h>
#include <boost/filesystem.hpp>

int initialize_cuda_handles(const int nDevices, 
                            const std::vector<further_split> further_split_for_gpu,
                            std::vector <cudaStream_t> &streams,
                            std::vector <cusparseHandle_t> &cusparseHandle,
                            std::vector <cublasHandle_t> &cublasHandle,
                            std::vector <cusparseMatDescr_t> &descr);

int aggregated_rows_gpu(const int nDevices,
                        const std::vector<further_split> further_split_for_gpu, 
                        further_split &rows_gpu_total);

int allocating_initializing_device_arrays(  const int nDevices,
                                            const size_t nRows,
                                            const size_t nColumns,
                                            const dim3 grid,
                                            const dim3 block,
                                            const std::vector <cudaStream_t> streams,
                                            const std::vector<further_split> further_split_for_gpu,
                                            const int* csrRowInd,
                                            const int* csrColInd,
                                            const float* csrVal,
                                            const float* lmsino_mask,
                                            float **lmsino_d,
                                            float **correlation_d, 
                                            float **update_d, 
                                            float **csrVal_d, 
                                            float **norm_d, 
                                            float **image_d,
                                            int   **csrRowInd_d, 
                                            int   **csrColInd_d);

int norm_calculation_hyb(   const Csr4Matrix& matrix,
                            const int nDevices,
                            const size_t nRows,
                            const size_t nColumns,
                            const dim3 grid,
                            const dim3 block,
                            const std::vector <cudaStream_t> streams,
                            const std::vector<further_split> further_split_for_gpu,
                            const further_split further_split_for_cpu,
                            int** csrRowInd_d,
                            int** csrColInd_d,
                            float** csrVal_d,
                            float** norm_d,
                            std::vector<float> &norm);

int norm_reduction_hyb( const int nDevices,
                        const size_t nColumns,
                        const std::vector <cudaStream_t> streams,
                        const std::vector<further_split> further_split_for_gpu,
                        const further_split further_split_for_cpu,
                        float **norm_d,
                        std::vector<float> &norm,
                        float *reduce_op_mem);

int calculating_setting_image_hyb(  const MpiData mpi,
                                    const int nDevices,
                                    const int nRows,
                                    const int nColumns,
                                    const dim3 grid,
                                    const dim3 block, 
                                    const std::vector<further_split> further_split_for_gpu,
                                    const further_split further_split_for_cpu,
                                    const std::vector <cublasHandle_t> cublasHandle,
                                    const std::vector<cudaStream_t> streams,
                                    const Vector<int> &lmsino,
                                    Vector<float>& image,
                                    int &iter_alt,
                                    float **image_d,
                                    float *image_mask,
                                    float **lmsino_d,
                                    float **norm_d,
                                    float *reduce_op_mem);

int calc_forward_proj_hyb(  const Csr4Matrix& matrix,
                            const int nDevices,
                            const size_t nRows,
                            const size_t nColumns,
                            const dim3 grid,
                            const dim3 block,
                            const std::vector<further_split> further_split_for_gpu,
                            const further_split further_split_for_cpu,
                            const std::vector<cudaStream_t> streams,
                            int **csrRowInd_d,
                            int **csrColInd_d,
                            float **csrVal_d,
                            float **correlation_d,
                            float **image_d,
                            float *image_mask,
                            std::vector<float> &correlation);

int calc_correlation_hyb(   const int nDevices,
                            const dim3 grid,
                            const dim3 block,
                            const std::vector<cudaStream_t> streams,
                            const std::vector <further_split> further_split_for_gpu,
                            const further_split further_split_for_cpu,
                            const Vector<int> &lmsino,
                            float **correlation_d,
                            float **lmsino_d,
                            std::vector<float> &correlation);

int calc_backward_proj_hyb( const Csr4Matrix& matrix,
                            const int nDevices, 
                            const size_t nColumns,
                            const dim3 grid,
                            const dim3 block,
                            const std::vector <cudaStream_t> streams,
                            const std::vector <further_split> further_split_for_gpu,
                            const further_split further_split_for_cpu,
                            int **csrRowInd_d,
                            int **csrColInd_d,
                            float **csrVal_d,
                            float **update_d,
                            float **correlation_d,
                            std::vector <float> &correlation,
                            std::vector<float> &update);

int redc_backward_proj_hyb( const int nDevices,
                            const size_t nColumns,
                            const std::vector<cudaStream_t> streams,
                            const std::vector <further_split> further_split_for_gpu,
                            const further_split further_split_for_cpu,
                            const std::vector <float> &update,
                            float **update_d,
                            float *reduce_op_mem);

int img_update_hyb( const int nDevices,
                    const size_t nColumns,
                    const dim3 grid,
                    const dim3 block,
                    const std::vector <cudaStream_t> streams,
                    const std::vector <further_split> further_split_for_gpu,
                    const further_split further_split_for_cpu,
                    float *image_mask,
                    float *reduce_op_mem,
                    std::vector <float> &update,
                    std::vector <float> &norm,
                    float **image_d,
                    float **update_d,
                    float **norm_d);

int img_sum_hyb(const int nDevices,
                const size_t nColumns,
                const std::vector <further_split> further_split_for_gpu,
                const std::vector <cublasHandle_t> cublasHandle,
                float **image_d,
                float *image_mask,
                float &image_sum);


int creating_chkpt_hyb( const size_t nColumns,
                        const std::vector <further_split> further_split_for_gpu,
                        const further_split further_split_for_cpu,
                        const int iter,
                        const int checkpointing,
                        const MpiData mpi,
                        Vector <float> &image,
                        float *image_mask,
                        float **image_d);

int delete_chkpt(const int checkpointing , const MpiData& mpi);


#endif