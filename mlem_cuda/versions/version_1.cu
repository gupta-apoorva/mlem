/*
This version only uses cusparse operations for calculating norm, forward projection and backward projection.
It also only make use of PAGABLE MEMORY for MPI_REDUCE OPERATIONS and is NOT CUDA AWARE.
*/

//#define _LARGEFILE64_SOURCE
#ifndef __APPLE__
#include <xmmintrin.h>
#else
#include <fenv.h>
#endif

#include "../helper_files/cudaKernels.cuh"
#include "../helper_files/helper.cuh"
#include "../../helper_files_common/structures.hpp"
#include "../../helper_files_common/csr4matrix.hpp"
#include "../../helper_files_common/vector.hpp"

#include <iostream>
#include <string>
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
#include <mpi.h>

/*
 * Compiler Switches
 * MESSUNG -> Supress human readable output and enable csv. generation
 */

#define IMG_CHECKPOINT_NAME "img.chkpt"
#define ITER_CHECKPOINT_NAME "iter.chkpt"
#define CHKPNT_INTEVALL 5

using namespace std;

TimingRuntime mlem(const MpiData& mpi, const std::vector<Range>& ranges,
          const Csr4Matrix& matrix, const Vector<int>& lmsino,
          Vector<float>& image, int nIterations, int checkpointing)
{
    TimingRuntime time_runtime(nIterations);
    // Creating the handles required for the cublas and cusparse and also the events required for time measurement
    cusparseHandle_t cusparseHandle;
    cusparseCreate(&cusparseHandle);
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Initializing the data on the CPU and putting it all together so it is easier/faster to load on the GPU 

    uint32_t nRows = matrix.rows();
    uint32_t nColumns = matrix.columns();
    int iter_alt = 0;

    auto& myrange = ranges[mpi.rank];
    uint32_t num_rows_this_rank = myrange.end - myrange.start;
    matrix.mapRows(myrange.start, num_rows_this_rank);

    uint32_t nnz_this_rank = get_nnz(myrange, matrix);

    float* csrVal = new float[nnz_this_rank];
    int* csrColInd = new int[nnz_this_rank];
    int* csrRowInd = new int[num_rows_this_rank + 1];

    start_time_measurement(start);
    csr_format_for_cuda(myrange, matrix, csrVal, csrRowInd, csrColInd);
    stop_time_measurement(start, stop, &time_runtime.struct_to_csr_vector_time);
    
    std::vector<float> lmsino_float(&lmsino[0], &lmsino[nRows]);
    float* lmsino_tmp = &lmsino_float[0];

    //int chkpt_int = 0;

    start_time_measurement(start);
    // Defining all the vectors needed on the device
    float *fwproj_d , *correlation_d, *update_d, *csrVal_d, *norm_d, *lmsino_d, *image_d;
    int *csrRowInd_d , *csrColInd_d;
    
    // Allocating all the vectors needed on the device            
                // Float vectors
    cudaMalloc((void**)&fwproj_d , nRows*sizeof(float));
    cudaMalloc((void**)&correlation_d , nRows*sizeof(float));
    cudaMalloc((void**)&update_d , nColumns*sizeof(float));
    cudaMalloc((void**)&csrVal_d , nnz_this_rank*sizeof(float));
    cudaMalloc((void**)&norm_d , nColumns*sizeof(float));
    cudaMalloc((void**)&lmsino_d , nRows*sizeof(float));
    cudaMalloc((void**)&image_d , nColumns*sizeof(float));
                // Int vectors
    cudaMalloc((void**)&csrColInd_d , nnz_this_rank*sizeof(int));
    cudaMalloc((void**)&csrRowInd_d , (num_rows_this_rank+1)*sizeof(int));

    // Copying the data from the host to the device where it is needed
    cudaMemcpy(csrVal_d, csrVal , nnz_this_rank*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(csrColInd_d, csrColInd , nnz_this_rank*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(csrRowInd_d, csrRowInd , (num_rows_this_rank+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(lmsino_d , lmsino_tmp , nRows*sizeof(float), cudaMemcpyHostToDevice);
    stop_time_measurement(start, stop, &time_runtime.alloc_copy_to_d_time);

    start_time_measurement(start);
    stop_time_measurement(start, stop, &time_runtime.further_par_time);
    
    // Defining the grid and block size for the computaion on cuda
    dim3 block = dim3(1024,1,1);
    int grid_x = ((std::max((uint32_t)num_rows_this_rank, nColumns) + block.x - 1)/block.x);
    int grid_y = 1;
    int grid_z = 1;
    dim3 grid = dim3(grid_x, grid_y, grid_z);

    // Allocating and initializing memory for mpi_allreduce operations
    size_t max_size_mem_required = std::max(nColumns, nRows);
    float* reduce_op_mem = new float[max_size_mem_required];

    // Initializing values for the csr mv on cuda 
    float alpha = 1.0;
    float beta = 0.0;

    // Calculating the norm (CalcColumnSums)
    cudaMemset(norm_d , 0, nColumns*sizeof(float));
    start_time_measurement(start);
    float* rowVector_d;
    cudaMalloc((void**)&rowVector_d , num_rows_this_rank*sizeof(float));
    d_set_value <<<grid, block>>> (rowVector_d , 1.0, num_rows_this_rank);

    cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, num_rows_this_rank, nColumns, 
    nnz_this_rank, &alpha, descrA, csrVal_d , csrRowInd_d, csrColInd_d, rowVector_d, &beta, norm_d); 
    stop_time_measurement(start, stop, &time_runtime.norm_calc_time);

    // Reducing the norm
    start_time_measurement(start);
    cudaMemcpy(reduce_op_mem, norm_d, nColumns*sizeof(float), cudaMemcpyDeviceToHost);
    MPI_Allreduce(MPI_IN_PLACE, reduce_op_mem, nColumns, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    cudaMemcpy(norm_d, reduce_op_mem, nColumns*sizeof(float), cudaMemcpyHostToDevice);
    stop_time_measurement(start, stop, &time_runtime.norm_redc_time);

    // Initializing the image
    float sumnorm = 0.0;
    float sumin = 0.0;

    start_time_measurement(start);
    cublasSasum(cublasHandle, nColumns, norm_d, 1, &sumnorm);
    cublasSasum(cublasHandle, nRows, lmsino_d, 1, &sumin);
    float initial = static_cast <float> (sumin/sumnorm);
    d_set_value <<<grid, block>>> (image_d , initial , nColumns);
    stop_time_measurement(start, stop, &time_runtime.calc_setting_image_time);

#ifndef MESSUNG
    std::cout << "MLEM [" << mpi.rank << "/" << mpi.size << "]: " <<
        std::setprecision(10) <<"sumnorm: " << sumnorm << std::endl; 

    std::cout << "MLEM [" << mpi.rank << "/" << mpi.size << "]: " <<
        std::setprecision(10) << "sumin: " << sumin << std::endl; 

    std::cout << "MLEM [" << mpi.rank << "/" << mpi.size << "]: " <<
        std::setprecision(10) << "initial: " << initial << std::endl; 
#endif

    float image_sum = 0.0;

    for (int iter = iter_alt; iter<nIterations ; iter++){                                                                                                                                                                                                                                                                                                                                                       
        
        cudaMemset(fwproj_d, 0, nRows*sizeof(float));
        cudaMemset(update_d, 0, nColumns*sizeof(float));
        cudaMemset(correlation_d, 0, nRows*sizeof(float));
    
        // calculating forward projection calcFwProj
        start_time_measurement(start);
        cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, num_rows_this_rank, nColumns, nnz_this_rank,
        &alpha, descrA, csrVal_d , csrRowInd_d, csrColInd_d, image_d, &beta, &fwproj_d[myrange.start]);
        stop_time_measurement(start, stop, &time_runtime.timing_loop[iter].fwproj_time);

        // reducing Forward Projection
        start_time_measurement(start);
        cudaMemcpy(reduce_op_mem, fwproj_d, nRows*sizeof(float), cudaMemcpyDeviceToHost);
        MPI_Allreduce(MPI_IN_PLACE, reduce_op_mem, nRows, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        cudaMemcpy(fwproj_d, reduce_op_mem, nRows*sizeof(float), cudaMemcpyHostToDevice);
        stop_time_measurement(start, stop, &time_runtime.timing_loop[iter].fwproj_redc_time);

        // calculating correlation
        start_time_measurement(start);
        d_calcCorrel <<<grid , block>>> (&fwproj_d[myrange.start], &lmsino_d[myrange.start], &correlation_d[myrange.start] , num_rows_this_rank);
        stop_time_measurement(start, stop, &time_runtime.timing_loop[iter].corr_calc_time);

        // Calculating backward projection calcBkProj
        start_time_measurement(start);
        cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, num_rows_this_rank, nColumns, nnz_this_rank,
        &alpha, descrA, csrVal_d , csrRowInd_d, csrColInd_d, &correlation_d[myrange.start], &beta, update_d);
        stop_time_measurement(start, stop, &time_runtime.timing_loop[iter].update_calc_time);

        // Reducing backward projection
        start_time_measurement(start);
        cudaMemcpy(reduce_op_mem, update_d, nColumns*sizeof(float), cudaMemcpyDeviceToHost);
        MPI_Allreduce(MPI_IN_PLACE, reduce_op_mem, nColumns, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        cudaMemcpy(update_d, reduce_op_mem, nColumns*sizeof(float), cudaMemcpyHostToDevice);
        stop_time_measurement(start, stop, &time_runtime.timing_loop[iter].update_redc_time);

        // Performing update        
        start_time_measurement(start);
        d_update <<<grid, block>>>(update_d, norm_d, image_d, nColumns);
        stop_time_measurement(start, stop, &time_runtime.timing_loop[iter].img_update_time);

        // Calculating Image Sum
        start_time_measurement(start);
        cublasSasum(cublasHandle, nColumns, image_d, 1, &image_sum);
        stop_time_measurement(start, stop, &time_runtime.timing_loop[iter].img_sum_time);

        // Outputting Results
        if (mpi.rank == 0)
        std::cout << "MLEM [" << mpi.rank << "/" << mpi.size << "]: Iter: "
                  << iter + 1 << ", "
                  << " Image sum: " << std::setprecision(10) << image_sum
                  << std::endl;
        
    }
    
    // Copying the image to the host to be outputted
    std::vector<float> image_float(&image[0], &image[nColumns]);
    float* image_tmp = &image_float[0];
    cudaMemcpy(image_tmp, image_d, nColumns*sizeof(float), cudaMemcpyDeviceToHost);
    
    //Freeing all the memory on the host
    if (csrVal) delete [] csrVal;
    if (csrRowInd) delete [] csrRowInd;
    if (csrColInd) delete [] csrColInd;
    if (reduce_op_mem) delete [] reduce_op_mem;    

    // Freeing all the allocated memory on the device
    if (fwproj_d) cudaFree(fwproj_d);
    if (correlation_d) cudaFree(correlation_d);
    if (update_d) cudaFree(update_d);
    if (csrVal_d) cudaFree(csrVal_d);
    if (norm_d) cudaFree(norm_d);
    if (lmsino_d) cudaFree(lmsino_d);
    if (image_d) cudaFree(image_d);
    if (csrRowInd_d) cudaFree(csrRowInd_d);
    if (csrColInd_d) cudaFree(csrColInd_d);

    return time_runtime;
}

int main(int argc, char *argv[])
{
    ProgramOptions progops = handleCommandLine(argc, argv);
#ifndef MESSUNG
    std::cout << "Matrix file: " << progops.mtxfilename << std::endl;
    std::cout << "Input file: "  << progops.infilename << std::endl;
    std::cout << "Output file: " << progops.outfilename << std::endl;
    std::cout << "Iterations: " << progops.iterations << std::endl;
    std::cout << "Checkpointing: " << progops.checkpointing << std::endl;
#endif
    Csr4Matrix matrix(progops.mtxfilename);
#ifndef MESSUNG
    std::cout << "Matrix rows (LORs): " << matrix.rows() << std::endl;
    std::cout << "Matrix cols (VOXs): " << matrix.columns() << std::endl;
#endif
    Vector<int> lmsino(progops.infilename);
    Vector<float> image(matrix.columns(), 0.0);

    MpiData mpi = initializeMpi(argc, argv);

    boost::filesystem::path full_path = make_prof_folder(argv[0], mpi);

    std::vector<Range> ranges = partition(mpi, matrix);

    TimingRuntime time_runtime = mlem(mpi, ranges, matrix, lmsino, image, progops.iterations, progops.checkpointing);

    time_writetofile(full_path, mpi, time_runtime, progops);

    if (mpi.rank == 0)
        image.writeToFile(progops.outfilename);

    // Remove chekcpoint
    if (progops.checkpointing == 1 && mpi.rank == 0){
        remove(IMG_CHECKPOINT_NAME);
        remove(ITER_CHECKPOINT_NAME);
    }

    MPI_Finalize();

    return 0;
}
