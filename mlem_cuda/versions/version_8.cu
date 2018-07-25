/*
For forward projection, backward projection and norm calculation it uses WARP version of the custom kernels.
It also only make use of PINNED MEMORY for MPI_REDUCE OPERATIONS and is NOT CUDA AWARE.

This version can run on smaller GPU's where the entire matrix could not be loaded at once.
Solves the further broken down parts on the GPU.

It implements the fused version of the code.
*/

//#define _LARGEFILE64_SOURCE
#ifndef __APPLE__
#include <xmmintrin.h>
#else
#include <fenv.h>
#endif

#include "../helper_files/cudaKernels.cuh"
#include "../helper_files/helper.cuh"
#include "../helper_files/helper_v_7_8.cuh"
#include "../../helper_files_common/structures.hpp"
#include "../../helper_files_common/csr4matrix.hpp"
#include "../../helper_files_common/vector.hpp"


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
    int num_rows_this_rank = myrange.end - myrange.start;
    matrix.mapRows(myrange.start, num_rows_this_rank);

    uint32_t nnz_this_rank = get_nnz(myrange, matrix);

    float* csrVal;
    int* csrColInd , *csrRowInd;
    cudaMallocHost((void**)&csrVal, nnz_this_rank*sizeof(float));
    cudaMallocHost((void**)&csrRowInd, (num_rows_this_rank +1)*sizeof(int));
    cudaMallocHost((void**)&csrColInd, nnz_this_rank*sizeof(int));

    start_time_measurement(start);
    csr_format_for_cuda(myrange, matrix, csrVal, csrRowInd, csrColInd);
    stop_time_measurement(start, stop, &time_runtime.struct_to_csr_vector_time);

    std::vector<float> lmsino_float(&lmsino[0], &lmsino[nRows]);
    float* lmsino_tmp = &lmsino_float[0];

    //int chkpt_int = 0;

    // Defining all the vectors needed on the device
    start_time_measurement(start);
    float *correlation_d, *update_d, *csrVal_d, *norm_d, *lmsino_d, *image_d;
    int *csrRowInd_d , *csrColInd_d;
    
    // Allocating all the vectors needed on the device            
                // Float vectors
    cudaMalloc((void**)&correlation_d , nRows*sizeof(float));
    cudaMalloc((void**)&update_d , nColumns*sizeof(float));
    cudaMalloc((void**)&norm_d , nColumns*sizeof(float));
    cudaMalloc((void**)&lmsino_d , nRows*sizeof(float));
    cudaMalloc((void**)&image_d , nColumns*sizeof(float));
                // Int vectors
    cudaMalloc((void**)&csrRowInd_d , (num_rows_this_rank+1)*sizeof(int));

    // Copying the data from the host to the device where it is needed
    cudaMemcpy(csrRowInd_d, csrRowInd , (num_rows_this_rank+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(lmsino_d , lmsino_tmp , nRows*sizeof(float), cudaMemcpyHostToDevice);
    stop_time_measurement(start, stop, &time_runtime.alloc_copy_to_d_time);

    start_time_measurement(start);
    // Splitting the matrix so that the code could run on even smaller graphics cards
    uint32_t max_ele_in_row = get_max_ele_in_row(myrange, matrix);
    int parts_to_further_break_matrix = get_parts_to_further_break_matrix(nnz_this_rank, max_ele_in_row);

    std::cout << "parts_to_further_break_matrix: " << parts_to_further_break_matrix << std::endl;   

    std::vector<Range> splitting_based_on_rows(parts_to_further_break_matrix);

    get_splitting_based_on_rows(myrange, matrix, nnz_this_rank, parts_to_further_break_matrix, mpi, splitting_based_on_rows);

    uint32_t max_nnz_in_part = get_max_nnz_in_part(myrange, matrix, splitting_based_on_rows, parts_to_further_break_matrix);
    stop_time_measurement(start, stop, &time_runtime.further_par_time);

    start_time_measurement(start);
    cudaMalloc((void**)&csrVal_d , max_nnz_in_part*sizeof(float));
    cudaMalloc((void**)&csrColInd_d , max_nnz_in_part*sizeof(int));
    stop_time_measurement(start, stop, &time_runtime.alloc_copy_to_d_time);

    // Defining the grid and block size for the computaion on cuda
    dim3 block = dim3(1024,1,1);
    int grid_x = ((std::max((uint32_t)num_rows_this_rank, nColumns) + block.x - 1)/block.x);
    int grid_y = 1;
    int grid_z = 1;
    dim3 grid = dim3(grid_x, grid_y, grid_z);

    // Initializing the vectors on the device that need to be initialized
    cudaMemset(norm_d , 0, nColumns*sizeof(float));

    // Allocating and initializing pinned memory for mpi_allreduce operations
    float* reduce_op_mem;
    cudaMallocHost((void**)&reduce_op_mem, std::max(nRows, nColumns)*sizeof(float));

    uint32_t x, y, z; 

    // Calculating the norm
    start_time_measurement(start);
    for (int i=0 ; i<parts_to_further_break_matrix; i++){

        cudaDeviceSynchronize();
        x = csrRowInd[splitting_based_on_rows[i].start];
        y = csrRowInd[splitting_based_on_rows[i].end] - csrRowInd[splitting_based_on_rows[i].start];
        z = splitting_based_on_rows[i].end - splitting_based_on_rows[i].start;

        if (mpi.rank == 0)
            std::cout << "x: " << x << " y: " << y << " z: " << z << std::endl;

        cudaMemcpy(csrVal_d, &csrVal[x] , y*sizeof(float) , cudaMemcpyHostToDevice);
        cudaMemcpy(csrColInd_d, &csrColInd[x] , y*sizeof(int), cudaMemcpyHostToDevice);

        trans_mat_unit_vec_mul_warp <<<grid, block>>> (x ,z, &csrRowInd_d[splitting_based_on_rows[i].start], csrColInd_d, csrVal_d, norm_d);
    }
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

// #ifndef MESSUNG
    std::cout << "MLEM [" << mpi.rank << "/" << mpi.size << "]: " <<
        std::setprecision(10) <<"sumnorm: " << sumnorm << std::endl; 

    std::cout << "MLEM [" << mpi.rank << "/" << mpi.size << "]: " <<
        std::setprecision(10) << "sumin: " << sumin << std::endl; 

    std::cout << "MLEM [" << mpi.rank << "/" << mpi.size << "]: " <<
        std::setprecision(10) << "initial: " << initial << std::endl; 
// #endif

    float image_sum = 0.0;

    for (int iter = iter_alt; iter<nIterations ; iter++){

        cudaMemset(correlation_d, 0, nRows*sizeof(float));
        cudaMemset(update_d, 0, nColumns*sizeof(float));
        
        // calculating forward projection
        start_time_measurement(start);
        for (int i=parts_to_further_break_matrix-1 ; i>=0 ; i--){

            cudaDeviceSynchronize();
            x = csrRowInd[splitting_based_on_rows[i].start];
            y = csrRowInd[splitting_based_on_rows[i].end] - csrRowInd[splitting_based_on_rows[i].start];
            z = splitting_based_on_rows[i].end - splitting_based_on_rows[i].start;

            
            if (i != parts_to_further_break_matrix-1){
                cudaMemcpy(csrVal_d, &csrVal[x] , y*sizeof(float) , cudaMemcpyHostToDevice);
                cudaMemcpy(csrColInd_d, &csrColInd[x] , y*sizeof(int), cudaMemcpyHostToDevice);
            }
            mat_vec_mul_warp <<<grid, block, block.x*sizeof(float)>>> (x, z ,&csrRowInd_d[splitting_based_on_rows[i].start] ,csrColInd_d ,csrVal_d ,image_d ,&correlation_d[myrange.start + splitting_based_on_rows[i].start]);   
        }
        stop_time_measurement(start, stop, &time_runtime.timing_loop[iter].fwproj_time);

        start_time_measurement(start);    
        stop_time_measurement(start, stop, &time_runtime.timing_loop[iter].fwproj_redc_time);    

        // calculating correlation
        start_time_measurement(start);
        d_calcCorrel <<<grid , block>>> (&correlation_d[myrange.start], &lmsino_d[myrange.start], &correlation_d[myrange.start] , num_rows_this_rank);
        stop_time_measurement(start, stop, &time_runtime.timing_loop[iter].corr_calc_time);

        // Calculating backward projection calcBkProj
        start_time_measurement(start);
        for (int i=0 ; i<parts_to_further_break_matrix; i++){

            cudaDeviceSynchronize();
            x = csrRowInd[splitting_based_on_rows[i].start];
            y = csrRowInd[splitting_based_on_rows[i].end] - csrRowInd[splitting_based_on_rows[i].start];
            z = splitting_based_on_rows[i].end - splitting_based_on_rows[i].start;

            if (i != 0){
                cudaMemcpy(csrVal_d, &csrVal[x] , y*sizeof(float) , cudaMemcpyHostToDevice);
                cudaMemcpy(csrColInd_d, &csrColInd[x] , y*sizeof(int), cudaMemcpyHostToDevice);
            }
            trans_mat_vec_mul_warp <<<grid, block>>> (x ,z, &csrRowInd_d[splitting_based_on_rows[i].start], csrColInd_d, csrVal_d, &correlation_d[myrange.start + splitting_based_on_rows[i].start], update_d);
        }   
        stop_time_measurement(start, stop, &time_runtime.timing_loop[iter].update_calc_time);

        // Reducing the BAckward Projection
        start_time_measurement(start);
        cudaMemcpy(reduce_op_mem, update_d, nColumns*sizeof(float), cudaMemcpyDeviceToHost);
        MPI_Allreduce(MPI_IN_PLACE, reduce_op_mem, nColumns, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        cudaMemcpy(update_d, reduce_op_mem, nColumns*sizeof(float), cudaMemcpyHostToDevice);
        stop_time_measurement(start, stop, &time_runtime.timing_loop[iter].update_redc_time);


        // Performing image update
        start_time_measurement(start);
        d_update <<<grid, block>>>(update_d, norm_d, image_d, nColumns);
        stop_time_measurement(start, stop, &time_runtime.timing_loop[iter].img_update_time);

        // Calculating the image sum
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
    if (csrVal) cudaFreeHost(csrVal);
    if (csrRowInd) cudaFreeHost(csrRowInd);
    if (csrColInd) cudaFreeHost(csrColInd);
    if (reduce_op_mem) cudaFreeHost(reduce_op_mem);    

    // Freeing all the allocated memory on the device
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