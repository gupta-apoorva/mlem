/*
For forward projection, backward projection and norm calculation it uses WARP version of the custom kernels.
It also only make use of PINNED MEMORY for MPI_REDUCE OPERATIONS and is NOT CUDA AWARE.

This is a hybrid multi gpu version. Breaks down the part assigned to a mpi process for the GPU and CPU 
depending on how many GPU's are connected.

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
#include "../helper_files/decideConfig.cuh"
#include "../helper_files/hybCalcFunc.cuh"
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
#include <nvml.h>

/*
 * Compiler Switches
 * MESSUNG -> Supress human readable output and enable csv. generation
 */


/*
* @brief The function that does all the heavy lifting and solve the problem
* @note
* @param mpi: The mpi data for this rank
* @param ranges: The rows alocated to each mpi process to solve
* @param matrix: The struct containing all the csr data
* @param lmsino: The x vector in Ax=y
* @param image: The y vector in AX = y
* @param nIterations: The number of iterations to perform
* @param checkpointing: Number of iteration after which to do checkpoint.
* @retval Struct TimingRuntime containing all the time data
*/
TimingRuntime mlem( const MpiData& mpi, 
                    const std::vector<Range>& ranges,
                    const Csr4Matrix& matrix, 
                    const Vector<int>& lmsino,
                    Vector<float>& image, 
                    int nIterations, 
                    int checkpointing)
{
        TimingRuntime time_runtime(nIterations);
        int nDevices; 

    // Finding out the major parameters required across the code. 
        size_t nRows = matrix.rows();
        size_t nColumns = matrix.columns();
        int iter_alt = 0;
        Range myrange = ranges[mpi.rank];
        size_t num_rows_this_rank = myrange.end - myrange.start;
        matrix.mapRows(myrange.start, num_rows_this_rank);

    // Initializing and spliting the data assigned to this mpi process between gpu/gpus and cpu
        std::vector<further_split> further_split_for_gpu;
        further_split further_split_for_cpu;

    // Time measurement with respect to cpu
        std::chrono::high_resolution_clock::time_point t_begin;
        std::chrono::high_resolution_clock::time_point t_end;    

    start_time_measurement_hyb(t_begin);    
        hybrid_splitting(   matrix, 
                            nRows, 
                            nColumns, 
                            num_rows_this_rank, 
                            myrange, 
                            further_split_for_gpu, 
                            further_split_for_cpu, 
                            nDevices);
    stop_time_measurement_hyb(t_begin, t_end, &time_runtime.further_par_time);

    // Total number of GPU's to use

    #ifndef MESSUNG  
        std::cout   << "MLEM [" << mpi.rank << "/" << mpi.size << "]: "
                    << "Num Devices to Use: " << nDevices 
                    << std::endl;
    #endif

    #ifndef MESSUNG  

        for(int device=0; device< nDevices ; ++device){
            std::cout   << "MLEM [" << mpi.rank << "/" << mpi.size << "]:" 
                        << " Device id: "<< further_split_for_gpu[device].id
                        << " Rows Start: "<< further_split_for_gpu[device].start 
                        << " Rows End: " << further_split_for_gpu[device].end 
                        << " NNZ: " << further_split_for_gpu[device].nnz 
                        << " Num Rows: " << further_split_for_gpu[device].num_rows 
                        << std::endl;
        }

        std::cout   << "MLEM [" << mpi.rank << "/" << mpi.size << "]:" 
                    << " CPU: " 
                    << " Rows Start: " << further_split_for_cpu.start 
                    << " Rows End: " << further_split_for_cpu.end 
                    << " NNZ: " << further_split_for_cpu.nnz 
                    << " Num Rows: " << further_split_for_cpu.num_rows 
                    << std::endl;
    #endif
        
    // Allocating and initializing the cuda handles for cublas, cusparse and streams
        std::vector <cudaStream_t> streams(nDevices);
        std::vector <cusparseHandle_t> cusparseHandle(nDevices);
        std::vector <cublasHandle_t> cublasHandle(nDevices);
        std::vector <cusparseMatDescr_t> descr(nDevices);

        initialize_cuda_handles(nDevices, 
                                further_split_for_gpu,
                                streams,
                                cusparseHandle,
                                cublasHandle,
                                descr);

    // Allocating and initializing pinned memory for mpi_allreduce operations
        float* reduce_op_mem;
        size_t reduce_op_mem_size = std::max(1,nDevices)*std::max(nRows, nColumns)*sizeof(float);
        cudaHostAlloc((void**)&reduce_op_mem, reduce_op_mem_size, cudaHostAllocDefault); 

    // Casting Vectors into float for GPU's
        float* image_mask = &image[0];

        std::vector<float> lmsino_float(&lmsino[0], &lmsino[nRows]);
        float* lmsino_mask = &lmsino_float[0];

    // Finding out the total amount of data that has to loaded on the gpu in total.
        further_split rows_gpu_total;

        aggregated_rows_gpu(nDevices, 
                            further_split_for_gpu, 
                            rows_gpu_total);

    #ifndef MESSUNG  
        std::cout   << "MLEM [" << mpi.rank << "/" << mpi.size << "]:" 
                    << " Rows GPU Total Start: " << rows_gpu_total.start 
                    << " Rows GPU Total End: " << rows_gpu_total.end 
                    << " NNZ GPU Total: " << rows_gpu_total.nnz
                    << " Num Rows GPU Total: " << rows_gpu_total.num_rows
                    << std::endl; 
    #endif

    // Initializing the data on the CPU and putting it all together so it is easier/faster to load on the GPU   
        float* csrVal;
        int* csrColInd , *csrRowInd; 
        cudaHostAlloc((void**)&csrRowInd, (num_rows_this_rank + 1)*sizeof(int), cudaHostAllocPortable);
        cudaHostAlloc((void**)&csrVal, rows_gpu_total.nnz*sizeof(float), cudaHostAllocPortable);
        cudaHostAlloc((void**)&csrColInd, rows_gpu_total.nnz*sizeof(int), cudaHostAllocPortable);

    // Converting the matrix structure to csr for GPU's
    start_time_measurement_hyb(t_begin);
        if (nDevices > 0)
            csr_format_for_cuda <further_split> (rows_gpu_total, matrix, csrVal, csrRowInd, csrColInd);
    stop_time_measurement_hyb(t_begin, t_end, &time_runtime.struct_to_csr_vector_time);

    // Defining the grid and block size for the computaion on cuda
        dim3 block = dim3(1024,1,1);
        int grid_x = ((std::max(num_rows_this_rank, nColumns) + block.x - 1)/block.x);
        int grid_y = 1;
        int grid_z = 1;
        dim3 grid = dim3(grid_x, grid_y, grid_z);

    // Defining all the vectors needed on the device
        float **lmsino_d = (float**)malloc(nDevices*sizeof(float));
        float **correlation_d = (float**)malloc(nDevices*sizeof(float));
        float **update_d = (float**)malloc(nDevices*sizeof(float));
        float **csrVal_d = (float**)malloc(nDevices*sizeof(float));
        float **norm_d = (float**)malloc(nDevices*sizeof(float));
        float **image_d = (float**)malloc(nDevices*sizeof(float));
        int   **csrRowInd_d = (int**)malloc(nDevices*sizeof(float));
        int   **csrColInd_d = (int**)malloc(nDevices*sizeof(float));

    start_time_measurement_hyb(t_begin);
        
        allocating_initializing_device_arrays(  nDevices,
                                                nRows,
                                                nColumns,
                                                grid,
                                                block, 
                                                streams,
                                                further_split_for_gpu,
                                                csrRowInd,
                                                csrColInd,
                                                csrVal,
                                                lmsino_mask,
                                                lmsino_d,
                                                correlation_d, 
                                                update_d, 
                                                csrVal_d, 
                                                norm_d, 
                                                image_d,
                                                csrRowInd_d, 
                                                csrColInd_d);

        // Vectors needed for solving the 2nd part on the CPU
        std::vector <float> norm(nColumns, 0.0);
        std::vector <float> correlation(nRows , 0.0);
        std::vector <float> update (nColumns , 0.0);

    stop_time_measurement_hyb(t_begin, t_end, &time_runtime.alloc_copy_to_d_time);


    for (int device =0 ; device < nDevices ; ++device){

        if (cudaGetLastError() != cudaSuccess){
            std::cout << "ERROR: Possibly more than 1 processes trying to use a single GPU and there is not enough memory on it." << std::endl;
            exit(1);
        }
    }


    start_time_measurement_hyb(t_begin);
        
        norm_calculation_hyb(   matrix,
                                nDevices,
                                nRows,
                                nColumns,
                                grid,
                                block,
                                streams,
                                further_split_for_gpu,
                                further_split_for_cpu,
                                csrRowInd_d,
                                csrColInd_d,
                                csrVal_d,
                                norm_d,
                                norm);  

    stop_time_measurement_hyb(t_begin, t_end, &time_runtime.norm_calc_time);

                    /* REDUCING THE NORM */
    start_time_measurement_hyb(t_begin);

        norm_reduction_hyb( nDevices,
                            nColumns,
                            streams,
                            further_split_for_gpu,
                            further_split_for_cpu,
                            norm_d,
                            norm,
                            reduce_op_mem);

    stop_time_measurement_hyb(t_begin, t_end, &time_runtime.norm_redc_time);

        
        /* Calculating the initial value using only 1 device as the vectors 
        are same on each device and if not then doing it on the cpu */
    start_time_measurement_hyb(t_begin);
        
        calculating_setting_image_hyb(  mpi,
                                        nDevices,
                                        nRows,
                                        nColumns,
                                        grid,
                                        block, 
                                        further_split_for_gpu,
                                        further_split_for_cpu,
                                        cublasHandle,
                                        streams,
                                        lmsino,
                                        image,
                                        iter_alt,
                                        image_d,
                                        image_mask,
                                        lmsino_d,
                                        norm_d,
                                        reduce_op_mem);
        
    stop_time_measurement_hyb(t_begin, t_end, &time_runtime.calc_setting_image_time);

    float image_sum = 0.0;

    for (int iter = iter_alt; iter<nIterations ; iter++){
        memset(reduce_op_mem, 0 , reduce_op_mem_size);
        std::fill(correlation.begin(), correlation.end(), 0.0);
        std::fill(update.begin(), update.end(), 0.0);

        start_time_measurement_hyb(t_begin);
            
            calc_forward_proj_hyb(  matrix,
                                    nDevices,
                                    nRows,
                                    nColumns,
                                    grid,
                                    block,
                                    further_split_for_gpu,
                                    further_split_for_cpu,
                                    streams,
                                    csrRowInd_d,
                                    csrColInd_d,
                                    csrVal_d,
                                    correlation_d,
                                    image_d,
                                    image_mask,
                                    correlation);

        stop_time_measurement_hyb(t_begin, t_end, &time_runtime.timing_loop[iter].fwproj_time);

        start_time_measurement_hyb(t_begin);    
        stop_time_measurement_hyb(t_begin, t_end, &time_runtime.timing_loop[iter].fwproj_redc_time);   

        start_time_measurement_hyb(t_begin);
            
            calc_correlation_hyb(   nDevices,
                                    grid,
                                    block,
                                    streams,
                                    further_split_for_gpu,
                                    further_split_for_cpu,
                                    lmsino,
                                    correlation_d,
                                    lmsino_d,
                                    correlation);

        stop_time_measurement_hyb(t_begin, t_end, &time_runtime.timing_loop[iter].corr_calc_time);

                            /*CALCULATING BACKWARD PROJECTION*/

            
        start_time_measurement_hyb(t_begin);
            
            calc_backward_proj_hyb( matrix,
                                    nDevices, 
                                    nColumns,
                                    grid,
                                    block,
                                    streams,
                                    further_split_for_gpu,
                                    further_split_for_cpu,
                                    csrRowInd_d,
                                    csrColInd_d,
                                    csrVal_d,
                                    update_d,
                                    correlation_d,
                                    correlation,
                                    update);

        stop_time_measurement_hyb(t_begin, t_end, &time_runtime.timing_loop[iter].update_calc_time);
                            
                            /*REDUCING THE UPDATE*/ 

            
        start_time_measurement_hyb(t_begin);
            
            redc_backward_proj_hyb( nDevices,
                                    nColumns,
                                    streams,
                                    further_split_for_gpu,
                                    further_split_for_cpu,
                                    update,
                                    update_d,
                                    reduce_op_mem);

        stop_time_measurement_hyb(t_begin, t_end, &time_runtime.timing_loop[iter].update_redc_time);


        start_time_measurement_hyb(t_begin);

            img_update_hyb( nDevices,
                            nColumns,
                            grid,
                            block,
                            streams,
                            further_split_for_gpu,
                            further_split_for_cpu,
                            image_mask,
                            reduce_op_mem,
                            update,
                            norm,
                            image_d,
                            update_d,
                            norm_d);
            
        stop_time_measurement_hyb(t_begin, t_end, &time_runtime.timing_loop[iter].img_update_time);

        start_time_measurement_hyb(t_begin);

        img_sum_hyb(nDevices,
                    nColumns,
                    further_split_for_gpu,
                    cublasHandle,
                    image_d,
                    image_mask,
                    image_sum);
            
        stop_time_measurement_hyb(t_begin, t_end, &time_runtime.timing_loop[iter].img_sum_time);

            // Outputting Results
            if (mpi.rank == 0)
                std::cout << "MLEM [" << mpi.rank << "/" << mpi.size << "]: Iter: "
                          << iter + 1 << ", "
                          << " Image sum: " << std::setprecision(10) << image_sum
                          << std::endl;

            // Creating Checkpoint
            creating_chkpt_hyb( nColumns,
                                further_split_for_gpu,
                                further_split_for_cpu,
                                iter,
                                checkpointing,
                                mpi,
                                image,
                                image_mask,
                                image_d);
        
    }

    // Finally coping the final image from the device
    if (further_split_for_cpu.num_rows == 0){
        cudaSetDevice(further_split_for_gpu[0].id); 
        cudaMemcpy(image_mask, image_d[0], nColumns*sizeof(float), cudaMemcpyDeviceToHost);
    }

    //Freeing all the memory on the host
    delete [] correlation_d;
    delete [] lmsino_d;
    delete [] update_d;
    delete [] csrVal_d;
    delete [] norm_d;
    delete [] image_d;
    delete [] csrRowInd_d;
    delete [] csrColInd_d;

    cudaFreeHost(csrVal);
    cudaFreeHost(csrRowInd);
    cudaFreeHost(csrColInd);
    cudaFreeHost(reduce_op_mem);  

    
    for (int device =0 ; device < nDevices ; ++device){
        cudaSetDevice(further_split_for_gpu[device].id); 

    // Deleting the streams, cublas handles, cusparse handles
        cudaStreamDestroy(streams[device]);
        cublasDestroy(cublasHandle[device]);
        cusparseDestroy(cusparseHandle[device]);
        cusparseDestroyMatDescr(descr[device]);
    
    // Freeing all the allocated memory on the each device
        cudaFree(correlation_d[device]);
        cudaFree(update_d[device]);
        cudaFree(csrVal_d[device]);
        cudaFree(norm_d[device]);
        cudaFree(lmsino_d[device]);
        cudaFree(image_d[device]);
        cudaFree(csrRowInd_d[device]);
        cudaFree(csrColInd_d[device]);
    }

    return time_runtime;
}


int main(int argc, char *argv[])
{
    omp_set_num_threads(omp_get_max_threads()); 

    std::cout << "omp_threads ::" << omp_get_max_threads() << std::endl; 

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
    delete_chkpt(progops.checkpointing, mpi);

    MPI_Finalize();

    return 0;
}
