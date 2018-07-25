#include "hybCalcFunc.cuh"

/*
* @brief Initializes the cuda handles
* @note
* @param nDevices: The number of GPU's to be used. For each MPI process this culd be different.
* @param further_split_for_gpu: vector of size nDevices containing the info about the the matrix part for each GPU
* @param streams: vector of size nDevices containing the cuda streams for each GPu for async ops.
* @param cusparseHandle: The vector of size nDevices containing the cusparse handles.
* @param cublasHandle: The vector of size nDevices containing the cublas handles.
* @param descr: The vector of size nDevices containing the cusparse descriptors for each GPU. 
* @retval Confirmation that the code exit normally.
*/
int initialize_cuda_handles(const int nDevices, 
                            const std::vector<further_split> further_split_for_gpu,
                            std::vector <cudaStream_t> &streams,
                            std::vector <cusparseHandle_t> &cusparseHandle,
                            std::vector <cublasHandle_t> &cublasHandle,
                            std::vector <cusparseMatDescr_t> &descr)
{
    
    for (size_t device =0 ; device < nDevices ; ++device){
        cudaSetDevice(further_split_for_gpu[device].id); 

        cudaStreamCreate(&streams[device]); 
        cusparseCreate(&cusparseHandle[device]); 
        cublasCreate(&cublasHandle[device]); 
        cublasSetStream(cublasHandle[device] ,streams[device]);

        cusparseCreateMatDescr(&descr[device]); 
        cusparseSetMatType(descr[device], CUSPARSE_MATRIX_TYPE_GENERAL); 
        cusparseSetMatIndexBase(descr[device], CUSPARSE_INDEX_BASE_ZERO); 
    }
    return 0;
}


/*
* @brief Find out the total rows to be solved on the GPU
* @note
* @param nDevices: The number of GPU's to be used. For each MPI process this culd be different.
* @param further_split_for_gpu: vector of size nDevices containing the info about the the matrix part for each GPU
* @param rows_gpu_total: The info about the rows that have to be loaded on the GPU's in total.
* @retval Confirmation that the code exit normally.
*/
int aggregated_rows_gpu(const int nDevices,
                        const std::vector<further_split> further_split_for_gpu, 
                        further_split &rows_gpu_total)
{
    
    if (nDevices > 0){
        rows_gpu_total.start = further_split_for_gpu[0].start;
        rows_gpu_total.end = further_split_for_gpu[nDevices - 1].end;
    }
    else{
        rows_gpu_total.start = 0;
        rows_gpu_total.end = 0;
    }

    if (nDevices > 0){
        for (int device = 0 ; device< nDevices ; ++device) {
            rows_gpu_total.num_rows += further_split_for_gpu[device].num_rows;
            rows_gpu_total.nnz += further_split_for_gpu[device].nnz;
        }   
    }
    return 0;
}


/*
* @brief Allocates memory and initialized the device arrays (by copying) data into them.
* @note
* @param nDevices: The number of GPU's to be used. For each MPI process this culd be different.
* @param nRows: The number of rows in the matrix.
* @param nColumns: The number of columns in the matrix.
* @param grid: The grid size to run GPU kernels.
* @param block: The block size to run GPU kernels.
* @param streams: vector of size nDevices containing the cuda streams for each GPu for async ops.
* @param further_split_for_gpu: vector of size nDevices containing the info about the the matrix part for each GPU
* @param csrRowInd: The csr Row array containing all the data to be loaded on the GPU.
* @param csrColInd: The csr Column array containing all the data to be loaded on the GPU.
* @param csrVal: The csr Values array containing all the data to be loaded on the GPU.
* @param lmsino_mask: The pointer to treat the lmsino as a float array instead of a vector.
* @param lmsino_d: The vector of size nDevices pointing to the lmsino array on the GPU.
* @param correlation_d: The vector of size nDevices pointing to the correlation array on the GPU.
* @param update_d: The vector of size nDevices pointing to the update array on the GPU.
* @param csrVal_d: The vector of size nDevices pointing to the csr Values on the GPU.
* @param norm_d: The vector of size nDevices pointing to the norm array on the GPU.
* @param image_d: The vector of size nDevices pointing to the image array on the GPU.
* @param csrRowInd_d: The vector of size nDevices pointing to the csr Row on the GPU.
* @param csrColInd_d: The vector of size nDevices pointing to the csr Column on the GPU.
* @retval Confirmation that the code exit normally.
*/
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
                                            int   **csrColInd_d)
{
    for (size_t device =0 ; device < nDevices ; ++device){
        cudaSetDevice(further_split_for_gpu[device].id); 
        
    // Allocating all the vectors needed on the device            
                // Float vectors
        cudaMalloc((void**)&correlation_d[device] , nRows*sizeof(float)); 
        cudaMalloc((void**)&update_d[device] , nColumns*sizeof(float));
        cudaMalloc((void**)&norm_d[device] , nColumns*sizeof(float));
        cudaMalloc((void**)&lmsino_d[device] , nRows*sizeof(float));
        cudaMalloc((void**)&image_d[device] , nColumns*sizeof(float)); 
        cudaMalloc((void**)&csrVal_d[device] , further_split_for_gpu[device].nnz*sizeof(float));
                    // Int vectors
        cudaMalloc((void**)&csrColInd_d[device] , further_split_for_gpu[device].nnz*sizeof(int));
        cudaMalloc((void**)&csrRowInd_d[device] , (further_split_for_gpu[device].num_rows + 1)*sizeof(int));

        // Finding out the proper index in the csrRow and csrVal+csrCol for each device
        size_t copy_nnz_index = 0;
        size_t copy_rows_index = 0;
        for (int i = 0; i < device ; i++){
            copy_nnz_index += further_split_for_gpu[i].nnz;
            copy_rows_index += further_split_for_gpu[i].num_rows;
        }

    // Copy the CsrRowInd on default stream(NULL) so the the next copies are overlapped.
        cudaMemcpyAsync(csrRowInd_d[device], 
                        &csrRowInd[copy_rows_index] , 
                        (further_split_for_gpu[device].num_rows + 1)*sizeof(int), 
                        cudaMemcpyHostToDevice, 
                        NULL);

    // Moving the index by the 1st value in csrRowInd to align it
        move_csrrow_ind <<<grid , block , 0 , NULL>>> ( csrRowInd_d[device], 
                                                        further_split_for_gpu[device].num_rows , 
                                                        csrRowInd[copy_rows_index]);

        cudaMemcpyAsync(lmsino_d[device] , 
                        lmsino_mask , 
                        nRows*sizeof(float), 
                        cudaMemcpyHostToDevice, 
                        streams[device]);

        cudaMemcpyAsync(csrVal_d[device], 
                        &csrVal[copy_nnz_index] , 
                        further_split_for_gpu[device].nnz*sizeof(float) , 
                        cudaMemcpyHostToDevice, 
                        streams[device]);

        cudaMemcpyAsync(csrColInd_d[device], 
                        &csrColInd[copy_nnz_index] ,
                        further_split_for_gpu[device].nnz*sizeof(int), 
                        cudaMemcpyHostToDevice, 
                        streams[device]);
    }

    // Performing the synchronization across all selected devices
    for (int device =0 ; device < nDevices ; ++device){
        cudaSetDevice(further_split_for_gpu[device].id); 
        cudaDeviceSynchronize(); 
    } 
    return 0;
}


/*
* @brief Calculates the norm 
* @note Could calculate in hybrid mode depending on the partitioning.
* @param matrix: The Csr4matrix containing all the data about the matrix.
* @param nDevices: The number of GPU's to be used. For each MPI process this culd be different.
* @param nRows: The number of rows in the matrix.
* @param nColumns: The number of columns in the matrix.
* @param grid: The grid size to run GPU kernels.
* @param block: The block size to run GPU kernels.
* @param streams: vector of size nDevices containing the cuda streams for each GPu for async ops.
* @param further_split_for_gpu: vector of size nDevices containing the info about the the matrix part for each GPU
* @param further_split_for_cpu: vector of size 1 containing the info about the the matrix part for the
* @param csrRowInd_d: The vector of size nDevices pointing to the csr Row on the GPU.
* @param csrColInd_d: The vector of size nDevices pointing to the csr Column on the GPU.
* @param csrVal_d: The vector of size nDevices pointing to the csr Values on the GPU.
* @param norm_d: The vector of size nDevices pointing to the norm array on the GPU.
* @param norm: The norm vector for the CPU.
* @retval Confirmation that the code exit normally.
*/
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
                            std::vector<float> &norm)
{

    for (size_t device =0 ; device < nDevices ; ++device){
        cudaSetDevice(further_split_for_gpu[device].id);    

    // Initializing the vectors on the device that need to be initialized
        cudaMemsetAsync(norm_d[device] , 0, nColumns*sizeof(float),streams[device]);

    // Calculating the norm
        trans_mat_unit_vec_mul_warp <<<grid, block, 0 , streams[device]>>> (0 , further_split_for_gpu[device].num_rows, 
            csrRowInd_d[device], csrColInd_d[device], csrVal_d[device], norm_d[device]); 
    }

    // CPU: Solving norm on part allocated to it
    if (further_split_for_cpu.num_rows > 0){
        #pragma omp parallel
        {
            Vector<float> private_norm(norm.size(), 0.0);

            #pragma omp for schedule(dynamic)
            for (int row= further_split_for_cpu.start ; row<further_split_for_cpu.end ; ++row) 
            {
                std::for_each(matrix.beginRow2(row), matrix.endRow2(row),[&](const RowElement<float>& e){
                    private_norm[e.column()] += e.value();
                });
            }
            
            #pragma omp critical
            {
                for (size_t i = 0; i < norm.size(); ++i) 
                    norm[i] += private_norm[i];
            }
        }
    }

    // Performing the synchronization across all selected devices
    for (int device =0 ; device < nDevices ; ++device){
        cudaSetDevice(further_split_for_gpu[device].id); 
        cudaDeviceSynchronize(); 
    } 

    return 0;
}


/*
* @brief Reduces the norm
* @note Could calculate in hybrid mode depending on the partitioning.
* @param nDevices: The number of GPU's to be used. For each MPI process this culd be different.
* @param nColumns: The number of columns in the matrix.
* @param streams: vector of size nDevices containing the cuda streams for each GPu for async ops.
* @param further_split_for_gpu: vector of size nDevices containing the info about the the matrix part for each GPU
* @param further_split_for_cpu: vector of size 1 containing the info about the the matrix part for the
* @param norm_d: The vector of size nDevices pointing to the norm array on the GPU.
* @param norm: The norm vector for the CPU.
* @param reduce_op_mem: The memory for performing the reduction operations.
* @retval Confirmation that the code exit normally.
*/
int norm_reduction_hyb( const int nDevices,
                        const size_t nColumns,
                        const std::vector <cudaStream_t> streams,
                        const std::vector<further_split> further_split_for_gpu,
                        const further_split further_split_for_cpu,
                        float **norm_d,
                        std::vector<float> &norm,
                        float *reduce_op_mem)
{   
    memset(reduce_op_mem, 0 , nDevices*nColumns*sizeof(float));

    for (size_t device =0 ; device < nDevices ; ++device){
            cudaSetDevice(further_split_for_gpu[device].id); 
        // Copying the norm back from device
            cudaMemcpyAsync(&reduce_op_mem[device*nColumns], 
                            norm_d[device], 
                            nColumns*sizeof(float), 
                            cudaMemcpyDeviceToHost, 
                            streams[device]); 
        }

    // Performing the synchronization across all selected devices
        for (int device =0 ; device < nDevices ; ++device){
            cudaSetDevice(further_split_for_gpu[device].id); 
            cudaDeviceSynchronize(); 
        }
        
    // Reducing the norm copied back by each device
        if (nDevices > 0){
            #pragma omp parallel for schedule(dynamic)
            for (size_t column= 0; column< nColumns ; column++ ){
                for (int device =1 ; device < nDevices ; ++device){
                    reduce_op_mem[column] += reduce_op_mem[device*nColumns + column];
                }
            }
        }
        
    // Adding the norm calculated by the cpu in the already reduced norm 
        if (further_split_for_cpu.num_rows > 0){
            #pragma omp parallel for schedule(dynamic)
            for (size_t column=0; column<nColumns; ++column){ 
                reduce_op_mem[column] += norm[column];
            } 
        }

    // Reducing the norm across all the mpi processes
        MPI_Allreduce(MPI_IN_PLACE, reduce_op_mem, nColumns, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    // Copying the reduced norm in each device
        for (int device =0 ; device < nDevices ; ++device){
            cudaSetDevice(further_split_for_gpu[device].id); 
            cudaMemcpyAsync(norm_d[device], reduce_op_mem, nColumns*sizeof(float), cudaMemcpyHostToDevice, streams[device]); 
        }

    // Coping the reduced norm in the cpu also
        if (further_split_for_cpu.num_rows > 0){
            #pragma omp parallel for schedule(dynamic)
            for (uint32_t column=0; column<nColumns; ++column) {
                norm[column] = reduce_op_mem[column];
            }
        }

    // Performing the synchronization across all selected devices
        for (int device =0 ; device < nDevices ; ++device){
            cudaSetDevice(further_split_for_gpu[device].id); 
            cudaDeviceSynchronize(); 
        } 

    return 0;
}


/* 
* @brief Initializing the image
* @note Could calculate in hybrid mode depending on the partitioning.
* @param mpi: The vector containing the mpi data  
* @param nDevices: The number of GPU's to be used. For each MPI process this culd be different.
* @param nRows: The number of rows in the matrix.
* @param nColumns: The number of columns in the matrix.
* @param grid: The grid size to run GPU kernels.
* @param block: The block size to run GPU kernels.
* @param further_split_for_gpu: vector of size nDevices containing the info about the the matrix part for each GPU
* @param further_split_for_cpu: vector of size 1 containing the info about the the matrix part for the
* @param cublasHandle: The vector of size nDevices containing the cublas handles.
* @param streams: vector of size nDevices containing the cuda streams for each GPu for async ops.
* @param lmsino: The vector containing the lmsino data.
* @param image: The vector containing the image data.
* @param iter_alt: The iterration number from which to start the code.
* @param image_d: The vector of size nDevices pointing to the image array on the GPU.
* @param image_mask: The pointer to treat the image as a float array instead of a vector.
* @param lmsino_d: The vector of size nDevices pointing to the lmsino array on the GPU.
* @param norm_d: The vector of size nDevices pointing to the norm array on the GPU.
* @param reduce_op_mem: The memory for performing the reduction operations.
* @retval Confirmation that the code exit normally.
*/
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
                                    float *reduce_op_mem)
{

    // Decide whether restart from checkpoint or fill intial estimates
    if(exists(IMG_CHECKPOINT_NAME) && exists(ITER_CHECKPOINT_NAME)){

        restore(image, IMG_CHECKPOINT_NAME, iter_alt, ITER_CHECKPOINT_NAME);

        for (int device =0 ; device < nDevices ; ++device){
            cudaSetDevice(further_split_for_gpu[device].id); 
            
            cudaMemcpy( image_d[device], 
                        image_mask, 
                        nColumns*sizeof(float), 
                        cudaMemcpyHostToDevice); CUDA_CHECK;
        }
#ifndef MESSUNG
        std::cout << "MLEM [" << mpi.rank << "/" << mpi.size << "]: "
                  << "I got Checkpoint at iternation " << iter_alt << "\n";
#endif
    }
    else{

        float sumnorm = 0.0;
        float sumin = 0.0;
        if (nDevices > 0){
            cudaSetDevice(further_split_for_gpu[0].id); 
            // Calculating the sumin
            cublasSasum(cublasHandle[0], nRows, lmsino_d[0], 1, &sumin); 
            // Calculating the sumnorm
            cublasSasum(cublasHandle[0], nColumns, norm_d[0], 1, &sumnorm); 
            cudaDeviceSynchronize(); 
        }
        else{
            for (size_t row=0; row<nRows; ++row) sumin += lmsino[row];
            for (size_t column=0; column<nColumns; ++column) sumnorm += reduce_op_mem[column];
        }

        float initial = static_cast <float> (sumin/sumnorm);

#ifndef MESSUNG
        std::cout << "MLEM [" << mpi.rank << "/" << mpi.size << "]: " <<
            std::setprecision(10) <<"sumnorm: " << sumnorm << std::endl; 

        std::cout << "MLEM [" << mpi.rank << "/" << mpi.size << "]: " <<
            std::setprecision(10) << "sumin: " << sumin << std::endl; 

        std::cout << "MLEM [" << mpi.rank << "/" << mpi.size << "]: " <<
            std::setprecision(10) << "initial: " << initial << std::endl; 
#endif

        // Initializing the image_d vector on each device with initial value
        for (int device =0 ; device < nDevices ; ++device){
            cudaSetDevice(further_split_for_gpu[device].id); 
            d_set_value <<<grid, block, 0 , streams[device]>>> (image_d[device] , initial , nColumns); 
        }

        // Initializing the image vector on cpu with initial value
        if (further_split_for_cpu.num_rows > 0){
            #pragma omp parallel for schedule(dynamic)
            for (size_t i=0; i<nColumns; ++i){
                image_mask[i] = initial;
            }
        }

         // Performing the synchronization across all selected devices
        for (int device =0 ; device < nDevices ; ++device){
            cudaSetDevice(further_split_for_gpu[device].id); 
            cudaDeviceSynchronize(); 
        } 
    }
    return 0;
}


/*
* @brief Calculates the forward projection
* @note Could calculate in hybrid mode depending on the partitioning.
* @param matrix: The Csr4matrix containing all the data about the matrix.
* @param nDevices: The number of GPU's to be used. For each MPI process this culd be different.
* @param nRows: The number of rows in the matrix.
* @param nColumns: The number of columns in the matrix.
* @param grid: The grid size to run GPU kernels.
* @param block: The block size to run GPU kernels.
* @param further_split_for_gpu: vector of size nDevices containing the info about the the matrix part for each GPU
* @param further_split_for_cpu: vector of size 1 containing the info about the the matrix part for the
* @param streams: vector of size nDevices containing the cuda streams for each GPu for async ops.
* @param csrRowInd_d: The vector of size nDevices pointing to the csr Row on the GPU.
* @param csrColInd_d: The vector of size nDevices pointing to the csr Column on the GPU.
* @param csrVal_d: The vector of size nDevices pointing to the csr Values on the GPU.
* @param correlation_d: The vector of size nDevices pointing to the correlation array on the GPU.
* @param image_d: The vector of size nDevices pointing to the image array on the GPU.
* @param image_mask: The pointer to treat the image as a float array instead of a vector.
* @param correlation: The correlation vector for the CPU.
* @retval Confirmation that the code exit normally.
*/
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
                            std::vector<float> &correlation)
{    
        for (int device =0 ; device < nDevices ; ++device){
            cudaSetDevice(further_split_for_gpu[device].id); 

            // Initialization
            cudaMemsetAsync(correlation_d[device], 0, nRows*sizeof(float), streams[device]); 
            
            // calculating forward projection GPU
            mat_vec_mul_warp <<<grid, block, block.x*sizeof(float) , streams[device]>>> (   0, 
                                                                        further_split_for_gpu[device].num_rows , 
                                                                        csrRowInd_d[device], 
                                                                        csrColInd_d[device], 
                                                                        csrVal_d[device] , 
                                                                        image_d[device] , 
                                                                        &correlation_d[device][further_split_for_gpu[device].start]);
        }

        // calculating forward projection CPU
        #pragma omp parallel for schedule(dynamic)
        for (size_t row=further_split_for_cpu.start ; row< further_split_for_cpu.end ; ++row) 
        {
            double res = 0.0;
            
            std::for_each(matrix.beginRow2(row), matrix.endRow2(row),
                          [&](const RowElement<float>& e){ res += e.value() * image_mask[e.column()]; });
            correlation[row] = (float)res;
        }

        // Performing the synchronization across all selected devices
        for (int device =0 ; device < nDevices ; ++device){
            cudaSetDevice(further_split_for_gpu[device].id); 
            cudaDeviceSynchronize(); 
        }
    return 0;
}


/*
* @brief Calculates the correlation
* @note Could calculate in hybrid mode depending on the partitioning.
* @param nDevices: The number of GPU's to be used. For each MPI process this culd be different.
* @param grid: The grid size to run GPU kernels.
* @param block: The block size to run GPU kernels.
* @param streams: vector of size nDevices containing the cuda streams for each GPu for async ops.
* @param further_split_for_gpu: vector of size nDevices containing the info about the the matrix part for each GPU
* @param further_split_for_cpu: vector of size 1 containing the info about the the matrix part for the
* @param lmsino: The vector containing the lmsino data.
* @param correlation_d: The vector of size nDevices pointing to the correlation array on the GPU.
* @param lmsino_d: The vector of size nDevices pointing to the lmsino array on the GPU.
* @param correlation: The correlation vector for the CPU.
* @retval Confirmation that the code exit normally.
*/
int calc_correlation_hyb(   const int nDevices,
                            const dim3 grid,
                            const dim3 block,
                            const std::vector<cudaStream_t> streams,
                            const std::vector <further_split> further_split_for_gpu,
                            const further_split further_split_for_cpu,
                            const Vector<int> &lmsino,
                            float **correlation_d,
                            float **lmsino_d,
                            std::vector<float> &correlation)
{
    for (int device =0 ; device < nDevices ; ++device){
            cudaSetDevice(further_split_for_gpu[device].id); 
            // Calculating Correlation on GPU
            d_calcCorrel <<<grid , block, 0 , streams[device]>>> (  &correlation_d[device][further_split_for_gpu[device].start], 
                                                                    &lmsino_d[device][further_split_for_gpu[device].start], 
                                                                    &correlation_d[device][further_split_for_gpu[device].start] , 
                                                                    further_split_for_gpu[device].num_rows); 
        }

        // calculating correlation on CPU
        #pragma omp parallel for schedule(dynamic)
        for (size_t row=further_split_for_cpu.start; row<further_split_for_cpu.end; ++row)
        {
            correlation[row] = (correlation[row] != 0.0) ? (float)(lmsino[row] / correlation[row]) : 0.0;
        } 

        // Performing the synchronization across all selected devices
        for (int device =0 ; device < nDevices ; ++device){
            cudaSetDevice(further_split_for_gpu[device].id); 
            cudaDeviceSynchronize(); 
        } 
    return 0;
}


/*
* @brief Calculates the Backward Projection
* @note Could calculate in hybrid mode depending on the partitioning.
* @param matrix: The Csr4matrix containing all the data about the matrix.
* @param nDevices: The number of GPU's to be used. For each MPI process this culd be different.
* @param nColumns: The number of columns in the matrix.
* @param grid: The grid size to run GPU kernels.
* @param block: The block size to run GPU kernels.
* @param streams: vector of size nDevices containing the cuda streams for each GPu for async ops.
* @param further_split_for_gpu: vector of size nDevices containing the info about the the matrix part for each GPU
* @param further_split_for_cpu: vector of size 1 containing the info about the the matrix part for the
* @param csrRowInd_d: The vector of size nDevices pointing to the csr Row on the GPU.
* @param csrColInd_d: The vector of size nDevices pointing to the csr Column on the GPU.
* @param csrVal_d: The vector of size nDevices pointing to the csr Values on the GPU.
* @param update_d: The vector of size nDevices pointing to the update array on the GPU.
* @param correlation_d: The vector of size nDevices pointing to the correlation array on the GPU.
* @param correlation: The correlation vector for the CPU.
* @param update: The update vector for the CPU.
* @retval Confirmation that the code exit normally.
*/
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
                            std::vector<float> &update)
{
    for (int device =0 ; device < nDevices ; ++device){
        cudaSetDevice(further_split_for_gpu[device].id); 
        
        // Calculating backward projection GPU
        cudaMemsetAsync(update_d[device], 0, nColumns*sizeof(float), streams[device]); 

        trans_mat_vec_mul_warp <<<grid, block, 0 , streams[device]>>> ( 0, 
                                                                        further_split_for_gpu[device].num_rows, 
                                                                        csrRowInd_d[device],    
                                                                        csrColInd_d[device], 
                                                                        csrVal_d[device],
                                                                        &correlation_d[device][further_split_for_gpu[device].start], 
                                                                        update_d[device]); 
    }

    // Calculating backward projection CPU
    #pragma omp parallel
    {
        Vector<double> private_update(update.size(), 0.0);

        #pragma omp for schedule(dynamic)
        for (uint32_t row=further_split_for_cpu.start; row<further_split_for_cpu.end; ++row) 
        {
            std::for_each(matrix.beginRow2(row), matrix.endRow2(row), [&](const RowElement<float>& e){
             private_update[e.column()] += (double)e.value() * (double)correlation[row]; });
        }
        
        #pragma omp critical
        {
            for (size_t i = 0; i < update.size(); ++i){
                update[i] += (float)private_update[i];
            }
        }
    }

    // Performing the synchronization across all selected devices
    for (int device =0 ; device < nDevices ; ++device){
        cudaSetDevice(further_split_for_gpu[device].id); 
        cudaDeviceSynchronize(); 
    } 
    return 0;
}


/* 
* @brief Reduces the Backward Projection
* @note Could calculate in hybrid mode depending on the partitioning.
* @param nDevices: The number of GPU's to be used. For each MPI process this culd be different.
* @param nColumns: The number of columns in the matrix.
* @param streams: vector of size nDevices containing the cuda streams for each GPu for async ops.
* @param further_split_for_gpu: vector of size nDevices containing the info about the the matrix part for each GPU
* @param further_split_for_cpu: vector of size 1 containing the info about the the matrix part for the
* @param update: The update vector for the CPU.
* @param update_d: The vector of size nDevices pointing to the update array on the GPU.
* @param reduce_op_mem: The memory for performing the reduction operations.
* @retval Confirmation that the code exit normally.
*/
int redc_backward_proj_hyb( const int nDevices,
                            const size_t nColumns,
                            const std::vector<cudaStream_t> streams,
                            const std::vector <further_split> further_split_for_gpu,
                            const further_split further_split_for_cpu,
                            const std::vector <float> &update,
                            float **update_d,
                            float *reduce_op_mem)
{
    for (int device =0 ; device < nDevices ; ++device){
        cudaSetDevice(further_split_for_gpu[device].id); 
        // Copying the calculated vector back
        cudaMemcpyAsync(&reduce_op_mem[device*nColumns], 
                        update_d[device], 
                        nColumns*sizeof(float), 
                        cudaMemcpyDeviceToHost, 
                        streams[device]); 
    }

    // Performing the synchronization across all selected devices
    for (int device =0 ; device < nDevices ; ++device){
        cudaSetDevice(further_split_for_gpu[device].id); 
        cudaDeviceSynchronize();
    } 
  
    // Reducing the update copied back by each device
    if (nDevices > 1){
        #pragma omp parallel for schedule(dynamic)
        for (size_t column= 0; column< nColumns ; ++column ){
            for (int device =1 ; device < nDevices ; ++device){
                reduce_op_mem[column] += reduce_op_mem[device*nColumns + column];
            }
        }
    }      

    // Adding the update calculated by the cpu into the already reduced update 
    if (further_split_for_cpu.num_rows > 0){
        #pragma omp parallel for schedule(dynamic)
        for (size_t column= 0; column< nColumns ; ++column ) 
            reduce_op_mem[column] += update[column];
    }

    // Reducing the update across all the mpi processes
    MPI_Allreduce(MPI_IN_PLACE, reduce_op_mem, nColumns, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    return 0;
}


/*
* @brief Updates the image
* @note Could calculate in hybrid mode depending on the partitioning.
* @param nDevices: The number of GPU's to be used. For each MPI process this culd be different.
* @param nColumns: The number of columns in the matrix.
* @param grid: The grid size to run GPU kernels.
* @param block: The block size to run GPU kernels.
* @param streams: vector of size nDevices containing the cuda streams for each GPu for async ops.
* @param further_split_for_gpu: vector of size nDevices containing the info about the the matrix part for each GPU
* @param further_split_for_cpu: vector of size 1 containing the info about the the matrix part for the
* @param image_mask: The pointer to treat the image as a float array instead of a vector.
* @param reduce_op_mem: The memory for performing the reduction operations.
* @param update: The update vector for the CPU.
* @param norm: The norm vector for the CPU.
* @param image_d: The vector of size nDevices pointing to the image array on the GPU.
* @param update_d: The vector of size nDevices pointing to the update array on the GPU.
* @param norm_d: The vector of size nDevices pointing to the norm array on the GPU.
* @retval Confirmation that the code exit normally.
*/
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
                    float **norm_d)
{
    if (further_split_for_cpu.num_rows > 0){
            #pragma omp parallel for schedule(dynamic)
            for (size_t i=0; i<update.size(); ++i){
                image_mask[i] *= (norm[i] != 0.0) ? (reduce_op_mem[i] / norm[i]) : reduce_op_mem[i];
            }

            for (int device =0 ; device < nDevices ; ++device){
                cudaSetDevice(further_split_for_gpu[device].id);            
                
                cudaMemcpyAsync(image_d[device], 
                                image_mask, 
                                nColumns*sizeof(float), 
                                cudaMemcpyHostToDevice, 
                                streams[device]);
            }
        }
        else{
            for (int device =0 ; device < nDevices ; ++device){
                cudaSetDevice(further_split_for_gpu[device].id); 
                
                cudaMemcpyAsync(update_d[device], 
                                reduce_op_mem, 
                                nColumns*sizeof(float), 
                                cudaMemcpyHostToDevice, 
                                streams[device]);
                
                d_update <<<grid, block, 0 , streams[device]>>>(update_d[device], 
                                                                norm_d[device], 
                                                                image_d[device], 
                                                                nColumns);
            }
        }

        for (int device =0 ; device < nDevices ; ++device){
            cudaSetDevice(further_split_for_gpu[device].id); 
            cudaDeviceSynchronize();
        }  
    return 0;
}


/*
* @brief Calculates the image sum
* @note Could calculate in hybrid mode depending on the partitioning.
* @param nDevices: The number of GPU's to be used. For each MPI process this culd be different.
* @param nColumns: The number of columns in the matrix.
* @param further_split_for_gpu: vector of size nDevices containing the info about the the matrix part for each GPU
* @param cublasHandle: The vector of size nDevices containing the cublas handles.
* @param image_d: The vector of size nDevices pointing to the image array on the GPU.
* @param image_mask: The pointer to treat the image as a float array instead of a vector.
* @param image_sum: The sum of the image.
* @retval Confirmation that the code exit normally.
*/
int img_sum_hyb(const int nDevices,
                const size_t nColumns,
                const std::vector <further_split> further_split_for_gpu,
                const std::vector <cublasHandle_t> cublasHandle,
                float **image_d,
                float *image_mask,
                float &image_sum)
{
    image_sum = 0.0;
    if (nDevices > 0){
        cudaSetDevice(further_split_for_gpu[0].id); 
        cublasSasum(cublasHandle[0], nColumns, image_d[0] , 1, &image_sum );
    }
    else{
        for(size_t i = 0; i < nColumns; ++i) image_sum += image_mask[i];
    }
    return 0;
}


/*
* @brief Creates the checkoint
* @note Could calculate in hybrid mode depending on the partitioning.
* @param nColumns: The number of columns in the matrix.
* @param further_split_for_gpu: vector of size nDevices containing the info about the the matrix part for each GPU
* @param further_split_for_cpu: vector of size 1 containing the info about the the matrix part for the
* @param iter: The current iteration number.
* @param checkpointing: The iteration number on which checkpoint will be created.
* @param mpi: The vector containing the mpi data 
* @param image: The vector containing the image data. 
* @param image_mask: The pointer to treat the image as a float array instead of a vector.
* @param image_d: The vector of size nDevices pointing to the image array on the GPU.
* @retval Confirmation that the code exit normally.
*/
int creating_chkpt_hyb( const size_t nColumns,
                        const std::vector <further_split> further_split_for_gpu,
                        const further_split further_split_for_cpu,
                        const int iter,
                        const int checkpointing,
                        const MpiData mpi,
                        Vector <float> &image,
                        float *image_mask,
                        float **image_d)
{
    if (checkpointing > 0){
        if ((iter + 1)%checkpointing == 0){
            if (further_split_for_cpu.num_rows > 0){
                checkPointNow(  mpi,
                                image,
                                iter+1,
                                IMG_CHECKPOINT_NAME,
                                ITER_CHECKPOINT_NAME);
                
                std::cout   << "MLEM [" << mpi.rank << "/" << mpi.size << "]: " 
                            << "creating checkpoint " << std::endl;
            }
            else{
                cudaSetDevice(further_split_for_gpu[0].id);
                
                cudaMemcpy( image_mask, 
                            image_d[0], 
                            nColumns*sizeof(float), 
                            cudaMemcpyDeviceToHost);

                checkPointNow(  mpi,
                                image,
                                iter+1,
                                IMG_CHECKPOINT_NAME,
                                ITER_CHECKPOINT_NAME);

                std::cout   << "MLEM [" << mpi.rank << "/" << mpi.size << "]: " 
                            << "creating checkpoint " << std::endl;
            }
        }
    }

    return 0;
}


/*
* @brief Used to delete the checkpoint when the code is exiting normally
* @note 
* @param checkpointing: The iteration number on which checkpoint will be created.
* @param mpi: The vector containing the mpi data 
* @retval Confirmation that the code exit normally.
*/
int delete_chkpt(const int checkpointing , const MpiData& mpi)
{
    if (checkpointing > 0 && mpi.rank == 0){
            remove(IMG_CHECKPOINT_NAME);
            remove(ITER_CHECKPOINT_NAME);
        }
    return 0;
}