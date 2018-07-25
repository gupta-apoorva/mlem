#include "cudaKernels.cuh"

// cuda error checking
string prev_file = "";
int prev_line = 0;
void cuda_check(string file, int line)
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        cout << endl << file << ", line " << line << ": " << cudaGetErrorString(e) << " (" << e << ")" << endl;
        if (prev_line>0) cout << "Previous CUDA call:" << endl << prev_file << ", line " << prev_line << endl;
        exit(1);
    }
    prev_file = file;
    prev_line = line;
}

/*
* @brief Calculate the correlation on the gpu
* @note It can only be launched from the host
* @param fwproj_d: The forward projection vector on the device
* @param lmsino_d: x vector on the device
* @param correlation_d: The Correlation vector on the device
* @param  nRows: The number of rows that needs to be processed
* @retval NONE
*/
__global__ void d_calcCorrel(   const float* fwproj_d, 
                                const float* lmsino_d, 
                                float* correlation_d, 
                                const uint32_t nRows)
{
    uint32_t i = threadIdx.x + blockIdx.x*blockDim.x;

    if (i < nRows){
    correlation_d[i] = (fwproj_d[i] != 0) ? (float)lmsino_d[i]/fwproj_d[i] : 0.0;  
    }
}


/*
* @brief Calculate the correlation on the gpu
* @note It can only be launched from the host
* @param update_d: The update vector on the device
* @param norm_d: norm vector on the device
* @param image_d: The image vector on the device
* @param  nRows: The number of columns that needs to be processed
* @retval NONE
*/
__global__ void d_update(   const float* update_d, 
                            const float* norm_d,  
                            float* image_d, 
                            const int cols)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    if (i < cols){
    image_d[i] *= (norm_d[i] != 0) ? update_d[i]/norm_d[i] : update_d[i];  
    }
}

/*
* @brief Sets the value in the device if required apart from zero
* @note It can only be launched from the host
* @param update_d: The vector which has to be initialized
* @param value: The value to be set
* @param  nRows: The number of elements to be initialized 
* @retval NONE
*/
__global__ void d_set_value(float* rowVector_d , 
                            const float value,  
                            const size_t num_elements)
{
    size_t i = threadIdx.x + blockIdx.x*blockDim.x;

    if (i<num_elements)
        rowVector_d[i] = value;
}

/*
* @brief Multiplies a csr matrix to a vector
* @note It can only be launched from the host
* @param result:The vector in which the result will be saved
* @param vec: The vector to be multiplied
* @param csrval_d: The csr values in a vector on the device
* @param csrRowInd_d: The csr row in a vector on the device
* @param csrColInd_d: csr column indexes in a vector on the device
* @param rows: The num of rows in the csr matrix
* @retval NONE
*/
__global__ void d_mat_vec_mul(  float* result, 
                                const float* vec, 
                                const float* csrVal_d, 
                                const int* csrRowInd_d, 
                                const int* csrColInd_d, 
                                const uint32_t rows)
{
    uint32_t i = threadIdx.x + blockIdx.x*blockDim.x;

    if (i<rows){
        int cols_start = csrRowInd_d[i];
        int nnz_ele_row = csrRowInd_d[i+1] - cols_start ;
        uint32_t col_num;
        float temp = 0;
        for (int j=0; j< nnz_ele_row ; j++){
            col_num = (uint32_t) csrColInd_d[cols_start + j];
            temp += vec[col_num]*csrVal_d[cols_start + j];
        }
        result[i] = temp;
    }
}

/*
* @brief Multiplies a csr matrix transpose to a vector
* @note It can only be launched from the host
* @param result:The vector in which the result will be saved
* @param vec: The vector to be multiplied
* @param csrval_d: The csr values in a vector on the device
* @param csrRowInd_d: The csr row in a vector on the device
* @param csrColInd_d: csr column indexes in a vector on the device
* @param rows: The num of rows in the csr matrix
* @retval NONE
*/
__global__ void d_tran_mat_vec_mul( float* result, 
                                    const float* vec , 
                                    const float* csrVal_d, 
                                    const int* csrRowInd_d, 
                                    const int* csrColInd_d, 
                                    const uint32_t rows)
{
    uint32_t i = threadIdx.x + blockIdx.x*blockDim.x;

    if (i<rows){
        int row_start = csrRowInd_d[i] ;
        int row_end = csrRowInd_d[i+1] ;
        for (int j=row_start; j< row_end ; j++){
            atomicAdd((float*)&result[csrColInd_d[j]], vec[i]*csrVal_d[j]);
        }
    }
}

/*
* @brief Multiplies a csr matrix transpose to a vector making use of warp to reduce the global reads
* @note It can only be launched from the host
* @param nnz_to_skip: useful when the full csrRowInd vector is there on the device but not the csrVal and csrColInd 
could be loaded. Then nnz has to be shifted in csrRowInd to get the correct results.
* @param result: The vector in which the result will be saved
* @param vector: The vector to be multiplied
* @param csrval_d: The csr values in a vector on the device
* @param csrRowInd_d: The csr row in a vector on the device
* @param csrColInd_d: csr column indexes in a vector on the device
* @param rows: The num of rows in the csr matrix
* @retval NONE
*/
__global__ void trans_mat_vec_mul_warp( const uint32_t nnz_to_skip, 
                                        const uint32_t rows,    
                                        const int* csrRowInd_d, 
                                        const int* csrColInd_d, 
                                        const float* csrVal_d, 
                                        const float* vector, 
                                        float* result)
{

uint32_t thread_id = blockDim.x * blockIdx.x + threadIdx.x;

uint32_t thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp

uint32_t warp_id = thread_id / WARP_SIZE; // global warp index
// total number of active warps
uint32_t num_warps = (blockDim.x / WARP_SIZE) * gridDim.x;
for(uint32_t row=warp_id; row < rows ; row+=num_warps)
    {
        uint32_t row_start = csrRowInd_d[row];
        uint32_t row_end = csrRowInd_d[row+1];
        for (uint32_t i=row_start+thread_lane; i < row_end;i+=WARP_SIZE)
            atomicAdd(&result[csrColInd_d[i - nnz_to_skip]], csrVal_d[i - nnz_to_skip] * vector[row]);
    }
}

/*
* @brief Multiplies a csr matrix transpose to a vector making use of warp to reduce the global reads
* @note It can only be launched from the host. The vector consist of only 1's.
* @param nnz_to_skip: useful when the full csrRowInd vector is there on the device but not the csrVal and csrColInd 
could be loaded. Then nnz has to be shifted in csrRowInd to get the correct results.
* @param result: The vector in which the result will be saved
* @param csrval_d: The csr values in a vector on the device
* @param csrRowInd_d: The csr row in a vector on the device
* @param csrColInd_d: csr column indexes in a vector on the device
* @param rows: The num of rows in the csr matrix
* @retval NONE
*/
__global__ void trans_mat_unit_vec_mul_warp(    const uint32_t nnz_to_skip, 
                                                const uint32_t rows, 
                                                const int* csrRowInd_d, 
                                                const int* csrColInd_d, 
                                                const float* csrVal_d, 
                                                float* result)
{
uint32_t thread_id = blockDim.x * blockIdx.x + threadIdx.x;

uint32_t thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp

uint32_t warp_id = thread_id / WARP_SIZE; // global warp index
// total number of active warps
uint32_t num_warps = (blockDim.x / WARP_SIZE) * gridDim.x;
for( uint32_t row=warp_id; row < rows ; row+=num_warps)
    {
    uint32_t row_start = csrRowInd_d[row];
    uint32_t row_end = csrRowInd_d[row + 1];
    for (uint32_t i=row_start+thread_lane; i < row_end; i+=WARP_SIZE)
        atomicAdd(&result[csrColInd_d[i - nnz_to_skip]], csrVal_d[i - nnz_to_skip] * 1.0);
    }
}

/*
* @brief Multiplies a csr matrix to a vector making use of warp to reduce the global reads
* @note It can only be launched from the host
* @param nnz_to_skip: useful when the full csrRowInd vector is there on the device but not the csrVal and csrColInd 
could be loaded. Then nnz has to be shifted in csrRowInd to get the correct results.
* @param result: The vector in which the result will be saved
* @param vector: The vector to be multiplied
* @param csrval_d: The csr values in a vector on the device
* @param csrRowInd_d: The csr row in a vector on the device
* @param csrColInd_d: csr column indexes in a vector on the device
* @param rows: The num of rows in the csr matrix
* @retval
*/
__global__ void mat_vec_mul_warp (  const uint32_t nnz_to_skip, 
                                    const int rows , 
                                    const int* csrRowInd_d , 
                                    const int* csrColInd_d , 
                                    const float* csrVal_d , 
                                    const float* vector , 
                                    float* result)
{
    extern __shared__ float vals [];

    uint32_t thread_id = blockDim.x * blockIdx.x + threadIdx.x;

    uint32_t thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp

    uint32_t warp_id = thread_id / WARP_SIZE; // global warp index
    // total number of active warps
    uint32_t num_warps = (blockDim.x / WARP_SIZE) * gridDim.x;
    // one warp per row
    for ( uint32_t row = warp_id; row < rows ; row += num_warps)
        {
            uint32_t row_start = csrRowInd_d [ row ];
            uint32_t row_end = csrRowInd_d [ row +1];
            // compute running sum per thread
            vals [ threadIdx.x ] = 0.0;
            for ( uint32_t jj = row_start + thread_lane ; jj < row_end ; jj += WARP_SIZE){
                vals [ threadIdx.x ] += csrVal_d [ jj - nnz_to_skip ] * vector [ csrColInd_d [ jj -nnz_to_skip]];
            }
            // first thread writes the result
            if ( thread_lane == 0){
                for (int i =1 ; i<WARP_SIZE ; i++)
                    vals[threadIdx.x] += vals[threadIdx.x + i];
                atomicAdd(&result[row], vals[threadIdx.x]);
            }

            __syncthreads();
        }
}

/*
* @brief Shifts the values in the vector by the required amount
* @note Used mostly to shift values in the csr Row vector
* @param csrRowInd_d: The csr row in a vector on the device
* @param rows: The num of rows in the csr matrix
* @param how_much_to_move: The amount by which the values needs to be shifted.
* @retval
*/
__global__ void move_csrrow_ind(int* csrRowInd_d, 
                                size_t rows, 
                                size_t how_much_to_move)
{
    size_t thread_id = blockDim.x * blockIdx.x + threadIdx.x;

    if (thread_id < rows + 1){
        csrRowInd_d[thread_id] = csrRowInd_d[thread_id] - how_much_to_move;
    }
}


