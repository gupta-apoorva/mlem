#include "decideConfig.cuh"


/*
* @brief Checks that the values extracted from the configuration file are in range or not
* @note
* @param nDevice: Number of devices reqested in the config file
* @param max_nnz: Maximum number of NNZ requested in the config file
* @param max_temp: Maximum temp of the device requested in the configuration file
* @param min_free_mem: Maximum free memory requested in the configuretion file
* @param max_mem_uti: Maximum percentage of memory utilization requested in the configuretion file
* @param max_gpu_uti: Maximum percentage of gpu utilization requested in the configuretion file
* @nnz_this_rank: number of nnz allocated to this mpi process
* @retval NONE
*/
void sanity_check_on_values(int64_t &nDevice, 
                            int64_t &max_nnz,
                            int64_t &max_temp, 
                            int64_t &min_free_mem, 
                            int64_t &max_mem_uti, 
                            int64_t &max_gpu_uti, 
                            const size_t nnz_this_rank)
{
    
    if (nDevice < 0 || nDevice > MAX_NUM_DEVICES_DEFAULT){
        std::cout << "max_devices_to_use OUT OF RANGE [0,8]. DEFAULT value = 3 will be used. " << std::endl;
        nDevice = MAX_NUM_DEVICES_DEFAULT;
    }

    if (max_nnz < 0 || max_nnz > nnz_this_rank){
        std::cout << "max_nnz_per_device OUT OF RANGE [0," << nnz_this_rank << 
            "]. DEFAULT value = nnz_this_rank will be used. " << std::endl; 
        max_nnz = nnz_this_rank;
    }

    if (max_temp < 0 || max_temp > MAX_TEMP_DEFAULT){
        std::cout << "max_temp_allowed OUT OF RANGE [0,100]. DEFAULT value = 100 will be used." << std::endl; 
        max_temp = MAX_TEMP_DEFAULT;
    }

    if (min_free_mem < MIN_FREE_MEM_DEFAULT || min_free_mem >100){
        std::cout << "min_free_mem_req OUT OF RANGE [0,100]. DEFAULT value = 0 will be used." << std::endl; 
        min_free_mem = MIN_FREE_MEM_DEFAULT; 
    }

    if (max_mem_uti < 0 || max_mem_uti >MAX_MEM_UTI_DEFAULT){
        std::cout << "max_mem_uti_allowed OUT OF RANGE [0,100]. DEFAULT value = 100 will be used." << std::endl; 
        max_mem_uti = MAX_MEM_UTI_DEFAULT; 
    }

    if (max_gpu_uti < 0 || max_gpu_uti >MAX_GPU_UTI_DEFAULT){
        std::cout << "max_gpu_uti_allowed OUT OF RANGE [0,100]. DEFAULT value = 100 will be used." << std::endl; 
        max_gpu_uti = MAX_GPU_UTI_DEFAULT; 
    }
}


/*
* @brief Reads the configuration file to find out the requested configuration
* @note
* @param nDevice: Number of devices reqested in the config file
* @param max_nnz: Maximum number of NNZ requested in the config file
* @param max_temp: Maximum temp of the device requested in the configuration file
* @param min_free_mem: Maximum free memory requested in the configuretion file
* @param max_mem_uti: Maximum percentage of memory utilization requested in the configuretion file
* @param max_gpu_uti: Maximum percentage of gpu utilization requested in the configuretion file
* @nnz_this_rank: number of nnz allocated to this mpi process
* @retval NONE
*/
int get_requested_config(   int64_t &nDevice, 
                            int64_t &max_nnz, 
                            int64_t &max_temp, 
                            int64_t &min_free_mem, 
                            int64_t &max_mem_uti, 
                            int64_t &max_gpu_uti, 
                            const size_t nnz_this_rank)
{

    std::ifstream f(CONFIG_FILE_NAME);
    // If the config file is found then read it otherwise use default values
    if (f.good()){

        // Name of the tags corresponding to which values will be searched
        char *max_devices = (char *)"max_devices_to_use";
        char *max_nnz_per_device = (char *)"max_nnz_per_device";
        char *max_temp_allowed = (char *) "max_temp_allowed";
        char *min_free_mem_req = (char *) "min_free_mem_req";
        char *max_mem_uti_allowed = (char *) "max_mem_uti_allowed";
        char *max_gpu_uti_allowed = (char *) "max_gpu_uti_allowed";

        // reading the file corresponding to each tag
        read_int (CONFIG_FILE_NAME, max_devices, &nDevice);
        read_int (CONFIG_FILE_NAME, max_nnz_per_device , &max_nnz);
        read_int (CONFIG_FILE_NAME, max_temp_allowed , &max_temp);
        read_int (CONFIG_FILE_NAME, min_free_mem_req , &min_free_mem);
        read_int (CONFIG_FILE_NAME, max_mem_uti_allowed , &max_mem_uti);
        read_int (CONFIG_FILE_NAME, max_gpu_uti_allowed , &max_gpu_uti);

        sanity_check_on_values( nDevice, 
                                max_nnz, 
                                max_temp, 
                                min_free_mem, 
                                max_mem_uti, 
                                max_gpu_uti, 
                                nnz_this_rank);
    }
    else{
        nDevice = MAX_NUM_DEVICES_DEFAULT;
        max_nnz = nnz_this_rank;
        max_temp = MAX_TEMP_DEFAULT;
        min_free_mem= MIN_FREE_MEM_DEFAULT;
        max_mem_uti = MAX_MEM_UTI_DEFAULT;
        max_gpu_uti = MAX_GPU_UTI_DEFAULT;
    }
    return 0;
}


/*
* @brief Find out the configuration that could be realistically used based on 
the requested configuration and on what is actually possible
* @note 
* @param further_split_for_gpu: the vector of struct which will contain what 
will be solved on each gpu
* @param nRows: number of rows in the matrix
* @param nColumns: number of columns in the matrix
* @param num_rows_this_rank: number of ranks this mi process has to solve
* @param nnz_this_rank: NNZ this mpi process has to process
* @param avg_nnz_each_gpu: The average number of NNZ each gpu has to process
* @param nDevices: number of devices that will actaully be used
* @retval Confirmation the code exited normally
*/
int get_config_to_use(  std::vector<further_split>& further_split_for_gpu,
                        const size_t nRows,
                        const size_t nColumns,
                        const size_t num_rows_this_rank,
                        const size_t nnz_this_rank,
                        size_t &avg_nnz_each_gpu,
                        int &nDevices)
{
    int64_t max_devices_req;
    int64_t max_nnz_per_device_req;
    int64_t max_temp;
    int64_t min_free_mem;
    int64_t max_gpu_uti;
    int64_t max_mem_uti;

    get_requested_config(   max_devices_req, 
                            max_nnz_per_device_req, 
                            max_temp, 
                            min_free_mem, 
                            max_mem_uti, 
                            max_gpu_uti, 
                            nnz_this_rank);

    // Finding out the devices this mpi process can detect
    int devices_connected;
    cudaGetDeviceCount(&devices_connected);

    #ifndef MESSUNG
        std::cout << "Devices Connected: " << devices_connected << std::endl;
    #endif

    // Deciding how many devices will be used
    int max_devices_to_use;
    max_devices_to_use = std::min(devices_connected, (int)max_devices_req);

    // Some temporary variables
    size_t free_bytes_device;
    size_t total_bytes_device;
    nvmlReturn_t NVML_CHECK;
    uint temp; 
    nvmlUtilization_t utilization;

    if (max_devices_to_use > 0)
    {
        for (int64_t device = 0 ; device < devices_connected ; ++device){
            
            // Exiting if limit of max num of possible devices are reached.
            int vector_size = further_split_for_gpu.size(); 
            if ( vector_size >= max_devices_to_use)
                break;

            // finding out the free and total memory on the device
            cudaSetDevice(device);
            cudaMemGetInfo(&free_bytes_device, &total_bytes_device);
            
            free_bytes_device = free_bytes_device;

            // Initialization for querying the device for temp and utilization 
            NVML_CHECK = nvmlInit();
            if (NVML_SUCCESS != NVML_CHECK )
                {std::cout << "Could not initialize NVML" << std::endl; exit(1);}

            nvmlDevice_t nvml_device;

            NVML_CHECK = nvmlDeviceGetHandleByIndex(device, &nvml_device);

            // Querying the temperature
            NVML_CHECK = nvmlDeviceGetTemperature(nvml_device, NVML_TEMPERATURE_GPU, &temp);
            if (NVML_SUCCESS != NVML_CHECK)
                {
                    temp = MAX_TEMP_DEFAULT;
                    std::cout << "Could not get device temp. DEFAULT will be used." << std::endl;}

            // Querying the device utilization
            NVML_CHECK = nvmlDeviceGetUtilizationRates(nvml_device, &utilization);
            if (NVML_SUCCESS != NVML_CHECK)
                {std::cout << "Could not get device utilization" << std::endl; exit(1);}
        
            // Deciding if the device conforms to the requested config or not
            if (free_bytes_device > size_t(min_free_mem/100*total_bytes_device) 
                && temp < max_temp 
                && utilization.gpu < max_gpu_uti 
                && utilization.memory < max_mem_uti)
                {
                    further_split this_gpu;
                    this_gpu.id = device;
                    this_gpu.free_bytes = free_bytes_device;
                    this_gpu.total_bytes = total_bytes_device;

                    further_split_for_gpu.push_back(this_gpu);
                }
        }
    }

    // Initializing the variable nDevices
    nDevices = further_split_for_gpu.size();

    size_t min_mem_across_gpu = UINTMAX_MAX;  

    // Finding out the minimum memory across selected gpu's
    for (int device = 0 ; device < nDevices ; ++device)
        if (further_split_for_gpu[device].free_bytes < min_mem_across_gpu)
            min_mem_across_gpu = further_split_for_gpu[device].free_bytes; 

    // Calculating the number of average nnz on each selected gpu
    if (nDevices > 0)
    {
        // This is the size of small arrays like image, lmsino etc.
        size_t fixed_mem_gpu = 2*nRows*sizeof(float) + 3*nColumns*sizeof(float) + (num_rows_this_rank+1)*sizeof(int);   
        size_t mem_remain_gpu_min = min_mem_across_gpu - fixed_mem_gpu;

        avg_nnz_each_gpu = (size_t)std::ceil(std::min(0.95*mem_remain_gpu_min/(1*sizeof(float) + 1*sizeof(int)), 
            std::min((double)nnz_this_rank/nDevices, (double)max_nnz_per_device_req)));
    }
    else
        avg_nnz_each_gpu = 0;

    return 0;
}


/*
* @brief Split the data further for each gpu
* @note 
* @param matrix: The struct containing all the csr data
* @param further_split_for_gpu: the vector of struct which will contain what 
will be solved on each gpu
* @param further_split_for_cpu: The struct which will contain the part that
needs to be solved on the cpu if required.
* @param myrange: The rows this mpi process has to solve
* @param nRows: number of rows in the matrix
* @param nColumns: number of columns in the matrix
* @param num_rows_this_rank: number of ranks this mi process has to solve
* @param nDevices: number of devices that will actaully be used
* @retval Confirmation the code exited normally
*/
int hybrid_splitting(   const Csr4Matrix& matrix,
                        const size_t nRows,
                        const size_t nColumns,
                        const size_t num_rows_this_rank,
                        const Range myrange, 
                        std::vector<further_split>& further_split_for_gpu,
                        further_split& further_split_for_cpu,
                        int &nDevices)
{

    // Finding the nnz for this rank
    size_t nnz_this_rank = get_nnz <Range> (myrange, matrix);

    // Getting the final configuration that will be used
    size_t avg_nnz_each_gpu = 0;

    get_config_to_use(  further_split_for_gpu, 
                        nRows, 
                        nColumns, 
                        num_rows_this_rank, 
                        nnz_this_rank, 
                        avg_nnz_each_gpu , 
                        nDevices);

    // Partitioning the matrix further for each gpu and for the cpu and if there are no devices connected then else
    if (nDevices > 0){
        int device = 0;
        size_t sum = 0;
        further_split_for_gpu[0].start = myrange.start;
        for (size_t row=myrange.start; row<myrange.end; ++row) {
            sum += matrix.elementsInRow(row);
            if (sum > avg_nnz_each_gpu * (device + 1)) {
                further_split_for_gpu[device].end = row + 1;
                device += 1;
                if (device == nDevices){break;}
                further_split_for_gpu[device].start = row + 1;
            }
            else{
                further_split_for_gpu[device].end = row + 1;
            }
        }
        if (nDevices > 1){
            sum = 0;
            for (int end_row = further_split_for_gpu[nDevices-1].start; end_row < myrange.end ; ++end_row ){
                sum += matrix.elementsInRow(end_row);
                if (sum >= avg_nnz_each_gpu){further_split_for_gpu[nDevices - 1].end = end_row + 1; break;}
                else{further_split_for_gpu[nDevices - 1].end = end_row + 1;}
            }
        }

        #pragma omp parallel for schedule(dynamic)
        for (device = 0 ; device < nDevices ; ++device ){
            further_split_for_gpu[device].nnz = get_nnz <further_split> (further_split_for_gpu[device], matrix);
            further_split_for_gpu[device].num_rows = further_split_for_gpu[device].end - further_split_for_gpu[device].start;
        }

        further_split_for_cpu.start = further_split_for_gpu[nDevices-1].end;
        further_split_for_cpu.end = myrange.end;
        further_split_for_cpu.num_rows = further_split_for_cpu.end - further_split_for_cpu.start;
        further_split_for_cpu.nnz = get_nnz <further_split> (further_split_for_cpu, matrix);
        }
    else{
        further_split_for_cpu.start = myrange.start;
        further_split_for_cpu.end = myrange.end;
        further_split_for_cpu.num_rows = further_split_for_cpu.end - further_split_for_cpu.start;
        further_split_for_cpu.nnz = get_nnz <further_split> (further_split_for_cpu, matrix);
    }

    return 0;
} 
