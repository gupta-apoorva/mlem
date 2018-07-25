#ifndef __DECIDECONFIG_HPP__
#define __DECIDECONFIG_HPP__


#define CONFIG_FILE_NAME "configuration/configuration.txt"

#define MAX_NUM_DEVICES_DEFAULT 3
#define MAX_TEMP_DEFAULT 100
#define MIN_FREE_MEM_DEFAULT 0
#define MAX_MEM_UTI_DEFAULT 100
#define MAX_GPU_UTI_DEFAULT 100

#include "helper.cuh"
#include "../../helper_files_common/structures.hpp"
#include "../../helper_files_common/csr4matrix.hpp"

#include <omp.h>
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
#include <boost/filesystem.hpp>
#include <nvml.h>


void sanity_check_on_values(int64_t &nDevice, 
                            int64_t &max_nnz,
                            int64_t &max_temp, 
                            int64_t &min_free_mem, 
                            int64_t &max_mem_uti, 
                            int64_t &max_gpu_uti, 
                            const size_t nnz_this_rank);

int get_requested_config(   int64_t &nDevice, 
                            int64_t &max_nnz, 
                            int64_t &max_temp, 
                            int64_t &min_free_mem, 
                            int64_t &max_mem_uti, 
                            int64_t &max_gpu_uti, 
                            const size_t nnz_this_rank);

int get_config_to_use(  std::vector<further_split>& further_split_for_gpu,
                        const size_t nRows,
                        const size_t nColumns,
                        const size_t num_rows_this_rank,
                        const size_t nnz_this_rank,
                        size_t &avg_nnz_each_gpu,
                        int &nDevices);

int hybrid_splitting(  const Csr4Matrix& matrix,
                        const size_t nRows,
                        const size_t nColumns,
                        const size_t num_rows_this_rank,
                        const Range myrange, 
                        std::vector<further_split>& further_split_for_gpu,
                        further_split& further_split_for_cpu,
                        int &nDevices);


#endif
