#ifndef _HELPER_V_7_8_CUH_
#define _HELPER_V_7_8_CUH_


#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <stdint.h>
#include <iomanip>

#include "../../helper_files_common/structures.hpp"
#include "../../helper_files_common/csr4matrix.hpp"
#include "../../helper_files_common/vector.hpp"


int get_parts_to_further_break_matrix(const uint32_t nnz_this_rank, const uint32_t max_ele_in_row);

void get_splitting_based_on_rows(const Range &myrange, const Csr4Matrix& matrix, const uint64_t nnz_this_rank, const int parts_to_further_break_matrix, const MpiData& mpi, std::vector<Range> &splitting_based_on_rows);

uint32_t get_max_nnz_in_part(const Range &myrange, const Csr4Matrix& matrix, const std::vector<Range> splitting_based_on_rows, const int parts_to_further_break_matrix);

#endif