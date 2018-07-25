#include "helper_v_7_8.cuh"


int get_parts_to_further_break_matrix(const uint32_t nnz_this_rank, const uint32_t max_ele_in_row)
{
    size_t free_bytes_device, total_bytes_device;

    cudaMemGetInfo(&free_bytes_device, &total_bytes_device);

    int parts_to_further_break_matrix = 0;

    // free_bytes_device = (size_t)free_bytes_device/2;

    std::cout << "Amount of Memory Required (Bytes): " << nnz_this_rank*sizeof(float) + nnz_this_rank*sizeof(int) << std::endl;

    std::cout << "Amount of Memory Available (Bytes): " << free_bytes_device << std::endl;

    if (free_bytes_device <  (nnz_this_rank*sizeof(float) + nnz_this_rank*sizeof(int)))
    {
        if (free_bytes_device < std::ceil(1.05*(float)max_ele_in_row*(1*sizeof(float) + 1*sizeof(int)))){
            std::cout << "WARNING: Graphics memory is not enough." << std::endl << std::endl << "EXITING" << std::endl; 
            exit(0);   
        }
        else
        {
            parts_to_further_break_matrix = (int)((size_t)(nnz_this_rank*sizeof(float) + nnz_this_rank*sizeof(int))/(size_t)(1.05*free_bytes_device));
            parts_to_further_break_matrix += 1;
        }
    }
    else
        parts_to_further_break_matrix += 1;

    return parts_to_further_break_matrix;
}

void get_splitting_based_on_rows(const Range &myrange, const Csr4Matrix& matrix, const uint64_t nnz_this_rank, const int parts_to_further_break_matrix, const MpiData& mpi, std::vector<Range>& splitting_based_on_rows)
{
    float avg_nnz_per_part = (float)nnz_this_rank / (float)parts_to_further_break_matrix;

    if (mpi.rank == 0)
        std::cout << "Avg nnz per part: " << std::setprecision(10) <<avg_nnz_per_part << std::endl;

    int idx = 0;
    uint32_t sum = 0;
    splitting_based_on_rows[0].start = 0;
    for (uint32_t row=myrange.start; row<myrange.end; ++row) 
    {
        sum += matrix.elementsInRow(row);
        if (sum > avg_nnz_per_part * (idx + 1)) 
        {
            splitting_based_on_rows[idx].end = row + 1 - myrange.start;
            idx += 1;
            splitting_based_on_rows[idx].start = row + 1 - myrange.start;
        }
    }
    splitting_based_on_rows[parts_to_further_break_matrix - 1].end = myrange.end - myrange.start;
}

uint32_t get_max_nnz_in_part(const Range &myrange, const Csr4Matrix& matrix, const std::vector<Range> splitting_based_on_rows, const int parts_to_further_break_matrix)
{
    uint32_t max_nnz_in_part = 0;
    for (int i=0 ; i< parts_to_further_break_matrix ; i++){
        uint32_t sum = 0;
        for (uint32_t row = (uint32_t)splitting_based_on_rows[i].start + myrange.start; row < (uint32_t)splitting_based_on_rows[i].end + myrange.start; ++row)
        {
            sum += matrix.elementsInRow(row);
        }
        if (sum > max_nnz_in_part)
            max_nnz_in_part = sum;
    }
    return max_nnz_in_part;
}
