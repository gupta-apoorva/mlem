# Explanation about different cuda versions of cudampicsr4mlem

## version_1

This version only uses cusparse operations for calculating norm, forward projection and backward projection.
It also only make use of PAGABLE MEMORY for MPI_REDUCE OPERATIONS and is NOT CUDA AWARE.

## version_2

This version only uses cusparse operations for calculating norm, forward projection and backward projection.
It also only make use of PINNED MEMORY for MPI_REDUCE OPERATIONS and is NOT CUDA AWARE.

## version_3

This version only uses cusparse operations for calculating norm, forward projection and backward projection.
It is CUDA AWARE so NO need to allocate memory for reduce operations.

## version_4

This version uses non warp kernels for calculating the norm , forward and backward projection.
It is CUDA AWARE so NO need to allocate memory for reduce operations.

## version_5

For forward projection, backward projection and norm calculation it uses WARP version of the custom kernels.
It is CUDA AWARE so NO need to allocate memory for reduce operations.
The block size is 1024

## version_6

For forward projection, backward projection and norm calculation it uses WARP version of the custom kernels.
It also only make use of PINNED MEMORY for MPI_REDUCE OPERATIONS and is NOT CUDA AWARE.
The block size is 64.

## version 7

For forward projection, backward projection and norm calculation it uses WARP version of the custom kernels.
It also only make use of PINNED MEMORY for MPI_REDUCE OPERATIONS and is NOT CUDA AWARE.

This version can run on smaller GPU's where the entire matrix could not be loaded at once.

## version 8

For forward projection, backward projection and norm calculation it uses WARP version of the custom kernels.
It also only make use of PINNED MEMORY for MPI_REDUCE OPERATIONS and is NOT CUDA AWARE.

This version can run on smaller GPU's where the entire matrix could not be loaded at once.
Solves the further broken down parts on the GPU.

It implements the fused version of the code.

## version 9

For forward projection, backward projection and norm calculation it uses WARP version of the custom kernels.
It also only make use of PINNED MEMORY for MPI_REDUCE OPERATIONS and is NOT CUDA AWARE.

This is a hybrid multi gpu version. Breaks down the part assigned to a mpi process for the GPU and CPU 
depending on how many GPU's are connected.

It implements the fused version of the code.
