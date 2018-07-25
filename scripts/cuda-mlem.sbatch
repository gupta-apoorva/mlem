#!/bin/bash

#SBATCH -o /home/hpc/pr63qi/ga58qob3/outfolder/cuda_mlem_test.%j.%N.out
#SBATCH -D /home/hpc/pr63qi/ga58qob3/mlem
#SBATCH -J cuda-mlem
#SBATCH --get-user-env
#SBATCH --partition=nvd
#SBATCH --nodes=2
#SBATCH --ntasks=4
#Sbatch --ntasks-per-node=2
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=end
#SBATCH --mail-user=apoorva.gupta@tum.de
#SBATCH --export=NONE
#SBATCH --time=00:30:00

source /etc/profile.d/modules.sh

module unload mpi.intel/5.1
module load mpi.intel/5.1_gcc
module load gcc/4.9
module load cuda/7.5
module load boost/1.65_gcc
module load binutils/2.25

mpirun -n 4 ~/mlem/cudampicsr4mlem ~/mlem/test.csr4 ~/mlem/sino65536.sino mlem-60.out 60 0  


