#!/bin/bash

#SBATCH -o /home/hpc/pr63qi/ga39pid2/outfile/mlemtest.%j.out
#SBATCH -D /home/hpc/pr63qi/ga39pid2/
#SBATCH -J mlem
#SBATCH --get-user-env
#SBATCH --cluster=mpp2
#SBATCH --ntasks=28
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=end
#SBATCH --mail-user=d.yang@tum.de
#SBATCH --export=NONE
#SBATCH --time=00:30:00

source /etc/profile.d/modules.sh
module load gcc/5
unset I_MPI_PMI_LIBRARY
export SLURM_CPU_BIND=none
mpiexec -genv OMP_NUM_THREADS 1 -genv I_MPI_PIN_PROCESSOR_LIST=0,1,2,3,7,8,9,10,14,15,16,17,21,22,23,24 -genv I_MPI_DEBUG 4 -genv I_MPI_PIN=1 -genv I_MPI_PIN_MODE=lib -n 12 ~/mlem/mpicsr4mlem ~/mlem/data/madpet2.p016.csr4 ~/mlem/data/Trues_Derenzo_GATE_rot_sm_200k.LMsino test.out 1 1



