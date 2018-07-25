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
mpirun -n 1 env
unset I_MPI_PIN_MAPPING
export SLURM_CPU_BIND=none
mpirun -n 1 -genv I_MPI_PIN_PROCESSOR_LIST=0,3 env


