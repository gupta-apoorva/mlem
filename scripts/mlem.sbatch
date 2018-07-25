#!/bin/bash

#SBATCH -o /home/hpc/pr63qi/ga39pid2/outfile/mlemtest.%j.out
#SBATCH -D /home/hpc/pr63qi/ga39pid2/
#SBATCH -J mlem
#SBATCH --get-user-env
#SBATCH --partition=snb
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=end
#SBATCH --mail-user=d.yang@tum.de
#SBATCH --export=NONE
#SBATCH --time=100:00

source /etc/profile.d/modules.sh
module load gcc/5
~/mlem/scripts/run.sh


