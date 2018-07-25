#/bin/bash
for y in {1..4}
do
    for x in 32 30 28 26 24 22
    do
        mpirun -np $x -npernode 1 -hostfile ~/nodelist32 -x OMP_NUM_THREADS=4 ~/mlem/mpicsr4mlem ~/mlem/data/madpet2.p016.csr4 ~/mlem/data/Trues_Derenzo_GATE_rot_sm_200k.LMsino test.out 10 1 &> ~/outfile/nativ_mpi_messung_pernode1_$y.$x.txt
        mpirun -np $x -npernode 1 -hostfile ~/nodelist32 -x OMP_NUM_THREADS=4 ~/mlem/laikcsr4mlem ~/mlem/data/madpet2.p016.csr4 ~/mlem/data/Trues_Derenzo_GATE_rot_sm_200k.LMsino test.out 10 &> ~/outfile/laik_mpi_messung_pernode1_$y.$x.txt
    done
    for x in 32 30 28 26 24 22 
    do
        z=$(($x*4))
        mpirun -np $z -npernode 4 -hostfile ~/nodelist32 -x OMP_NUM_THREADS=1 ~/mlem/mpicsr4mlem ~/mlem/data/madpet2.p016.csr4 ~/mlem/data/Trues_Derenzo_GATE_rot_sm_200k.LMsino test.out 10 1 &> ~/outfile/nativ_mpi_messung_pernode4_$y.$x.txt
        mpirun -np $z -npernode 4 -hostfile ~/nodelist32 -x OMP_NUM_THREADS=1 ~/mlem/laikcsr4mlem ~/mlem/data/madpet2.p016.csr4 ~/mlem/data/Trues_Derenzo_GATE_rot_sm_200k.LMsino test.out 10 &> ~/outfile/laik_mpi_messung_pernode4_$y.$x.txt
    done
done
