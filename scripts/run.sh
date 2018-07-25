#/bin/bash
for y in {1..4}
do
    for x in 16 14 12 10 8 
    do
        mpiexec -genv OMP_NUM_THREADS 1 -genv I_MPI_DOMAIN cache -genv I_MPI_PIN=1 -genv I_MPI_PIN_MODE=lib -n $x ~/mlem/mpicsr4mlem ~/mlem/data/madpet2.p016.csr4 ~/mlem/data/Trues_Derenzo_GATE_rot_sm_200k.LMsino test.out 10 1 >> ~/outfile/test/nativ_mpi_messung.$x.txt 2>&1
        mpiexec -genv OMP_NUM_THREADS 1 -genv I_MPI_DOMAIN cache -genv I_MPI_PIN=1 -genv LAIK_LOG=2 -genv I_MPI_PIN_MODE=lib -n $x ~/mlem/laikcsr4mlem ~/mlem/data/madpet2.p016.csr4 ~/mlem/data/Trues_Derenzo_GATE_rot_sm_200k.LMsino test.out 10 >> ~/outfile/test/laik_messung.$x.txt 2>&1
    done
done
