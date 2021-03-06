#/bin/bash
for y in {1..4}
do
    for x in 28 26 24 22 20 18 16 14 12 10 8 6 4 2 
    do
        mpiexec -genv OMP_NUM_THREADS 1 -genv I_MPI_PIN_DOMAIN numa -genv I_MPI_PIN=1 -genv I_MPI_DEBUG 4 -genv I_MPI_PIN_MODE=lib -n $x ~/mlem/mpicsr4mlem ~/mlem/data/madpet2.p016.csr4 ~/mlem/data/Trues_Derenzo_GATE_rot_sm_200k.LMsino test.out 10 1 >> ~/outfile/test/nativ_mpi_messung.$x.txt 
        mpiexec -genv OMP_NUM_THREADS 1 -genv I_MPI_PIN_DOMAIN numa -genv I_MPI_PIN=1 -genv I_MPI_DEBUG 4 -genv LAIK_LOG=2 -genv I_MPI_PIN_MODE=lib -n $x ~/mlem/laikcsr4mlem ~/mlem/data/madpet2.p016.csr4 ~/mlem/data/Trues_Derenzo_GATE_rot_sm_200k.LMsino test.out 10 >> ~/outfile/test/laik_messung.$x.txt
    done
done
