#/bin/bash
for y in {1..4}
do
    for x in 16 14 12 10 8 
    do
        mpiexec -genv OMP_NUM_THREADS 1 -genv I_MPI_PIN_DOMAIN numa -genv I_MPI_PIN 1 -genv LAIK_LOG=2 -genv I_MPI_PIN_MODE=lib -genv SHRINK_ITER=5 -genv SHRINK_FROM=1 -genv SHRINK_TO=1 -n $x ~/mlem/laikcsr4mlem-repart ~/mlem/data/madpet2.p016.csr4 ~/mlem/data/Trues_Derenzo_GATE_rot_sm_200k.LMsino test.out 10 >> ~/outfile/shrink/laik-repart_messung.$x.txt 
    done
done
