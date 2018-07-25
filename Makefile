CC      = mpicxx
NVCC    = nvcc
RM      = rm -f

-include Makefile.config

MLEM_CU_VER_DIR = mlem_cuda/versions
MLEM_MPI_DIR = mlem_mpi
MLEM_OMP_DIR = mlem_openmp
MLEM_SEQ_DIR = mlem_seq

HELPER_FILES_CU_DIR = mlem_cuda/helper_files
HELPER_FILES_COMMON = helper_files_common

OMP_FLAGS = -fopenmp
CFLAGS  = -O3 -std=c++11 -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__
LFLAGS  = -O3
CUFLAGS = -lcublas -lcusparse -lmpi -lnvidia-ml -lboost_system -lboost_filesystem -I/usr/lib/openmpi/include -I/usr/lib/openmpi/include/openmpi -I/usr/include/ #-DMESSUNG --gpu-architecture=sm_50

SOURCES = csr4matrix.cpp scannerconfig.cpp
HEADERS = csr4matrix.hpp vector.hpp matrixelement.hpp scannerconfig.hpp
OBJECTS = $(SOURCES:%.cpp=%.o) profiling.o

CUDA_SOURCES = helper.cu decideConfig.cu cudaKernels.cu hybCalcFunc.cu helper_v_7_8.cu
CUDA_HEADERS = helper.cuh decideConfig.cuh cudaKernels.cuh hybCalcFunc.cuh helper_v_7_8.cuh
CUDA_OBJECTS = $(CUDA_SOURCES:%.cu=%.o)

all: version_1 version_2 version_3 version_4 version_5 version_6 version_7 version_8 version_9 mpicsr4mlem fusedmpimlem openmpcsr4mlem seqcsr4mlem #laikcsr4mlem-repart laikcsr4mlem	

version_1: version_1.o $(OBJECTS) $(CUDA_OBJECTS)
	$(NVCC) -lgomp $(CUFLAGS) $(LFLAGS) $(DEFS) -o $@ version_1.o $(OBJECTS) $(CUDA_OBJECTS)

version_1.o: $(MLEM_CU_VER_DIR)/version_1.cu
	$(NVCC) -Xcompiler $(OMP_FLAGS) $(CUFLAGS) $(CFLAGS) $(DEFS) -I. -o $@ -c $<

version_2: version_2.o $(OBJECTS) $(CUDA_OBJECTS)
	$(NVCC) -lgomp $(CUFLAGS) $(LFLAGS) $(DEFS) -o $@ version_2.o $(OBJECTS) $(CUDA_OBJECTS)

version_2.o: $(MLEM_CU_VER_DIR)/version_2.cu
	$(NVCC) -Xcompiler $(OMP_FLAGS) $(CUFLAGS) $(CFLAGS) $(DEFS) -I. -o $@ -c $<

version_3: version_3.o $(OBJECTS) $(CUDA_OBJECTS)
	$(NVCC) -lgomp $(CUFLAGS) $(LFLAGS) $(DEFS) -o $@ version_3.o $(OBJECTS) $(CUDA_OBJECTS)

version_3.o: $(MLEM_CU_VER_DIR)/version_3.cu
	$(NVCC) -Xcompiler $(OMP_FLAGS) $(CUFLAGS) $(CFLAGS) $(DEFS) -I. -o $@ -c $<

version_4: version_4.o $(OBJECTS) $(CUDA_OBJECTS)
	$(NVCC) -lgomp $(CUFLAGS) $(LFLAGS) $(DEFS) -o $@ version_4.o $(OBJECTS) $(CUDA_OBJECTS)

version_4.o: $(MLEM_CU_VER_DIR)/version_4.cu
	$(NVCC) -Xcompiler $(OMP_FLAGS) $(CUFLAGS) $(CFLAGS) $(DEFS) -I. -o $@ -c $<

version_5: version_5.o $(OBJECTS) $(CUDA_OBJECTS)
	$(NVCC) -lgomp $(CUFLAGS) $(LFLAGS) $(DEFS) -o $@ version_5.o $(OBJECTS) $(CUDA_OBJECTS)

version_5.o: $(MLEM_CU_VER_DIR)/version_5.cu
	$(NVCC) -Xcompiler $(OMP_FLAGS) $(CUFLAGS) $(CFLAGS) $(DEFS) -I. -o $@ -c $<

version_6: version_6.o $(OBJECTS) $(CUDA_OBJECTS)
	$(NVCC) -lgomp $(CUFLAGS) $(LFLAGS) $(DEFS) -o $@ version_6.o $(OBJECTS) $(CUDA_OBJECTS)

version_6.o: $(MLEM_CU_VER_DIR)/version_6.cu
	$(NVCC) -Xcompiler $(OMP_FLAGS) $(CUFLAGS) $(CFLAGS) $(DEFS) -I. -o $@ -c $<

version_7: version_7.o $(OBJECTS) $(CUDA_OBJECTS)
	$(NVCC) -lgomp $(CUFLAGS) $(LFLAGS) $(DEFS) -o $@ version_7.o $(OBJECTS) $(CUDA_OBJECTS) 

version_7.o: $(MLEM_CU_VER_DIR)/version_7.cu
	$(NVCC) -Xcompiler $(OMP_FLAGS) $(CUFLAGS) $(CFLAGS) $(DEFS) -I. -o $@ -c $<

version_8: version_8.o $(OBJECTS) $(CUDA_OBJECTS)
	$(NVCC) -lgomp $(CUFLAGS) $(LFLAGS) $(DEFS) -o $@ version_8.o $(OBJECTS) $(CUDA_OBJECTS)

version_8.o: $(MLEM_CU_VER_DIR)/version_8.cu
	$(NVCC) -Xcompiler $(OMP_FLAGS) $(CUFLAGS) $(CFLAGS) $(DEFS) -I. -o $@ -c $<

version_9: version_9.o $(OBJECTS) $(CUDA_OBJECTS)
	$(NVCC) -lgomp $(CUFLAGS) $(LFLAGS) $(DEFS) -o $@ version_9.o $(OBJECTS) $(CUDA_OBJECTS)

version_9.o: $(MLEM_CU_VER_DIR)/version_9.cu
	$(NVCC) -Xcompiler $(OMP_FLAGS) $(CUFLAGS) $(CFLAGS) $(DEFS) -I. -o $@ -c $<
	
mpicsr4mlem: mpicsr4mlem.o $(OBJECTS)
	$(CC) $(CFLAGS) $(DEFS) -o $@ mpicsr4mlem.o $(OBJECTS)

fusedmpimlem: fusedmpimlem.o $(OBJECTS)
	$(CC) $(LFLAGS) $(DEFS) -o $@ $< $(OBJECTS)

openmpcsr4mlem: openmpcsr4mlem.o $(OBJECTS)
	$(CC) $(LFLAGS) $(DEFS) $(OMP_FLAGS) -o $@ $< $(OBJECTS)

seqcsr4mlem: seqcsr4mlem.o $(OBJECTS)
	$(CC) $(LFLAGS) $(DEFS) -o $@ $< $(OBJECTS)

%.o: $(MLEM_SEQ_DIR)/%.cpp
	$(CC) $(CFLAGS) $(OMP_FLAGS) $(DEFS) -o $@ -c $<

%.o: $(MLEM_OMP_DIR)/%.cpp
	$(CC) $(CFLAGS) $(OMP_FLAGS) $(DEFS) -o $@ -c $<

%.o: $(MLEM_MPI_DIR)/%.cpp
	$(CC) $(CFLAGS) $(OMP_FLAGS) $(DEFS) -o $@ -c $<

%.o: $(HELPER_FILES_COMMON)/%.cpp
	$(CC) $(CFLAGS) $(OMP_FLAGS) $(DEFS) -o $@ -c $<

%.o: $(HELPER_FILES_COMMON)/%.c
	$(CC) $(CFLAGS) $(OMP_FLAGS) $(DEFS) -o $@ -c $< #-no-legacy-libc

%.o: $(HELPER_FILES_CU_DIR)/%.cu
	$(NVCC) -Xcompiler $(OMP_FLAGS) $(CUFLAGS) $(CFLAGS) $(DEFS) -o $@ -c $< 

clean:
	- $(RM) *.o version_* fusedmpimlem mpicsr4mlem openmpcsr4mlem seqcsr4mlem

distclean: clean
	- $(RM) *.c~ *.h~



