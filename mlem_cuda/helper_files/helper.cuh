#ifndef __HELPER_HPP__
#define __HELPER_HPP__


#include <chrono>
#include <stdint.h>
#include <string>
#include <iomanip>
#include <sys/stat.h>
#include <stdio.h>
#include <sys/time.h>
#include <cassert>
#include <algorithm>
#include <mpi.h>
#include <omp.h>
#include <ctime>
#include <boost/filesystem.hpp>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include "../../helper_files_common/structures.hpp"
#include "../../helper_files_common/csr4matrix.hpp"
#include "../../helper_files_common/vector.hpp"

#define MAX_LINE_LENGTH 1024

#define READ_ERROR(szMessage, szVarName, szFileName, nLine) \
	{ char szTmp[80]; \
    if( nLine ) \
		sprintf( szTmp, " %s  File: %s   Variable: %s  Line: %d", szMessage, szFileName, szVarName, nLine ); \
    else \
  		sprintf( szTmp, " %s  File: %s   Variable: %s ", szMessage, szFileName, szVarName); \
    error( szTmp ); \
  }
    
int64_t min_int( const int64_t n1, const int64_t n2 );

void error(char* error_name);

void read_string( const char* szFileName, const char* szVarName, char*   pVariable);

void read_int( const char* szFileName, const char* szVarName, int64_t* pVariable);

size_t get_max_ele_in_row(const Range &myrange, const Csr4Matrix& matrix);

template <class T=Range> size_t get_nnz(const T &myrange, const Csr4Matrix& matrix);

template <class T=Range> void csr_format_for_cuda(const T &myrange, const Csr4Matrix& matrix, float* csrVal, int* csrRowInd, int* csrColInd);

double wtime();

bool exists(const std::string& name);

void checkPointNow(const MpiData& mpi, Vector<float>& image, int iter, const std::string& img_chkpt_name, const std::string& iter_chkpt_name);

void restore( Vector<float>& image, const std::string& img_chkpt_name, int& iter, const std::string& iter_chkpt_name);

ProgramOptions handleCommandLine(int argc, char *argv[]);

MpiData initializeMpi(int argc, char *argv[]);

std::vector<Range> partition( const MpiData& mpi, const Csr4Matrix& matrix);

void start_time_measurement_hyb(std::chrono::high_resolution_clock::time_point &t_begin);

void stop_time_measurement_hyb(std::chrono::high_resolution_clock::time_point &t_begin, std::chrono::high_resolution_clock::time_point &t_end, float* milliseconds);

void start_time_measurement(cudaEvent_t &start);

void stop_time_measurement(cudaEvent_t &start, cudaEvent_t &stop, float* milliseconds);

boost::filesystem::path make_prof_folder(std::string exe_file, MpiData mpi);

void time_writetofile(boost::filesystem::path full_path, MpiData mpi ,TimingRuntime time_runtime, ProgramOptions progops);

#endif
