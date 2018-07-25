#include "helper.cuh"

/*
* @brief compares 2 int64_t values and returns the minimum
* @note
* @param n1: first int64_t value
* @param n2: second int64_t value
* @retval The minimum of the 2 values
*/

int64_t min_int(    const int64_t n1, 
                    const int64_t n2 )
{
    if( n1 < n2 ) return n1;
    return n2;
}

/*
@brief A small function that prints the error and exits the code
@param error_name: The output that needs to be printed before calling it quits
@retval NONE
*/
void error(char* error_name){
  std::cout << error_name << std::endl;
  exit(1);
}


/*
* @brief The function searches the datafile fh for the line defining the variable szVarName
* @note
* @param szFileName: File name of the configuration file
* @param szVarName: The parameter that needs to be searched in the file
* @retval The value of the variable found as char
*/
char* find_string(  const char* szFileName, 
                    const char *szVarName )
{ 
  int nLine = 0;
  int i;
  FILE *fh = NULL;
  
  static char szBuffer[MAX_LINE_LENGTH];    /* containes the line read  */
                                             /* from the datafile        */

  char* szLine = szBuffer;
  char* szValue = NULL;
  char* szName = NULL;

  /* open file */
  fh = fopen( szFileName, "rt" );
  if( fh == 0 ) 
  READ_ERROR("Could not open file", szVarName, szFileName, 0);

    /* searching */
  while( ! feof(fh) )
  {
    if ( !fgets( szLine, MAX_LINE_LENGTH, fh ))
        READ_ERROR("Could not read text from file.", szVarName, szFileName, 0);
    ++nLine;

    /* remove comments */
    for( i = 0; i < strlen(szLine); i++)
      if( szLine[i] == '#' )
      {
            szLine[i] = '\0'; 
            break;
      }

    /* remove empty lines */
    while( isspace( (int)*szLine ) && *szLine) ++szLine;
      if( strlen( szLine ) == 0) continue; 

    /* now, the name can be extracted */
    szName = szLine;
    szValue = szLine;
    while( (isalnum( (int)*szValue ) || *szValue == '_') && *szValue) ++szValue;
    
    /* is the value for the respective name missing? */
    if( *szValue == '\n' || strlen( szValue) == 0)  
        READ_ERROR("wrong format", szName, szFileName, nLine);
    
    *szValue = 0;       /* complete szName! at the right place */
    ++szValue;
          
    /* read next line if the correct name wasn't found */
    if( strcmp( szVarName, szName)) continue;

    /* remove all leading blnkets and tabs from the value string  */
    while( isspace( (int)*szValue) ) ++szValue;
        if( *szValue == '\n' || strlen( szValue) == 0)  
            READ_ERROR("wrong format", szName, szFileName, nLine);
    
    fclose(fh);
    return szValue;
  }  
    // In case the string is not found in the configuration file
    return NULL;   
} 

/*
* @brief Open the file to read its contents to load the value of the configuration in the variable  
* @note
* @param szFileName: File name of the configuration file
* @param szVarName: The parameter that needs to be searched in the file
* @param pVariable: The variable in which the extracted value will be stored.
* @retval NONE
*/

void read_int(  const char* szFileName, 
                const char* szVarName, 
                int64_t* pVariable)
{
    char* szValue = NULL; /* string containing the read variable value */

    if( szVarName  == 0 )  error((char *)"null pointer given as varable name" );
    if( szFileName == 0 )  error((char *)"null pointer given as filename" );
    if( pVariable  == 0 )  error((char *)"null pointer given as variable" );

    if( szVarName[0] == '*' )
        szValue = find_string( szFileName, szVarName +1 );
    else
        szValue = find_string( szFileName, szVarName );

    if (szValue){
        if( sscanf( szValue, "%ld", pVariable) == 0)
            READ_ERROR("wrong format", szVarName, szFileName, 0);

            printf( "File: %s\t\t%s%s= %ld\n", szFileName, 
                                      szVarName,
                                      &("               "[min_int( strlen(szVarName), 15)]), 
                                      *pVariable );
    }
    else{
        printf("Did Not Find The String \t%s = DEFAULT\n", szVarName );
        *pVariable = -1;
    }
}

/*
* @brief Finds out the maximum nnz in the marked rows
* @note
* @param myrange: The range of rows that needs loaded into vectors
* @param matrix: The original data read from the file and put into matrix
* @retval maximum nnz found in the row
*/

size_t get_max_ele_in_row(  const Range &myrange, 
                            const Csr4Matrix& matrix)
{
    size_t max_ele_in_row = 0; 
    for (int row=myrange.start; row<myrange.end; ++row) 
    {  
        if (matrix.elementsInRow(row) > max_ele_in_row)
            max_ele_in_row = (size_t)matrix.elementsInRow(row);
    }
    return max_ele_in_row;
}

/*
* @brief Finds out the total nnz in the rows
* @note template function
* @param myrange: The range of rows that needs loaded into vectors
* @param matrix: The original data read from the file and put into matrix
* @retval nnz in the marked rows
*/

template <class T>
size_t get_nnz( const T &myrange, 
                const Csr4Matrix& matrix)
{
    size_t nnz_this_rank = 0;
    for (size_t row = myrange.start; row<myrange.end; row++){
        nnz_this_rank += matrix.elementsInRow(row);
    }
    return nnz_this_rank;
}

/*
* @brief Converts the original data into vectors for the GPU
* @note Should be sufficient to call just once. Very expensive. template function
* @param myrange: The range of rows that needs loaded into vectors
* @param matrix: The original data read from the file and put into matrix
* @param carVal: The csr values in a vector 
* @param csrRowInd: The csr row in a vector
* @param csrColInd: csr column indexes in a vector.
* @retval NONE
*/
template <class T>
void csr_format_for_cuda(   const T &myrange, 
                            const Csr4Matrix& matrix, 
                            float* csrVal, 
                            int* csrRowInd, 
                            int* csrColInd)
{   
    int index = 0;
    csrRowInd[index] = 0;

    for (int row=myrange.start; row<myrange.end; ++row) {
        csrRowInd[row - myrange.start + 1] = csrRowInd[row - myrange.start] + (int)matrix.elementsInRow(row);

        std::for_each(matrix.beginRow2(row), matrix.endRow2(row),[&](const RowElement<float>& e){ 
            csrVal[index] = e.value();
            csrColInd[index] = (int)e.column() ;
            index = index + 1; });
    }
	MPI_Barrier(MPI_COMM_WORLD);
}


/** 
 * @brief  Simple Time Measurement Function
 * @note   This function should not be called very often since gettimeofday ()
 * is quite expensive. Can be also changed to other time implementation such 
 * as walltime.
 * 
 * @retval Current Real Time
 */
double wtime()
{
    struct timeval tv;
    gettimeofday(&tv, 0);

    return tv.tv_sec+1e-6*tv.tv_usec;
}

/** 
 * @brief  Check if a file exists. 
 * @note   
 * @param  name: The file name.
 * @retval true, if file exists. Otherwise false. 
 */
bool exists(const std::string& name)
{
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

/** 
 * @brief  Create a checkpoint for the given MLEM code.
 * @note   
 * @param  mpi: the MPI Control structure
 * @param  image: Image vector
 * @param  iter: current iteration number
 * @param  img_chkpt_name: checkpoint file name for image vector
 * @param  iter_chkpt_name: checkpoint file name for the iteration number
 * @retval None
 */
void checkPointNow( const MpiData& mpi,
                    Vector<float>& image,
                    int iter,
                    const std::string& img_chkpt_name,
                    const std::string& iter_chkpt_name)
{
    if(mpi.rank>0) {
        return;
    }

    // Use the synchonous version to ensure that the file is flush
    image.writeToFileSync(img_chkpt_name);

    std::ofstream myfile(iter_chkpt_name,
                         std::ofstream::out | std::ofstream::binary);
    if (!myfile.good())
        throw std::runtime_error(std::string("Cannot open file ")
                                 + iter_chkpt_name
                                 + std::string("; reason: ")
                                 + std::string(strerror(errno)));
    myfile.write(reinterpret_cast<char*>(&iter), sizeof(int));
    myfile.flush();
    
    //The fsync() is required to ensure the file flush. 
    fsync(image.GetFd(*myfile.rdbuf()));
    myfile.close();
}

/** 
 * @brief Restore working vectors from a given checkpoint
 * @note   
 * @param  image: reference to the image vector
 * @param  img_chkpt_name: name to the image vector checkpoint file
 * @param  iter: reference to the iteration vector
 * @param  iter_chkpt_name: name to the iteration vector file name
 * @retval None
 */
void restore(   Vector<float>& image,
                const std::string& img_chkpt_name,
                int& iter,
                const std::string& iter_chkpt_name)
{
    image.readFromFile(img_chkpt_name);
    
    std::ifstream myFile(iter_chkpt_name.c_str(),
                         std::ifstream::in | std::ifstream::binary);
    if(myFile.good()) {
        myFile.read(reinterpret_cast<char*>(&iter), sizeof(int));
        
    } else {
        throw std::runtime_error(
                    std::string("Cannot open file ") +
                    iter_chkpt_name +
                    std::string("; reason: ") +
                    std::string(strerror(errno)));
    }
    myFile.close();
}

/** 
 * @brief  Simple Commandline Options Handler
 * @note   
 * @param  argc: 
 * @param  *argv[]: 
 * @retval 
 */
ProgramOptions handleCommandLine(   int argc, 
                                    char *argv[])
{
    if (argc != 6)
        throw std::runtime_error("wrong number of command line parameters");

    ProgramOptions progoptions;
    progoptions.mtxfilename = std::string(argv[1]);
    progoptions.infilename  = std::string(argv[2]);
    progoptions.outfilename = std::string(argv[3]);
    progoptions.iterations = std::stoi(argv[4]);
    progoptions.checkpointing = std::stoi(argv[5]);
    return progoptions;
}

/** 
 * @brief  Simple implementation to initialize MPI Communication
 * @note   
 * @param  argc: 
 * @param  *argv[]: 
 * @retval 
 */
MpiData initializeMpi(int argc, char *argv[])
{
    MpiData mpidata;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpidata.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpidata.size);

    return mpidata;
}

/** 
 * @brief  Simple Partitioner
 * @note   This partiitoner operates on the number of NZs per row. 
 * @param  mpi: the MPI data structure
 * @param  matrix: the input sparse matrix
 * @retval a range of on which the current MPI rank should work on. s
 */
std::vector<Range> partition(   const MpiData& mpi,
                                const Csr4Matrix& matrix)
{
    float avgElemsPerRank = (float)matrix.elements() / (float)mpi.size;

#ifndef MESSUNG
    std::cout << "MLEM [" << mpi.rank << "/" << mpi.size << "]: " 
        << "Matrix elements: " << matrix.elements()
        << ", avg elements per rank: " << avgElemsPerRank << std::endl;
#endif
    std::vector<Range> ranges(mpi.size);
    int idx = 0;
    size_t sum = 0;
    ranges[0].start = 0;
    for (size_t row=0; row<matrix.rows(); ++row) {
        sum += matrix.elementsInRow(row);
        if (sum > avgElemsPerRank * (idx + 1)) {
            ranges[idx].end = row + 1;
            idx += 1;
            ranges[idx].start = row + 1;
        }
    }
    ranges[mpi.size - 1].end = matrix.rows();


#ifndef MESSUNG
    for (size_t i=0; i<ranges.size(); ++i) {
        std::cout << "MLEM [" << mpi.rank << "/" << mpi.size << "]:"
                  << "Range " << i <<" from " << ranges[i].start << " to " <<ranges[i].end << std::endl;
    }
#endif

    return ranges;
}

/*
* @brief Starts the time measurement
* @note THis is onlu for hybrid code. Synchronized with CPU. Might give error if used in other codes.
* @param t_begin: variable in which the start time will be stored
* @retval NONE
*/
void start_time_measurement_hyb(std::chrono::high_resolution_clock::time_point &t_begin){
    t_begin = std::chrono::high_resolution_clock::now();
}

/*
* @brief Stops the time measurement
* @note THis is onlu for hybrid code.  Synchronized with CPU. Might give error if used in other codes.
* @param t_begin: variable in which the start time will be passed
* @param t_end: variable in which the stop time will be stored
* @param milliseconds: The difference between start and stop
* @retval NONE
*/
void stop_time_measurement_hyb(std::chrono::high_resolution_clock::time_point &t_begin, std::chrono::high_resolution_clock::time_point &t_end, float* milliseconds){
    t_end = std::chrono::high_resolution_clock::now();
    *milliseconds = (float)(std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count()/(float)1000);
}

/*
* @brief Starts the time measurement
* @note
* @param start: variable in which the start time will be stored
* @retval NONE
*/
void start_time_measurement(cudaEvent_t &start){
    cudaEventRecord(start);
}

/*
* @brief Stops the time measurement
* @note
* @param start: variable in which the start time will be passed
* @param stop: variable in which the stop time will be stored
* @param milliseconds: The difference between start and stop
* @retval NONE
*/
void stop_time_measurement(cudaEvent_t &start, cudaEvent_t &stop, float* milliseconds){
    float local_ms = 0;
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&local_ms, start, stop);
    *milliseconds += local_ms;
}


/*
* @brief Creates a folder to store the perf measurement files
* @note
* @param exe_file: The name of the executable file bcoz that 
* will be name of the output folder
* @retval The full path of the created folder
*/
boost::filesystem::path make_prof_folder(std::string exe_file, MpiData mpi){
    char buffer[100];
    time_t rawtime;
    struct tm * timeinfo;

    if (mpi.rank == 0)
    {
    time (&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(buffer,sizeof(buffer),"%d-%m-%Y %I:%M:%S",timeinfo);
    }

    MPI_Bcast(&buffer, sizeof(buffer), MPI_CHAR, 0, MPI_COMM_WORLD);

    std::string str(buffer);

    boost::filesystem::path full_path = boost::filesystem::system_complete(("prof_" + exe_file.substr(2) + "/" + buffer).c_str());

    if (mpi.rank ==0){
        if (!boost::filesystem::exists(full_path))
            boost::filesystem::create_directories(full_path);
    }
    return full_path;
}

/*
* @brief Outputs the perf measurement into files
* @note
* @param full_path: The name of the folder in which files will be saved 
* @param mpi: The struct containing the mpi data
* @param time_runtime: The struct containg all the time measurements
* @param progops: The struct containg the command line arguments
* @retval None
*/
void time_writetofile(boost::filesystem::path full_path, MpiData mpi ,TimingRuntime time_runtime, ProgramOptions progops){
    
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    std::string file_name = full_path.string() + "/rank_" + std::to_string(mpi.rank) + ".txt";

    std::ofstream myfile;
    myfile.open(file_name);

    myfile << processor_name << std::endl;
    myfile << "struct_to_csr_vector_time: " << std::fixed << std::setprecision(3) << time_runtime.struct_to_csr_vector_time << " ms" << std::endl;
    myfile << "alloc_copy_to_d_time: " << std::fixed << std::setprecision(3) << time_runtime.alloc_copy_to_d_time << " ms"<< std::endl;
    myfile << "further_par_time: " << std::fixed << std::setprecision(3) << time_runtime.further_par_time << " ms" << std::endl;
    myfile << "norm_calc_time: " << std::fixed << std::setprecision(3) << time_runtime.norm_calc_time << " ms"<< std::endl;
    myfile << "norm_redc_time: " << std::fixed << std::setprecision(3) << time_runtime.norm_redc_time << " ms"<< std::endl;
    myfile << "calc_setting_image_time: " << std::fixed << std::setprecision(3) << time_runtime.calc_setting_image_time << " ms"<< std::endl;

    myfile << std::endl;
    myfile  << std::setw(14) << std::left << "fwproj " 
            << std::setw(17) << std::left << ", fwproj_redc " 
            << std::setw(17) << std::left << ", corr_calc " 
            << std::setw(17) << std::left << ", update_calc " 
            << std::setw(17) << std::left << ", update_redc " 
            << std::setw(17) << std::left << ", img_update " 
            << std::setw(17) << std::left << ", img_sum " 
            << std::endl;
    for (int iter = 0; iter<progops.iterations ; iter++){ 
        myfile  << std::setw(14) << std::left << std::fixed << std::setprecision(3) << time_runtime.timing_loop[iter].fwproj_time << ", " 
                << std::setw(15) << std::left << std::fixed << std::setprecision(3) << time_runtime.timing_loop[iter].fwproj_redc_time << ", " 
                << std::setw(15) << std::left << std::fixed << std::setprecision(3) << time_runtime.timing_loop[iter].corr_calc_time << ", " 
                << std::setw(15) << std::left << std::fixed << std::setprecision(3) << time_runtime.timing_loop[iter].update_calc_time << ", "
                << std::setw(15) << std::left << std::fixed << std::setprecision(3) << time_runtime.timing_loop[iter].update_redc_time << ", " 
                << std::setw(15) << std::left << std::fixed << std::setprecision(3) << time_runtime.timing_loop[iter].img_update_time << ", " 
                << std::setw(15) << std::left << std::fixed << std::setprecision(3) << time_runtime.timing_loop[iter].img_sum_time 
                << std::endl;
    }
    myfile.close();
}


template size_t get_nnz <Range> (Range const&, Csr4Matrix const&);

template size_t get_nnz <further_split> (further_split const&, Csr4Matrix const&);

template void csr_format_for_cuda <Range> (const Range&, const Csr4Matrix&, float*, int*, int*);

template void csr_format_for_cuda <further_split> (const further_split&, const Csr4Matrix&, float*, int*, int*);
