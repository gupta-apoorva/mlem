#include "../helper_files_common/csr4matrix.hpp"
#include "../helper_files_common/vector.hpp"

#include <omp.h>

#include <iostream>
#include <string>
#include <stdexcept>
#include <cassert>
#include <algorithm>
#include <chrono>


struct ProgramOptions
{
    std::string mtxfilename;
    std::string infilename;
    std::string outfilename;
    int iterations;
};


ProgramOptions handleCommandLine(int argc, char *argv[])
{
    if (argc != 5)
        throw std::runtime_error("wrong number of command line parameters");

    ProgramOptions progops;
    progops.mtxfilename = std::string(argv[1]);
    progops.infilename  = std::string(argv[2]);
    progops.outfilename = std::string(argv[3]);
    progops.iterations = std::stoi(argv[4]);

    return progops;
}


void calcColumnSums(const Csr4Matrix& matrix, Vector<float>& norm)
{
    assert(matrix.columns() == norm.size());

    std::fill(norm.ptr(), norm.ptr() + norm.size(), 0.0);

    for (uint32_t row=0; row<matrix.rows(); ++row) {
        std::for_each(matrix.beginRow2(row), matrix.endRow2(row),
                      [&](const RowElement<float>& e){ norm[e.column()] += e.value(); });
    }
    // norm.writeToFile("norm-0.out");
}


void initImage(const Vector<float>& norm, Vector<float>& image, const Vector<int>& lmsino)
{
    // Sum up norm vector, creating sum of all matrix elements
    double sumnorm = 0.0;
    for (size_t i=0; i<norm.size(); ++i) sumnorm += norm[i];
    double sumin = 0.0;
    for (size_t i=0; i<lmsino.size(); ++i) sumin += lmsino[i];

    float initial = static_cast<float>(sumin / sumnorm);

    std::cout << "INIT: sumnorm=" << sumnorm << ", sumin=" << sumin
              << ", initial value=" << initial << std::endl;

    for (size_t i=0; i<image.size(); ++i) image[i] = initial;
}


void calcFwProj(const Csr4Matrix& matrix, const Vector<float>& input, Vector<float>& result)
{
    // Direct access to arrays is supported e.g. const float* in = input.ptr();

    assert(matrix.columns() == input.size() && matrix.rows() == result.size());

#pragma omp parallel for schedule(dynamic)
    for (uint32_t row=0; row<matrix.rows(); ++row) {
        float res = 0.0;
        
        std::for_each(matrix.beginRow2(row), matrix.endRow2(row),
                      [&](const RowElement<float>& e){ res += e.value() * input[e.column()]; });
        /*
        auto start = matrix.beginRow2(row);
        auto end = matrix.endRow2(row);
        for (auto it=start; it!=end; ++it) {
            RowElement<float> e = *it;
            res += e.value() * input[e.column()];
        }
        */
        result[row] = res;
    }
}


void calcCorrel(const Vector<float>& fwproj, const Vector<int>& input, Vector<float>& correlation)
{
    assert(fwproj.size() == input.size() && input.size() == correlation.size());

    for (size_t i=0; i<fwproj.size(); ++i)
        correlation[i] = (fwproj[i] != 0.0) ? (input[i] / fwproj[i]) : 0.0;
}

/* Version 1: require gcc 6.1, brings OpenMP 4.5
   #pragma omp parallel for reduction(+:udpate.prt()[:udpate.size()])
*/

// Version 2, critical section
void calcBkProj(const Csr4Matrix& matrix, const Vector<float>& correlation, Vector<float>& update)
{
    assert(matrix.rows() == correlation.size() && matrix.columns() == update.size());

    std::fill(update.ptr(), update.ptr() + update.size(), 0.0);

    #pragma omp parallel
    {
        Vector<float> private_update(update.size(), 0);

        #pragma omp for schedule(dynamic)
        for (uint32_t row=0; row<matrix.rows(); ++row) {
            std::for_each(matrix.beginRow2(row), matrix.endRow2(row),
                [&](const RowElement<float>& e) { private_update[e.column()] += e.value() * correlation[row]; });
        }

        #pragma omp critical
        {
            for (size_t i = 0; i < update.size(); ++i) update[i] += private_update[i];
        }
    }
}

// Version 2, avoid critical section, perform reduction by hand, may have cache issues
void calcBkProj3(const Csr4Matrix& matrix, const Vector<float>& correlation, Vector<float>& update)
{
    assert(matrix.rows() == correlation.size() && matrix.columns() == update.size());

    auto length = update.size();
    std::fill(update.ptr(), update.ptr() + length, 0);

    std::vector<float> proxy_update;

    #pragma omp parallel
    {
        const int nthreads = omp_get_num_threads();
        const int tidx = omp_get_thread_num();

        #pragma omp single
        proxy_update.resize(nthreads * length, 0);

        #pragma omp for
        for (uint32_t row=0; row<matrix.rows(); ++row) {
            std::for_each(matrix.beginRow2(row), matrix.endRow2(row),
                [&](const RowElement<float>& e) {
                    proxy_update[tidx * length + e.column()] += e.value() * correlation[row];
            });
        }

        #pragma omp for
        for (size_t i = 0; i < length; ++i) {
            for (int t = 0; t < nthreads; ++t) {
                update[i] += proxy_update[t * length + i];
            }
        }
    }
}


void calcUpdate(const Vector<float>& update, const Vector<float>& norm, Vector<float>& image)
{
    assert(image.size() == update.size() && image.size() == norm.size());

    for (size_t i=0; i<update.size(); ++i)
        image[i] *= (norm[i] != 0.0) ? (update[i] / norm[i]) : update[i];
}


void mlem(const Csr4Matrix& matrix, const Vector<int>& lmsino,
          Vector<float>& image, int nIterations)
{
    uint32_t nRows = matrix.rows();
    uint32_t nColumns = matrix.columns();

    // Allocate temporary vectors
    Vector<float> fwproj(nRows, 0.0);
    Vector<float> correlation(nRows, 0.0);
    Vector<float> update(nColumns, 0.0);

    // Calculate column sums ("norm")
    Vector<float> norm(nColumns, 0.0);
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    calcColumnSums(matrix, norm);
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Calculated norm, elapsed time: " << elapsed_seconds.count() << "s\n";

    // Fill image with initial estimate
    initImage(norm, image, lmsino);

    std::cout << "Starting " << nIterations << " MLEM iterations" << std::endl;

    start = std::chrono::system_clock::now();
    for (int iter=0; iter<nIterations; ++iter) {
        calcFwProj(matrix, image, fwproj);
        calcCorrel(fwproj, lmsino, correlation);
        calcBkProj(matrix, correlation, update);
        calcUpdate(update, norm, image);

        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;

        double sum = 0;
        for(size_t i = 0; i < image.size(); ++i) sum += image[i];
  #ifndef MESSUNG
        std::cout << "Finished iteration " << iter + 1 << ", ";
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s, image sum "
                  << sum << "\n";
  #else 
        std::cout<<iter<<","<<elapsed_seconds.count()<<std::endl;
    #endif
        start = end;
    }
}


int main(int argc, char *argv[])
{
    ProgramOptions progops = handleCommandLine(argc, argv);
    std::cout << "Matrix file: " << progops.mtxfilename << std::endl;
    std::cout << "Input file: "  << progops.infilename << std::endl;
    std::cout << "Output file: " << progops.outfilename << std::endl;
    std::cout << "Iterations: " << progops.iterations << std::endl;

    Csr4Matrix matrix(progops.mtxfilename);
    Vector<int> lmsino(progops.infilename);
    Vector<float> image(matrix.columns(), 0.0);

    matrix.mapRows(0, matrix.rows());

    mlem(matrix, lmsino, image, progops.iterations);

    image.writeToFile(progops.outfilename);

    return 0;
}
