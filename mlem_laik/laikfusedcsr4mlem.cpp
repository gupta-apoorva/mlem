#include "../helper_files_common/csr4matrix.hpp"
#include "../helper_files_common/vector.hpp"

extern "C" {
#include "laik-backend-mpi.h"
#include "laik-internal.h"
}

// C++ additions to LAIK header
inline Laik_DataFlow operator|(Laik_DataFlow a, Laik_DataFlow b)
{
    return static_cast<Laik_DataFlow>(static_cast<int>(a) | static_cast<int>(b));
}

#include <iostream>
#include <string>
#include <stdexcept>
#include <cassert>
#include <algorithm>
#include <chrono>
#include <unordered_map>
#ifdef MESSUNG 
#include <stdio.h>
#endif


struct ProgramOptions
{
    std::string mtxfilename;
    std::string infilename;
    std::string outfilename;
    int iterations;
    int checkpointing;
};



struct Range
{
    int start;
    int end;
};


struct SubRow
{
    int row;
    int offset;
};


struct SubRowSlice
{
    SubRow from;
    SubRow to;
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


#if 0
// For element-wise weighted partitioning: number of elems in row
double getEW(Laik_Index* i, const void* d)
{
    // SpM* m = (SpM*) d;
    const Csr4Matrix* m = static_cast<const Csr4Matrix*>(d);
    int ii = i->i[0];  // first dimension of Laik_Index

    // Return (float) (m->row[ii + 1] - m->row[ii]);
    return m->elementsInRow(ii);
}
#endif

struct RowData {
    // local row index vector
    std::vector<uint64_t> rowIdx;

    // global row numbers
    size_t fromRow;
    size_t toRow;
};

struct Slice {
    uint64_t from;
    uint64_t to;
};

bool operator==(const Slice& lhs, const Slice& rhs) {
    return lhs.from == rhs.from && lhs.to == rhs.to;
}

struct Hash
{
    std::size_t operator()(const Slice& s) const noexcept
    {
        std::size_t h1 = std::hash<uint64_t>{}(s.from);
        std::size_t h2 = std::hash<uint64_t>{}(s.to);
        return h1 ^ (h2 < 1);
    }
};

std::unordered_map<Slice, RowData, Hash> cache;


RowData cache_get(const Csr4Matrix& matrix, const Slice& slice)
{
    auto search = cache.find(slice);
    if (search != cache.end()) return search->second;

    const uint32_t rowCount = matrix.rows();
    const uint64_t* origRowIdx = matrix.getRowIdx();
    auto it = std::upper_bound(origRowIdx, origRowIdx + rowCount, slice.from);
    if (std::distance(origRowIdx, it) == 0) {
        // Original row index vector is missing first element "0"
        size_t fromRow = 0;
        size_t toRow = std::distance(origRowIdx,
            std::lower_bound(origRowIdx, origRowIdx + rowCount, slice.to - 1));
        size_t len = toRow - fromRow + 2;
        RowData rv = { std::vector<uint64_t>(len), fromRow, toRow };
        std::copy(origRowIdx, origRowIdx + len - 1, std::begin(rv.rowIdx) + 1);
        rv.rowIdx[0] = slice.from;
        rv.rowIdx.back() = slice.to;
        cache[slice] = rv;
        return rv;
    } // else

    size_t fromRow = std::distance(origRowIdx, it - 1);
    size_t toRow = std::distance(origRowIdx,
        std::upper_bound(origRowIdx, origRowIdx + rowCount, slice.to - 1));
    size_t len = toRow - fromRow + 1;  // number of rows + 1
    RowData rv { std::vector<uint64_t>(len), fromRow, toRow };
    std::copy(origRowIdx + fromRow, origRowIdx + fromRow + len, std::begin(rv.rowIdx));
    rv.rowIdx[0] = slice.from;
    rv.rowIdx.back() = slice.to;
    cache[slice] = rv;
    return rv;
}


void calcColumnSums(Laik_Partitioning* p,
                    const Csr4Matrix& matrix, Laik_Data* norm)
{
    laik_switchto_new(norm, laik_All,
                      LAIK_DF_Init | LAIK_DF_ReduceOut | LAIK_DF_Sum);

    float* res;
    laik_map_def1(norm, (void**) &res, 0);

    // Loop over all local slices

    for (int sNo = 0; ; sNo++) {
        Slice slice;
        Laik_TaskSlice* slc = laik_my_slice_1d(p, sNo, &slice.from, &slice.to);
        if (slc == 0) break;

        RowData rv = cache_get(matrix, slice);
        auto fromRow = rv.fromRow;
        auto toRow = rv.toRow;
        matrix.mapRows(fromRow, toRow - fromRow + 1);

        for (size_t r = 0; r < rv.rowIdx.size() - 1; r++) {
            std::for_each(matrix.beginRow(r, rv.rowIdx), matrix.endRow(r, rv.rowIdx),
                [&](const RowElement<float>& e){ res[e.column()] += e.value();
            });
        }

        float s = 0.0;
        for (uint i = 0; i < matrix.columns(); i++) s += res[i];
#ifndef MESSUNG
        laik_log(LAIK_LL_Info, "Range %d - %d: Sum: %lf \n", fromRow, toRow, s);
#endif
    }

    laik_switchto_new(norm, laik_All, LAIK_DF_CopyIn);

    laik_map_def1(norm, (void**) &res, 0);

#ifndef MESSUNG
    float s = 0.0;
    for(uint i = 0; i < matrix.columns(); i++) s += res[i];
    laik_log(LAIK_LL_Info, "Norm_Sum: %lf", s);
#endif
}


void initImage(Laik_Data* norm, Vector<float>& image, const Vector<int>& lmsino)
{
    float* n;
    laik_map_def1(norm, (void**) &n, 0);

    // Sum up norm vector, creating sum of all matrix elements
    float sumnorm = 0.0;
    for (size_t i=0; i<image.size(); ++i) sumnorm += n[i];
    float sumin = 0.0;
    for (size_t i=0; i<lmsino.size(); ++i) sumin += lmsino[i];

    float initial = static_cast<float>(sumin / sumnorm);

#ifndef MESSUNG
    laik_log(LAIK_LL_Info, "Init: SumNorm= %lf, sumin= %lf, initial value = %lf\n", sumnorm, sumin, initial);
#endif
    for (size_t i=0; i<image.size(); ++i) image[i] = initial;
}

void fused(
    Laik_Partitioning* p, 
    const Csr4Matrix& matrix, 
    Laik_Data* update,
    Vector<float>& image,
    const Vector<int>& lmsino, 
    Laik_Data* norm
){
    laik_switchto_new(update, laik_All, LAIK_DF_Init | LAIK_DF_ReduceOut | LAIK_DF_Sum);

    float* upd,*nm;
    Vector<float> correlation(matrix.rows(), 0);

    laik_map_def1(update, (void**)&upd, 0);

    for(int sNo = 0; ; sNo++) {

        Slice slice;
        Laik_TaskSlice* slc = laik_my_slice_1d(p, sNo, &slice.from, &slice.to);
        if (slc == 0) break;

        RowData rv = cache_get(matrix, slice);
        auto fromRow = rv.fromRow;
        auto toRow = rv.toRow;
        matrix.mapRows(fromRow, toRow - fromRow + 1);

        for(size_t r = 0; r < rv.rowIdx.size() - 1; r++) {
            std::for_each(matrix.beginRow(r, rv.rowIdx), matrix.endRow(r, rv.rowIdx),
                [&](const RowElement<float>& e){ correlation[r + fromRow] += (float)e.value() * image[e.column()]; });
        }

        for (size_t i = 0; i < correlation.size(); ++i)
            correlation[i] = (correlation[i] != 0) ? (lmsino[i] / correlation[i]) : 0;

        for (size_t r = 0; r < rv.rowIdx.size() - 1; r++) {
            std::for_each(matrix.beginRow(r, rv.rowIdx), matrix.endRow(r, rv.rowIdx),
                [&](const RowElement<float>& e){ upd[e.column()] += (float)e.value() * correlation[r + fromRow]; });
        }

    }

    laik_switchto_new(update, laik_All, LAIK_DF_CopyIn);
    laik_map_def1(update, (void**)&upd, 0);
    laik_map_def1(norm, (void**)nm, 0);

    for (size_t i = 0; i < image.size(); ++i)
        image[i] *= (nm[i] != 0) ? (upd[i] / nm[i]) : upd[i];
}

void mlem(Laik_Instance* inst, Laik_Group* world, Laik_Partitioning* part,
          const Csr4Matrix& matrix, const Vector<int>& lmsino,
          Vector<float>& image, int nIterations)
{
    uint32_t nRows = matrix.rows();
    uint32_t nColumns = matrix.columns();
    std::chrono::duration<float> compute_time, total_time;

    // Allocate temporary vectors
    Laik_Data* fwproj = laik_new_data_1d(world, laik_Float, nRows);
    Vector<float> correlation(nRows, 0.0); // could update vector fwproj instead
    //Vector<float> update(nColumns, 0.0);
    Laik_Data* update = laik_new_data_1d(world, laik_Float, nColumns);

    // Calculate column sums ("norm")
    Laik_Data* norm = laik_new_data_1d(world, laik_Float, nColumns);
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    calcColumnSums(part, matrix, norm);
    end = std::chrono::system_clock::now();
    std::chrono::duration<float> elapsed_seconds = end - start;

#ifndef MESSUNG
    laik_log(LAIK_LL_Info, "Calculated norm, elapsed time: %f\n", elapsed_seconds.count());
#endif

    // Fill image with initial estimate
    initImage(norm, image, lmsino);

#ifndef MESSUNG
    laik_log(LAIK_LL_Info, "Starting %d MLEM Iterations\n", nIterations);
#endif
    for (int iter=0; iter<nIterations; ++iter) {

        compute_time = std::chrono::duration<float>::zero();
        total_time = std::chrono::duration<float>::zero();
        
     fused(
        part,
        matrix, 
        fwproj,
        image,
        lmsino, 
        norm
    );
        // Debug: sum over image values
        float s = 0.0;
        for(uint i = 0; i < matrix.columns(); i++) s += image[i];
#ifndef MESSUNG
        laik_log(LAIK_LL_Info, "Finished Iteration %d, Time: %f(%f/%f/%f)\n",
                 iter + 1,
                 total_time.count(),
                 compute_time.count(),
                 laik_get_total_time() - laik_get_backend_time(),
                 laik_get_backend_time()
                 );
#else
        printf("%d, %lf, %lf, %lf, %lf\n",
               iter + 1,
               total_time.count(),
               compute_time.count(),
               laik_get_total_time() - laik_get_backend_time(),
               laik_get_backend_time() );
#endif

        laik_reset_profiling(inst);
#ifndef MESSUNG
        laik_log(LAIK_LL_Info, "Image Sum: %f\n", s);
#endif
    }
}


void mlemPartitioner(Laik_Partitioner* pr,
                     Laik_BorderArray* ba, Laik_BorderArray* oldBA)
{
    // Laik_Space* space = ba->space;  // unused
    Laik_Group* g = ba->group;
    Csr4Matrix* mtx = static_cast<Csr4Matrix*>(pr->data);

    int sliceCount = g->size;
    uint64_t elementsPerSlice = (mtx->elements() + sliceCount - 1) / sliceCount;
    uint64_t elementCount = 0;
    uint64_t task = 0;

    SubRowSlice* slice_data = new SubRowSlice[sliceCount];  // FIXME memory leak

    Laik_Slice sl;
    sl.from = { 0, 0, 0 }; // Laik_Index
    sl.to = { 0, 0, 0, };

    int startRow = 0;
    int startOffset = 0;

    for (uint32_t row = 0; row < mtx->rows(); ++row) {
        elementCount += mtx->elementsInRow(row);
        if (elementCount >= sl.from.i[0] + elementsPerSlice) {
            sl.to.i[0] = std::min(sl.from.i[0] + elementsPerSlice, mtx->elements());

            uint32_t offset = sl.from.i[0] + elementsPerSlice - elementCount + mtx->elementsInRow(row);

            slice_data[task].from.row = startRow;
            slice_data[task].from.offset = startOffset;
            slice_data[task].to.row = row;
            slice_data[task].to.offset = offset;

            if (offset == mtx->elementsInRow(row)) {
                slice_data[task].to.row = row + 1;
                slice_data[task].to.offset = 0;
            }

            laik_append_slice(ba, task, &sl, 0, (void*)&slice_data[task]);
            ++task;
            sl.from = sl.to;
            startRow = row;
            startOffset = offset;
        }
    }
}


int main(int argc, char *argv[])
{
    ProgramOptions progops = handleCommandLine(argc, argv);
#ifndef MESSUNG
    std::cout << "Matrix file: " << progops.mtxfilename << std::endl;
    std::cout << "Input file: "  << progops.infilename << std::endl;
    std::cout << "Output file: " << progops.outfilename << std::endl;
    std::cout << "Iterations: " << progops.iterations << std::endl;
#endif
    Csr4Matrix matrix(progops.mtxfilename);
    //Csr4Matrix matrix = Csr4Matrix::testMatrix();

#ifndef MESSUNG
    std::cout << "Matrix rows (LORs): " << matrix.rows() << std::endl;
    std::cout << "Matrix cols (VOXs): " << matrix.columns() << std::endl;
#endif
    Vector<int> lmsino(progops.infilename);
    Vector<float> image(matrix.columns(), 0.0);

    Laik_Instance* inst = laik_init_mpi(&argc, &argv);
    Laik_Group* world = laik_world(inst);

    laik_enable_profiling(inst);

    // 1d space, block partitioning of matrix elements
    Laik_Space* space = laik_new_space_1d(inst, matrix.elements());
    Laik_Partitioner* part = laik_new_block_partitioner1();
    Laik_Partitioning* p = laik_new_partitioning(world, space, part, nullptr);

    mlem(inst, world, p, matrix, lmsino, image, progops.iterations);

    if (laik_myid(world) == 0)
        image.writeToFile(progops.outfilename);

    laik_finalize(inst);

    return 0;
}
