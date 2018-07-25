/**
 * symhandler.cpp
 *
 * Created on: Feb 21, 2011
 *     Author: kuestner
 */

#include <cstdlib>

#include "scannerconfig.hpp"
#include "symhandler.hpp"

SymHandler::SymHandler(const ScannerConfig& scancfg)
{
    // Init for voxel syms
    IVec3 gridDim = scancfg.getGridDim();
    nvx = gridDim.x; nvx1 = nvx - 1;
    nvy = gridDim.y; nvy1 = nvy - 1;
    nvz = gridDim.z; nvz1 = nvz - 1;
    sliceArea = nvx * nvy;

    // Init for detector syms
    initLUTs(scancfg);
    nCrystals = scancfg.nCrystals();
}

SymHandler::~SymHandler()
{
    delete[] lut[0];
    delete[] lut;
}

void SymHandler::initLUTs(const ScannerConfig& scfg)
{
    // i) Find out which symmetry axis the scanner supports
    SymConfig presSyms;
    // Symmetry to yz plane (i.e. x-axis) is always present. This is due to
    // the way we define the scanner geometry (in Monte Carlo simulations as
    // well as the DRF and LOI model): the first detector block is always
    // aligned along the x axis with its center precisely on the axis
    presSyms.x = true;
    // Symmetry in xz plane (y-axis) is not present, if there are an odd
    // number of detector blocks (e.g. 3)
    if (scfg.nBlocks() % 2 != 0) // if is odd
        presSyms.y = false;
    else presSyms.y = true;
    // Symmetry in xy plane (z-axis) is always present
    // NOTE A detector can be symmetric to itself
    presSyms.z = true;

    int nCrystals = scfg.nCrystals();

    lut = new int*[8];
    lut[0] = new int[8 * nCrystals];
    for (int i=1; i<8; ++i) lut[i] = &lut[0][i * nCrystals];

    // Init LUT 0 (0, 0, 0)
    for (int i=0; i<nCrystals; ++i)
        lut[0][i] = i;

    int voxelsInLayer = scfg.blocksize() * scfg.nBlocks();
    int voxelsInRing =  scfg.blocksize() * scfg.nBlocks() * scfg.nLayers();

    // Init LUT 1 (1, 0, 0) - x sym
    int xoff = scfg.blocksize() - 1;
    for (size_t r=0; r<scfg.nRings(); ++r) {
        for (size_t l=0; l<scfg.nLayers(); ++l) {
            for (int v=0; v<voxelsInLayer; ++v) {
                lut[1][v + l * voxelsInLayer + r * voxelsInRing]
                    = (voxelsInLayer - v + xoff) % voxelsInLayer
                    + l * voxelsInLayer + r * voxelsInRing;
            }
        }
    }

    // Init LUT 2 (0, 1, 0) - y sym
    int yoff = scfg.blocksize() * (scfg.nBlocks() / 2 + 1) - 1;
    for (size_t r=0; r<scfg.nRings(); ++r) {
        for (size_t l=0; l<scfg.nLayers(); ++l) {
            for (int v=0; v<voxelsInLayer; ++v) {
                lut[2][v + l * voxelsInLayer + r * voxelsInRing]
                    = (voxelsInLayer - v + yoff) % voxelsInLayer
                    + l * voxelsInLayer + r * voxelsInRing;
            }
        }
    }

    // Init LUT 4 (0, 0, 1) - z sym
    for (size_t r=0; r<scfg.nRings(); ++r) {
        for (int v=0; v<voxelsInRing; ++v) {
            lut[4][v + r * voxelsInRing]
                = v + (scfg.nRings() - 1 - r) * voxelsInRing;
        }
    }

    // Build LUT 3 (1, 1, 0) - xy sym
    // with the help of LUT 1 and 2
    for (int i=0; i<nCrystals; ++i)
        lut[3][i] = lut[2][lut[1][i]];

    // Build LUT 5 (1, 0, 1) - xz sym
    // with the help of LUT 1 and 4
    for (int i=0; i<nCrystals; ++i)
        lut[5][i] = lut[4][lut[1][i]];

    // Build LUT 6 (0, 1, 1) - yz sym
    // with the help of LUT 2 and 4
    for (int i=0; i<nCrystals; ++i)
        lut[6][i] = lut[4][lut[2][i]];

    // Build LUT 7 (1, 1, 1) - xyz sym
    // with the help of LUT 1, 2 and 4
    for (int i=0; i<nCrystals; ++i)
        lut[7][i] = lut[4][lut[2][lut[1][i]]];
}
