/**
 * dispcsr4spy.cpp
 *
 * Created on: Feb 21, 2011
 *     Author: kuestner
 */

#include "scannerconfig.hpp"
#include <vector>

class SymHandler
{
    public:
        SymHandler(const ScannerConfig& scancfg);
        ~SymHandler();
        std::vector<SymConfig> generateSyms(const SymConfig& baseSym);
        int transformVoxel(int voxelnr, const SymConfig& sym) const;
        int transformDetector(int detnr, const SymConfig& sym) const;
        int transformLor(int lornr, const SymConfig& sym) const;
    private:
        void initLUTs(const ScannerConfig& scancfg);

        int nvx, nvy, nvz, nvx1, nvy1, nvz1, sliceArea;
        int **lut;
        int nCrystals;
};

/// This method generates all possible symmetry settings from \p baseSym
/**
    For example, (1, 1, 0) (that is symmetries in x and y, but not in z) will be
    transformed into the list (1, 0, 0), (0, 1, 0) and (1, 1, 0).
    \attention Note that (0, 0, 0) will not be generated.
*/
inline
std::vector<SymConfig> SymHandler::generateSyms(const SymConfig& baseSym)
{
    std::vector<SymConfig> result;
    // NOTE C++ standard says (int)true == 1 and (int)false == 0
    // Also (bool)0 == false and bool(x) == true for all x != 0
    for (int x=0; x<2 && x<(int)baseSym.x+1; ++x) {
        for (int y=0; y<2 && y<(int)baseSym.y+1; ++y) {
            for (int z=0; z<2 && z<(int)baseSym.z+1; ++z) {
                if (x==0 && y==0 && z==0) continue;
                result.push_back(SymConfig(x, y, z));
            }
        }
    }
    return result;
}

inline
int SymHandler::transformVoxel(int voxelnr, const SymConfig& sym) const
{
    // Breaking voxel number into voxel coordinates
    // NOTE Various optimizations possible (modulo, etc.) Using C stdlib
    // div is one of them.
    div_t e = div(voxelnr, sliceArea);
    int z = e.quot;
    e = div(e.rem, nvx);
    int y = e.quot;
    int x = e.rem;

    // Mirroring voxel coordinates
    // NOTE Ifs should not be in the center of a loop nest
    // But branch is always same way (if outer code is smart)
    // Various optimizations possible
    // No if when using multiplication
    // NOTE Formulas are correct even if nvx is an odd number
    if (sym.x) x = nvx1 - x;
    if (sym.y) y = nvy1 - y;
    if (sym.z) z = nvz1 - z;

    // Reconstructing voxel number
    return z * sliceArea + y * nvx + x;
}

inline
int SymHandler::transformLor(int lornr, const SymConfig& sym) const
{
    div_t e = div(lornr, nCrystals);
    int d1 = transformDetector(e.quot, sym);
    int d2 = transformDetector(e.rem,  sym);
    return (d1 < d2) ? (d1 * nCrystals + d2) : (d2 * nCrystals + d1);
}

inline
int SymHandler::transformDetector(int detnr, const SymConfig& sym) const
{
    // NOTE Two possible alternatives:
    // i) 3 LUTs for x, y, z used one after the other
    // ii) 7 LUTs, one for each symmetry
    // Alternative (i) uses less space but takes more steps
    // We implement (ii)

    // This if should be avoided by calling code
    // if (!sym.x && !sym.y && !sym.z) return detnr;

    // This line supersedes the previous if as it contains lut[0][...]
    // return lut[sym.x][sym.y][sym.z][detnr];
    return lut[(int)sym.x + ((int)sym.y<<1) + ((int)sym.z<<2)][detnr];
}
