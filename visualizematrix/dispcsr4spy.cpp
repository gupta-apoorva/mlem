/**
 * dispcsr4spy.cpp
 *
 * Created on: Feb 18, 2011
 *     Author: kuestner
 */

#include <cstdlib>
#include <stdint.h>

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
#include <string>
using std::string;
#include <vector>
using std::vector;
#include <algorithm>
#include <stdexcept>
using std::runtime_error;

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <SDL2/SDL.h>

#include "csr4matrix.hpp"
#include "symhandler.hpp"
#include "timers.hpp"


struct ProgramOptions
{
    string mtxfilename;
    int r, c, w, h;
    bool genSyms;
    bool sqrtScale;
};

struct Rect // same as SDL_Rect { int x,y,w,h; }
{
    int r, c, w, h;
};

template <typename T>
T divideInto(const T& dividend, const T& divisor)
{
    return (dividend + divisor - 1) / divisor;
}

/**
    Function generates all possible symmetry settings from summary baseSym
    E.g. (1, 1, 0) (x, y, no z) will be transformed into the list
    (1, 0, 0), (0, 1, 0) and (1, 1, 0)
    Note that (0, 0, 0) will not be generated
*/
vector<SymConfig> generateSyms(const SymConfig& baseSym)
{
    vector<SymConfig> result;
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

void fillTiles(const Csr4Matrix& mtx, const Rect& rect,
               int ratio, const ProgramOptions& args,
               vector<vector<int> >& tiles)
{
    SymConfig mtxsyms = mtx.symconfig();
    SymHandler symhan(mtx.scannerConfig());
    vector<SymConfig> symsList;
    if (args.genSyms) symsList = generateSyms(mtxsyms);
    // else list stays empty

    // NOTE v1 is slightly slower than v2

    // --------------- version 1 --------------------
    
    // for (int r=0; r<mtx.rows(); ++r) {
    //     vector<int> tr;
    //     for (int i=0; i<symsList.size(); ++i) {
    //         tr.push_back(symhan.transformLor(r, symsList[i]));
    //     }
    //     typedef Csr4Matrix::RowIterator RowItor;
    //     for (RowItor it=mtx.beginRow(r); it!=mtx.endRow(r); ++it) {
    //         int c = (*it).column();
    //         // Process element at hand
    //         if (r >= rect.r && r < rect.r + rect.h &&
    //             c >= rect.c && c < rect.c + rect.w) {
    //             ++tiles[(r-rect.r)/ratio][(c-rect.c)/ratio];
    //         }
    //         // Process generated (indirect, dependent) elements
    //          for (int i=0; i<symsList.size(); ++i) {
    //              int ttr = tr[i];
    //              int tc = symhan.transformVoxel(c, symsList[i]);
    //              if (ttr >= rect.r && ttr < rect.r + rect.h &&
    //                  tc >= rect.c && tc < rect.c + rect.w) {
    //                  ++tiles[(ttr-rect.r)/ratio][(tc-rect.c)/ratio];
    //              }
    //          }
    //     }
    // }

    // --------------- version 2 --------------------

    for (int r=rect.r; r<rect.r+rect.h; ++r) {
        int rr = (r - rect.r) / ratio;
        typedef Csr4Matrix::RowIterator RowItor;
        for (RowItor it=mtx.beginRow(r); it!=mtx.endRow(r); ++it) {
            int c = (*it).column();
            if (c < rect.c || c >= rect.c + rect.w) continue;
            int cc = (c - rect.c) / ratio;
            ++tiles[rr][cc];
        }
    }

    for (size_t i=0; i<symsList.size(); ++i) {
        for (size_t r=0; r<mtx.rows(); ++r) {
            int rt = symhan.transformLor(r, symsList[i]);
            if (rt < rect.r || rt >= rect.r + rect.h) continue;
            int rr = rt / ratio;
            typedef Csr4Matrix::RowIterator RowItor;
            for (RowItor it=mtx.beginRow(r); it!=mtx.endRow(r); ++it) {
                int c = (*it).column();
                int ct = symhan.transformVoxel(c, symsList[i]);
                if (ct < rect.c || ct >= rect.c + rect.w) continue;
                int cc = (ct - rect.c) / ratio;
                ++tiles[rr][cc];
            }
        }
    }
}

void drawPlane(SDL_Renderer *renderer,
               const vector<vector<int> >& tiles,
               float threshold, int maxWeight, bool sqrtScale)
{
    size_t h = tiles.size();
    size_t w = tiles[0].size();

    // Clear white screen
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_RenderClear(renderer);
    SDL_RenderPresent(renderer);

    for (size_t y=0; y<h; ++y) for (size_t x=0; x<w; ++x) {
        double value = (double)tiles[y][x] / (double)maxWeight;
//        cout<<value<<"   ";
        if (sqrtScale) value = sqrt(value);
        if (value != 0 && value >= threshold) {
            uint8_t color = (uint8_t)(255 - value * 255);
            SDL_SetRenderDrawColor(renderer, color, color, color, 255);
            SDL_RenderDrawPoint(renderer, x, y);
        }
    }
    SDL_RenderPresent(renderer);
}

void parseCommandLine(int argc, char *argv[], ProgramOptions& ops)
{
    po::options_description desc("Options");
    desc.add_options()
            ("help", "show this help message")
            ("matrix,m", po::value<string>(&ops.mtxfilename),
             "system matrix file (type csr, version 4) [required]")
            ("row,r", po::value<int>(&ops.r)->default_value(0),
             "Select matrix display window start row")
            ("column,c", po::value<int>(&ops.c)->default_value(0),
             "Select matrix display window start column")
            ("width,w", po::value<int>(&ops.w)->default_value(-1),
             "Select matrix display window width")
            ("height,h", po::value<int>(&ops.h)->default_value(-1),
             "Select matrix display window height")
            ("symmetries,s", po::bool_switch(&ops.genSyms)->default_value(false),
             "Generate matrix elements ommited due to scanner symmetries")
            ("usr-sqrt-scaling,u", po::bool_switch(&ops.sqrtScale)
             ->default_value(false), "Apply square root to final pixel color")
            ;

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout<<"Usage: "<<argv[0]<<" [options]\n";
        cout<<desc;
        exit(EXIT_SUCCESS);
    }

    string prophelp = string("Type '") + string(argv[0])
            + string(" --help' to get more help.");

    if (!vm.count("matrix"))
        throw runtime_error(string("Parameter 'matrix' is mandatory.\n")
                            + prophelp);
}

const ProgramOptions handleCommandLine(int argc, char *argv[])
{
    ProgramOptions ops;
    try {
        parseCommandLine(argc, argv, ops);
    }
    catch (std::exception& e) {
        cout<<e.what()<<"\n";
        exit(EXIT_FAILURE);
    }
    return ops;
}


int main(int argc, char *argv[])
{
    ProgramOptions args = handleCommandLine(argc, argv);

    string mtxfilename(args.mtxfilename);

    Csr4Matrix mtx(mtxfilename);

    // Handle some errors for command line arguments
    string prophelp = string("Type '") + string(argv[0])
            + string(" --help' to get more help.");
    if (args.r < 0 || args.r >= (int)mtx.rows()) {
        cerr<<"Paramter 'row' must be in [0, matrixrows]"<<endl;
        return EXIT_FAILURE;
    }
    if (args.c < 0 || args.c >= (int)mtx.columns()) {
        cerr<<"Paramter 'column' must be in [0, matrixcolumns]"<<endl;
        return EXIT_FAILURE;
    }
    // Handle gracefully
    if (args.w == -1) args.w = (int)mtx.columns();
    if (args.h == -1) args.h = (int)mtx.rows();
    if (args.w < 0 || args.c + args.w > (int)mtx.columns() ||
            args.h < 0 || args.r + args.h > (int)mtx.rows()) {
        cout<<"Display window will be modified. See below."<<endl;
    }

    SymConfig symcfg = mtx.symconfig();

    cout<<"Matrix statistics:"<<endl;
    cout<<" rows: "<<mtx.rows()<<endl;
    cout<<" columns: "<<mtx.columns()<<endl;
    cout<<" elements: "<<mtx.elements()<<endl;
    cout<<"Matrix symmetry configuration:"<<endl;
    cout<<" x symmetry: "<<((symcfg.x)?"yes":"no")<<endl;
    cout<<" y symmetry: "<<((symcfg.y)?"yes":"no")<<endl;
    cout<<" z symmetry: "<<((symcfg.z)?"yes":"no")<<endl;

    // Determine matrix display window from command line parameters and matrix
    // file
    Rect disp;
    disp.r = args.r;
    disp.c = args.c;
    disp.h = std::min(args.h, (int)mtx.rows());
    if (disp.r + disp.h >= (int)mtx.rows()) disp.h = mtx.rows() - disp.r;
    disp.w = std::min(args.w, (int)mtx.columns());
    if (disp.c + disp.w >= (int)mtx.columns()) disp.w = mtx.columns() - disp.c;

    cout<<"Displaying rows "<<disp.r<<" to "<<(disp.r + disp.h)<<endl;
    cout<<"Displaying columns "<<disp.c<<" to "<<(disp.c + disp.w)<<endl;

    // Matrix is mapped onto tiles (factor 'ratio'). Tiles are mapped 1:1 onto
    // pixels.
    // SDL window is 750 x 750 pixels (tiles) at max
    const int nTilesMax = 750;

    // Find matrix to tiles, so that tiles correspond to square matrix ares
    int longerEdge = std::max(disp.w, disp.h);
    // Divide bigger edge into nTiles tiles (buckets)
    int ratio = divideInto(longerEdge, nTilesMax);
    // Calc number of tiles (buckets) used in width and height
    // If e.g. longerEdge was columns then nTilesW should be nTiles again
    int nTilesW = divideInto(disp.w, ratio);
    int nTilesH = divideInto(disp.h, ratio);
    cout<<"One pixel equals a square of "<<ratio<<" matrix rows and columns"
       <<endl;
    cout<<"Display window: "<<nTilesW<<" x "<<nTilesH<<endl;

    vector<vector<int> > tiles; // initializes with zero
    tiles.resize(nTilesH);
    for (size_t i=0; i<tiles.size(); ++i) tiles[i].resize(nTilesW);

    cout<<"Analyzing ... ";
    cout<<std::flush; //cout.flush();
    double start = secs();
    fillTiles(mtx, disp, ratio, args, tiles);
    double end = secs();
    cout<<"done, time: "<<end-start<<" s"<<endl;

    int maxWeight = 0;
    for (size_t r=0; r<tiles.size(); ++r)
        for (size_t c=0; c<tiles[0].size(); ++c)
            if (tiles[r][c] > maxWeight) maxWeight = tiles[r][c];

    cout<<"Max tile weight: "<<maxWeight<<endl;

    // Start SDL 2.0

    // Start graphics
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        cerr<<"Unable to initialize SDL: "<<string(SDL_GetError())<<endl;
        return EXIT_FAILURE;
    }
    atexit(SDL_Quit);

    string title = string("Spy plot of ") + mtxfilename;
    SDL_Window *window = SDL_CreateWindow(title.c_str(),
                                          SDL_WINDOWPOS_UNDEFINED,
                                          SDL_WINDOWPOS_UNDEFINED,
                                          nTilesW, nTilesH,
                                          0//SDL_WINDOW_OPENGL
                                          );
    if (!window) {
        cerr<<"Unable to set video mode "<<nTilesW<<" x "<<nTilesH
           <<": "<<string(SDL_GetError())<<endl;
        return EXIT_FAILURE;
    }

    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, 0);
    if (!renderer) {
        cerr<<"Unable to create SDL renderer: "<<string(SDL_GetError())<<endl;
        return EXIT_FAILURE;
    }

    // Clear white screen
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_RenderClear(renderer);
    SDL_RenderPresent(renderer);

    if (args.sqrtScale)
        cout<<"Applying square root scaling"<<endl;

    // Display threshold
    float threshold = 0;
    drawPlane(renderer, tiles, threshold, maxWeight, args.sqrtScale);

    // Main display loop
    SDL_Event event;
    bool running = true;
    while (running && SDL_WaitEvent(&event)) {
        switch (event.type) {
        case SDL_KEYDOWN:
            switch (event.key.keysym.sym) {
            case SDLK_PLUS:
            case SDLK_KP_PLUS:
                if (threshold >= 0.05) threshold -= 0.05;
                else threshold = 0;
                drawPlane(renderer, tiles, threshold, maxWeight,
                          args.sqrtScale);
                break;
            case SDLK_MINUS:
            case SDLK_KP_MINUS:
                if (threshold < 1) threshold += 0.05;
                drawPlane(renderer, tiles, threshold, maxWeight,
                          args.sqrtScale);
                break;
            case SDLK_ESCAPE:
                running = false;
                break;
            default:
                break;
            }
            break;
        case SDL_QUIT:
            running = false;
            break;
        }
    }

    SDL_DestroyWindow(window);
    SDL_Quit();

    return EXIT_SUCCESS;
}
