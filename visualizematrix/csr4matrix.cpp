/**
	csr4matrix.cpp

	Created on: Oct 15, 2009
		Author: kuestner
*/

#include <string>
using std::string;
#include <sstream>
using std::stringstream;
#include <stdexcept>
using std::runtime_error;

#include <stdint.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>

#include "csr4matrix.hpp"


Csr4Matrix::Csr4Matrix(const string& filename)
{
	// Cannot use the C++ iostream library, as it does not provide a way
	// to access the underlying file descriptor
	fileDescriptor = open(filename.c_str(), O_RDONLY);
	if (fileDescriptor < 0)
		throw runtime_error(string("Cannot open file ") + filename);

	// Alternative for seek would be getting the file size via stat
	fileSize = lseek(fileDescriptor, 0, SEEK_END);
	lseek(fileDescriptor, 0, SEEK_SET);

	if (fileSize < minHeaderLength)
		throw runtime_error(string("File too short ") + filename);

	map = (char*)mmap(0, fileSize, PROT_READ, MAP_PRIVATE, fileDescriptor, 0);
	if (map == MAP_FAILED)
		throw runtime_error(string("Cannot map file ") + filename);

	// Read header data
	// NOTE This code contains magic numbers (in the form of 'sizeof(uint32_t)'
	// etc). They need to match with those in class CsrWriter
	string magic = string((char*)map, 4);
	uint8_t version = *((uint8_t*)map + 4);

	flags = *((uint8_t*)(map + 5));
	if (flags & xsymmask) symcfg.x = true;
	if (flags & ysymmask) symcfg.y = true;
	if (flags & zsymmask) symcfg.z = true;

	nRows = *((uint32_t*)(map + 8));
	nColumns = *((uint32_t*)(map + 12));
	nnz = *((uint64_t*)(map + 16));

	uint32_t scanConfigBytes = *((uint32_t*)(map + 24));

	if (magic != string("PCSR")) {
		stringstream ss;
		ss<<"Input file '"<<filename<<"' is not a CSR matrix";
		throw runtime_error(ss.str());
	}

	if (version != 4) {
		stringstream ss;
		ss<<"Input file '"<<filename<<"' has wrong CSR version (need v4)";
		throw runtime_error(ss.str());
	}

	uint32_t size = minHeaderLength + scanConfigBytes
		+ nRows * sizeof(uint64_t) + nnz * sizeof(RowElement<float>);
	if (fileSize < size) {
		stringstream ss;
		ss<<"Input file '"<<filename<<"' is too short: "<<fileSize<<" B"
			<<" instead of at least "<<size<<" B";
		throw runtime_error(ss.str());
	}

	if (scanConfigBytes < minScanConfigBytes) {
		throw runtime_error(string(
			"Cannot find scanner geometry configuration"));
	}

	// Start reading scanner geometry configuration
	char* p = map + minHeaderLength; // p helper pointer
	uint32_t nLayers = *((uint32_t*)p); p += sizeof(uint32_t);
	for (uint32_t i=0; i<nLayers; ++i) {
		float w  = *((float*)p); p+= sizeof(float);
		float h  = *((float*)p); p+= sizeof(float);
		float d  = *((float*)p); p+= sizeof(float);
		float r  = *((float*)p); p+= sizeof(float);
		float mu = *((float*)p); p+= sizeof(float);
		scancfg.addLayer(LayerConfig(w, h, d, r, mu));
	}
	uint32_t nBlocks = *((uint32_t*)p); p += sizeof(uint32_t);
	scancfg.setNBlocks(nBlocks);
	uint32_t blocksize = *((uint32_t*)p); p += sizeof(uint32_t);
	for (uint32_t i=0; i<blocksize-1; ++i) { // nr of gaps is one less
		float gap =  *((float*)p); p+= sizeof(float);
		scancfg.addBlockGap(gap);
	}
	uint32_t nRings = *((uint32_t*)p); p += sizeof(uint32_t);
	for (uint32_t i=0; i<nRings-1; ++i) { // nr of gaps is one less
		float gap =  *((float*)p); p+= sizeof(float);
		scancfg.addRingGap(gap);
	}
	float w = *((float*)p); p+= sizeof(float);
	float h = *((float*)p); p+= sizeof(float);
	float d = *((float*)p); p+= sizeof(float);
	scancfg.setVoxelSize(Vector3d<float>(w, h, d));
	uint32_t nx = *((uint32_t*)p); p += sizeof(uint32_t);
	uint32_t ny = *((uint32_t*)p); p += sizeof(uint32_t);
	uint32_t nz = *((uint32_t*)p); p += sizeof(uint32_t);
	scancfg.setGridDim(IVec3(nx, ny, nz));

	if (p - (map + minHeaderLength) != scanConfigBytes) {
		throw runtime_error(string(
			"Error reading scanner geometry configuration"));
	}

	rowidx = (uint64_t*)(map + minHeaderLength + scanConfigBytes);
	data = (RowElement<float>*)(map + minHeaderLength
		+ scanConfigBytes + nRows * sizeof(uint64_t));
}

Csr4Matrix::~Csr4Matrix()
{
	munmap(map, fileSize);
	close(fileDescriptor);
}

uint32_t Csr4Matrix::elementsInRow(uint32_t rowNr) const
{
	if (rowNr == 0) return rowidx[rowNr];
	else return rowidx[rowNr] - rowidx[rowNr - 1];
}

Csr4Matrix::RowIterator Csr4Matrix::beginRow(uint32_t rowNr) const {
	if (rowNr == 0) return RowIterator(data);
	else return RowIterator(&data[rowidx[rowNr - 1]]);
}

Csr4Matrix::RowIterator Csr4Matrix::endRow(uint32_t rowNr) const {
	return RowIterator(&data[rowidx[rowNr]]);
}
