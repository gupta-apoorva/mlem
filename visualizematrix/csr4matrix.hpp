/**
	csr4matrix.hpp

	Created on: Oct 15, 2009
		Author: kuestner
*/

#pragma once

#include "matrixelement.hpp"
#include "scannerconfig.hpp"

# include <boost/iterator/iterator_facade.hpp>

#include <string>
#include <stdint.h>

/// A Matrix in csr4 format
class Csr4Matrix {
public:
	Csr4Matrix(const std::string& filename);
	~Csr4Matrix();

	uint32_t rows() const { return nRows; }
	uint32_t columns() const { return nColumns; }
	uint64_t elements() const { return nnz; }
	uint32_t elementsInRow(uint32_t rowNr) const;
	const SymConfig symconfig() const { return symcfg; }
	const ScannerConfig scannerConfig() const { return scancfg; }

    // allow access to raw pointers
    const RowElement<float>* getData() const { return data; }
    const uint64_t* getRowIdx() const { return rowidx; }

    void setData(RowElement<float>* d) { data = d; }

	class RowIterator // a const iterator
     : public std::iterator<std::forward_iterator_tag, RowElement<float> > {
	public:
		explicit RowIterator(RowElement<float>* pelem_) : pelem(pelem_) {}
		RowIterator(const RowIterator& i) : pelem(i.pelem) {}
		RowIterator& operator=(const RowIterator& rhs) {
			pelem = rhs.pelem;
			return (*this);
		}
		bool operator==(const RowIterator& rhs) {
			return (pelem == rhs.pelem);
		}
		bool operator!=(const RowIterator& rhs) {
			return (pelem != rhs.pelem);
		}
	    RowIterator& operator++() { // prefix increment
			++pelem;
			return (*this);
		}
		RowIterator operator++(int) { // postfix increment
			RowIterator tmp(*this);
			++(*this); // call prefix increment
			return tmp;
		}
        const RowElement<float>& operator*() const {
			return *pelem;
		}
        const RowElement<float>* operator->() const {
            return pelem;
        }

	private:
		RowElement<float>* pelem;
	};


    template <class Value>
    class node_iter : public boost::iterator_facade <
        node_iter<Value>,
        Value,
        //boost::forward_traversal_tag
        //boost::random_access_traversal_tag,
        std::random_access_iterator_tag,
        Value&
        >
    {
    private:
        // Define the type of the base class
        typedef Value& Ref;
        typedef boost::iterator_facade<node_iter<Value>, Value,
            boost::random_access_traversal_tag, Ref> base_type;

    public:
        // The following type definitions are common public typedefs:
        typedef node_iter<Value> this_type;
        typedef typename base_type::difference_type difference_type;
        typedef typename base_type::reference reference;
        //typedef typename Map::const_iterator iterator_type;

        //node_iter() : p_elem(0) {}
        explicit node_iter(Value* p) : p_elem(p) {}

        const Value* get() const { return p_elem; }

     private:
        friend class boost::iterator_core_access;

        reference dereference() const { return *p_elem; }

        bool equal(node_iter<Value> const& other) const {
            return this->p_elem == other.p_elem;
        }

        void increment() { ++p_elem; }

        void decrement() { --p_elem; }

        void advance(difference_type n) { p_elem += n; }

        difference_type distance_to(const this_type &rhs) const {
            return rhs.p_elem - p_elem;
        }

        Value* p_elem;
    };
    typedef node_iter<RowElement<float> > node_iterator;
    typedef node_iter<const RowElement<float> > node_const_iterator;

    node_iterator beginRow2(uint32_t rowNr) const {
        if (rowNr == 0) return node_iterator(data);
        else return node_iterator(&data[rowidx[rowNr - 1]]);
    }
    node_iterator endRow2(uint32_t rowNr) const {
        return node_iterator(&data[rowidx[rowNr]]);
    }

    node_const_iterator beginRowConst2(uint32_t rowNr) const {
        if (rowNr == 0) return node_const_iterator(data);
        else return node_const_iterator(&data[rowidx[rowNr - 1]]);
    }
    node_const_iterator endRowConst2(uint32_t rowNr) const {
        return node_const_iterator(&data[rowidx[rowNr]]);
    }

	RowIterator beginRow(uint32_t rowNr) const;
	RowIterator endRow(uint32_t rowNr) const;

private:
	static const uint32_t minHeaderLength = 28;
	static const uint32_t minScanConfigBytes = 60;
	static const uint8_t xsymmask = 1;
	static const uint8_t ysymmask = 2;
	static const uint8_t zsymmask = 4;
	int fileDescriptor;
	off_t fileSize;
	char* map;
	uint8_t flags;
	uint32_t nRows;
	uint32_t nColumns;
	uint64_t nnz;
	uint64_t* rowidx;
	RowElement<float>* data;
	SymConfig symcfg;
	ScannerConfig scancfg;
};
