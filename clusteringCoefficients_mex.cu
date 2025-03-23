
#include <mex.h>
#include "matrix.h"

#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include "CooMatrix.h"
#include "helpFunctions.h"
#include "ThrustSystem.h"
#include "TypeDefinitions.h"


struct SparseDotProductForUndirected_Functor : public thrust::unary_function<thrust::tuple<b12::NrBoxes, b12::NrBoxes,
                                                                                           b12::NrBoxes, b12::NrBoxes,
                                                                                           b12::NrBoxes, b12::NrBoxes>,
                                                                             b12::NrPoints>
{
  const b12::NrBoxes * rowIndices_begin_;
  
  __host__ __device__
  SparseDotProductForUndirected_Functor(const b12::NrBoxes * rowIndices_begin) 
      : rowIndices_begin_(rowIndices_begin) {};
  
  __host__ __device__
  b12::NrPoints operator()(thrust::tuple<b12::NrBoxes, b12::NrBoxes,
                                         b12::NrBoxes, b12::NrBoxes,
                                         b12::NrBoxes, b12::NrBoxes> t) const
  {
    // thrust::get<0>(t) ... i
    // thrust::get<1>(t) ... begin of column i
    // thrust::get<2>(t) ... end of column i
    // thrust::get<3>(t) ... j
    // thrust::get<4>(t) ... begin of column j
    // thrust::get<5>(t) ... end of column j
    
    b12::NrPoints res = 0;
    
    if (thrust::get<0>(t) != thrust::get<3>(t)) {
      b12::NrBoxes colI = *(rowIndices_begin_ + thrust::get<1>(t));
      b12::NrBoxes colJ = *(rowIndices_begin_ + thrust::get<4>(t));
      while (thrust::get<1>(t) != thrust::get<2>(t) && thrust::get<4>(t) != thrust::get<5>(t)) {
        if (colI < colJ) {
          colI = *(rowIndices_begin_ + (++thrust::get<1>(t)));
        } else if (colI > colJ) {
          colJ = *(rowIndices_begin_ + (++thrust::get<4>(t)));
        } else { // rowI == colJ
          if (colI != thrust::get<0>(t) && colJ != thrust::get<3>(t)) {
            ++res;
          }
          colI = *(rowIndices_begin_ + (++thrust::get<1>(t)));
          colJ = *(rowIndices_begin_ + (++thrust::get<4>(t)));
        }
      }
    }
    
    return res;
  }
};

struct SparseDotProductForDirected_Functor : public thrust::unary_function<thrust::tuple<b12::NrBoxes, b12::NrBoxes,
                                                                                         b12::NrBoxes, b12::NrBoxes,
                                                                                         b12::NrBoxes, b12::NrBoxes>,
                                                                           b12::NrPoints>
{
  const b12::NrBoxes * csrColumnIndices_begin_, * cscRowIndices_begin_;
  
  __host__ __device__
  SparseDotProductForDirected_Functor(const b12::NrBoxes * csrColumnIndices_begin,
                                      const b12::NrBoxes * cscRowIndices_begin) 
      : csrColumnIndices_begin_(csrColumnIndices_begin), cscRowIndices_begin_(cscRowIndices_begin) {};
  
  __host__ __device__
  b12::NrPoints operator()(thrust::tuple<b12::NrBoxes, b12::NrBoxes,
                                         b12::NrBoxes, b12::NrBoxes,
                                         b12::NrBoxes, b12::NrBoxes> t) const
  {
    // thrust::get<0>(t) ... i
    // thrust::get<1>(t) ... begin of row i (in CSR matrix)
    // thrust::get<2>(t) ... end of row i (in CSR matrix)
    // thrust::get<3>(t) ... j
    // thrust::get<4>(t) ... begin of column j (in CSC matrix)
    // thrust::get<5>(t) ... end of column j (in CSC matrix)
    
    b12::NrPoints res = 0;
    
    if (thrust::get<0>(t) != thrust::get<3>(t)) {
      b12::NrBoxes rowI = *(csrColumnIndices_begin_ + thrust::get<1>(t));
      b12::NrBoxes colJ = *(cscRowIndices_begin_ + thrust::get<4>(t));
      while (thrust::get<1>(t) != thrust::get<2>(t) && thrust::get<4>(t) != thrust::get<5>(t)) {
        if (rowI < colJ) {
          rowI = *(csrColumnIndices_begin_ + (++thrust::get<1>(t)));
        } else if (rowI > colJ) {
          colJ = *(cscRowIndices_begin_ + (++thrust::get<4>(t)));
        } else { // rowI == colJ
          if (rowI != thrust::get<0>(t) && colJ != thrust::get<3>(t)) {
            ++res;
          }
          rowI = *(csrColumnIndices_begin_ + (++thrust::get<1>(t)));
          colJ = *(cscRowIndices_begin_ + (++thrust::get<4>(t)));
        }
      }
    }
    
    return res;
  }
};

struct SetInt_RowIndexPairsPerColumn_AllIndexPairs_Undirected_Functor : public thrust::unary_function<thrust::tuple<int,
                                                                                                                    int>,
                                                                                                      b12::NrPoints>
{
  const b12::NrBoxes * rowIndices_begin_, * columnIndices_begin_;
  b12::NrPoints nnzMatrix_;
  
  __host__ __device__
  SetInt_RowIndexPairsPerColumn_AllIndexPairs_Undirected_Functor(const b12::NrBoxes * rowIndices_begin, 
                                                                 const b12::NrBoxes * columnIndices_begin, 
                                                                 b12::NrPoints nnzMatrix)
      : rowIndices_begin_(rowIndices_begin), columnIndices_begin_(columnIndices_begin), nnzMatrix_(nnzMatrix) {};
  
  __host__ __device__
  b12::NrPoints operator()(thrust::tuple<int, int> t) const
  {
    int ind = thrust::get<0>(t);
    int nnz = thrust::get<1>(t) - thrust::get<0>(t);
    
    auto zi_begin = thrust::make_zip_iterator(
      thrust::make_tuple(
        thrust::make_permutation_iterator(
          rowIndices_begin_ + ind,
          thrust::make_transform_iterator(thrust::make_counting_iterator(b12::NrPoints(0)),
                                          thrust::placeholders::_1 % nnz)),
        thrust::make_permutation_iterator(
          rowIndices_begin_ + ind,
          thrust::make_transform_iterator(thrust::make_counting_iterator(b12::NrPoints(0)),
                                          thrust::placeholders::_1 / nnz))));
    
    auto inds_begin = thrust::make_zip_iterator(
      thrust::make_tuple(
        rowIndices_begin_,
        columnIndices_begin_));
    
    auto res_begin = thrust::make_discard_iterator();
    
    auto res_end = thrust::set_intersection(thrust::seq,
                                            zi_begin, zi_begin + nnz*nnz,
                                            inds_begin, inds_begin + nnzMatrix_,
                                            res_begin,
                                            b12::ColumnMajorOrderingFunctor());
    
    return res_end - res_begin;
  }
};

struct SetInt_RowIndexPairsPerColumn_AllIndexPairs_Directed_Functor : public thrust::unary_function<thrust::tuple<int,
                                                                                                                  int>,
                                                                                                    b12::NrPoints>
{
  const b12::NrBoxes * rowIndices_begin_, * columnIndices_begin_, * rowIndices_sym_begin_;
  b12::NrPoints nnzMatrix_;
  
  __host__ __device__
  SetInt_RowIndexPairsPerColumn_AllIndexPairs_Directed_Functor(const b12::NrBoxes * rowIndices_begin, 
                                                               const b12::NrBoxes * columnIndices_begin, 
                                                               const b12::NrBoxes * rowIndices_sym_begin, 
                                                               b12::NrPoints nnzMatrix)
      : rowIndices_begin_(rowIndices_begin), columnIndices_begin_(columnIndices_begin), 
        rowIndices_sym_begin_(rowIndices_sym_begin), nnzMatrix_(nnzMatrix) {};
  
  __host__ __device__
  b12::NrPoints operator()(thrust::tuple<int, int> t) const
  {
    int ind = thrust::get<0>(t);
    int nnz = thrust::get<1>(t) - thrust::get<0>(t);
    
    auto zi_begin = thrust::make_zip_iterator(
      thrust::make_tuple(
        thrust::make_permutation_iterator(
          rowIndices_sym_begin_ + ind,
          thrust::make_transform_iterator(thrust::make_counting_iterator(b12::NrPoints(0)),
                                          thrust::placeholders::_1 % nnz)),
        thrust::make_permutation_iterator(
          rowIndices_sym_begin_ + ind,
          thrust::make_transform_iterator(thrust::make_counting_iterator(b12::NrPoints(0)),
                                          thrust::placeholders::_1 / nnz))));
    
    auto inds_begin = thrust::make_zip_iterator(
      thrust::make_tuple(
        rowIndices_begin_,
        columnIndices_begin_));
    
    auto res_begin = thrust::make_discard_iterator();
    
    auto res_end = thrust::set_intersection(thrust::seq,
                                            zi_begin, zi_begin + nnz*nnz,
                                            inds_begin, inds_begin + nnzMatrix_,
                                            res_begin,
                                            b12::ColumnMajorOrderingFunctor());
    
    return res_end - res_begin;
  }
};

struct AreThereIndexPairs_02_and_01_Functor : public thrust::unary_function<thrust::tuple<b12::NrBoxes,
                                                                                          b12::NrBoxes,
                                                                                          b12::NrBoxes>,
                                                                            b12::NrPoints>
{
  const b12::NrBoxes * rowIndices_begin_, * columnIndices_begin_;
  b12::NrPoints nnzMatrix_;
  bool isUndirected_;
  
  __host__ __device__
  AreThereIndexPairs_02_and_01_Functor(const b12::NrBoxes * rowIndices_begin, const b12::NrBoxes * columnIndices_begin,
                                       b12::NrPoints nnzMatrix, bool isUndirected)
      : rowIndices_begin_(rowIndices_begin), columnIndices_begin_(columnIndices_begin),
        nnzMatrix_(nnzMatrix), isUndirected_(isUndirected) {};
  
  __host__ __device__
  b12::NrPoints operator()(thrust::tuple<b12::NrBoxes, b12::NrBoxes, b12::NrBoxes> t) const
  {
    // thrust::get<0>(t) ... zu ueberpreufen
    // thrust::get<1>(t) ... vorhandene Zeile / getroffen von thrust::get<2>(t)
    // thrust::get<2>(t) ... vorhandene Spalte
    
    if (thrust::get<0>(t) == thrust::get<1>(t) || 
        thrust::get<0>(t) == thrust::get<2>(t) || 
        thrust::get<1>(t) == thrust::get<2>(t)) {
      
      return 0;
      
    } else {
      
      auto inds_begin = thrust::make_zip_iterator(
        thrust::make_tuple(
          rowIndices_begin_,
          columnIndices_begin_));
      
      // gibt es Weg von thrust::get<2>(t) zu thrust::get<0>(t)
      bool is0NeighbourOf2 = thrust::binary_search(thrust::seq,
                                                   inds_begin, inds_begin + nnzMatrix_,
                                                   thrust::make_tuple(thrust::get<0>(t),
                                                                      thrust::get<2>(t)),
                                                   b12::ColumnMajorOrderingFunctor());
      if (! isUndirected_) {
        is0NeighbourOf2 = is0NeighbourOf2 || thrust::binary_search(thrust::seq,
                                                                   inds_begin, inds_begin + nnzMatrix_,
                                                                   thrust::make_tuple(thrust::get<2>(t),
                                                                                      thrust::get<0>(t)),
                                                                   b12::ColumnMajorOrderingFunctor());
      }
      
      if (is0NeighbourOf2) {
        // gibt es Weg von thrust::get<1>(t) zu thrust::get<0>(t)
        bool way1to0 = thrust::binary_search(thrust::seq,
                                             inds_begin, inds_begin + nnzMatrix_,
                                             thrust::make_tuple(thrust::get<0>(t),
                                                                thrust::get<1>(t)),
                                             b12::ColumnMajorOrderingFunctor());
        return way1to0 ? 1 : 0;
      } else {
        return 0;
      }
    }
  }
};

struct IsThereIndexPair_12_Functor : public thrust::unary_function<thrust::tuple<bool,
                                                                                 b12::NrBoxes,
                                                                                 b12::NrBoxes,
                                                                                 b12::NrBoxes>,
                                                                   b12::NrPoints>
{
  const b12::NrBoxes * rowIndices_begin_, * columnIndices_begin_;
  b12::NrPoints nnzMatrix_;
  bool isUndirected_;
  
  __host__ __device__
  IsThereIndexPair_12_Functor(const b12::NrBoxes * rowIndices_begin,
                              const b12::NrBoxes * columnIndices_begin,
                              b12::NrPoints nnzMatrix, bool isUndirected)
      : rowIndices_begin_(rowIndices_begin), columnIndices_begin_(columnIndices_begin),
        nnzMatrix_(nnzMatrix), isUndirected_(isUndirected) {};
  
  __host__ __device__
  b12::NrPoints operator()(thrust::tuple<bool, b12::NrBoxes, b12::NrBoxes, b12::NrBoxes> t) const
  {
    // thrust::get<0>(t) ... gibt an, ob Zeilenindex nicht zu weit ist
    // thrust::get<1>(t) ... zu ueberpreufen
    // thrust::get<2>(t) ... vorhandene Zeile / Nachbar von thrust::get<3>(t)
    // thrust::get<3>(t) ... vorhandene Spalte
    
    bool temp = isUndirected_ ? thrust::get<1>(t) >= thrust::get<2>(t) : thrust::get<1>(t) == thrust::get<2>(t);
    
    if (! thrust::get<0>(t) || temp || thrust::get<1>(t) == thrust::get<3>(t) || thrust::get<2>(t) == thrust::get<3>(t)) {
      
      return 0;
      
    } else {
      
      auto inds_begin = thrust::make_zip_iterator(
        thrust::make_tuple(
          rowIndices_begin_,
          columnIndices_begin_));
        
      // gibt es Weg von thrust::get<2>(t) zu thrust::get<1>(t)
      bool res =  thrust::binary_search(thrust::seq,
                                        inds_begin, inds_begin + nnzMatrix_,
                                        thrust::make_tuple(thrust::get<1>(t), thrust::get<2>(t)),
                                        b12::ColumnMajorOrderingFunctor());
      
      return res ? 1 : 0;
    }
  }
};

struct Is_1_RowIndex_Functor : public thrust::unary_function<thrust::tuple<bool,
                                                                           b12::NrBoxes, b12::NrBoxes, b12::NrBoxes,
                                                                           b12::NrBoxes, b12::NrBoxes>,
                                                             b12::NrPoints>
{
  const b12::NrBoxes * rowIndices_begin_;
  bool isUndirected_;
  
  __host__ __device__
  Is_1_RowIndex_Functor(const b12::NrBoxes * rowIndices_begin, bool isUndirected)
      : rowIndices_begin_(rowIndices_begin), isUndirected_(isUndirected) {};
  
  __host__ __device__
  b12::NrPoints operator()(thrust::tuple<bool, 
                                         b12::NrBoxes, b12::NrBoxes, b12::NrBoxes, 
                                         b12::NrBoxes, b12::NrBoxes> t) const
  {
    // thrust::get<0>(t) ... gibt an, ob Zeilenindex nicht zu weit ist
    // thrust::get<1>(t) ... zu ueberpreufen
    // thrust::get<2>(t) ... vorhandene Zeile / getroffen von thrust::get<3>(t)
    // thrust::get<3>(t) ... vorhandene Spalte
    // [rowIndices_begin_ + thrust::get<4>(t), rowIndices_begin_ + thrust::get<5>(t)) ... Spaltenbereich, in dem thrust::get<2>(t) stehen
    
    return thrust::get<0>(t) && 
           thrust::get<2>(t) != thrust::get<3>(t) && 
           thrust::get<1>(t) != thrust::get<3>(t) && 
           (isUndirected_ ? thrust::get<1>(t) < thrust::get<2>(t) : thrust::get<1>(t) != thrust::get<2>(t)) &&
           thrust::binary_search(thrust::seq,
                                 rowIndices_begin_ + thrust::get<4>(t), rowIndices_begin_ + thrust::get<5>(t), // range to search in
                                 thrust::get<1>(t)); // value to search
  }
};

struct AreValuesDifferent_Functor : public thrust::unary_function<thrust::tuple<b12::NrBoxes,
                                                                                b12::NrBoxes,
                                                                                b12::NrBoxes>,
                                                                  bool>
{
  bool isUndirected_;
  
  __host__ __device__
  AreValuesDifferent_Functor(bool isUndirected) : isUndirected_(isUndirected) {};
  
  __host__ __device__
  bool operator()(thrust::tuple<b12::NrBoxes, b12::NrBoxes, b12::NrBoxes> t) const
  {
    return (isUndirected_ ? thrust::get<0>(t) < thrust::get<1>(t) : thrust::get<0>(t) != thrust::get<1>(t)) && 
           thrust::get<0>(t) != thrust::get<2>(t) &&
           thrust::get<1>(t) != thrust::get<2>(t);
  }
};

struct AreAllTrue_Functor : public thrust::unary_function<thrust::tuple<bool, bool, bool>, b12::NrPoints>
{
  __host__ __device__
  b12::NrPoints operator()(thrust::tuple<bool, bool, bool> t) const
  {
    return (thrust::get<0>(t) && thrust::get<1>(t) && thrust::get<2>(t)) ? 1 : 0;
  }
};


template<b12::Architecture A>
void makeSymmetric(typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& rowIndices,
                   typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& columnIndices)
{
  uint64_t nnz = rowIndices.size();
  
  rowIndices.resize(nnz * 2);
  columnIndices.resize(nnz * 2);
  
  thrust::copy(rowIndices.begin(), rowIndices.begin() + nnz, columnIndices.begin() + nnz);
  thrust::copy(columnIndices.begin(), columnIndices.begin() + nnz, rowIndices.begin() + nnz);
  
  auto inds_begin = thrust::make_zip_iterator(thrust::make_tuple(rowIndices.begin(), columnIndices.begin()));
  
  thrust::sort(typename b12::ThrustSystem<A>::execution_policy(),
               inds_begin, inds_begin + nnz * 2,
               b12::ColumnMajorOrderingFunctor());
  
  auto it = thrust::unique(typename b12::ThrustSystem<A>::execution_policy(),
                           inds_begin, inds_begin + nnz * 2);
  
  nnz = it - inds_begin;
  
  rowIndices.resize(nnz);
  columnIndices.resize(nnz);
  
  cudaThreadSynchronize(); // block until kernel is finished
}


template<b12::Architecture A>
void removeDiagonal(typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& rowIndices,
                    typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& columnIndices)
{
  auto ind_begin = thrust::make_zip_iterator(thrust::make_tuple(rowIndices.begin(),
                                                                columnIndices.begin()));
  auto ind_end = thrust::remove_if(typename b12::ThrustSystem<A>::execution_policy(),
                                   ind_begin, ind_begin + rowIndices.size(),
                                   b12::makeBinaryTransformIterator(rowIndices.begin(),
                                                                    columnIndices.begin(),
                                                                    thrust::equal_to<b12::NrBoxes>()),
                                   thrust::identity<bool>());
  rowIndices.resize(ind_end - ind_begin);
  columnIndices.resize(ind_end - ind_begin);
  
  cudaThreadSynchronize(); // block until kernel is finished
}


template<b12::Architecture A>
void numberOfNeighbours(uint64_t nnz,
                        const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& rowIndices,
                        const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& columnIndices,
                        typename b12::ThrustSystem<A>::Vector<double>& k)
{
  k.resize(nnz);
  
  // only add 1 if elements are not on the diagonal of the matrix
  auto itPair = thrust::reduce_by_key(typename b12::ThrustSystem<A>::execution_policy(),
                                      columnIndices.begin(), columnIndices.end(), // keys
                                      thrust::make_transform_iterator(
                                        b12::makeBinaryTransformIterator(rowIndices.begin(),
                                                                         columnIndices.begin(),
                                                                         thrust::not_equal_to<b12::NrBoxes>()),
                                        thrust::identity<double>()), // 0 if i == j, else 1
                                      thrust::make_discard_iterator(),
                                      k.begin());
  
  k.resize(itPair.second - k.begin());
  k.shrink_to_fit();
  
  cudaThreadSynchronize(); // block until kernel is finished
}


template<b12::Architecture A>
void clusteringCoefficients(bool isUndirected,
                            const typename b12::ThrustSystem<A>::Vector<double>& e,
                            const typename b12::ThrustSystem<A>::Vector<double>& k,
                            typename b12::ThrustSystem<A>::Vector<double>& C)
{
  C.resize(e.size());
  
  if (isUndirected) {
    thrust::transform(typename b12::ThrustSystem<A>::execution_policy(),
                      e.begin(), e.end(), // first arguments
                      k.begin(), // second arguments
                      C.begin(), // result
                      2.0 * thrust::placeholders::_1 / (thrust::placeholders::_2 * (thrust::placeholders::_2 - 1.0)));
  } else {
    thrust::transform(typename b12::ThrustSystem<A>::execution_policy(),
                      e.begin(), e.end(), // first arguments
                      k.begin(), // second arguments
                      C.begin(), // result
                      thrust::placeholders::_1 / (thrust::placeholders::_2 * (thrust::placeholders::_2 - 1.0)));
  }
  
  cudaThreadSynchronize(); // block until kernel is finished
}


template<b12::Architecture A>
void dot_product(const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& rowIndices,
                 const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& columnIndices,
                 const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& csrVector,
                 const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& cscVector,
                 const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& rowIndices_sym,
                 const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& columnIndices_sym,
                 const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& cscVector_sym,
                 mwIndex nRows, mwIndex nColumns, uint64_t nnz, bool isUndirected, bool isSymmetric,
                 typename b12::ThrustSystem<A>::Vector<double>& e)
{
  typename b12::ThrustSystem<A>::Vector<double> eTemp(nnz);
  // existing columns, i.e. without empty columns
  typename b12::ThrustSystem<A>::Vector<b12::NrBoxes> col(nnz);
  
  if (isUndirected || isSymmetric) {
    // ti_begin[.] = SparseDotProductForUndirected_Functor(rowIndices_sym[.],
    //                                                     cscVector_sym[rowIndices_sym[.]],
    //                                                     cscVector_sym[rowIndices_sym[.] + 1],
    //                                                     columnIndices_sym[.],
    //                                                     cscVector_sym[columnIndices_sym[.]],
    //                                                     cscVector_sym[columnIndices_sym[.] + 1])
    auto ti_begin = thrust::make_transform_iterator(
      thrust::make_zip_iterator(
        thrust::make_tuple(
          rowIndices_sym.begin(),
          thrust::make_permutation_iterator(cscVector_sym.begin(), rowIndices_sym.begin()),
          thrust::make_permutation_iterator(cscVector_sym.begin() + 1, rowIndices_sym.begin()),
          columnIndices_sym.begin(),
          thrust::make_permutation_iterator(cscVector_sym.begin(), columnIndices_sym.begin()),
          thrust::make_permutation_iterator(cscVector_sym.begin() + 1, columnIndices_sym.begin()))),
      SparseDotProductForUndirected_Functor(thrust::raw_pointer_cast(rowIndices_sym.data())));
    
    auto it_pair = thrust::reduce_by_key(typename b12::ThrustSystem<A>::execution_policy(),
                                         columnIndices_sym.begin(), columnIndices_sym.end(),
                                         ti_begin,
                                         col.begin(),
                                         eTemp.begin());
    
    col.resize(it_pair.first - col.begin());
    eTemp.resize(it_pair.second - eTemp.begin());
    
    if (isUndirected) {
      thrust::transform(typename b12::ThrustSystem<A>::execution_policy(),
                        eTemp.begin(), eTemp.end(),
                        eTemp.begin(),
                        thrust::placeholders::_1 / 2.0);
    }
  } else {
    // ti_begin[.] = SparseDotProductForDirected_Functor(rowIndices_sym[.],
    //                                                   csrVector[rowIndices_sym[.]],
    //                                                   csrVector[rowIndices_sym[.] + 1],
    //                                                   columnIndices_sym[.],
    //                                                   cscVector_sym[columnIndices_sym[.]],
    //                                                   cscVector_sym[columnIndices_sym[.] + 1])
    auto ti_begin = thrust::make_transform_iterator(
      thrust::make_zip_iterator(
        thrust::make_tuple(
          rowIndices_sym.begin(),
          thrust::make_permutation_iterator(csrVector.begin(), rowIndices_sym.begin()),
          thrust::make_permutation_iterator(csrVector.begin() + 1, rowIndices_sym.begin()),
          columnIndices_sym.begin(),
          thrust::make_permutation_iterator(cscVector_sym.begin(), columnIndices_sym.begin()),
          thrust::make_permutation_iterator(cscVector_sym.begin() + 1, columnIndices_sym.begin()))),
      SparseDotProductForDirected_Functor(thrust::raw_pointer_cast(columnIndices.data()),
                                          thrust::raw_pointer_cast(rowIndices_sym.data())));
    
    auto it_pair = thrust::reduce_by_key(typename b12::ThrustSystem<A>::execution_policy(),
                                         columnIndices_sym.begin(), columnIndices_sym.end(),
                                         ti_begin,
                                         col.begin(),
                                         eTemp.begin());
    
    col.resize(it_pair.first - col.begin());
    eTemp.resize(it_pair.second - eTemp.begin());
  }
  
  // scatter into e
  e.assign(nColumns, 0.0);
  thrust::scatter(typename b12::ThrustSystem<A>::execution_policy(),
                  eTemp.begin(), eTemp.end(),
                  col.begin(),
                  e.begin());
  
  cudaThreadSynchronize(); // block until kernel is finished
}


template<b12::Architecture A>
void set_intersection_serialOverColumns(const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& rowIndices,
                                        const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& columnIndices,
                                        const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& csrVector,
                                        const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& cscVector,
                                        const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& rowIndices_sym,
                                        const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& columnIndices_sym,
                                        const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& cscVector_sym,
                                        mwIndex nRows, mwIndex nColumns, uint64_t nnz, bool isUndirected, bool isSymmetric,
                                        typename b12::ThrustSystem<A>::Vector<double>& e)
{
  // set intersection of two sets of index pairs
  // 1) all actual index pairs in the matrix (without diagonal) -> [inds_begin, inds_end)
  // 2) all nnz row indices of one column in variation with themselves, i.e. if i_1, ..., i_nnz are the row indices of one column, the set is
  //    (i_1, i_1), ..., (i_nnz, i_1), (i_1, i_2), ..., (i_nnz, i_2), ..., (i_nnz, i_nnz) -> [zi_begin, zi_begin + nnz^2)
  // for directed graphs: take 1) as before, but 2) from symmetric part of the matrix
  
  e.assign(nColumns, 0);
  
  auto inds_begin = thrust::make_zip_iterator(
    thrust::make_tuple(
      rowIndices.begin(),
      columnIndices.begin()));
  
  auto inds_end = thrust::make_zip_iterator(
    thrust::make_tuple(
      rowIndices.end(),
      columnIndices.end()));
  
  if (isUndirected || isSymmetric) {
    inds_begin = thrust::make_zip_iterator(
      thrust::make_tuple(
        rowIndices_sym.begin(),
        columnIndices_sym.begin()));
    
    inds_end = thrust::make_zip_iterator(
      thrust::make_tuple(
        rowIndices_sym.end(),
        columnIndices_sym.end()));
  }
    
  for (int i = 0; i < cscVector_sym.size()-1; ++i) {
    
    b12::NrPoints nnz = cscVector_sym[i + 1] - cscVector_sym[i];
    
    // zi_begin[.] = (rowIndices_sym[(. mod nnz) + cscVector_sym[i]],
    //                rowIndices_sym[(.  /  nnz) + cscVector_sym[i]])
    auto zi_begin = thrust::make_zip_iterator(
      thrust::make_tuple(
        thrust::make_permutation_iterator(
          rowIndices_sym.begin() + cscVector_sym[i],
          thrust::make_transform_iterator(thrust::make_counting_iterator(b12::NrPoints(0)), thrust::placeholders::_1 % nnz)),
        thrust::make_permutation_iterator(
          rowIndices_sym.begin() + cscVector_sym[i],
          thrust::make_transform_iterator(thrust::make_counting_iterator(b12::NrPoints(0)), thrust::placeholders::_1 / nnz))));
    
    auto res_begin = thrust::make_discard_iterator();
    
    auto res_end = thrust::set_intersection(typename b12::ThrustSystem<A>::execution_policy(),
                                            zi_begin, zi_begin + nnz*nnz,
                                            inds_begin, inds_end,
                                            res_begin,
                                            b12::ColumnMajorOrderingFunctor());
    
    e[i] = res_end - res_begin;
  }
    
  if (isUndirected) {
    thrust::transform(typename b12::ThrustSystem<A>::execution_policy(),
                      e.begin(), e.end(),
                      e.begin(),
                      thrust::placeholders::_1 / 2);
  }
  
  cudaThreadSynchronize(); // block until kernel is finished
}


template<b12::Architecture A>
void set_intersection_parallelOverColumns(const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& rowIndices,
                                          const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& columnIndices,
                                          const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& csrVector,
                                          const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& cscVector,
                                          const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& rowIndices_sym,
                                          const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& columnIndices_sym,
                                          const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& cscVector_sym,
                                          mwIndex nRows, mwIndex nColumns, uint64_t nnz, bool isUndirected, bool isSymmetric,
                                          typename b12::ThrustSystem<A>::Vector<double>& e)
{
  // set intersection of two sets of index pairs
  // 1) all actual index pairs in the matrix (without diagonal) -> [inds_begin, inds_end)
  // 2) all nnz row indices of one column in variation with themselves, i.e. if i_1, ..., i_nnz are the row indices of one column, the set is
  //    (i_1, i_1), ..., (i_nnz, i_1), (i_1, i_2), ..., (i_nnz, i_2), ..., (i_nnz, i_nnz) -> [zi_begin, zi_begin + nnz^2)
  // for directed graphs: take 1) as before, but 2) from symmetric part of the matrix
  
  e.resize(nColumns);
  
  // zi_begin[.] = (cscVector_sym[.],
  //                cscVector_sym[. + 1])
  auto zi_begin = thrust::make_zip_iterator(thrust::make_tuple(cscVector_sym.begin(), cscVector_sym.begin() + 1));
  
  if (isUndirected || isSymmetric) {
    thrust::transform(typename b12::ThrustSystem<A>::execution_policy(),
                      zi_begin, zi_begin + nColumns,
                      e.begin(),
                      SetInt_RowIndexPairsPerColumn_AllIndexPairs_Undirected_Functor(thrust::raw_pointer_cast(rowIndices_sym.data()),
                                                                                     thrust::raw_pointer_cast(columnIndices_sym.data()),
                                                                                     columnIndices_sym.size()));
    
    if (isUndirected) {
      thrust::transform(typename b12::ThrustSystem<A>::execution_policy(),
                        e.begin(), e.end(),
                        e.begin(),
                        thrust::placeholders::_1 / 2);
    }
  } else {
    thrust::transform(typename b12::ThrustSystem<A>::execution_policy(),
                      zi_begin, zi_begin + nColumns,
                      e.begin(),
                      SetInt_RowIndexPairsPerColumn_AllIndexPairs_Directed_Functor(thrust::raw_pointer_cast(rowIndices.data()),
                                                                                   thrust::raw_pointer_cast(columnIndices.data()),
                                                                                   thrust::raw_pointer_cast(rowIndices_sym.data()),
                                                                                   columnIndices.size()));
  }
  
  cudaThreadSynchronize(); // block until kernel is finished
}


template<b12::Architecture A>
void all_triangles_reduce_by_key(const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& rowIndices,
                                 const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& columnIndices,
                                 const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& csrVector,
                                 const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& cscVector,
                                 const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& rowIndices_sym,
                                 const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& columnIndices_sym,
                                 const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& cscVector_sym,
                                 mwIndex nRows, mwIndex nColumns, uint64_t nnz, bool isUndirected, bool isSymmetric,
                                 typename b12::ThrustSystem<A>::Vector<double>& e)
{
  // for each existing index pair (i,j) of the matrix's symmetric part, the triples (0,i,j), ..., (n-1,i,j) are progressed
  // for each triple (k,i,j), 1 is added if k != i != j != k and (k,j) and (k,i) are index pairs of the matrix
  // (found out by binary a search), 0 otherwise
  
  e.resize(b12::ThrustSystem<A>::Memory::getFreeBytes(uint64_t(8) << 30) / sizeof(b12::NrPoints) / 6);
  typename b12::ThrustSystem<A>::Vector<b12::NrBoxes> c(e.size());
  
  auto functor = AreThereIndexPairs_02_and_01_Functor(thrust::raw_pointer_cast(rowIndices.data()),
                                                      thrust::raw_pointer_cast(columnIndices.data()),
                                                      columnIndices.size(),
                                                      isUndirected || isSymmetric);
  if (isUndirected || isSymmetric) {
    functor = AreThereIndexPairs_02_and_01_Functor(thrust::raw_pointer_cast(rowIndices_sym.data()),
                                                   thrust::raw_pointer_cast(columnIndices_sym.data()),
                                                   columnIndices_sym.size(),
                                                   isUndirected || isSymmetric);
  }
  
  // ti_begin[.] = AreThereIndexPairs_02_and_01_Functor(. mod nColumns,
  //                                                    rowIndices_sym[. / nColumns],
  //                                                    columnIndices_sym[. / nColumns],
  //                                                    isUndirected || isSymmetric)
  auto ti_begin = thrust::make_transform_iterator(
    thrust::make_zip_iterator(
      thrust::make_tuple(
        thrust::make_transform_iterator(thrust::make_counting_iterator(b12::NrPoints(0)),
                                        thrust::placeholders::_1 % nColumns),
        thrust::make_permutation_iterator(rowIndices_sym.begin(),
                                          thrust::make_transform_iterator(thrust::make_counting_iterator(b12::NrPoints(0)),
                                                                          thrust::placeholders::_1 / nColumns)),
        thrust::make_permutation_iterator(columnIndices_sym.begin(),
                                          thrust::make_transform_iterator(thrust::make_counting_iterator(b12::NrPoints(0)),
                                                                          thrust::placeholders::_1 / nColumns)))),
    functor);
  
  // key_begin[.] = columnIndices_sym[. / nColumns]
  auto key_begin = thrust::make_permutation_iterator(columnIndices_sym.begin(),
                                                     thrust::make_transform_iterator(
                                                       thrust::make_counting_iterator(b12::NrPoints(0)),
                                                       thrust::placeholders::_1 / nColumns));
  
  auto it_pair = thrust::make_pair(c.begin(), e.begin());
  
  for (int64_t i = 0, length = 1;
       it_pair.second != e.end() && i < nColumns * columnIndices_sym.size() && length != 0;
       i += length, ti_begin += length, key_begin += length) {
    
    length = thrust::minimum<int64_t>()(nColumns * columnIndices_sym.size() - i, e.end() - it_pair.second);
    
    it_pair = thrust::reduce_by_key(typename b12::ThrustSystem<A>::execution_policy(),
                                    key_begin, key_begin + length,
                                    ti_begin,
                                    it_pair.first,
                                    it_pair.second);
  }
  
  thrust::inclusive_scan_by_key(typename b12::ThrustSystem<A>::execution_policy(),
                                thrust::make_reverse_iterator(it_pair.first), c.rend(),
                                thrust::make_reverse_iterator(it_pair.second),
                                thrust::make_reverse_iterator(it_pair.second));
  
  auto new_end = thrust::unique_by_key(typename b12::ThrustSystem<A>::execution_policy(),
                                       c.begin(), it_pair.first,
                                       e.begin());
  
  c.resize(new_end.first - c.begin());
  e.resize(new_end.second - e.begin());
  
  if (isUndirected) {
    thrust::transform(typename b12::ThrustSystem<A>::execution_policy(),
                      e.begin(), e.end(),
                      e.begin(),
                      thrust::placeholders::_1 / 2);
  }
  
  cudaThreadSynchronize(); // block until kernel is finished
}


template<b12::Architecture A>
void all_triangles_count_if(const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& rowIndices,
                            const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& columnIndices,
                            const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& csrVector,
                            const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& cscVector,
                            const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& rowIndices_sym,
                            const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& columnIndices_sym,
                            const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& cscVector_sym,
                            mwIndex nRows, mwIndex nColumns, uint64_t nnz, bool isUndirected, bool isSymmetric,
                            typename b12::ThrustSystem<A>::Vector<double>& e)
{
  // for each existing index pair (i,j) of the matrix's symmetric part, the triples (0,i,j), ..., (n-1,i,j) are progressed
  // a triple (k,i,j) is counted, if k != i != j != k and (k,j) and (k,i) are index pairs of the matrix (found by binary search)
  
  e.resize(nColumns);
  
  auto functor = AreThereIndexPairs_02_and_01_Functor(thrust::raw_pointer_cast(rowIndices.data()),
                                                      thrust::raw_pointer_cast(columnIndices.data()),
                                                      columnIndices.size(),
                                                      isUndirected);
  if (isUndirected || isSymmetric) {
    functor = AreThereIndexPairs_02_and_01_Functor(thrust::raw_pointer_cast(rowIndices_sym.data()),
                                                   thrust::raw_pointer_cast(columnIndices_sym.data()),
                                                   columnIndices_sym.size(),
                                                   isUndirected);
  }
  
  // ti_begin[.] = AreThereIndexPairs_02_and_01_Functor(. mod nColumns,
  //                                                    rowIndices_sym[. / nColumns],
  //                                                    columnIndices_sym[. / nColumns],
  //                                                    isUndirected)
  auto ti_begin = thrust::make_transform_iterator(
    thrust::make_zip_iterator(
      thrust::make_tuple(
        thrust::make_transform_iterator(thrust::make_counting_iterator(b12::NrPoints(0)),
                                        thrust::placeholders::_1 % nColumns),
        thrust::make_permutation_iterator(rowIndices_sym.begin(),
                                          thrust::make_transform_iterator(thrust::make_counting_iterator(b12::NrPoints(0)),
                                                                          thrust::placeholders::_1 / nColumns)),
        thrust::make_permutation_iterator(columnIndices_sym.begin(),
                                          thrust::make_transform_iterator(thrust::make_counting_iterator(b12::NrPoints(0)),
                                                                          thrust::placeholders::_1 / nColumns)))),
    functor);
  
  for (int i = 0; i < nColumns; ti_begin += (cscVector_sym[i+1] - cscVector_sym[i]) * nColumns, ++i) {
    e[i] = thrust::count_if(typename b12::ThrustSystem<A>::execution_policy(),
                            ti_begin, ti_begin + (cscVector_sym[i+1] - cscVector_sym[i]) * nColumns,
                            thrust::identity<bool>());
  }
  
  if (isUndirected) {
    thrust::transform(typename b12::ThrustSystem<A>::execution_policy(),
                      e.begin(), e.end(),
                      e.begin(),
                      thrust::placeholders::_1 / 2);
  }
  
  cudaThreadSynchronize(); // block until kernel is finished
}


template<b12::Architecture A>
void triangles_binary_search_full_field(const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& rowIndices,
                                        const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& columnIndices,
                                        const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& csrVector,
                                        const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& cscVector,
                                        const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& rowIndices_sym,
                                        const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& columnIndices_sym,
                                        const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& cscVector_sym,
                                        mwIndex nRows, mwIndex nColumns, uint64_t nnz, bool isUndirected, bool isSymmetric,
                                        typename b12::ThrustSystem<A>::Vector<double>& e)
{
  // if, for each column j, i_1, ..., i_n are the row indices, then the quadruplets (1<=n, i_1, i_1, j), (2<=n, i_2, i_1, j), ..., (maxOutDegree<=n, i_maxOutDegree, i_1, j), 
  // (1<=n, i_1, i_2, j), (2<=n, i_2, i_2, j), ..., (maxOutDegree<=n, i_maxOutDegree, i_2, j), ..., (maxOutDegree<=n, i_maxOutDegree, i_n, j) are progressed
  // for each quadruplet (b,k,i,j), 1 is added if b is true and k != i != j != k and (k,i) are index pairs of the matrix (found out by binary a search), 0 otherwise;
  // if isUndirected is true, the additional condition k<i must be true for a 1.
  
  e.resize(b12::ThrustSystem<A>::Memory::getFreeBytes(uint64_t(8) << 30) / sizeof(b12::NrPoints) / 5);
  typename b12::ThrustSystem<A>::Vector<b12::NrBoxes> c(e.size());
  
  // nnz_begin[.] = cscVector_sym[. + 1] - cscVector_sym[.]
  auto nnz_begin = b12::makeBinaryTransformIterator(cscVector_sym.begin() + 1, cscVector_sym.begin(),
                                                    thrust::minus<b12::NrBoxes>());
  
  b12::NrBoxes maxOutDegree = * thrust::max_element(typename b12::ThrustSystem<A>::execution_policy(),
                                                    nnz_begin, nnz_begin + nColumns);
  
  // delay_row_begin[.] = rowIndices_sym[. / maxOutDegree]
  auto delay_row_begin = thrust::make_permutation_iterator(
    rowIndices_sym.begin(),
    thrust::make_transform_iterator(thrust::make_counting_iterator(b12::NrPoints(0)),
                                    thrust::placeholders::_1 / b12::NrPoints(maxOutDegree)));
  
  // delay_column_begin[.] = columnIndices_sym[. / maxOutDegree]
  auto delay_column_begin = thrust::make_permutation_iterator(
    columnIndices_sym.begin(),
    thrust::make_transform_iterator(thrust::make_counting_iterator(b12::NrPoints(0)),
                                    thrust::placeholders::_1 / b12::NrPoints(maxOutDegree)));
  
  // specific_row_begin[.] = rowIndices_sym[(cscVector_sym[delay_column_begin[.]] + (. mod maxOutDegree)) mod nnzMatrix]
  auto specific_row_begin = thrust::make_permutation_iterator(
    rowIndices_sym.begin(),
    thrust::make_transform_iterator(
      b12::makeBinaryTransformIterator(
        thrust::make_permutation_iterator(cscVector_sym.begin(), delay_column_begin),
        thrust::make_transform_iterator(thrust::make_counting_iterator(b12::NrPoints(0)), thrust::placeholders::_1 % b12::NrPoints(maxOutDegree)),
        thrust::plus<b12::NrPoints>()),
      thrust::placeholders::_1 % b12::NrPoints(rowIndices_sym.size())));
  
  // check_overhang_begin[.] = (. mod maxOutDegree) < nnz_begin[delay_column_begin[.]]
  auto check_overhang_begin = makeBinaryTransformIterator(
    thrust::make_transform_iterator(thrust::make_counting_iterator(b12::NrPoints(0)), thrust::placeholders::_1 % b12::NrPoints(maxOutDegree)),
    thrust::make_permutation_iterator(nnz_begin, delay_column_begin),
    thrust::less<b12::NrPoints>());
  
  auto functor = IsThereIndexPair_12_Functor(thrust::raw_pointer_cast(rowIndices.data()),
                                             thrust::raw_pointer_cast(columnIndices.data()),
                                             columnIndices.size(),
                                             isUndirected);
  if (isUndirected || isSymmetric) {
    functor = IsThereIndexPair_12_Functor(thrust::raw_pointer_cast(rowIndices_sym.data()),
                                          thrust::raw_pointer_cast(columnIndices_sym.data()),
                                          columnIndices_sym.size(),
                                          isUndirected);
  }
  
  // ti_begin[.] = IsThereIndexPair_12_Functor(check_overhang_begin[.],
  //                                           specific_row_begin[.],
  //                                           delay_row_begin[.],
  //                                           delay_column_begin[.])
  auto ti_begin = thrust::make_transform_iterator(
    thrust::make_zip_iterator(
      thrust::make_tuple(
        check_overhang_begin,
        specific_row_begin,
        delay_row_begin,
        delay_column_begin)),
    functor);
  
  auto it_pair = thrust::make_pair(c.begin(), e.begin());
  
  for (int64_t i = 0, length = 1;
       it_pair.second != e.end() && i < maxOutDegree * columnIndices_sym.size() && length != 0;
       i += length, ti_begin += length, delay_column_begin += length) {
    
    length = thrust::minimum<int64_t>()(maxOutDegree * columnIndices_sym.size() - i, e.end() - it_pair.second);
    
    it_pair = thrust::reduce_by_key(typename b12::ThrustSystem<A>::execution_policy(),
                                    delay_column_begin, delay_column_begin + length,
                                    ti_begin,
                                    it_pair.first,
                                    it_pair.second);
  }
  
  thrust::inclusive_scan_by_key(typename b12::ThrustSystem<A>::execution_policy(),
                                thrust::make_reverse_iterator(it_pair.first), c.rend(),
                                thrust::make_reverse_iterator(it_pair.second),
                                thrust::make_reverse_iterator(it_pair.second));
  
  auto new_end = thrust::unique_by_key(typename b12::ThrustSystem<A>::execution_policy(),
                                       c.begin(), it_pair.first,
                                       e.begin());
  
  c.resize(new_end.first - c.begin());
  e.resize(new_end.second - e.begin());
  
  cudaThreadSynchronize(); // block until kernel is finished
}


template<b12::Architecture A>
void triangles_binary_search_segmented_field(const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& rowIndices,
                                             const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& columnIndices,
                                             const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& csrVector,
                                             const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& cscVector,
                                             const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& rowIndices_sym,
                                             const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& columnIndices_sym,
                                             const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& cscVector_sym,
                                             mwIndex nRows, mwIndex nColumns, uint64_t nnz, bool isUndirected, bool isSymmetric,
                                             typename b12::ThrustSystem<A>::Vector<double>& e)
{
  // if, for each column j, i_1, ..., i_n are the row indices, then the sextuples (1<=n, i_1, i_1, j, ., .), (2<=n, i_2, i_1, j, ., .), ..., (maxOutDegree<=n, i_maxOutDegree, i_1, j, ., .), 
    // (1<=n, i_1, i_2, j, ., .), (2<=n, i_2, i_2, j, ., .), ..., (maxOutDegree<=n, i_maxOutDegree, i_2, j, ., .), ..., (maxOutDegree<=n, i_maxOutDegree, i_n, j, ., .) are progressed
    // for each sextuplet (b,k,i,j,r1,r2), 1 is added if b is true and k != i != j != k and k is a row index (found out by binary a search) in rowIndices_begin_+[r1,r2) that is covering the row indices of column i, 0 otherwise;
    // if isUndirected is true, the additional condition k<i must be true for a 1.
    
    e.resize(b12::ThrustSystem<A>::Memory::getFreeBytes(uint64_t(8) << 30) / sizeof(b12::NrPoints) / 5);
    typename b12::ThrustSystem<A>::Vector<b12::NrBoxes> c(e.size());
    
    // nnz_begin[.] = cscVector_sym[. + 1] - cscVector_sym[.]
    auto nnz_begin = b12::makeBinaryTransformIterator(cscVector_sym.begin() + 1, cscVector_sym.begin(), thrust::minus<b12::NrBoxes>());
    
    b12::NrBoxes maxOutDegree = * thrust::max_element(typename b12::ThrustSystem<A>::execution_policy(),
                                                      nnz_begin, nnz_begin + nColumns);
    
    // delay_row_begin[.] = rowIndices_sym[. / maxOutDegree]
    auto delay_row_begin = thrust::make_permutation_iterator(
      rowIndices_sym.begin(),
      thrust::make_transform_iterator(thrust::make_counting_iterator(b12::NrPoints(0)),
                                      thrust::placeholders::_1 / b12::NrPoints(maxOutDegree)));
    
    // delay_column_begin[.] = columnIndices_sym[. / maxOutDegree]
    auto delay_column_begin = thrust::make_permutation_iterator(
      columnIndices_sym.begin(),
      thrust::make_transform_iterator(thrust::make_counting_iterator(b12::NrPoints(0)),
                                      thrust::placeholders::_1 / b12::NrPoints(maxOutDegree)));
    
    // specific_row_begin[.] = rowIndices_sym[(cscVector_sym[delay_column_begin[.]] + (. mod maxOutDegree)) mod nnzMatrix]
    auto specific_row_begin = thrust::make_permutation_iterator(
      rowIndices_sym.begin(),
      thrust::make_transform_iterator(
        b12::makeBinaryTransformIterator(
          thrust::make_permutation_iterator(cscVector_sym.begin(), delay_column_begin),
          thrust::make_transform_iterator(thrust::make_counting_iterator(b12::NrPoints(0)), thrust::placeholders::_1 % b12::NrPoints(maxOutDegree)),
          thrust::plus<b12::NrPoints>()),
        thrust::placeholders::_1 % b12::NrPoints(rowIndices_sym.size())));
    
    // check_overhang_begin[.] = (. mod maxOutDegree) < nnz_begin[delay_column_begin[.]]
    auto check_overhang_begin = b12::makeBinaryTransformIterator(
      thrust::make_transform_iterator(thrust::make_counting_iterator(b12::NrPoints(0)), thrust::placeholders::_1 % b12::NrPoints(maxOutDegree)),
      thrust::make_permutation_iterator(nnz_begin, delay_column_begin),
      thrust::less<b12::NrPoints>());
    
    // ti_begin[.] = Is_1_RowIndex_Functor(check_overhang_begin[.],
    //                                     specific_row_begin[.],
    //                                     delay_row_begin[.],
    //                                     delay_column_begin[.],
    //                                     cscVector[delay_row_begin[.]],
    //                                     cscVector[delay_row_begin[.] + 1])
    auto ti_begin = thrust::make_transform_iterator(
      thrust::make_zip_iterator(
        thrust::make_tuple(
          check_overhang_begin,
          specific_row_begin,
          delay_row_begin,
          delay_column_begin,
          thrust::make_permutation_iterator(cscVector.begin(), delay_row_begin),
          thrust::make_permutation_iterator(cscVector.begin() + 1, delay_row_begin))),
      Is_1_RowIndex_Functor(thrust::raw_pointer_cast(rowIndices.data()),
                            isUndirected));
    
    if (isUndirected || isSymmetric) {
      ti_begin = thrust::make_transform_iterator(
        thrust::make_zip_iterator(
          thrust::make_tuple(
            check_overhang_begin,
            specific_row_begin,
            delay_row_begin,
            delay_column_begin,
            thrust::make_permutation_iterator(cscVector_sym.begin(), delay_row_begin),
            thrust::make_permutation_iterator(cscVector_sym.begin() + 1, delay_row_begin))),
        Is_1_RowIndex_Functor(thrust::raw_pointer_cast(rowIndices_sym.data()),
                              isUndirected));
    }
    
    auto it_pair = thrust::make_pair(c.begin(), e.begin());
    
    for (int64_t i = 0, length = 1;
         it_pair.second != e.end() && i < maxOutDegree * columnIndices_sym.size() && length != 0;
         i += length, ti_begin += length, delay_column_begin += length) {
      
      length = thrust::minimum<int64_t>()(maxOutDegree * columnIndices_sym.size() - i, e.end() - it_pair.second);
      
      it_pair = thrust::reduce_by_key(typename b12::ThrustSystem<A>::execution_policy(),
                                      delay_column_begin, delay_column_begin + length,
                                      ti_begin,
                                      it_pair.first,
                                      it_pair.second);
    }
    
    thrust::inclusive_scan_by_key(typename b12::ThrustSystem<A>::execution_policy(),
                                  thrust::make_reverse_iterator(it_pair.first), c.rend(),
                                  thrust::make_reverse_iterator(it_pair.second),
                                  thrust::make_reverse_iterator(it_pair.second));
    
    auto new_end = thrust::unique_by_key(typename b12::ThrustSystem<A>::execution_policy(),
                                         c.begin(), it_pair.first,
                                         e.begin());
    
    c.resize(new_end.first - c.begin());
    e.resize(new_end.second - e.begin());
    
    cudaThreadSynchronize(); // block until kernel is finished
}


template<b12::Architecture A>
void triangles_vectorized_binary_search_full_field(const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& rowIndices,
                                                   const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& columnIndices,
                                                   const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& csrVector,
                                                   const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& cscVector,
                                                   const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& rowIndices_sym,
                                                   const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& columnIndices_sym,
                                                   const typename b12::ThrustSystem<A>::Vector<b12::NrBoxes>& cscVector_sym,
                                                   mwIndex nRows, mwIndex nColumns, uint64_t nnz, bool isUndirected, bool isSymmetric,
                                                   typename b12::ThrustSystem<A>::Vector<double>& e)
{
  //TODO
  
  e.resize(b12::ThrustSystem<A>::Memory::getFreeBytes(uint64_t(8) << 30) / sizeof(b12::NrPoints) / 5);
  typename b12::ThrustSystem<A>::Vector<b12::NrBoxes> c(e.size());
  typename b12::ThrustSystem<A>::Vector<bool> b(e.size());
  
  // nnz_begin[.] = cscVector_sym[. + 1] - cscVector_sym[.]
  auto nnz_begin = b12::makeBinaryTransformIterator(cscVector_sym.begin() + 1, cscVector_sym.begin(), thrust::minus<b12::NrBoxes>());
  
  b12::NrBoxes maxOutDegree = * thrust::max_element(typename b12::ThrustSystem<A>::execution_policy(),
                                                    nnz_begin, nnz_begin + nColumns);
  
  // delay_row_begin[.] = rowIndices_sym[. / maxOutDegree]
  auto delay_row_begin = thrust::make_permutation_iterator(
    rowIndices_sym.begin(),
    thrust::make_transform_iterator(thrust::make_counting_iterator(b12::NrPoints(0)),
                                    thrust::placeholders::_1 / b12::NrPoints(maxOutDegree)));
  
  // delay_column_begin[.] = columnIndices_sym[. / maxOutDegree]
  auto delay_column_begin = thrust::make_permutation_iterator(
    columnIndices_sym.begin(),
    thrust::make_transform_iterator(thrust::make_counting_iterator(b12::NrPoints(0)),
                                    thrust::placeholders::_1 / b12::NrPoints(maxOutDegree)));
  
  // specific_row_begin[.] = rowIndices_sym[(cscVector_sym[delay_column_begin[.]] + (. mod maxOutDegree)) mod nnzMatrix]
  auto specific_row_begin = thrust::make_permutation_iterator(
    rowIndices_sym.begin(),
    thrust::make_transform_iterator(
      b12::makeBinaryTransformIterator(
        thrust::make_permutation_iterator(cscVector_sym.begin(), delay_column_begin),
        thrust::make_transform_iterator(thrust::make_counting_iterator(b12::NrPoints(0)), thrust::placeholders::_1 % b12::NrPoints(maxOutDegree)),
        thrust::plus<b12::NrPoints>()),
      thrust::placeholders::_1 % b12::NrPoints(rowIndices_sym.size())));
  
  // check_overhang_begin[.] = (. mod maxOutDegree) < nnz_begin[delay_column_begin[.]]
  auto check_overhang_begin = b12::makeBinaryTransformIterator(
    thrust::make_transform_iterator(thrust::make_counting_iterator(b12::NrPoints(0)), thrust::placeholders::_1 % b12::NrPoints(maxOutDegree)),
    thrust::make_permutation_iterator(nnz_begin, delay_column_begin),
    thrust::less<b12::NrPoints>());
  
  // ti_begin[.] = AreValuesDifferent_Functor(specific_row_begin[.],
  //                                          delay_row_begin[.],
  //                                          delay_column_begin[.])
  auto ti_begin = thrust::make_transform_iterator(
    thrust::make_zip_iterator(
      thrust::make_tuple(
        specific_row_begin,
        delay_row_begin,
        delay_column_begin)),
    AreValuesDifferent_Functor(isUndirected));
  
  // pairs_begin[.] = (specific_row_begin[.],
  //                   delay_row_begin[.])
  auto pairs_begin = thrust::make_zip_iterator(thrust::make_tuple(specific_row_begin, delay_row_begin));
  
  auto it_pair = thrust::make_pair(c.begin(), e.begin());
  
  auto inds_begin = thrust::make_zip_iterator(
    thrust::make_tuple(
      rowIndices.begin(),
      columnIndices.begin()));
    
  auto inds_end = thrust::make_zip_iterator(
    thrust::make_tuple(
      rowIndices.end(),
      columnIndices.end()));
  
  if (isUndirected || isSymmetric) {
    inds_begin = thrust::make_zip_iterator(
      thrust::make_tuple(
        rowIndices_sym.begin(),
        columnIndices_sym.begin()));
    
    inds_end = thrust::make_zip_iterator(
      thrust::make_tuple(
        rowIndices_sym.end(),
        columnIndices_sym.end()));
  }
  
  for (int64_t i = 0, length = 1;
       it_pair.second != e.end() && i < maxOutDegree * columnIndices_sym.size() && length != 0;
       i += length, ti_begin += length, delay_column_begin += length, check_overhang_begin += length, pairs_begin += length) {
    
    length = thrust::minimum<int64_t>()(maxOutDegree * columnIndices_sym.size() - i, e.end() - it_pair.second);
    
    thrust::binary_search(typename b12::ThrustSystem<A>::execution_policy(),
                          inds_begin, inds_end,
                          pairs_begin, pairs_begin + length,
                          b.begin(),
                          b12::ColumnMajorOrderingFunctor());
    
    it_pair = thrust::reduce_by_key(typename b12::ThrustSystem<A>::execution_policy(),
                                    delay_column_begin, delay_column_begin + length,
                                    thrust::make_transform_iterator(
                                      thrust::make_zip_iterator(
                                        thrust::make_tuple(
                                          b.begin(), check_overhang_begin, ti_begin)),
                                      AreAllTrue_Functor()),
                                    it_pair.first,
                                    it_pair.second,
                                    thrust::equal_to<b12::NrBoxes>(),
                                    thrust::plus<b12::NrPoints>());
  }
  
  thrust::inclusive_scan_by_key(typename b12::ThrustSystem<A>::execution_policy(),
                                thrust::make_reverse_iterator(it_pair.first), c.rend(),
                                thrust::make_reverse_iterator(it_pair.second),
                                thrust::make_reverse_iterator(it_pair.second));
  
  auto new_end = thrust::unique_by_key(typename b12::ThrustSystem<A>::execution_policy(),
                                      c.begin(), it_pair.first,
                                      e.begin());
  
  c.resize(new_end.first - c.begin());
  e.resize(new_end.second - e.begin());
  
  cudaThreadSynchronize(); // block until kernel is finished
}


template<b12::Architecture A>
inline void computeClusteringCoefficients(mwIndex * row_begin, mwIndex * csc_begin,
                                          mwIndex nRows, mwIndex nColumns, uint64_t nnz,
                                          const std::string& method, bool isUndirected, bool isSymmetric,
                                          double * resC, double * resE, double * resK)
{
  // _sym vectors are not necessarily symmetric!!!
  typename b12::ThrustSystem<A>::Vector<b12::NrBoxes> rowIndices_sym(row_begin, row_begin + nnz);
  typename b12::ThrustSystem<A>::Vector<b12::NrBoxes> cscVector_sym(csc_begin, csc_begin + nColumns + 1);
  
  typename b12::ThrustSystem<A>::Vector<b12::NrBoxes> columnIndices_sym(cscVector_sym.begin(), cscVector_sym.end());
  b12::extendCompressedIndexVector<A>(columnIndices_sym);
  
  if (method.compare("set_intersection_serialOverColumns") == 0 || 
      method.compare("set_intersection_parallelOverColumns") == 0) {
    removeDiagonal<A>(rowIndices_sym, columnIndices_sym);
    // adapt cscVector_sym
    cscVector_sym.assign(columnIndices_sym.begin(), columnIndices_sym.end());
    b12::compressIndexVector<A>(cscVector_sym, nColumns);
  }
  
  typename b12::ThrustSystem<A>::Vector<b12::NrBoxes> rowIndices;
  typename b12::ThrustSystem<A>::Vector<b12::NrBoxes> columnIndices;
  typename b12::ThrustSystem<A>::Vector<b12::NrBoxes> csrVector;
  typename b12::ThrustSystem<A>::Vector<b12::NrBoxes> cscVector;
  
  // only use nonsymmetric matrix, too, if graph shall be directed and matrix is not symmetric
  if (! isUndirected && ! isSymmetric) {
    rowIndices.assign(rowIndices_sym.begin(), rowIndices_sym.end());
    columnIndices.assign(columnIndices_sym.begin(), columnIndices_sym.end());
    cscVector.assign(cscVector_sym.begin(), cscVector_sym.end());
    if (method.compare("dot_product") == 0) {
      auto inds_begin = thrust::make_zip_iterator(thrust::make_tuple(rowIndices.begin(), columnIndices.begin()));
      thrust::sort(typename b12::ThrustSystem<A>::execution_policy(),
                   inds_begin, inds_begin + columnIndices.size(),
                   b12::RowMajorOrderingFunctor());
      csrVector.assign(rowIndices.begin(), rowIndices.end());
      b12::compressIndexVector<A>(csrVector, nRows);
      // cscVector is not required here
    }
  }
  
  // make matrix symmetric
  makeSymmetric<A>(rowIndices_sym, columnIndices_sym);
  // adapt cscVector_sym
  cscVector_sym.assign(columnIndices_sym.begin(), columnIndices_sym.end());
  b12::compressIndexVector<A>(cscVector_sym, nColumns);
  
  
  // number of triangles
  typename b12::ThrustSystem<A>::Vector<double> e;
  
  if (method.compare("dot_product") == 0) {
    dot_product<A>(rowIndices, columnIndices, csrVector, cscVector, 
                   rowIndices_sym, columnIndices_sym, cscVector_sym, 
                   nRows, nColumns, nnz, isUndirected, isSymmetric, e);
  } else if (method.compare("set_intersection_serialOverColumns") == 0) {
    set_intersection_serialOverColumns<A>(rowIndices, columnIndices, csrVector, cscVector, 
                                          rowIndices_sym, columnIndices_sym, cscVector_sym, 
                                          nRows, nColumns, nnz, isUndirected, isSymmetric, e);
  } else if (method.compare("set_intersection_parallelOverColumns") == 0) {
    set_intersection_parallelOverColumns<A>(rowIndices, columnIndices, csrVector, cscVector, 
                                            rowIndices_sym, columnIndices_sym, cscVector_sym, 
                                            nRows, nColumns, nnz, isUndirected, isSymmetric, e);
  } else if (method.compare("all_triangles_reduce_by_key") == 0) {
    all_triangles_reduce_by_key<A>(rowIndices, columnIndices, csrVector, cscVector, 
                                   rowIndices_sym, columnIndices_sym, cscVector_sym, 
                                   nRows, nColumns, nnz, isUndirected, isSymmetric, e);
  } else if (method.compare("all_triangles_count_if") == 0) {
    all_triangles_count_if<A>(rowIndices, columnIndices, csrVector, cscVector, 
                              rowIndices_sym, columnIndices_sym, cscVector_sym, 
                              nRows, nColumns, nnz, isUndirected, isSymmetric, e);
  } else if (method.compare("triangles_binary_search_full_field") == 0) {
    triangles_binary_search_full_field<A>(rowIndices, columnIndices, csrVector, cscVector, 
                                          rowIndices_sym, columnIndices_sym, cscVector_sym, 
                                          nRows, nColumns, nnz, isUndirected, isSymmetric, e);
  } else if (method.compare("triangles_binary_search_segmented_field") == 0) {
    triangles_binary_search_segmented_field<A>(rowIndices, columnIndices, csrVector, cscVector, 
                                               rowIndices_sym, columnIndices_sym, cscVector_sym, 
                                               nRows, nColumns, nnz, isUndirected, isSymmetric, e);
  } else if (method.compare("triangles_vectorized_binary_search_full_field") == 0) {
    triangles_vectorized_binary_search_full_field<A>(rowIndices, columnIndices, csrVector, cscVector, 
                                                     rowIndices_sym, columnIndices_sym, cscVector_sym, 
                                                     nRows, nColumns, nnz, isUndirected, isSymmetric, e);
  }
  
  typename b12::ThrustSystem<A>::Vector<b12::NrBoxes> cm(columnIndices_sym);
  thrust::unique(typename b12::ThrustSystem<A>::execution_policy(), cm.begin(), cm.end());
  
  // number of neighbours
  typename b12::ThrustSystem<A>::Vector<double> k;
  numberOfNeighbours<A>(nnz, rowIndices_sym, columnIndices_sym, k);
  
  // clustering coefficients
  typename b12::ThrustSystem<A>::Vector<double> C;
  clusteringCoefficients<A>(isUndirected, e, k, C);
  
  // copy results to output
  // missing columns in cm (e.g. because a column in the matrix was empty) are factored in by scattering
  {
    typename b12::ThrustSystem<A>::Vector<double> temp(nColumns, 0.0);
    thrust::scatter(typename b12::ThrustSystem<A>::execution_policy(),
                    C.begin(), C.end(),
                    cm.begin(),
                    temp.begin());
    thrust::copy(temp.begin(), temp.end(), resC);
  }
  {
    typename b12::ThrustSystem<A>::Vector<double> temp(nColumns, 0.0);
    thrust::scatter(typename b12::ThrustSystem<A>::execution_policy(),
                    e.begin(), e.end(),
                    cm.begin(),
                    temp.begin());
    thrust::copy(temp.begin(), temp.end(), resE);
  }
  {
    typename b12::ThrustSystem<A>::Vector<double> temp(nColumns, 0.0);
    thrust::scatter(typename b12::ThrustSystem<A>::execution_policy(),
                    k.begin(), k.end(),
                    cm.begin(),
                    temp.begin());
    thrust::copy(temp.begin(), temp.end(), resK);
  }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  // prhs[0] ... matrix (in CSC format)
  // prhs[1] ... bool indicating whether the matrix is undirected
  // prhs[2] ... bool indicating whether the algortihm shall be run on GPU
  // prhs[3] ... bool indicating whether the matrix is symmetric
  // prhs[4] ... method for calculating
  
  mwIndex * row_begin = mxGetIr(prhs[0]);
  mwIndex * csc_begin = mxGetJc(prhs[0]);
  
  mwIndex nRows    = * mxGetDimensions(prhs[0]);
  mwIndex nColumns = *(mxGetDimensions(prhs[0]) + 1);
  
  uint64_t nnz = csc_begin[nColumns];
  
  bool isUndirected = *((bool *) mxGetData(prhs[1]));
  bool onGPU        = *((bool *) mxGetData(prhs[2]));
  bool isSymmetric  = *((bool *) mxGetData(prhs[3]));
  char method[255];
  mxGetString(prhs[4], method, sizeof(method) + 1);
  
  if (isUndirected && nRows != nColumns) {
    mexErrMsgTxt("Matrix must be square.");
  }
  
  // initialised with 0's
  plhs[0] = mxCreateDoubleMatrix(nColumns, 1, mxREAL);
  plhs[1] = mxCreateDoubleMatrix(nColumns, 1, mxREAL);
  plhs[2] = mxCreateDoubleMatrix(nColumns, 1, mxREAL);
  
//   std::cout << "Output created." << std::endl;
  
  if (onGPU) {
    computeClusteringCoefficients<b12::CUDA>(row_begin, csc_begin, nRows, nColumns, nnz, std::string(method), isUndirected, isSymmetric,
                                             mxGetPr(plhs[0]), mxGetPr(plhs[1]), mxGetPr(plhs[2]));
  } else {
    computeClusteringCoefficients<b12::OMP>(row_begin, csc_begin, nRows, nColumns, nnz, std::string(method), isUndirected, isSymmetric,
                                            mxGetPr(plhs[0]), mxGetPr(plhs[1]), mxGetPr(plhs[2]));
  }
}
