
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/pair.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include "Grids.h" 
#include "Map_Henon.h"
#include "Map_Lorenz.h"
#include "Map_StandardBO2.h"

// #include "CooMatrix.h"
// #include "TypeDefinitions.h"
// #include "ThrustSystem.h"

#define INCLUDE_IMPLICIT_BOX_TREE_MEMBER_DEFINITIONS
#include "ImplicitBoxTree.h"
#undef INCLUDE_IMPLICIT_BOX_TREE_MEMBER_DEFINITIONS

#include <typeinfo>
#include <chrono>
#include <string>

using namespace b12;

struct SetInt_RowIndexPairsPerColumn_AllIndexPairs_Undirected_Functor : public thrust::unary_function<thrust::tuple<int,int>,NrPoints>
{
  NrBoxes * rowIndices_begin_, * columnIndices_begin_;
  NrPoints nnzMatrix_;
  
  __host__ __device__
  SetInt_RowIndexPairsPerColumn_AllIndexPairs_Undirected_Functor(NrBoxes * rowIndices_begin, NrBoxes * columnIndices_begin, NrPoints nnzMatrix)
      : rowIndices_begin_(rowIndices_begin), columnIndices_begin_(columnIndices_begin), nnzMatrix_(nnzMatrix) {};
  
  __host__ __device__
  NrPoints operator()(thrust::tuple<int,int> t) const
  {
    int ind = thrust::get<0>(t);
    int nnz = thrust::get<1>(t) - thrust::get<0>(t);
    
    auto zi_begin = thrust::make_zip_iterator(
      thrust::make_tuple(
        thrust::make_permutation_iterator(
          rowIndices_begin_ + ind,
          thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)), thrust::placeholders::_1 % nnz)),
        thrust::make_permutation_iterator(
          rowIndices_begin_ + ind,
          thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)), thrust::placeholders::_1 / nnz))));
    
    auto inds_begin = thrust::make_zip_iterator(
      thrust::make_tuple(
        rowIndices_begin_,
        columnIndices_begin_));
    
    auto res_begin = thrust::make_discard_iterator();
    
    auto res_end = thrust::set_intersection(thrust::seq,
                                            zi_begin, zi_begin + nnz*nnz,
                                            inds_begin, inds_begin + nnzMatrix_,
                                            res_begin,
                                            ColumnMajorOrderingFunctor());
    
    return res_end - res_begin;
  }
};

struct SetInt_RowIndexPairsPerColumn_AllIndexPairs_Directed_Functor : public thrust::unary_function<thrust::tuple<int,int>,NrPoints>
{
  NrBoxes * rowIndices_begin_, * columnIndices_begin_, * rowIndices_sym_begin_;
  NrPoints nnzMatrix_;
  
  __host__ __device__
  SetInt_RowIndexPairsPerColumn_AllIndexPairs_Directed_Functor(NrBoxes * rowIndices_begin, NrBoxes * columnIndices_begin, NrBoxes * rowIndices_sym_begin, NrPoints nnzMatrix)
      : rowIndices_begin_(rowIndices_begin), columnIndices_begin_(columnIndices_begin), rowIndices_sym_begin_(rowIndices_sym_begin), nnzMatrix_(nnzMatrix) {};
  
  __host__ __device__
  NrPoints operator()(thrust::tuple<int,int> t) const
  {
    int ind = thrust::get<0>(t);
    int nnz = thrust::get<1>(t) - thrust::get<0>(t);
    
    auto zi_begin = thrust::make_zip_iterator(
      thrust::make_tuple(
        thrust::make_permutation_iterator(
          rowIndices_sym_begin_ + ind,
          thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)), thrust::placeholders::_1 % nnz)),
        thrust::make_permutation_iterator(
          rowIndices_sym_begin_ + ind,
          thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)), thrust::placeholders::_1 / nnz))));
    
    auto inds_begin = thrust::make_zip_iterator(
      thrust::make_tuple(
        rowIndices_begin_,
        columnIndices_begin_));
    
    auto res_begin = thrust::make_discard_iterator();
    
    auto res_end = thrust::set_intersection(thrust::seq,
                                            zi_begin, zi_begin + nnz*nnz,
                                            inds_begin, inds_begin + nnzMatrix_,
                                            res_begin,
                                            ColumnMajorOrderingFunctor());
    
    return res_end - res_begin;
  }
};

struct AreThereIndexPairs_02_and_01_Functor : public thrust::unary_function<thrust::tuple<NrBoxes,NrBoxes,NrBoxes>,NrPoints>
{
  NrBoxes * rowIndices_begin_, * columnIndices_begin_;
  NrPoints nnzMatrix_;
  bool isUndirected_;
  
  __host__ __device__
  AreThereIndexPairs_02_and_01_Functor(NrBoxes * rowIndices_begin, NrBoxes * columnIndices_begin, NrPoints nnzMatrix, bool isUndirected)
      : rowIndices_begin_(rowIndices_begin), columnIndices_begin_(columnIndices_begin), nnzMatrix_(nnzMatrix), isUndirected_(isUndirected) {};
  
  __host__ __device__
  NrPoints operator()(thrust::tuple<NrBoxes,NrBoxes,NrBoxes> t) const
  {
    // thrust::get<0>(t) ... zu ueberpreufen
    // thrust::get<1>(t) ... vorhandene Zeile / getroffen von thrust::get<2>(t)
    // thrust::get<2>(t) ... vorhandene Spalte
    
    if (thrust::get<0>(t) == thrust::get<1>(t) || thrust::get<0>(t) == thrust::get<2>(t) || thrust::get<1>(t) == thrust::get<2>(t)) {
      
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
                                                   ColumnMajorOrderingFunctor());
      if (! isUndirected_) {
        is0NeighbourOf2 = is0NeighbourOf2 || thrust::binary_search(thrust::seq,
                                                                   inds_begin, inds_begin + nnzMatrix_,
                                                                   thrust::make_tuple(thrust::get<2>(t),
                                                                                      thrust::get<0>(t)),
                                                                   ColumnMajorOrderingFunctor());
      }
      
      if (is0NeighbourOf2) {
        // gibt es Weg von thrust::get<1>(t) zu thrust::get<0>(t)
        bool way1to0 = thrust::binary_search(thrust::seq,
                                             inds_begin, inds_begin + nnzMatrix_,
                                             thrust::make_tuple(thrust::get<0>(t),
                                                                thrust::get<1>(t)),
                                             ColumnMajorOrderingFunctor());
        return way1to0 ? 1 : 0;
      } else {
        return 0;
      }
    }
  }
};

struct IsThereIndexPair_12_Functor : public thrust::unary_function<thrust::tuple<bool,NrBoxes,NrBoxes,NrBoxes>,NrPoints>
{
  NrBoxes * rowIndices_begin_, * columnIndices_begin_;
  NrPoints nnzMatrix_;
  bool isUndirected_;
  
  __host__ __device__
  IsThereIndexPair_12_Functor(NrBoxes * rowIndices_begin, NrBoxes * columnIndices_begin, NrPoints nnzMatrix, bool isUndirected)
      : rowIndices_begin_(rowIndices_begin), columnIndices_begin_(columnIndices_begin), nnzMatrix_(nnzMatrix), isUndirected_(isUndirected) {};
  
  __host__ __device__
  NrPoints operator()(thrust::tuple<bool,NrBoxes,NrBoxes,NrBoxes> t) const
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
                                        ColumnMajorOrderingFunctor());
      
      return res ? 1 : 0;
    }
  }
};

struct Is_1_RowIndex_Functor : public thrust::unary_function<thrust::tuple<bool,NrBoxes,NrBoxes,NrBoxes,NrBoxes,NrBoxes>,NrPoints>
{
  NrBoxes * rowIndices_begin_;
  bool isUndirected_;
  
  __host__ __device__
  Is_1_RowIndex_Functor(NrBoxes * rowIndices_begin, bool isUndirected)
      : rowIndices_begin_(rowIndices_begin), isUndirected_(isUndirected) {};
  
  __host__ __device__
  NrPoints operator()(thrust::tuple<bool,NrBoxes,NrBoxes,NrBoxes,NrBoxes,NrBoxes> t) const
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

struct AreValuesDifferent_Functor : public thrust::unary_function<thrust::tuple<NrBoxes,NrBoxes,NrBoxes>,bool>
{
  bool isUndirected_;
  
  __host__ __device__
  AreValuesDifferent_Functor(bool isUndirected) : isUndirected_(isUndirected) {};
  
  __host__ __device__
  bool operator()(thrust::tuple<NrBoxes,NrBoxes,NrBoxes> t) const
  {
    return (isUndirected_ ? thrust::get<0>(t) < thrust::get<1>(t) : thrust::get<0>(t) != thrust::get<1>(t)) && 
           thrust::get<0>(t) != thrust::get<2>(t) &&
           thrust::get<1>(t) != thrust::get<2>(t);
  }
};

struct AreAllTrue_Functor : public thrust::unary_function<thrust::tuple<bool,bool,bool>,NrPoints>
{
  __host__ __device__
  NrPoints operator()(thrust::tuple<bool,bool,bool> t) const
  {
    return (thrust::get<0>(t) && thrust::get<1>(t) && thrust::get<2>(t)) ? 1 : 0;
  }
};

struct SparseDotProductForUndirected_Functor : public thrust::unary_function<thrust::tuple<NrBoxes,NrBoxes,NrBoxes,NrBoxes,NrBoxes,NrBoxes>,NrPoints>
{
  NrBoxes * rowIndices_begin_;
  
  __host__ __device__
  SparseDotProductForUndirected_Functor(NrBoxes * rowIndices_begin) 
      : rowIndices_begin_(rowIndices_begin) {};
  
  __host__ __device__
  NrPoints operator()(thrust::tuple<NrBoxes,NrBoxes,NrBoxes,NrBoxes,NrBoxes,NrBoxes> t) const
  {
    // thrust::get<0>(t) ... i
    // thrust::get<1>(t) ... begin of column i
    // thrust::get<2>(t) ... end of column i
    // thrust::get<3>(t) ... j
    // thrust::get<4>(t) ... begin of column j
    // thrust::get<5>(t) ... end of column j
    
    NrPoints res = 0;
    
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

struct SparseDotProductForDirected_Functor : public thrust::unary_function<thrust::tuple<NrBoxes,NrBoxes,NrBoxes,NrBoxes,NrBoxes,NrBoxes>,NrPoints>
{
  NrBoxes * csrColumnIndices_begin_, * cscRowIndices_begin_;
  
  __host__ __device__
  SparseDotProductForDirected_Functor(NrBoxes * csrColumnIndices_begin, NrBoxes * cscRowIndices_begin) 
      : csrColumnIndices_begin_(csrColumnIndices_begin), cscRowIndices_begin_(cscRowIndices_begin) {};
  
  __host__ __device__
  NrPoints operator()(thrust::tuple<NrBoxes,NrBoxes,NrBoxes,NrBoxes,NrBoxes,NrBoxes> t) const
  {
    // thrust::get<0>(t) ... i
    // thrust::get<1>(t) ... begin of row i (in CSR matrix)
    // thrust::get<2>(t) ... end of row i (in CSR matrix)
    // thrust::get<3>(t) ... j
    // thrust::get<4>(t) ... begin of column j (in CSC matrix)
    // thrust::get<5>(t) ... end of column j (in CSC matrix)
    
    NrPoints res = 0;
    
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

template<Architecture A>
void makeSymmetric(typename b12::CooMatrix<A, bool>& P)
{
  uint64_t nnz = P.rowIndices_.size();
  
  P.rowIndices_.resize(nnz * 2);
  P.columnIndices_.resize(nnz * 2);
  
  thrust::copy(P.rowIndices_.begin(), P.rowIndices_.begin() + nnz, P.columnIndices_.begin() + nnz);
  thrust::copy(P.columnIndices_.begin(), P.columnIndices_.begin() + nnz, P.rowIndices_.begin() + nnz);
  
  auto inds_begin = thrust::make_zip_iterator(thrust::make_tuple(P.rowIndices_.begin(), P.columnIndices_.begin()));
  
  thrust::sort(typename b12::ThrustSystem<A>::execution_policy(),
               inds_begin, inds_begin + nnz * 2,
               b12::ColumnMajorOrderingFunctor());
  
  auto it = thrust::unique(typename b12::ThrustSystem<A>::execution_policy(),
                           inds_begin, inds_begin + nnz * 2);
  
  nnz = it - inds_begin;
  
  P.rowIndices_.resize(nnz);
  P.columnIndices_.resize(nnz);
  
  cudaThreadSynchronize(); // block until kernel is finished
}

template<Architecture A>
void removeDiagonal(typename b12::CooMatrix<A, bool>& P)
{
  auto P_begin = thrust::make_zip_iterator(thrust::make_tuple(P.rowIndices_.begin(), P.columnIndices_.begin()));
  auto P_end = thrust::remove_if(typename ThrustSystem<A>::execution_policy(),
                                 P_begin, P_begin + P.rowIndices_.size(),
                                 makeBinaryTransformIterator(P.rowIndices_.begin(), P.columnIndices_.begin(), thrust::equal_to<NrBoxes>()),
                                 thrust::identity<bool>());
  P.resize(P_end - P_begin);
  cudaThreadSynchronize(); // block until kernel is finished
}

template<Architecture A>
void printVector(const typename ThrustSystem<A>::Vector<NrPoints>& e)
{
  for (auto i : e) {
    std::cout << i << "\t";
  }
  std::cout << std::endl;
}

int main(int argc, char *argv[])
{
  const Architecture A = CUDA;
  const Dimension DIM = 2;
  int d = 0;
  double K = 1.0;
  int s = 1;
  bool isUndirected = false;
  
  if (argc >= 2) {
    d = atoi(argv[1]);
    if (argc >= 3) {
      K = atof(argv[2]);
      if (argc >= 4) {
        s = atoi(argv[3]);
        if (argc >= 5) {
          isUndirected = atoi(argv[4]);
        }
      }
    }
  }
  
  ThrustSystem<A>::Vector<double> center(DIM, 0.0);
  center[0] = 0.5;
  ThrustSystem<A>::Vector<double> radius(DIM, 0.5);
  
  ImplicitBoxTree<A, DIM, double, 32> ibt(center, radius); // surrounding box: [-3, 3]^2
  StandardBO2<DIM, double> map(K, s);
  
  for (int i = 0; i < d; ++i) {
    ibt.subdivide(Flag::NONE);
  }
  
//   ImplicitBoxTree<A, 2, double, 64> ibt("../Skripte/test/henon_rga_" + (d < 10 ? std::string("0") : std::string(""))   + std::to_string(d) + ".json");
//   Henon<2, double> map(K, 0.3, s);
  
  InnerGrid<DIM, double> grid(50); // an inner grid scheme with 10^DIM points for each box
  
  auto tStart = std::chrono::system_clock::now();
  CooMatrix<A, bool> P = ibt.transitionMatrixForLeaves<bool>(grid, map);
  P.values_.clear();
  P.values_.shrink_to_fit();
  cudaThreadSynchronize(); // block until kernel is finished
  std::chrono::duration<double> tDiff = std::chrono::system_clock::now() - tStart;
  std::cout << "Matrix: \t " << tDiff.count() << " s" << std::endl;
  
  if (isUndirected) {
    tStart = std::chrono::system_clock::now();
    makeSymmetric(P);
    cudaThreadSynchronize(); // block until kernel is finished
    tDiff = std::chrono::system_clock::now() - tStart;
    std::cout << "Make sym.: \t " << tDiff.count() << " s" << std::endl;
  }
  
  tStart = std::chrono::system_clock::now();
  auto PwoD = P;
  removeDiagonal<A>(PwoD);
  tDiff = std::chrono::system_clock::now() - tStart;
  std::cout << "Elim. diag.: \t " << tDiff.count() << " s" << std::endl;
  
  tStart = std::chrono::system_clock::now();
  auto Psym = P;
  makeSymmetric(Psym);
  cudaThreadSynchronize(); // block until kernel is finished
  tDiff = std::chrono::system_clock::now() - tStart;
  std::cout << "Make sym.: \t " << tDiff.count() << " s" << std::endl;
  
  tStart = std::chrono::system_clock::now();
  auto PsymwoD = Psym;
  removeDiagonal<A>(PsymwoD);
  tDiff = std::chrono::system_clock::now() - tStart;
  std::cout << "Elim. diag.: \t " << tDiff.count() << " s" << std::endl;
  
  tStart = std::chrono::system_clock::now();
  typename ThrustSystem<A>::Vector<NrBoxes> cscVector(P.columnIndices_.begin(), P.columnIndices_.end());
  compressIndexVector<A>(cscVector, P.nColumns_);
  cudaThreadSynchronize(); // block until kernel is finished
  tDiff = std::chrono::system_clock::now() - tStart;
  std::cout << "cscVector vector: \t " << tDiff.count() << " s" << std::endl;
  
  tStart = std::chrono::system_clock::now();
  typename ThrustSystem<A>::Vector<NrBoxes> cscVectorwoD(PwoD.columnIndices_.begin(), PwoD.columnIndices_.end());
  compressIndexVector<A>(cscVectorwoD, PwoD.nColumns_);
  cudaThreadSynchronize(); // block until kernel is finished
  tDiff = std::chrono::system_clock::now() - tStart;
  std::cout << "cscVectorwoD vector: \t " << tDiff.count() << " s" << std::endl << std::endl;
  
  tStart = std::chrono::system_clock::now();
  typename ThrustSystem<A>::Vector<NrBoxes> cscVectorSym(Psym.columnIndices_.begin(), Psym.columnIndices_.end());
  compressIndexVector<A>(cscVectorSym, Psym.nColumns_);
  cudaThreadSynchronize(); // block until kernel is finished
  tDiff = std::chrono::system_clock::now() - tStart;
  std::cout << "cscVectorSym vector: \t " << tDiff.count() << " s" << std::endl;
  
  tStart = std::chrono::system_clock::now();
  typename ThrustSystem<A>::Vector<NrBoxes> cscVectorSymwoD(PsymwoD.columnIndices_.begin(), PsymwoD.columnIndices_.end());
  compressIndexVector<A>(cscVectorSymwoD, PsymwoD.nColumns_);
  cudaThreadSynchronize(); // block until kernel is finished
  tDiff = std::chrono::system_clock::now() - tStart;
  std::cout << "cscVectorSymwoD vector: \t " << tDiff.count() << " s" << std::endl << std::endl;
  
//   P = PwoD;
//   std::cout << P << std::endl;
  
  typename ThrustSystem<A>::Vector<NrPoints> test;
  
// for each (parallelly executed) pair [i,j] compute the dot product of row i and column j sequentially ----------------------------------------
  {
    auto tStart = std::chrono::system_clock::now();
    
    typename ThrustSystem<A>::Vector<NrPoints> e(P.rowIndices_.size());
    typename ThrustSystem<A>::Vector<NrBoxes> c(e.size());
    typename ThrustSystem<A>::Vector<NrBoxes> k(e.size());
    
    if (isUndirected) {
      
      // ti_begin[.] = SparseDotProductForUndirected_Functor(P.rowIndices_[.],
      //                                                     cscVector[P.rowIndices_[.]],
      //                                                     cscVector[P.rowIndices_[.] + 1],
      //                                                     P.columnIndices_[.],
      //                                                     cscVector[P.columnIndices_[.]],
      //                                                     cscVector[P.columnIndices_[.] + 1])
      auto ti_begin = thrust::make_transform_iterator(
        thrust::make_zip_iterator(
          thrust::make_tuple(
            P.rowIndices_.begin(),
            thrust::make_permutation_iterator(cscVector.begin(), P.rowIndices_.begin()),
            thrust::make_permutation_iterator(cscVector.begin() + 1, P.rowIndices_.begin()),
            P.columnIndices_.begin(),
            thrust::make_permutation_iterator(cscVector.begin(), P.columnIndices_.begin()),
            thrust::make_permutation_iterator(cscVector.begin() + 1, P.columnIndices_.begin()))),
        SparseDotProductForUndirected_Functor(thrust::raw_pointer_cast(P.rowIndices_.data())));
      
      auto it_pair = thrust::reduce_by_key(typename ThrustSystem<A>::execution_policy(),
                                           P.columnIndices_.begin(), P.columnIndices_.end(),
                                           ti_begin,
                                           c.begin(),
                                           e.begin());
      
      c.resize(it_pair.first - c.begin());
      e.resize(it_pair.second - e.begin());
      
      thrust::transform(typename ThrustSystem<A>::execution_policy(),
                        e.begin(), e.end(),
                        e.begin(),
                        thrust::placeholders::_1 / 2);
    } else {
      
      typename ThrustSystem<A>::Vector<NrBoxes> csrVector(P.rowIndices_.begin(), P.rowIndices_.end());
      typename ThrustSystem<A>::Vector<NrBoxes> columnIndices(P.columnIndices_.begin(), P.columnIndices_.end());
      
      auto inds_begin = thrust::make_zip_iterator(thrust::make_tuple(csrVector.begin(), columnIndices.begin()));
      
      thrust::sort(typename ThrustSystem<A>::execution_policy(),
                   inds_begin, inds_begin + P.rowIndices_.size(),
                   RowMajorOrderingFunctor());
      
      compressIndexVector<A>(csrVector, P.nRows_);
      
      // ti_begin[.] = SparseDotProductForDirected_Functor(Psym.rowIndices_[.],
      //                                                   csrVector[Psym.rowIndices_[.]],
      //                                                   csrVector[Psym.rowIndices_[.] + 1],
      //                                                   Psym.columnIndices_[.],
      //                                                   cscVectorSym[Psym.columnIndices_[.]],
      //                                                   cscVectorSym[Psym.columnIndices_[.] + 1])
      auto ti_begin = thrust::make_transform_iterator(
        thrust::make_zip_iterator(
          thrust::make_tuple(
            Psym.rowIndices_.begin(),
            thrust::make_permutation_iterator(csrVector.begin(), Psym.rowIndices_.begin()),
            thrust::make_permutation_iterator(csrVector.begin() + 1, Psym.rowIndices_.begin()),
            Psym.columnIndices_.begin(),
            thrust::make_permutation_iterator(cscVectorSym.begin(), Psym.columnIndices_.begin()),
            thrust::make_permutation_iterator(cscVectorSym.begin() + 1, Psym.columnIndices_.begin()))),
        SparseDotProductForDirected_Functor(thrust::raw_pointer_cast(columnIndices.data()),
                                            thrust::raw_pointer_cast(Psym.rowIndices_.data())));
      
      auto it_pair = thrust::reduce_by_key(typename ThrustSystem<A>::execution_policy(),
                                           Psym.columnIndices_.begin(), Psym.columnIndices_.end(),
                                           ti_begin,
                                           c.begin(),
                                           e.begin());
      
      c.resize(it_pair.first - c.begin());
      e.resize(it_pair.second - e.begin());
    }
    
    auto it_pair = thrust::reduce_by_key(typename ThrustSystem<A>::execution_policy(),
                                         Psym.columnIndices_.begin(), Psym.columnIndices_.end(),
                                         thrust::make_constant_iterator(NrBoxes(1)),
                                         thrust::make_discard_iterator(),
                                         k.begin());
    
    k.resize(it_pair.second - k.begin());
    
    cudaThreadSynchronize(); // block until kernel is finished
    std::chrono::duration<double> tDiff = std::chrono::system_clock::now() - tStart;
    std::cout << "for each (parallelly executed) pair [i,j] compute the dot product of row i and column j sequentially:" << std::endl << tDiff.count() << " s";
    std::cout << std::endl;
//     printVector<A>(e);
    test.assign(e.begin(), e.end());
  }
// ---------------------------------------------------------------------------------------------------------------------------------------------

  
// serial iteration over all columns / parallel in each column using set_intersection ----------------------------------------------------------
  {
    // set intersection of two sets of index pairs
    // 1) all actual index pairs in the matrix (without diagonal) -> [inds_begin, inds_end)
    // 2) all nnz row indices of one column in variation with themselves, i.e. if i_1, ..., i_nnz are the row indices of one column, the set is
    //    (i_1, i_1), ..., (i_nnz, i_1), (i_1, i_2), ..., (i_nnz, i_2), ..., (i_nnz, i_nnz) -> [zi_begin, zi_begin + nnz^2)
    // for directed graphs: take 1) as before, but 2) from symmetric part of the matrix
    
    auto tStart = std::chrono::system_clock::now();
    
    auto inds_begin = thrust::make_zip_iterator(
      thrust::make_tuple(
        PwoD.rowIndices_.begin(),
        PwoD.columnIndices_.begin()));
    
    auto inds_end = thrust::make_zip_iterator(
      thrust::make_tuple(
        PwoD.rowIndices_.end(),
        PwoD.columnIndices_.end()));
    
    typename ThrustSystem<A>::Vector<NrPoints> e(PwoD.nColumns_, 0);
    
    if (isUndirected) {
      for (int i = 0; i < cscVectorwoD.size()-1; ++i) {
        
        NrPoints nnz = cscVectorwoD[i + 1] - cscVectorwoD[i];
        
        // zi_begin[.] = (PwoD.rowIndices_[(. mod nnz) + cscVectorwoD[i]],
        //                PwoD.rowIndices_[(.  /  nnz) + cscVectorwoD[i]])
        auto zi_begin = thrust::make_zip_iterator(
          thrust::make_tuple(
            thrust::make_permutation_iterator(
              PwoD.rowIndices_.begin() + cscVectorwoD[i],
              thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)), thrust::placeholders::_1 % nnz)),
            thrust::make_permutation_iterator(
              PwoD.rowIndices_.begin() + cscVectorwoD[i],
              thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)), thrust::placeholders::_1 / nnz))));
        
        auto res_begin = thrust::make_discard_iterator();
        
        auto res_end = thrust::set_intersection(typename ThrustSystem<A>::execution_policy(),
                                                zi_begin, zi_begin + nnz*nnz,
                                                inds_begin, inds_end,
                                                res_begin,
                                                ColumnMajorOrderingFunctor());
        
        e[i] = res_end - res_begin;
      }
      
      thrust::transform(typename ThrustSystem<A>::execution_policy(),
                        e.begin(), e.end(),
                        e.begin(),
                        thrust::placeholders::_1 / 2);
    } else {
      for (int i = 0; i < cscVectorSymwoD.size()-1; ++i) {
        
        NrPoints nnz = cscVectorSymwoD[i + 1] - cscVectorSymwoD[i];
        
        // zi_begin[.] = (PsymwoD.rowIndices_[(. mod nnz) + cscVectorSymwoD[i]],
        //                PsymwoD.rowIndices_[(.  /  nnz) + cscVectorSymwoD[i]])
        auto zi_begin = thrust::make_zip_iterator(
          thrust::make_tuple(
            thrust::make_permutation_iterator(
              PsymwoD.rowIndices_.begin() + cscVectorSymwoD[i],
              thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)), thrust::placeholders::_1 % nnz)),
            thrust::make_permutation_iterator(
              PsymwoD.rowIndices_.begin() + cscVectorSymwoD[i],
              thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)), thrust::placeholders::_1 / nnz))));
        
        auto res_begin = thrust::make_discard_iterator();
        
        auto res_end = thrust::set_intersection(typename ThrustSystem<A>::execution_policy(),
                                                zi_begin, zi_begin + nnz*nnz,
                                                inds_begin, inds_end,
                                                res_begin,
                                                ColumnMajorOrderingFunctor());
        
        e[i] = res_end - res_begin;
      }
    }
    
    cudaThreadSynchronize(); // block until kernel is finished
    std::chrono::duration<double> tDiff = std::chrono::system_clock::now() - tStart;
    std::cout << "serial iteration over all columns / parallel in each column using set_intersection:" << std::endl << tDiff.count() << " s";
    std::cout << ",\t " << (thrust::equal(typename ThrustSystem<A>::execution_policy(), e.begin(), e.end(), test.begin()) ? "" : "not ") << "equal" << std::endl;
//     printVector<A>(e);
    // test.assign(e.begin(), e.end());
  }
// ---------------------------------------------------------------------------------------------------------------------------------------------
  
  
  
// parallel iteration over all columns / serial in each column using set_intersection ----------------------------------------------------------
  {
    // set intersection of two sets of index pairs
    // 1) all actual index pairs in the matrix (without diagonal) -> [inds_begin, inds_end)
    // 2) all nnz row indices of one column in variation with themselves, i.e. if i_1, ..., i_nnz are the row indices of one column, the set is
    //    (i_1, i_1), ..., (i_nnz, i_1), (i_1, i_2), ..., (i_nnz, i_2), ..., (i_nnz, i_nnz) -> [zi_begin, zi_begin + nnz^2)
    // for directed graphs: take 1) as before, but 2) from symmetric part of the matrix
    
    auto tStart = std::chrono::system_clock::now();
    
    typename ThrustSystem<A>::Vector<NrPoints> e(PwoD.nColumns_);
    
    if (isUndirected) {
      // zi_begin[.] = (cscVectorwoD[.],
      //                cscVectorwoD[. + 1])
      auto zi_begin = thrust::make_zip_iterator(thrust::make_tuple(cscVectorwoD.begin(), cscVectorwoD.begin() + 1));
      
      thrust::transform(typename ThrustSystem<A>::execution_policy(),
                        zi_begin, zi_begin + PwoD.nColumns_,
                        e.begin(),
                        SetInt_RowIndexPairsPerColumn_AllIndexPairs_Undirected_Functor(thrust::raw_pointer_cast(PwoD.rowIndices_.data()),
                                                                                       thrust::raw_pointer_cast(PwoD.columnIndices_.data()),
                                                                                       PwoD.columnIndices_.size()));
      
      thrust::transform(typename ThrustSystem<A>::execution_policy(),
                        e.begin(), e.end(),
                        e.begin(),
                        thrust::placeholders::_1 / 2);
    } else {
      // zi_begin[.] = (cscVectorSymwoD[.],
      //                cscVectorSymwoD[. + 1])
      auto zi_begin = thrust::make_zip_iterator(thrust::make_tuple(cscVectorSymwoD.begin(), cscVectorSymwoD.begin() + 1));
      
      thrust::transform(typename ThrustSystem<A>::execution_policy(),
                        zi_begin, zi_begin + PsymwoD.nColumns_,
                        e.begin(),
                        SetInt_RowIndexPairsPerColumn_AllIndexPairs_Directed_Functor(thrust::raw_pointer_cast(PwoD.rowIndices_.data()),
                                                                                     thrust::raw_pointer_cast(PwoD.columnIndices_.data()),
                                                                                     thrust::raw_pointer_cast(PsymwoD.rowIndices_.data()),
                                                                                     PwoD.columnIndices_.size()));
    }
    
    cudaThreadSynchronize(); // block until kernel is finished
    std::chrono::duration<double> tDiff = std::chrono::system_clock::now() - tStart;
    std::cout << "parallel iteration over all columns / serial in each column using set_intersection:" << std::endl << tDiff.count() << " s";
    std::cout << ",\t " << (thrust::equal(typename ThrustSystem<A>::execution_policy(), e.begin(), e.end(), test.begin()) ? "" : "not ") << "equal" << std::endl;
//     printVector<A>(e);
    // test.assign(e.begin(), e.end());
  }
// ---------------------------------------------------------------------------------------------------------------------------------------------
  
  
  
// iteration over all possible triangles made out of existing pairs using reduce_by_key --------------------------------------------------------
  {
    // for each existing index pair (i,j) of the matrix's symmetric part, the triples (0,i,j), ..., (n-1,i,j) are progressed
    // for each triple (k,i,j), 1 is added if k != i != j != k and (k,j) and (k,i) are index pairs of the matrix (found out by binary a search), 0 otherwise
    
    auto tStart = std::chrono::system_clock::now();
    
    typename ThrustSystem<A>::Vector<NrPoints> e(ThrustSystem<A>::Memory::getFreeBytes(uint64_t(8) << 30) / sizeof(NrPoints) / 6);
    typename ThrustSystem<A>::Vector<NrBoxes> c(e.size());
    
    if (isUndirected) {
      // ti_begin[.] = AreThereIndexPairs_02_and_01_Functor(. mod P.nColumns_,
      //                                                    P.rowIndices_[. / P.nColumns_],
      //                                                    P.columnIndices_[. / P.nColumns_],
      //                                                    isUndirected)
      auto ti_begin = thrust::make_transform_iterator(
        thrust::make_zip_iterator(
          thrust::make_tuple(
            thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)), thrust::placeholders::_1 % P.nColumns_),
            thrust::make_permutation_iterator(P.rowIndices_.begin(),
                                              thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)),
                                                                              thrust::placeholders::_1 / P.nColumns_)),
            thrust::make_permutation_iterator(P.columnIndices_.begin(),
                                              thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)),
                                                                              thrust::placeholders::_1 / P.nColumns_)))),
        AreThereIndexPairs_02_and_01_Functor(thrust::raw_pointer_cast(P.rowIndices_.data()),
                                             thrust::raw_pointer_cast(P.columnIndices_.data()),
                                             P.columnIndices_.size(),
                                             isUndirected));
      
      // key_begin[.] = P.columnIndices_[. / P.nColumns_]
      auto key_begin = thrust::make_permutation_iterator(P.columnIndices_.begin(),
                                                         thrust::make_transform_iterator(
                                                           thrust::make_counting_iterator(NrPoints(0)),
                                                           thrust::placeholders::_1 / P.nColumns_));
      
      auto it_pair = thrust::make_pair(c.begin(), e.begin());
      
      for (int64_t i = 0, length = 1;
          it_pair.second != e.end() && i < P.nColumns_ * P.columnIndices_.size() && length != 0;
          i += length, ti_begin += length, key_begin += length) {
        
        length = thrust::minimum<int64_t>()(P.nColumns_ * P.columnIndices_.size() - i, e.end() - it_pair.second);
        
        it_pair = thrust::reduce_by_key(typename ThrustSystem<A>::execution_policy(),
                                        key_begin, key_begin + length,
                                        ti_begin,
                                        it_pair.first,
                                        it_pair.second);
      }
      
      thrust::inclusive_scan_by_key(typename ThrustSystem<A>::execution_policy(),
                                    thrust::make_reverse_iterator(it_pair.first), c.rend(),
                                    thrust::make_reverse_iterator(it_pair.second),
                                    thrust::make_reverse_iterator(it_pair.second));
      
      auto new_end = thrust::unique_by_key(typename ThrustSystem<A>::execution_policy(),
                                           c.begin(), it_pair.first,
                                           e.begin());
      
      c.resize(new_end.first - c.begin());
      e.resize(new_end.second - e.begin());
      
      thrust::transform(typename ThrustSystem<A>::execution_policy(),
                        e.begin(), e.end(),
                        e.begin(),
                        thrust::placeholders::_1 / 2);
    } else {
      // ti_begin[.] = AreThereIndexPairs_02_and_01_Functor(. mod Psym.nColumns_,
      //                                                    Psym.rowIndices_[. / Psym.nColumns_],
      //                                                    Psym.columnIndices_[. / Psym.nColumns_],
      //                                                    isUndirected)
      // but constructor takes indices of original matrix as search field
      auto ti_begin = thrust::make_transform_iterator(
        thrust::make_zip_iterator(
          thrust::make_tuple(
            thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)), thrust::placeholders::_1 % Psym.nColumns_),
            thrust::make_permutation_iterator(Psym.rowIndices_.begin(),
                                              thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)),
                                                                              thrust::placeholders::_1 / Psym.nColumns_)),
            thrust::make_permutation_iterator(Psym.columnIndices_.begin(),
                                              thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)),
                                                                              thrust::placeholders::_1 / Psym.nColumns_)))),
        AreThereIndexPairs_02_and_01_Functor(thrust::raw_pointer_cast(P.rowIndices_.data()),
                                             thrust::raw_pointer_cast(P.columnIndices_.data()),
                                             P.columnIndices_.size(),
                                             isUndirected));
      
      // key_begin[.] = Psym.columnIndices_[. / Psym.nColumns_]
      auto key_begin = thrust::make_permutation_iterator(Psym.columnIndices_.begin(),
                                                         thrust::make_transform_iterator(
                                                           thrust::make_counting_iterator(NrPoints(0)),
                                                           thrust::placeholders::_1 / Psym.nColumns_));
      
      auto it_pair = thrust::make_pair(c.begin(), e.begin());
      
      for (int64_t i = 0, length = 1;
          it_pair.second != e.end() && i < Psym.nColumns_ * Psym.columnIndices_.size() && length != 0;
          i += length, ti_begin += length, key_begin += length) {
        
        length = thrust::minimum<int64_t>()(Psym.nColumns_ * Psym.columnIndices_.size() - i, e.end() - it_pair.second);
        
        it_pair = thrust::reduce_by_key(typename ThrustSystem<A>::execution_policy(),
                                        key_begin, key_begin + length,
                                        ti_begin,
                                        it_pair.first,
                                        it_pair.second);
      }
      
      thrust::inclusive_scan_by_key(typename ThrustSystem<A>::execution_policy(),
                                    thrust::make_reverse_iterator(it_pair.first), c.rend(),
                                    thrust::make_reverse_iterator(it_pair.second),
                                    thrust::make_reverse_iterator(it_pair.second));
      
      auto new_end = thrust::unique_by_key(typename ThrustSystem<A>::execution_policy(),
                                           c.begin(), it_pair.first,
                                           e.begin());
      
      c.resize(new_end.first - c.begin());
      e.resize(new_end.second - e.begin());
    }
    
    cudaThreadSynchronize(); // block until kernel is finished
    std::chrono::duration<double> tDiff = std::chrono::system_clock::now() - tStart;
    std::cout << "iteration over all possible triangles made out of existing pairs using reduce_by_key:" << std::endl << tDiff.count() << " s";
    std::cout << ",\t " << (thrust::equal(typename ThrustSystem<A>::execution_policy(), e.begin(), e.end(), test.begin()) ? "" : "not ") << "equal" << std::endl;
//     printVector<A>(e);
    // test.assign(e.begin(), e.end());
  }
// ---------------------------------------------------------------------------------------------------------------------------------------------
  
  
  
// iteration over all possible triangles made out of existing pairs using count_if -------------------------------------------------------------
  {
    // for each existing index pair (i,j) of the matrix's symmetric part, the triples (0,i,j), ..., (n-1,i,j) are progressed
    // a triple (k,i,j) is counted, if k != i != j != k and (k,j) and (k,i) are index pairs of the matrix (found out by binary a search)
    
    auto tStart = std::chrono::system_clock::now();
    
    typename ThrustSystem<A>::Vector<NrPoints> e(P.nColumns_);
    
    if (isUndirected) {
      // ti_begin[.] = AreThereIndexPairs_02_and_01_Functor(. mod P.nColumns_,
      //                                                    P.rowIndices_[. / P.nColumns_],
      //                                                    P.columnIndices_[. / P.nColumns_],
      //                                                    isUndirected)
      auto ti_begin = thrust::make_transform_iterator(
        thrust::make_zip_iterator(
          thrust::make_tuple(
            thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)), thrust::placeholders::_1 % P.nColumns_),
            thrust::make_permutation_iterator(P.rowIndices_.begin(),
                                              thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)),
                                                                              thrust::placeholders::_1 / P.nColumns_)),
            thrust::make_permutation_iterator(P.columnIndices_.begin(),
                                              thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)),
                                                                              thrust::placeholders::_1 / P.nColumns_)))),
        AreThereIndexPairs_02_and_01_Functor(thrust::raw_pointer_cast(P.rowIndices_.data()),
                                             thrust::raw_pointer_cast(P.columnIndices_.data()),
                                             P.columnIndices_.size(),
                                             isUndirected));
      
      for (int i = 0; i < P.nColumns_; ti_begin += (cscVector[i+1]-cscVector[i]) * P.nColumns_, ++i) {
        e[i] = thrust::count_if(typename ThrustSystem<A>::execution_policy(),
                                ti_begin, ti_begin + (cscVector[i+1]-cscVector[i]) * P.nColumns_,
                                thrust::identity<bool>());
      }
      
      thrust::transform(typename ThrustSystem<A>::execution_policy(),
                        e.begin(), e.end(),
                        e.begin(),
                        thrust::placeholders::_1 / 2);
    } else {
      // ti_begin[.] = AreThereIndexPairs_02_and_01_Functor(. mod Psym.nColumns_,
      //                                                    Psym.rowIndices_[. / Psym.nColumns_],
      //                                                    Psym.columnIndices_[. / Psym.nColumns_],
      //                                                    isUndirected)
      auto ti_begin = thrust::make_transform_iterator(
        thrust::make_zip_iterator(
          thrust::make_tuple(
            thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)), thrust::placeholders::_1 % Psym.nColumns_),
            thrust::make_permutation_iterator(Psym.rowIndices_.begin(),
                                              thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)),
                                                                              thrust::placeholders::_1 / Psym.nColumns_)),
            thrust::make_permutation_iterator(Psym.columnIndices_.begin(),
                                              thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)),
                                                                              thrust::placeholders::_1 / Psym.nColumns_)))),
        AreThereIndexPairs_02_and_01_Functor(thrust::raw_pointer_cast(P.rowIndices_.data()),
                                             thrust::raw_pointer_cast(P.columnIndices_.data()),
                                             P.columnIndices_.size(),
                                             isUndirected));
      
      for (int i = 0; i < Psym.nColumns_; ti_begin += (cscVectorSym[i+1]-cscVectorSym[i]) * Psym.nColumns_, ++i) {
        e[i] = thrust::count_if(typename ThrustSystem<A>::execution_policy(),
                                ti_begin, ti_begin + (cscVectorSym[i+1]-cscVectorSym[i]) * Psym.nColumns_,
                                thrust::identity<bool>());
      }
    }
    
    cudaThreadSynchronize(); // block until kernel is finished
    std::chrono::duration<double> tDiff = std::chrono::system_clock::now() - tStart;
    std::cout << "iteration over all possible triangles made out of existing pairs using count_if:" << std::endl << tDiff.count() << " s";
    std::cout << ",\t " << (thrust::equal(typename ThrustSystem<A>::execution_policy(), e.begin(), e.end(), test.begin()) ? "" : "not ") << "equal" << std::endl;
//     printVector<A>(e);
    // test.assign(e.begin(), e.end());
  }
// ---------------------------------------------------------------------------------------------------------------------------------------------
  
  
  
// iteration over (max{outDegrees}) possible triangles made out of existing pairs using reduce_by_key (binary search in full field in each thread) -------------
  {
    // if, for each column j, i_1, ..., i_n are the row indices, then the quadruplets (1<=n, i_1, i_1, j), (2<=n, i_2, i_1, j), ..., (maxOutDegree<=n, i_maxOutDegree, i_1, j), 
    // (1<=n, i_1, i_2, j), (2<=n, i_2, i_2, j), ..., (maxOutDegree<=n, i_maxOutDegree, i_2, j), ..., (maxOutDegree<=n, i_maxOutDegree, i_n, j) are progressed
    // for each quadruplet (b,k,i,j), 1 is added if b is true and k != i != j != k and (k,i) are index pairs of the matrix (found out by binary a search), 0 otherwise;
    // if isUndirected is true, the additional condition k<i must be true for a 1.
    
    auto tStart = std::chrono::system_clock::now();
    
    typename ThrustSystem<A>::Vector<NrPoints> e(ThrustSystem<A>::Memory::getFreeBytes(uint64_t(8) << 30) / sizeof(NrPoints) / 5);
    typename ThrustSystem<A>::Vector<NrBoxes> c(e.size());
    
    if (isUndirected) {
      // nnz_begin[.] = cscVector[. + 1] - cscVector[.]
      auto nnz_begin = makeBinaryTransformIterator(cscVector.begin() + 1, cscVector.begin(), thrust::minus<NrBoxes>());
      
      NrBoxes maxOutDegree = * thrust::max_element(typename ThrustSystem<A>::execution_policy(),
                                                   nnz_begin, nnz_begin + P.nColumns_);
      
      // delay_row_begin[.] = P.rowIndices_[. / maxOutDegree]
      auto delay_row_begin = thrust::make_permutation_iterator(
        P.rowIndices_.begin(),
        thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)),
                                        thrust::placeholders::_1 / NrPoints(maxOutDegree)));
      
      // delay_column_begin[.] = P.columnIndices_[. / maxOutDegree]
      auto delay_column_begin = thrust::make_permutation_iterator(
        P.columnIndices_.begin(),
        thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)),
                                        thrust::placeholders::_1 / NrPoints(maxOutDegree)));
      
      // specific_row_begin[.] = P.rowIndices_[(cscVector[delay_column_begin[.]] + (. mod maxOutDegree)) mod nnzMatrix]
      auto specific_row_begin = thrust::make_permutation_iterator(
        P.rowIndices_.begin(),
        thrust::make_transform_iterator(
          makeBinaryTransformIterator(
            thrust::make_permutation_iterator(cscVector.begin(), delay_column_begin),
            thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)), thrust::placeholders::_1 % NrPoints(maxOutDegree)),
            thrust::plus<NrPoints>()),
          thrust::placeholders::_1 % NrPoints(P.rowIndices_.size())));
      
      // check_overhang_begin[.] = (. mod maxOutDegree) < nnz_begin[delay_column_begin[.]]
      auto check_overhang_begin = makeBinaryTransformIterator(
        thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)), thrust::placeholders::_1 % NrPoints(maxOutDegree)),
        thrust::make_permutation_iterator(nnz_begin, delay_column_begin),
        thrust::less<NrPoints>());
      
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
        IsThereIndexPair_12_Functor(thrust::raw_pointer_cast(P.rowIndices_.data()),
                                    thrust::raw_pointer_cast(P.columnIndices_.data()),
                                    P.columnIndices_.size(),
                                    isUndirected));
      
      auto it_pair = thrust::make_pair(c.begin(), e.begin());
      
      for (int64_t i = 0, length = 1;
          it_pair.second != e.end() && i < maxOutDegree * P.columnIndices_.size() && length != 0;
          i += length, ti_begin += length, delay_column_begin += length) {
        
        length = thrust::minimum<int64_t>()(maxOutDegree * P.columnIndices_.size() - i, e.end() - it_pair.second);
        
        it_pair = thrust::reduce_by_key(typename ThrustSystem<A>::execution_policy(),
                                        delay_column_begin, delay_column_begin + length,
                                        ti_begin,
                                        it_pair.first,
                                        it_pair.second);
      }
      
      thrust::inclusive_scan_by_key(typename ThrustSystem<A>::execution_policy(),
                                    thrust::make_reverse_iterator(it_pair.first), c.rend(),
                                    thrust::make_reverse_iterator(it_pair.second),
                                    thrust::make_reverse_iterator(it_pair.second));
      
      auto new_end = thrust::unique_by_key(typename ThrustSystem<A>::execution_policy(),
                                           c.begin(), it_pair.first,
                                           e.begin());
      
      c.resize(new_end.first - c.begin());
      e.resize(new_end.second - e.begin());
    } else {
      // nnz_begin[.] = cscVectorSym[. + 1] - cscVectorSym[.]
      auto nnz_begin = makeBinaryTransformIterator(cscVectorSym.begin() + 1, cscVectorSym.begin(), thrust::minus<NrBoxes>());
      
      NrBoxes maxOutDegree = * thrust::max_element(typename ThrustSystem<A>::execution_policy(),
                                                   nnz_begin, nnz_begin + Psym.nColumns_);
      
      // delay_row_begin[.] = Psym.rowIndices_[. / maxOutDegree]
      auto delay_row_begin = thrust::make_permutation_iterator(
        Psym.rowIndices_.begin(),
        thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)),
                                        thrust::placeholders::_1 / NrPoints(maxOutDegree)));
      
      // delay_column_begin[.] = Psym.columnIndices_[. / maxOutDegree]
      auto delay_column_begin = thrust::make_permutation_iterator(
        Psym.columnIndices_.begin(),
        thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)),
                                        thrust::placeholders::_1 / NrPoints(maxOutDegree)));
      
      // specific_row_begin[.] = Psym.rowIndices_[(cscVectorSym[delay_column_begin[.]] + (. mod maxOutDegree)) mod nnzMatrix]
      auto specific_row_begin = thrust::make_permutation_iterator(
        Psym.rowIndices_.begin(),
        thrust::make_transform_iterator(
          makeBinaryTransformIterator(
            thrust::make_permutation_iterator(cscVectorSym.begin(), delay_column_begin),
            thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)), thrust::placeholders::_1 % NrPoints(maxOutDegree)),
            thrust::plus<NrPoints>()),
          thrust::placeholders::_1 % NrPoints(Psym.rowIndices_.size())));
      
      // check_overhang_begin[.] = (. mod maxOutDegree) < nnz_begin[delay_column_begin[.]]
      auto check_overhang_begin = makeBinaryTransformIterator(
        thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)), thrust::placeholders::_1 % NrPoints(maxOutDegree)),
        thrust::make_permutation_iterator(nnz_begin, delay_column_begin),
        thrust::less<NrPoints>());
      
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
        IsThereIndexPair_12_Functor(thrust::raw_pointer_cast(P.rowIndices_.data()),
                                    thrust::raw_pointer_cast(P.columnIndices_.data()),
                                    P.columnIndices_.size(),
                                    isUndirected));
      
      auto it_pair = thrust::make_pair(c.begin(), e.begin());
      
      for (int64_t i = 0, length = 1;
          it_pair.second != e.end() && i < maxOutDegree * Psym.columnIndices_.size() && length != 0;
          i += length, ti_begin += length, delay_column_begin += length) {
        
        length = thrust::minimum<int64_t>()(maxOutDegree * Psym.columnIndices_.size() - i, e.end() - it_pair.second);
        
        it_pair = thrust::reduce_by_key(typename ThrustSystem<A>::execution_policy(),
                                        delay_column_begin, delay_column_begin + length,
                                        ti_begin,
                                        it_pair.first,
                                        it_pair.second);
      }
      
      thrust::inclusive_scan_by_key(typename ThrustSystem<A>::execution_policy(),
                                    thrust::make_reverse_iterator(it_pair.first), c.rend(),
                                    thrust::make_reverse_iterator(it_pair.second),
                                    thrust::make_reverse_iterator(it_pair.second));
      
      auto new_end = thrust::unique_by_key(typename ThrustSystem<A>::execution_policy(),
                                           c.begin(), it_pair.first,
                                           e.begin());
      
      c.resize(new_end.first - c.begin());
      e.resize(new_end.second - e.begin());
    }
    
    cudaThreadSynchronize(); // block until kernel is finished
    std::chrono::duration<double> tDiff = std::chrono::system_clock::now() - tStart;
    std::cout << "iteration over (max{outDegrees}) possible triangles made out of existing pairs using reduce_by_key (binary search in full field in each thread):" << std::endl << tDiff.count() << " s";
    std::cout << ",\t " << (thrust::equal(typename ThrustSystem<A>::execution_policy(), e.begin(), e.end(), test.begin()) ? "" : "not ") << "equal" << std::endl;
//     printVector<A>(e);
    // test.assign(e.begin(), e.end());
  }
// ---------------------------------------------------------------------------------------------------------------------------------------------
  
  
  
// iteration over (max{outDegrees}) possible triangles made out of existing pairs using reduce_by_key (binary search in segmented field in each thread) -------------
  {
    // if, for each column j, i_1, ..., i_n are the row indices, then the sextuples (1<=n, i_1, i_1, j, ., .), (2<=n, i_2, i_1, j, ., .), ..., (maxOutDegree<=n, i_maxOutDegree, i_1, j, ., .), 
    // (1<=n, i_1, i_2, j, ., .), (2<=n, i_2, i_2, j, ., .), ..., (maxOutDegree<=n, i_maxOutDegree, i_2, j, ., .), ..., (maxOutDegree<=n, i_maxOutDegree, i_n, j, ., .) are progressed
    // for each sextuplet (b,k,i,j,r1,r2), 1 is added if b is true and k != i != j != k and k is a row index (found out by binary a search) in rowIndices_begin_+[r1,r2) that is covering the row indices of column i, 0 otherwise;
    // if isUndirected is true, the additional condition k<i must be true for a 1.
    
    auto tStart = std::chrono::system_clock::now();
    
    typename ThrustSystem<A>::Vector<NrPoints> e(ThrustSystem<A>::Memory::getFreeBytes(uint64_t(8) << 30) / sizeof(NrPoints) / 5);
    typename ThrustSystem<A>::Vector<NrBoxes> c(e.size());
    
    if (isUndirected) {
      // nnz_begin[.] = cscVector[. + 1] - cscVector[.]
      auto nnz_begin = makeBinaryTransformIterator(cscVector.begin() + 1, cscVector.begin(), thrust::minus<NrBoxes>());
      
      NrBoxes maxOutDegree = * thrust::max_element(typename ThrustSystem<A>::execution_policy(),
                                                   nnz_begin, nnz_begin + P.nColumns_);
      
      // delay_row_begin[.] = P.rowIndices_[. / maxOutDegree]
      auto delay_row_begin = thrust::make_permutation_iterator(
        P.rowIndices_.begin(),
        thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)),
                                        thrust::placeholders::_1 / NrPoints(maxOutDegree)));
      
      // delay_column_begin[.] = P.columnIndices_[. / maxOutDegree]
      auto delay_column_begin = thrust::make_permutation_iterator(
        P.columnIndices_.begin(),
        thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)),
                                        thrust::placeholders::_1 / NrPoints(maxOutDegree)));
      
      // specific_row_begin[.] = P.rowIndices_[(cscVector[delay_column_begin[.]] + (. mod maxOutDegree)) mod nnzMatrix]
      auto specific_row_begin = thrust::make_permutation_iterator(
        P.rowIndices_.begin(),
        thrust::make_transform_iterator(
          makeBinaryTransformIterator(
            thrust::make_permutation_iterator(cscVector.begin(), delay_column_begin),
            thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)), thrust::placeholders::_1 % NrPoints(maxOutDegree)),
            thrust::plus<NrPoints>()),
          thrust::placeholders::_1 % NrPoints(P.rowIndices_.size())));
      
      // check_overhang_begin[.] = (. mod maxOutDegree) < nnz_begin[delay_column_begin[.]]
      auto check_overhang_begin = makeBinaryTransformIterator(
        thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)), thrust::placeholders::_1 % NrPoints(maxOutDegree)),
        thrust::make_permutation_iterator(nnz_begin, delay_column_begin),
        thrust::less<NrPoints>());
      
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
        Is_1_RowIndex_Functor(thrust::raw_pointer_cast(P.rowIndices_.data()),
                              isUndirected));
      
      auto it_pair = thrust::make_pair(c.begin(), e.begin());
      
      for (int64_t i = 0, length = 1;
          it_pair.second != e.end() && i < maxOutDegree * P.columnIndices_.size() && length != 0;
          i += length, ti_begin += length, delay_column_begin += length) {
        
        length = thrust::minimum<int64_t>()(maxOutDegree * P.columnIndices_.size() - i, e.end() - it_pair.second);
        
        it_pair = thrust::reduce_by_key(typename ThrustSystem<A>::execution_policy(),
                                        delay_column_begin, delay_column_begin + length,
                                        ti_begin,
                                        it_pair.first,
                                        it_pair.second);
      }
      
      thrust::inclusive_scan_by_key(typename ThrustSystem<A>::execution_policy(),
                                    thrust::make_reverse_iterator(it_pair.first), c.rend(),
                                    thrust::make_reverse_iterator(it_pair.second),
                                    thrust::make_reverse_iterator(it_pair.second));
      
      auto new_end = thrust::unique_by_key(typename ThrustSystem<A>::execution_policy(),
                                           c.begin(), it_pair.first,
                                           e.begin());
      
      c.resize(new_end.first - c.begin());
      e.resize(new_end.second - e.begin());
    } else {
      // nnz_begin[.] = cscVectorSym[. + 1] - cscVectorSym[.]
      auto nnz_begin = makeBinaryTransformIterator(cscVectorSym.begin() + 1, cscVectorSym.begin(), thrust::minus<NrBoxes>());
      
      NrBoxes maxOutDegree = * thrust::max_element(typename ThrustSystem<A>::execution_policy(),
                                                   nnz_begin, nnz_begin + Psym.nColumns_);
      
      // delay_row_begin[.] = Psym.rowIndices_[. / maxOutDegree]
      auto delay_row_begin = thrust::make_permutation_iterator(
        Psym.rowIndices_.begin(),
        thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)),
                                        thrust::placeholders::_1 / NrPoints(maxOutDegree)));
      
      // delay_column_begin[.] = Psym.columnIndices_[. / maxOutDegree]
      auto delay_column_begin = thrust::make_permutation_iterator(
        Psym.columnIndices_.begin(),
        thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)),
                                        thrust::placeholders::_1 / NrPoints(maxOutDegree)));
      
      // specific_row_begin[.] = Psym.rowIndices_[(cscVectorSym[delay_column_begin[.]] + (. mod maxOutDegree)) mod nnzMatrix]
      auto specific_row_begin = thrust::make_permutation_iterator(
        Psym.rowIndices_.begin(),
        thrust::make_transform_iterator(
          makeBinaryTransformIterator(
            thrust::make_permutation_iterator(cscVectorSym.begin(), delay_column_begin),
            thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)), thrust::placeholders::_1 % NrPoints(maxOutDegree)),
            thrust::plus<NrPoints>()),
          thrust::placeholders::_1 % NrPoints(Psym.rowIndices_.size())));
      
      // check_overhang_begin[.] = (. mod maxOutDegree) < nnz_begin[delay_column_begin[.]]
      auto check_overhang_begin = makeBinaryTransformIterator(
        thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)), thrust::placeholders::_1 % NrPoints(maxOutDegree)),
        thrust::make_permutation_iterator(nnz_begin, delay_column_begin),
        thrust::less<NrPoints>());
      
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
        Is_1_RowIndex_Functor(thrust::raw_pointer_cast(P.rowIndices_.data()),
                              isUndirected));
      
      auto it_pair = thrust::make_pair(c.begin(), e.begin());
      
      for (int64_t i = 0, length = 1;
          it_pair.second != e.end() && i < maxOutDegree * Psym.columnIndices_.size() && length != 0;
          i += length, ti_begin += length, delay_column_begin += length) {
        
        length = thrust::minimum<int64_t>()(maxOutDegree * Psym.columnIndices_.size() - i, e.end() - it_pair.second);
        
        it_pair = thrust::reduce_by_key(typename ThrustSystem<A>::execution_policy(),
                                        delay_column_begin, delay_column_begin + length,
                                        ti_begin,
                                        it_pair.first,
                                        it_pair.second);
      }
      
      thrust::inclusive_scan_by_key(typename ThrustSystem<A>::execution_policy(),
                                    thrust::make_reverse_iterator(it_pair.first), c.rend(),
                                    thrust::make_reverse_iterator(it_pair.second),
                                    thrust::make_reverse_iterator(it_pair.second));
      
      auto new_end = thrust::unique_by_key(typename ThrustSystem<A>::execution_policy(),
                                           c.begin(), it_pair.first,
                                           e.begin());
      
      c.resize(new_end.first - c.begin());
      e.resize(new_end.second - e.begin());
    }
    
    cudaThreadSynchronize(); // block until kernel is finished
    std::chrono::duration<double> tDiff = std::chrono::system_clock::now() - tStart;
    std::cout << "iteration over (max{outDegrees}) possible triangles made out of existing pairs using reduce_by_key (binary search in segmented field in each thread):" << std::endl << tDiff.count() << " s";
    std::cout << ",\t " << (thrust::equal(typename ThrustSystem<A>::execution_policy(), e.begin(), e.end(), test.begin()) ? "" : "not ") << "equal" << std::endl;
//     printVector<A>(e);
    // test.assign(e.begin(), e.end());
  }
// ---------------------------------------------------------------------------------------------------------------------------------------------
  
  
  
// iteration over (max{outDegrees}) possible triangles made out of existing pairs using reduce_by_key (vectorized binary search in full field) ------------
  {
    //TODO
    
    auto tStart = std::chrono::system_clock::now();
    
    typename ThrustSystem<A>::Vector<NrPoints> e(ThrustSystem<A>::Memory::getFreeBytes(uint64_t(8) << 30) / sizeof(NrPoints) / 5);
    typename ThrustSystem<A>::Vector<NrBoxes> c(e.size());
    typename ThrustSystem<A>::Vector<bool> b(e.size());
    
    if (isUndirected) {
      auto inds_begin = thrust::make_zip_iterator(
        thrust::make_tuple(
          P.rowIndices_.begin(),
          P.columnIndices_.begin()));
      
      auto inds_end = thrust::make_zip_iterator(
        thrust::make_tuple(
          P.rowIndices_.end(),
          P.columnIndices_.end()));
      
      // nnz_begin[.] = cscVector[. + 1] - cscVector[.]
      auto nnz_begin = makeBinaryTransformIterator(cscVector.begin() + 1, cscVector.begin(), thrust::minus<NrBoxes>());
      
      NrBoxes maxOutDegree = * thrust::max_element(typename ThrustSystem<A>::execution_policy(),
                                                   nnz_begin, nnz_begin + P.nColumns_);
      
      // delay_row_begin[.] = P.rowIndices_[. / maxOutDegree]
      auto delay_row_begin = thrust::make_permutation_iterator(
        P.rowIndices_.begin(),
        thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)),
                                        thrust::placeholders::_1 / NrPoints(maxOutDegree)));
      
      // delay_column_begin[.] = P.columnIndices_[. / maxOutDegree]
      auto delay_column_begin = thrust::make_permutation_iterator(
        P.columnIndices_.begin(),
        thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)),
                                        thrust::placeholders::_1 / NrPoints(maxOutDegree)));
      
      // specific_row_begin[.] = P.rowIndices_[(cscVector[delay_column_begin[.]] + (. mod maxOutDegree)) mod nnzMatrix]
      auto specific_row_begin = thrust::make_permutation_iterator(
        P.rowIndices_.begin(),
        thrust::make_transform_iterator(
          makeBinaryTransformIterator(
            thrust::make_permutation_iterator(cscVector.begin(), delay_column_begin),
            thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)), thrust::placeholders::_1 % NrPoints(maxOutDegree)),
            thrust::plus<NrPoints>()),
          thrust::placeholders::_1 % NrPoints(P.rowIndices_.size())));
      
      // check_overhang_begin[.] = (. mod maxOutDegree) < nnz_begin[delay_column_begin[.]]
      auto check_overhang_begin = makeBinaryTransformIterator(
        thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)), thrust::placeholders::_1 % NrPoints(maxOutDegree)),
        thrust::make_permutation_iterator(nnz_begin, delay_column_begin),
        thrust::less<NrPoints>());
      
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
      
      for (int64_t i = 0, length = 1;
          it_pair.second != e.end() && i < maxOutDegree * P.columnIndices_.size() && length != 0;
          i += length, ti_begin += length, delay_column_begin += length, check_overhang_begin += length, pairs_begin += length) {
        
        length = thrust::minimum<int64_t>()(maxOutDegree * P.columnIndices_.size() - i, e.end() - it_pair.second);
        
        thrust::binary_search(typename ThrustSystem<A>::execution_policy(),
                              inds_begin, inds_end,
                              pairs_begin, pairs_begin + length,
                              b.begin(),
                              ColumnMajorOrderingFunctor());
        
        it_pair = thrust::reduce_by_key(typename ThrustSystem<A>::execution_policy(),
                                        delay_column_begin, delay_column_begin + length,
                                        thrust::make_transform_iterator(
                                          thrust::make_zip_iterator(
                                            thrust::make_tuple(
                                              b.begin(), check_overhang_begin, ti_begin)),
                                          AreAllTrue_Functor()),
                                        it_pair.first,
                                        it_pair.second,
                                        thrust::equal_to<NrBoxes>(),
                                        thrust::plus<NrPoints>());
      }
      
      thrust::inclusive_scan_by_key(typename ThrustSystem<A>::execution_policy(),
                                    thrust::make_reverse_iterator(it_pair.first), c.rend(),
                                    thrust::make_reverse_iterator(it_pair.second),
                                    thrust::make_reverse_iterator(it_pair.second));
      
      auto new_end = thrust::unique_by_key(typename ThrustSystem<A>::execution_policy(),
                                          c.begin(), it_pair.first,
                                          e.begin());
      
      c.resize(new_end.first - c.begin());
      e.resize(new_end.second - e.begin());
    } else {
      auto inds_begin = thrust::make_zip_iterator(
        thrust::make_tuple(
          P.rowIndices_.begin(),
          P.columnIndices_.begin()));
      
      auto inds_end = thrust::make_zip_iterator(
        thrust::make_tuple(
          P.rowIndices_.end(),
          P.columnIndices_.end()));
      
      // nnz_begin[.] = cscVectorSym[. + 1] - cscVectorSym[.]
      auto nnz_begin = makeBinaryTransformIterator(cscVectorSym.begin() + 1, cscVectorSym.begin(), thrust::minus<NrBoxes>());
      
      NrBoxes maxOutDegree = * thrust::max_element(typename ThrustSystem<A>::execution_policy(),
                                                   nnz_begin, nnz_begin + Psym.nColumns_);
      
      // delay_row_begin[.] = Psym.rowIndices_[. / maxOutDegree]
      auto delay_row_begin = thrust::make_permutation_iterator(
        Psym.rowIndices_.begin(),
        thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)),
                                        thrust::placeholders::_1 / NrPoints(maxOutDegree)));
      
      // delay_column_begin[.] = Psym.columnIndices_[. / maxOutDegree]
      auto delay_column_begin = thrust::make_permutation_iterator(
        Psym.columnIndices_.begin(),
        thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)),
                                        thrust::placeholders::_1 / NrPoints(maxOutDegree)));
      
      // specific_row_begin[.] = Psym.rowIndices_[(cscVectorSym[delay_column_begin[.]] + (. mod maxOutDegree)) mod nnzMatrix]
      auto specific_row_begin = thrust::make_permutation_iterator(
        Psym.rowIndices_.begin(),
        thrust::make_transform_iterator(
          makeBinaryTransformIterator(
            thrust::make_permutation_iterator(cscVectorSym.begin(), delay_column_begin),
            thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)), thrust::placeholders::_1 % NrPoints(maxOutDegree)),
            thrust::plus<NrPoints>()),
          thrust::placeholders::_1 % NrPoints(Psym.rowIndices_.size())));
      
      // check_overhang_begin[.] = (. mod maxOutDegree) < nnz_begin[delay_column_begin[.]]
      auto check_overhang_begin = makeBinaryTransformIterator(
        thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)), thrust::placeholders::_1 % NrPoints(maxOutDegree)),
        thrust::make_permutation_iterator(nnz_begin, delay_column_begin),
        thrust::less<NrPoints>());
      
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
      
      for (int64_t i = 0, length = 1;
          it_pair.second != e.end() && i < maxOutDegree * Psym.columnIndices_.size() && length != 0;
          i += length, ti_begin += length, delay_column_begin += length, check_overhang_begin += length, pairs_begin += length) {
        
        length = thrust::minimum<int64_t>()(maxOutDegree * Psym.columnIndices_.size() - i, e.end() - it_pair.second);
        
        thrust::binary_search(typename ThrustSystem<A>::execution_policy(),
                              inds_begin, inds_end,
                              pairs_begin, pairs_begin + length,
                              b.begin(),
                              ColumnMajorOrderingFunctor());
        
        it_pair = thrust::reduce_by_key(typename ThrustSystem<A>::execution_policy(),
                                        delay_column_begin, delay_column_begin + length,
                                        thrust::make_transform_iterator(
                                          thrust::make_zip_iterator(
                                            thrust::make_tuple(
                                              b.begin(), check_overhang_begin, ti_begin)),
                                          AreAllTrue_Functor()),
                                        it_pair.first,
                                        it_pair.second,
                                        thrust::equal_to<NrBoxes>(),
                                        thrust::plus<NrPoints>());
      }
      
      thrust::inclusive_scan_by_key(typename ThrustSystem<A>::execution_policy(),
                                    thrust::make_reverse_iterator(it_pair.first), c.rend(),
                                    thrust::make_reverse_iterator(it_pair.second),
                                    thrust::make_reverse_iterator(it_pair.second));
      
      auto new_end = thrust::unique_by_key(typename ThrustSystem<A>::execution_policy(),
                                          c.begin(), it_pair.first,
                                          e.begin());
      
      c.resize(new_end.first - c.begin());
      e.resize(new_end.second - e.begin());
    }
    
    cudaThreadSynchronize(); // block until kernel is finished
    std::chrono::duration<double> tDiff = std::chrono::system_clock::now() - tStart;
    std::cout << "iteration over (max{outDegrees}) possible triangles made out of existing pairs using reduce_by_key (vectorized binary search in full field):" << std::endl << tDiff.count() << " s";
    std::cout << ",\t " << (thrust::equal(typename ThrustSystem<A>::execution_policy(), e.begin(), e.end(), test.begin()) ? "" : "not ") << "equal" << std::endl;
//     printVector<A>(e);
    // test.assign(e.begin(), e.end());
  }
// ---------------------------------------------------------------------------------------------------------------------------------------------
  
  return 0;
}