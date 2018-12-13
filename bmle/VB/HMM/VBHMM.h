#ifndef VBFACTORANALYSERMIXTURE_H
#define VBFACTORANALYSERMIXTURE_H
//
//
//
#include <limits>
#include <vector>
#include <random>
#include <math.h>
#include <chrono>
#include <memory>
#include <tuple>
//#include <cmath.h>
// Eigen
#include <Eigen/Core>
#include <Eigen/Eigen>
//
// ITK
//
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIterator.h>
#include <itkNiftiImageIO.h>
#include <itkOrientImageFilter.h>
#include <itkSpatialOrientation.h>
//
//#include "EM.h"
#include "VBPosterior.h"
//
//
//
/** \class VBFactorAnalyserMixture
 *
 * \brief  Expaectation-Maximization algorithm
 * 
 * Dim is the number of dimensions
 * input:
 *   Y [p x n]: entry data. the matrix is supposed to be normalized.
 *              p = Dim, 
 *              n is the size of the dataset
 *              k < Dim dimension reduction
 * 
 */
template< int Dim  >
class VBFactorAnalyserMixture
{
 
 public:
  /** Constructor. */
  explicit VBFactorAnalyserMixture( const std::list< Eigen::Matrix< double, Dim, 1 > >& );
    
  /** Destructor */
  virtual ~VBFactorAnalyserMixture(){};

  //
  // Accessors
  // posterior probabilities

  //
  // Functions
  // main algorithn
  void   ExpectationMaximization();


  //
  // Accessors
  using Var_post = std::tuple< VP_hyper<Dim>, VP_qlambs<Dim>, VP_qxisi<Dim>, VP_qnu<Dim>, VP_qsi<Dim>  >;
  enum Vpost {HYPER, QLAMBS, QXISI, QNU, QSI};

 private:
  //
  // private member function
  //

  //
  // Data set size
  std::size_t n_{0};
  // Data set
  std::vector< Eigen::Matrix< double, Dim, 1 > >  Y_; 

  //
  // variational posteriors and hyper parameters
  Var_post variational_posteriors_;

  //
  // log marginal likelihood lower bound
  double F_{-1.e-36};
  std::list< double > F_history_;
};

//
//
//
template < int Dim  >
VBFactorAnalyserMixture< Dim >::VBFactorAnalyserMixture( const std::list< Eigen::Matrix< double, Dim, 1 > >& Y ):
n_{Y.size()}
{
  //
  //
  Y_ = std::vector< Eigen::Matrix< double, Dim, 1 > >( std::begin(Y), std::end(Y) );
  
  //
  //
  variational_posteriors_ = std::make_tuple( VP_hyper <Dim>( Y_ ),
					     VP_qlambs<Dim>( Y_ ),
					     VP_qxisi <Dim>( Y_ ),
					     VP_qnu   <Dim>( Y_ ),
					     VP_qsi   <Dim>( Y_ ) );
  //
  // Initialization
  std::get< QXISI >(variational_posteriors_).Expectation( variational_posteriors_ );
  std::get< QSI   >(variational_posteriors_).Expectation( variational_posteriors_ );
  std::get< QNU   >(variational_posteriors_).Expectation( variational_posteriors_ );
  // Lower bound history
  F_history_.push_back( F_ );
}
//
//
//
template < int Dim  > void
VBFactorAnalyserMixture< Dim >::ExpectationMaximization()
{
  double
    dF    =  1.e06,
    F_old = -1.e-40;
  while ( fabs(dF) > 1.e-3  )
    {
      //
      // E step
      std::get< QXISI  >(variational_posteriors_).Expectation( variational_posteriors_ );
      std::get< QLAMBS >(variational_posteriors_).Expectation( variational_posteriors_ );
      std::get< QSI    >(variational_posteriors_).Expectation( variational_posteriors_ );
      std::get< QNU    >(variational_posteriors_).Expectation( variational_posteriors_ );
      std::get< HYPER  >(variational_posteriors_).Expectation( variational_posteriors_ );

      //
      //
      F_old = F_;
      // Formaula (4.29)
      F_    = 0.;
      F_   += std::get< HYPER  >(variational_posteriors_).get_F();
      F_   += std::get< QNU    >(variational_posteriors_).get_F();
      F_   += std::get< QXISI  >(variational_posteriors_).get_F();
      //
      dF    = F_ - F_old;
      F_history_.push_back( F_ );
    }
}
#endif
