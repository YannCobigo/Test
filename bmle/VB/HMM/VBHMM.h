#ifndef VBHiddenMarkovModel_H
#define VBHiddenMarkovModel_H
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
#include "VBPosterior.h"
//
//
//
namespace VB
{
  namespace HMM
  {
    /** \class Hidden_Markov_Model
     *
     * \brief  Expaectation-Maximization algorithm
     * 
     * Dim is the number of dimensions
     * input:
     *   Dim: dimension of the measure
     *   S: number of state accessible to the Markov Chain
     *   n: number of cases (subjects)
     *   Y [p x n]: entry data. the matrix is supposed to be normalized.
     *              p = Dim, 
     *              n is the size of the dataset
     * 
     */
    template< int Dim, int S >
      class Hidden_Markov_Model
    {
 
    public:
      /** Constructor. */
      explicit Hidden_Markov_Model( const std::vector< std::vector< Eigen::Matrix< double, Dim, 1 > > >& );
    
      /** Destructor */
      virtual ~Hidden_Markov_Model(){};

      //
      // Accessors
      // posterior probabilities

      //
      // Functions
      // main algorithn
      void   ExpectationMaximization();


      //
      // Accessors
      using Var_post = std::tuple< 
	VB::HMM::VP_qsi<Dim,S>, 
	VB::HMM::VP_qdch<Dim,S>, 
	VB::HMM::VP_qgau<Dim,S>  >;
      enum Vpost {QSI,QDCH,QGAU};

    private:
      //
      // private member function
      //

      //
      // Data set size
      std::size_t n_{0};
      // Data set
      std::vector< std::vector< Eigen::Matrix< double, Dim, 1 > > >  Y_; 

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
    template < int Dim, int S >
      Hidden_Markov_Model< Dim, S >::Hidden_Markov_Model( const std::vector< std::vector< Eigen::Matrix< double, Dim, 1 > > >& Y ):
      Y_{Y}, n_{Y.size()}
    {
      //
      //
      variational_posteriors_ = std::make_tuple( VB::HMM::VP_qsi <Dim,S>( Y_ ),
						 VB::HMM::VP_qdch<Dim,S>( Y_ ),
						 VB::HMM::VP_qgau<Dim,S>( Y_ ) );
      //
      // Initialization
      std::get< QSI  >(variational_posteriors_).Expectation( variational_posteriors_ );
      std::get< QDCH >(variational_posteriors_).Expectation( variational_posteriors_ );
      std::get< QGAU >(variational_posteriors_).Expectation( variational_posteriors_ );
      // Lower bound history
      F_history_.push_back( F_ );
    }
    //
    //
    //
    template < int Dim, int S > void
      Hidden_Markov_Model< Dim, S >::ExpectationMaximization()
      {
	double
	  dF    =  1.e06,
	  F_old = -1.e-40;
	while ( fabs(dF) > 1.e-3  )
	  {
	    //
	    // M step
	    std::get< QSI  >(variational_posteriors_).Maximization( variational_posteriors_ );
	    std::get< QDCH >(variational_posteriors_).Maximization( variational_posteriors_ );
	    std::get< QGAU >(variational_posteriors_).Maximization( variational_posteriors_ );
	    //
	    // E step
	    std::get< QSI  >(variational_posteriors_).Expectation( variational_posteriors_ );
	    std::get< QDCH >(variational_posteriors_).Expectation( variational_posteriors_ );
	    std::get< QGAU >(variational_posteriors_).Expectation( variational_posteriors_ );

	    //
	    //
	    F_old = F_;
	    // Formaula (4.29)
	    F_    = 0.;
	    F_   += std::get< QSI  >(variational_posteriors_).get_F();
	    F_   += std::get< QDCH >(variational_posteriors_).get_F();
	    F_   += std::get< QGAU >(variational_posteriors_).get_F();
	    //
	    dF    = F_ - F_old;
	    std::cout << dF << std::endl;
	    F_history_.push_back( F_ );
	  }
      }
  }
}
#endif
