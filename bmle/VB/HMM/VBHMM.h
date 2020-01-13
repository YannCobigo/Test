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
     * \brief  Expectation-Maximization algorithm
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
      explicit Hidden_Markov_Model( const std::vector< std::vector< Eigen::Matrix< double, Dim, 1 > > >& ,
				    const std::vector< std::vector< Eigen::Matrix< double, 1, 1 > > >& );
    
      /** Destructor */
      ~Hidden_Markov_Model(){};

      //
      // Accessors
      // posterior probabilities

      //
      // Functions
      // main algorithn
      void   ExpectationMaximization();



    private:
      //
      // private member function
      //

      //
      // Data set size
      std::size_t n_{0};
      // Data set
      std::vector< std::vector< Eigen::Matrix< double, Dim, 1 > > >  Y_; 
      std::vector< std::vector< Eigen::Matrix< double, 1, 1 > > >  Age_; 

      //
      // variational posteriors and hyper parameters
      //rm      Var_post variational_posteriors_;
      std::shared_ptr< VB::HMM::VP_qsi <Dim,S> > qsi_;
      std::shared_ptr< VB::HMM::VP_qdch<Dim,S> > qdch_;
      std::shared_ptr< VB::HMM::VP_qgau<Dim,S> > qgau_;

      //
      // log marginal likelihood lower bound
      double F_{-1.e-36};
      std::list< double > F_history_;
    };

    //
    //
    //
    template < int Dim, int S >
      Hidden_Markov_Model< Dim, S >::Hidden_Markov_Model( const std::vector< std::vector< Eigen::Matrix< double, Dim, 1 > > >& Y,
							  const std::vector< std::vector< Eigen::Matrix< double, 1, 1 > > >& Age ):
      Y_{Y}, Age_{Age}, n_{Y.size()}
    {
      //
      //
      qsi_  = std::make_shared< VB::HMM::VP_qsi <Dim,S> >( Y_ );
      qdch_ = std::make_shared< VB::HMM::VP_qdch<Dim,S> >( qsi_, Y_ );
      qgau_ = std::make_shared< VB::HMM::VP_qgau<Dim,S> >( qsi_, Y_ );
      // set dependencies
      qsi_->set(qdch_,qgau_);
      qdch_->set(qsi_);
      qgau_->set(qsi_,qdch_);
      // Initialization
      qsi_->Expectation();
      qdch_->Expectation();
      qgau_->Expectation();
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
	int iteration = 0;
	while ( fabs(dF) > 1.e-6 )
	  {
	    std::cout << "Begining iteration: " << ++iteration << std::endl;
	    //
	    // M step
	    qsi_->Maximization();
	    qdch_->Maximization();
	    qgau_->Maximization();
	    //
	    // E step
	    qsi_->Expectation();
	    qdch_->Expectation();
	    qgau_->Expectation();

	    //
	    //
	    F_old = F_;
	    // Formaula (4.29)
	    F_    = 0.;
	    F_   += qsi_->get_F();
	    F_   += qdch_->get_F();
	    F_   += qgau_->get_F();
	    //
	    dF    = F_ - F_old;
	    F_history_.push_back( F_ );
	    //
	    //
	    std::cout << "Ending iteration: " << iteration << std::endl;
	    std::cout << "Lower bound: " << F_  << std::endl;
	    std::cout << "Delta Lower bound: " << dF  << std::endl;
	  }
      }
  }
}
#endif
