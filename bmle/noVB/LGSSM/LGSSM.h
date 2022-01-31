#ifndef Linear_Gaussian_State_Space_Model_H
#define Linear_Gaussian_State_Space_Model_H
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
#include "Posterior.h"
//
//
//
namespace noVB
{
  namespace LGSSM
  {
    /** \class Linear_Gaussian_State_Space_Model
     *
     * \brief  Expectation-Maximization algorithm for the Linear
     *         State-Space model.
     * 
     * Dim is the number of dimensions
     * input:
     *   Dim: dimension of the measure
     *   S: dimension of state accessible to the Markov Chain
     *   n: number of cases (subjects)
     *   Y [p x n]: entry data. the matrix is supposed to be normalized.
     *              p = Dim, 
     *              n is the size of the dataset
     * 
     */
    template< int Dim, int S >
      class Linear_Gaussian_State_Space_Model
    {
 
    public:
      /** Constructor. */
      explicit Linear_Gaussian_State_Space_Model( const std::vector< std::vector< Eigen::Matrix< double, Dim, 1 > > >& ,
				    const std::vector< std::vector< Eigen::Matrix< double, 1, 1 > > >& );
    
      /** Destructor */
      ~Linear_Gaussian_State_Space_Model(){};

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
      std::shared_ptr< noVB::LGSSM::P_qsi <Dim,S> > qsi_;
      std::shared_ptr< noVB::LGSSM::P_qdch<Dim,S> > qdch_;
      std::shared_ptr< noVB::LGSSM::P_qgau<Dim,S> > qgau_;

      //
      // log marginal likelihood lower bound
      double L_{-1.e-36};
      std::list< double > L_history_;
    };

    //
    //
    //
    template < int Dim, int S >
      Linear_Gaussian_State_Space_Model< Dim, S >::Linear_Gaussian_State_Space_Model( const std::vector< std::vector< Eigen::Matrix< double, Dim, 1 > > >& Y,
										      const std::vector< std::vector< Eigen::Matrix< double, 1, 1 > > >& Age ):
      Y_{Y}, Age_{Age}, n_{Y.size()}
    {
      //
      //
      qsi_  = std::make_shared< noVB::LGSSM::P_qsi <Dim,S> >( Y_ );
      qdch_ = std::make_shared< noVB::LGSSM::P_qdch<Dim,S> >( qsi_, Y_ );
      qgau_ = std::make_shared< noVB::LGSSM::P_qgau<Dim,S> >( qsi_, Y_, Age_ );
      // set dependencies
      qsi_->set(qdch_,qgau_);
    }
    //
    //
    //
    template < int Dim, int S > void
      Linear_Gaussian_State_Space_Model< Dim, S >::ExpectationMaximization()
      {
	double
	  dL    =  1.e06,
	  L_old = -1.e-40;
	int iteration = 0;

	//
	// Access the states
//	const std::vector< std::vector< Eigen::Matrix < double, S , 1 > > > &_s_         = qsi_->get_s();
//	const std::vector< std::vector< Eigen::Matrix < double, S , 1 > > > &_ln_gamma_  = qgau_->get_ln_gamma();
//	const                           Eigen::Matrix < double, S , 1 >     &_pi_        = qdch_->get_pi();
//	const                           Eigen::Matrix < double, S , S >     &_A_         = qdch_->get_A();
	//
	while ( fabs(dL) > 1.e-10 )
	  {
	    std::cout << "Begining iteration: " << ++iteration << std::endl;
	    //
	    // E step
	    // Update of the state probability
	    qsi_->Expectation();
	    qdch_->Expectation();
	    qgau_->Expectation();
	    //
	    // M step
	    // Update of the transition and emission probabilities
	    qsi_->Maximization();
	    qdch_->Maximization();
	    qgau_->Maximization();
	    
	    //
	    // Build the posterior probability
	    L_old = L_;
	    L_  = qsi_->get_L();
	    L_ += qdch_->get_L();
	    L_ += qgau_->get_L();
			
	    //
	    //
	    dL    = L_ - L_old;
	    L_history_.push_back( L_ );
	    //
	    //
	    std::cout << "#################" << std::endl;
	    std::cout << "Ending iteration: " << iteration << std::endl;
	    std::cout << "Lower bound: " << L_  << std::endl;
	    std::cout << "Delta Lower bound: " << dL  << std::endl;
	  }
      }
  }
}
#endif
