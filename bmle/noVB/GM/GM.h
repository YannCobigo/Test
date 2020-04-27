#ifndef GAUSSIANMIXTURE_H
#define GAUSSIANMIXTURE_H
//
//
//
#include <limits>
#include <vector>
#include <random>
#include <math.h>
#include <chrono>
// GSL - GNU Scientific Library
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_sf_gamma.h>
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
#include "Tools.h"
//
#define ln_2    0.69314718055994529L
#define ln_2_pi 1.8378770664093453L
#define ln_pi   1.1447298858494002L
//
//
//
namespace noVB
{
  namespace GM
  {
    /** \class GaussianMixture
     *
     * \brief  Expectation-Maximization algorithm
     * 
     * Dim is the number of dimensions
     * K is the number of Gaussians in the mixture
     *
     */
    template< int Dim, int K >
      class GaussianMixture
    {
 
    public:
      /** Constructor. */
      explicit GaussianMixture( const std::list< Eigen::Matrix< double, Dim, 1 > >&  );
    
      /** Destructor */
      virtual ~GaussianMixture(){};

      //
      // Accessors
      // posterior probabilities

      //
      // Functions
      // main algorithn
      void   ExpectationMaximization();
      // Multivariate gaussian
      double gaussian( const Eigen::Matrix< double, Dim, 1 >& , 
		       const Eigen::Matrix< double, Dim, 1 >& , 
		       const Eigen::Matrix< double, Dim, Dim >& ) const;


      //
      // Accessors
      // posterior probabilities
      const std::vector< std::vector< double > >&
	get_posterior_probabilities() const { return gamma_; };
      // posterior responsability pi
      //const double get_pi( const int KK ) const { return exp(ln_pi_[KK]); };
      // 
      // Print cluster statistics
      void         get_cluster_statistics( const int ) const;

    private:
      //
      // private member function
      //
      

      //
      // Data set
      std::list< Eigen::Matrix< double, Dim, 1 > >     data_set_; 
      // Data set size
      std::size_t                                      data_set_size_{0};

      //
      // Mean of gaussians
      std::vector< Eigen::Matrix< double, Dim, 1 > >   mu_;
      std::vector< Eigen::Matrix< double, Dim, 1 > >   mu_new_;
      // Covariance of gaussians
      std::vector< Eigen::Matrix< double, Dim, Dim > > covariance_;
      // Mixture coefficients of gaussians
      std::vector< double >                            pi_;
      // Posterior probability 
      std::vector< std::vector< double > >             gamma_;
      // effective number of point assigned to a gaussian
      std::vector< double >                            N_;
      
      //
      // Convergence criteria
      double epsilon_{1.e-3};
      // (2pi)^d
      double two_pi_dim_{1.};
    };

    //
    //
    //
    template < int Dim, int K >
      GaussianMixture< Dim, K >::GaussianMixture( const std::list< Eigen::Matrix< double, Dim, 1 > >& X ):
      data_set_{X}, data_set_size_{X.size()}
    {
      //
      //
      mu_.resize(K);
      mu_new_.resize(K);
      covariance_.resize(K);
      pi_.resize(K);
      gamma_.resize(K);
      N_.resize(K);
      
      //
      // Initializarion
      std::default_random_engine generator;
      std::normal_distribution<double> distribution( 0.0, 1.0 );
      //
      for ( int k = 0 ; k < K ; k++ )
	{
	  // mu
	  for ( int d = 0 ; d < Dim ; d++ )
	    mu_[k](d,0) = distribution(generator);
	  // Covariance
	  covariance_[k] = 1.e+2 * Eigen::Matrix< double, Dim, Dim >::Identity();
	  // mixture coefficients
	  pi_[k] = 1. / static_cast< double >(K);
	  N_[k]  = 0;
	}
      //
      for ( int d = 0 ; d < Dim ; d++ )
	two_pi_dim_ *= 2 * M_PI;
    }
    //
    //
    //
    template < int Dim, int K > void
      GaussianMixture< Dim, K >::ExpectationMaximization()
      {
	//
	// resize the posterior probabilities
	data_set_size_ = data_set_.size();
	for ( int k = 0 ; k < K ; k++ )
	  gamma_[k].resize( data_set_size_ );
	//
	double 
	  old_P = 1.e+16,
	  P     = 1.e+18;

	//
	// Main loop
	int conv = 0;
	while ( fabs( old_P - P ) > epsilon_ || ++conv < 20 )
	  {
	    //
	    // E-step
	    // Re-initialize the counts of data per gaussians
	    old_P = P;
	    P     = 0.;
	    //
	    for ( int k = 0 ; k < K ; k++ )
	      {
		//
		// reinitialize
		N_[k]      = 0.;
		mu_new_[k] = Eigen::Matrix< double, Dim, 1 >::Zero();
	      }
	    //
	    // Calculation of the posterior probability
	    typename std::list< Eigen::Matrix< double, Dim, 1 > >::const_iterator it_x = data_set_.begin();
	    auto start = std::chrono::steady_clock::now();
	    for ( int x = 0 ; x < data_set_size_ ; x++ )
	      {
		// denominator
		double marginal = 0.;
		for ( int k = 0 ; k < K ; k++ )
		  marginal += gamma_[k][x] = pi_[k] * gaussian( (*it_x), mu_[k], covariance_[k] );
		// posterior
		if ( !std::isfinite(marginal) )
		  exit(-1);
		for ( int k = 0 ; k < K ; k++ )
		  {
		    gamma_[k][x] /= marginal;
		    N_[k]        += gamma_[k][x];
		    mu_new_[k]   += gamma_[k][x] * (*it_x);
		  }
		//
		++it_x;
	      }
 
	    //
	    // M-step
	    for ( int k = 0 ; k < K ; k++ )
	      {
		// Mu
		mu_[k]  = mu_new_[k];
		mu_[k] /= N_[k];
		// Cov new
		covariance_[k] = Eigen::Matrix< double, Dim, Dim >::Zero();
		//
		it_x = data_set_.begin();
		for ( int x = 0 ; x < data_set_size_ ; x++ )
		  {
		    covariance_[k] += gamma_[k][x] * (*it_x - mu_[k]) * (*it_x - mu_[k]).transpose();
		    ++it_x;
		  }
		covariance_[k] /= N_[k];
		// Pi new
		pi_[k] = N_[k] / static_cast< double >(data_set_size_);
	      }

	    //
	    // Log likelihood
	    it_x = data_set_.begin();
	    for ( int x = 0 ; x < data_set_size_ ; x++ )
	      {
		double sum = 0.;
		for ( int k = 0 ; k < K ; k++ )
		  sum += pi_[k] * gaussian( (*it_x), mu_[k], covariance_[k]) * sqrt( covariance_[k].determinant() );
		//
		P += log(sum);
		++it_x;
	      }
	    std::cout << "P: " << P 
		      << " DeltaP: " << fabs( old_P - P )
		      << " iteration: " << conv 
		      << std::endl;

	    auto end   = std::chrono::steady_clock::now();
	    auto duration = std::chrono::duration_cast< std::chrono::microseconds > 
	      ( end - start ).count();
	    std::cout << duration << " mus"<< std::endl;
	    //
	    for ( int k = 0 ; k < K ; k++ )
	      std::cout 
		<< "k = " << k << "\n"
		<< "pi = " << pi_[k] << "\n"
		<< "mu = \n" << mu_[k] << "\n"
		<< "Cov = \n" << covariance_[k] << "\n";
	  }
	//
	for ( int k = 0 ; k < K ; k++ )
	  std::cout 
	    << "k = " << k << "\n"
	    << "pi = " << pi_[k] << "\n"
	    << "mu = \n" << mu_[k] << "\n"
	    << "Cov = \n" << covariance_[k] << "\n";
      }
    //
    //
    //
    template < int Dim, int K > double
      GaussianMixture< Dim, K >::gaussian( const Eigen::Matrix< double, Dim, 1 >&   X, 
					   const Eigen::Matrix< double, Dim, 1 >&   Mu, 
					   const Eigen::Matrix< double, Dim, Dim >& Cov ) const
      {
//	//
//	//
//	double 
//	  arg        = ((X-Mu).transpose() * Cov.inverse() * (X-Mu))(0,0)/2.,
//	  two_pi_dim = 1.;
//	for ( int d = 0 ; d < Dim ; d++ )
//	  two_pi_dim *= 2 * M_PI;
//	//      
//	return exp( - arg ) * sqrt( Cov.determinant() * two_pi_dim); 
  double 
    arg  = ((X-Mu).transpose() * Cov.inverse() * (X-Mu))(0,0) * 0.5,
    norm = Cov.determinant() * two_pi_dim_ ;
//  std::cout 
//    << "arg " << arg
//    << " exp( - arg ) " << exp( - arg )
//    << " norm " << norm 
//    << " sqrt( norm ) " << sqrt( norm )
//    << "\n result1 " << exp( - arg ) / sqrt( norm )
//    << "\n result2 " << exp( - arg ) / sqrt( two_pi_dim_ ) / Cov.determinant()
//    << std::endl;
  //      
  //return exp( - arg ) / sqrt( norm ); 
  return exp( - arg ) / Cov.determinant(); 
      }
    //
    //
    //
    template < int Dim, int K > void
      GaussianMixture< Dim, K >::get_cluster_statistics( const int KK ) const
      {
//	//
//	//
//	std::cout 
//	  << " - The cluster ["<<KK<<"] represents " << exp(ln_pi_[KK])
//	  << " of the statistics ("<< N_[KK] << " voxels)."
//	  << " The center of the cluster is:\n"
//	  << m_[KK]
//	  << "\n The variance is: \n"
//	  << S_[KK]
//	  << std::endl;
      }
  }
}
#endif
