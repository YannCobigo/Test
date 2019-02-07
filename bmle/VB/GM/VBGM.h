#ifndef VBGAUSSIANMIXTURE_H
#define VBGAUSSIANMIXTURE_H
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
namespace VB
{
  namespace GM
  {
    /** \class VBGaussianMixture
     *
     * \brief  Expaectation-Maximization algorithm
     * 
     * Dim is the number of dimensions
     * K is the number of Gaussians in the mixture
     *
     */
    template< int Dim, int K >
      class VBGaussianMixture
    {
 
    public:
      /** Constructor. */
      explicit VBGaussianMixture( const std::list< Eigen::Matrix< double, Dim, 1 > >&  );
    
      /** Destructor */
      virtual ~VBGaussianMixture(){};

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
      // Responsability
      double responsability( const double, const double,
			     const Eigen::Matrix< double, Dim, 1 >&, 
			     const Eigen::Matrix< double, Dim, 1 >&, 
			     const Eigen::Matrix< double, Dim, Dim >& ) const;
      // Digamma
      double digamma( const double ) const;
      // Normalisation function for Dirichlet distribution
      double Dirichlet_partition_func( const double ) const;
      double Dirichlet_partition_func( std::vector< double > Alpha ) const;
      double ln_Dirichlet_partition_func( const double ) const;
      double ln_Dirichlet_partition_func( std::vector< double > Alpha ) const;
      // log Normalisation function for Dirichlet distribution
      double ln_Wishart_partition_func( const Eigen::Matrix< double, Dim, Dim >,
					const double ) const;
      // Entropy function for Dirichlet distribution
      double Wishart_entropy( const Eigen::Matrix< double, Dim, Dim >,
			      const double, const double ) const;
      // Logarithm of sum of exponetials
      double ln_sum_exp( const std::vector< std::vector< double > >, const int ) const;


      //
      // Accessors
      // posterior probabilities
      const std::vector< std::vector< double > >&
	get_posterior_probabilities() const { return gamma_; };

    private:
      //
      // private member function
      //

      //
      // Data set
      std::list< Eigen::Matrix< double, Dim, 1 > >  data_set_; 
      // Data set size
      std::size_t data_set_size_{0};

      //
      // Dirichlet prior
      double                                          alpha0_{ 1.e+0 };
      double                                          alpha_hat_{0.};
      std::vector< double >                           alpha_;
      //
      // Gaussian-Wishart prior
      // Wishart on the precision
      double                                           nu0_{50.};
      std::vector< double >                            nu_;
      Eigen::Matrix< double, Dim, Dim >                W0_;
      // Gaussian
      double                                           beta0_{1.e-0};
      std::vector< double >                            beta_;
      // Gaussian-Wishart
      std::vector< Eigen::Matrix< double, Dim, 1 > >   m0_;
      std::vector< Eigen::Matrix< double, Dim, 1 > >   m_;
      std::vector< Eigen::Matrix< double, Dim, Dim > > S_;
      std::vector< Eigen::Matrix< double, Dim, Dim > > W_;
      //
      // Posterior probability 
      // responsability
      std::vector< std::vector< double > >             gamma_;
      std::vector< std::vector< double > >             ln_gamma_;
      // 
      std::vector< double >                            pi_;
      std::vector< double >                            ln_pi_;
      // 
      std::vector< double >                            lambda_;
      std::vector< double >                            ln_lambda_;
      // Mixture coefficients of gaussians
      std::vector< Eigen::Matrix< double, Dim, 1 > >   x_mean_;

      //
      // effective number of point assigned to a gaussian
      std::vector< double >                            N_;
      // Convergence criteria
      double                                           epsilon_{1.e-6};
      double                                           variational_lower_bound_{0.};
    };

    //
    //
    //
    template < int Dim, int K >
      VBGaussianMixture< Dim, K >::VBGaussianMixture( const std::list< Eigen::Matrix< double, Dim, 1 > >& X ):
      data_set_{X}, data_set_size_{X.size()}
    {
      //
      // Dirichlet
      alpha_.resize(K);
      // Wishart
      nu_.resize(K);
      // Gaussian
      beta_.resize(K);
      // Gaussian-Wishart
      m0_.resize(K);
      m_.resize(K);
      S_.resize(K);
      W_.resize(K);
      //
      gamma_.resize(K);
      ln_gamma_.resize(K);
      pi_.resize(K);
      ln_pi_.resize(K);
      lambda_.resize(K);
      ln_lambda_.resize(K);
      x_mean_.resize(K);
      N_.resize(K);


      //
      // Initializarion
      //nu0_ = 1.e+6;// /*static_cast<double>(Dim) * */ static_cast< double >( data_set_size_ );
      nu0_ = static_cast<double>( 1./*Dim *  data_set_size_*/ );
      //
      // Creating a Definit positive random matrix
      W0_  = 1.e-02 / nu0_ * Eigen::Matrix< double, Dim, Dim >::Identity();
      //Eigen::MatrixXd random = Eigen::MatrixXd::Random( Dim, Dim );
      //Eigen::JacobiSVD< Eigen::MatrixXd > 
      //	svd( random, Eigen::ComputeThinU | Eigen::ComputeThinV );
      //W0_ = Eigen::Matrix< double, Dim, Dim >::Identity();
      //for ( int u = 0 ; u < Dim ; u++ )
      //	W0_(u,u) = svd.singularValues()(u);
      //W0_ /= nu0_;
      //std::cout << W0_ << std::endl;
      //
      for ( int k = 0 ; k < K ; k++ )
	{
	  //
	  //
	  alpha_[k]   = alpha0_ / static_cast< double >(K);
	  alpha_hat_ += alpha_[k];
	  // Wishart
	  nu_[k]      = nu0_ + N_[k];
	  // Gaussian
	  beta_[k]    = beta0_ + N_[k];
	  // Gaussian-Wishart
	  //m0_[k]      = mean_init.get_means()[k];
	  m0_[k]      = Eigen::Matrix< double, Dim, 1 >::Random();
	  m_[k]       = m0_[k];
	  // Covariance
	  W_[k]       = W0_;
	  // mixture coefficients
	  ln_pi_[k]      = exp( 1./static_cast< double >(K) );
	  ln_lambda_[k]  = 0.;
	  //
	  //
	  gamma_[k].resize( data_set_size_ );
	  ln_gamma_[k].resize( data_set_size_ );
	}
    }
    //
    //
    //
    template < int Dim, int K > void
      VBGaussianMixture< Dim, K >::ExpectationMaximization()
      {
	//
	double 
	  P               =  1.e+15,  
	  old_P           =  1.e+16,
	  old_lower_bound =  1.e+16,
	  ln_Calpha0      =  ln_Dirichlet_partition_func(alpha0_), 
	  ln_B0           = -ln_Wishart_partition_func( W0_, nu0_ ),
	  old_N[K];
	//
	Eigen::Matrix< double, Dim, Dim > W0_inv = W0_.inverse();


	//
	// Main loop
	int iteration = 0;
	//while ( ++iteration < 10000 )
	while ( fabs( old_lower_bound - variational_lower_bound_ ) > epsilon_ && ++iteration < 100000 )
	  {
	    //
	    // Loop initialization
	    //++iteration;

	    //
	    // Parameters reset
	    for ( int k = 0 ; k < K ; k++ )
	      {
		old_N[k]        = N_[k];
		old_P           = P;
		P               = 0.;
		old_lower_bound = variational_lower_bound_;
		//
		N_[k]      = 0.;
		x_mean_[k] = Eigen::Matrix< double, Dim, 1 >::Zero();
		S_[k]      = Eigen::Matrix< double, Dim, Dim >::Zero();
	      }

	    //
	    // start time
	    auto start = std::chrono::steady_clock::now();

	    //
	    // E-step
	    // Calculation of the posterior probability
	    typename std::list< Eigen::Matrix< double, Dim, 1 > >::const_iterator it_x = data_set_.begin();
	    //
	    for ( int x = 0 ; x < data_set_size_ ; x++ )
	      {
		// Maginal
		double ln_marginal = 0.;
		for ( int j = 0 ; j < K ; j++ )
		  {
		    ln_gamma_[j][x]  = ln_pi_[j] + 0.5 * ln_lambda_[j];
		    ln_gamma_[j][x] -= 0.5 * static_cast< double >(Dim) / beta_[j];
		    ln_gamma_[j][x] -= 0.5 * nu_[j] * (((*it_x)-m_[j]).transpose() * W_[j] * ((*it_x)-m_[j]))(0,0);
		  }
		//
		//if ( std::isfinite(marginal) )
		//  marginal = log( marginal );
		//else
		ln_marginal = ln_sum_exp( ln_gamma_, x );
		// Posterior
		for ( int k = 0 ; k < K ; k++ )
		  {
		    //std::cout << "ln_gamma_[k][x] " << ln_gamma_[k][x] << std::endl;
		    ln_gamma_[k][x] -= ln_marginal;
		    gamma_[k][x]     = exp( ln_gamma_[k][x] );
		    N_[k]           += gamma_[k][x];
		    x_mean_[k]      += gamma_[k][x] * (*it_x);
		  }
		// next datum
		++it_x;
	      }

	    //
	    // M-step
	    alpha_hat_ = 0.;
	    for ( int k = 0 ; k < K ; k++ )
	      {
		//
		//
		//N_[k]      += 1.e-16;
		x_mean_[k] /= N_[k];

		//
		//
		alpha_hat_ += alpha_[k] = alpha0_ + N_[k];
		beta_[k]    = beta0_  + N_[k];
		nu_[k]      = nu0_    + N_[k];
		// 
		m_[k]       = ( beta0_*m0_[k] + N_[k]*x_mean_[k] )/beta_[k];

		//
		// Covariance
		it_x = data_set_.begin();
		for ( int x = 0 ; x < data_set_size_ ; x++ )
		  {
		    S_[k]  += gamma_[k][x] * (*it_x - x_mean_[k]) * (*it_x - x_mean_[k]).transpose();
		    ++it_x;
		  }	  
		Eigen::Matrix< double, Dim, Dim > 
		  W_inv    = (x_mean_[k]-m0_[k]) * (x_mean_[k]-m0_[k]).transpose() * beta0_ * N_[k] / (beta_[k]);
		W_inv     += W0_inv +  S_[k];
		W_[k]      = W_inv.inverse() /*+ 1.e-16 * Eigen::Matrix< double, Dim, Dim >::Identity()*/;
		S_[k]     /= N_[k];
	      }
	    // Parameters reset
	    double 
	      ln_W_determinant[K], psi_D[K];
	    for ( int k = 0 ; k < K ; k++ )
	      {
		psi_D[k] = 0.;
		for ( int d = 0 ; d < Dim ; d++ )
		  psi_D[k] += digamma( (nu_[k] + 1. - d) * 0.5 );
		ln_W_determinant[k] = NeuroBayes::ln_determinant( W_[k] );
		//
		ln_pi_[k]     = digamma(alpha_[k]) - digamma(alpha_hat_);
		ln_lambda_[k] = psi_D[k] + static_cast< double >(Dim) * ln_2 + ln_W_determinant[k];
	      }

	    //
	    // Variational lower bound
	    if ( true )
	      {
		double
		  ln_Calpha = ln_Dirichlet_partition_func( alpha_ ), 
		  L1, L2, L3, L4, L5, L6, L7 = 0.;
		L1 = L2 = L3 = L4 = L5 = L6 = L7;
		variational_lower_bound_    = 0;
		for ( int k = 0 ; k < K ; k++ )
		  {
		    // E[ln p(X|Z, μ, Λ)]
		    double 
   		      L1_part = ln_lambda_[k] - static_cast<double>(Dim)*( ln_2_pi + 1. / beta_[k] );
		    L1_part  -= nu_[k] * (S_[k]*W_[k]).trace();
		    L1_part  -= nu_[k] * (x_mean_[k]-m_[k]).transpose() * W_[k] * (x_mean_[k]-m_[k]);
		    L1       += N_[k]  * L1_part;
		    // E[ln p(π)]
		    L3 += ln_pi_[k];
		    // E[ln p(μ, Λ)]
		    double 
		      L4_part = static_cast<double>(Dim)*( log(beta0_) - ln_2_pi - beta0_/beta_[k] ) + ln_lambda_[k];
		    L4_part  -= beta0_*nu_[k]*(m_[k]-m0_[k]).transpose() * W_[k] * (m_[k]-m0_[k]);
		    L4_part  -= nu_[k]*( W0_inv * W_[k] ).trace();
		    L4_part  += ln_lambda_[k]*( nu0_ - 1 - static_cast<double>(Dim) );
		    L4       += 0.5 * L4_part;
		    // E[ln q(π)]
		    L6 += ln_pi_[k]*(alpha0_ - 1);
		    // E[ln q(μ, Λ)]
		    L7 += ln_lambda_[k] + static_cast<double>(Dim)*(log(beta_[k]) - ln_2_pi -1) ;
		    L7 *= 0.5;
		    L7 -= Wishart_entropy(W_[k], nu_[k], ln_lambda_[k]);
		    //
		    it_x = data_set_.begin();
		    for ( int x = 0 ; x < data_set_size_ ; x++ )
		      {
			// E[ln p(Z|π)]
			L2 += gamma_[k][x] * ln_pi_[k];
			// E[ln q(Z)]
			L5 += gamma_[k][x] * ln_gamma_[k][x];
		      }
		  }
		//
		variational_lower_bound_  = 0.5*L1 + L2 + ln_Calpha0 + (alpha0_ -1)*L3;
		variational_lower_bound_ += L4 + static_cast<double>(K)*ln_B0;
		variational_lower_bound_ -= L5 + L6 + ln_Calpha + L7;
		//
		std::cout << "Variational lower bound = " << variational_lower_bound_;
		std::cout << " && difference last iter: " 
			  <<   old_lower_bound - variational_lower_bound_  
			  << std::endl;
	      }
      
	    //
	    // count time
	    for ( int k = 0 ; k < K ; k++ )
	      {
		P += old_N[k] - N_[k];
		//
		std::cout 
		  << " pi_[" << k <<"] " << exp(ln_pi_[k])
		  << " N_["  << k <<"] " << N_[k]
		  << " alpha_["  << k <<"] " << alpha_[k]
		  << " beta_["  << k <<"] " << beta_[k]
		  << " nu_["  << k <<"] \n" << nu_[k]
		  << "\n m_["<< k <<"] \n" << m_[k]
		  //<< "\n xm_["<< k <<"] \n" << x_mean_[k]
		  << "\n W_["<< k <<"] \n" << W_[k]
		  << "\n S_["<< k <<"] \n" << S_[k]
		  << std::endl;
	      }
	    auto end   = std::chrono::steady_clock::now();
	    auto duration = std::chrono::duration_cast< std::chrono::microseconds > 
	      ( end - start ).count();
	    std::cout << "Iteration: " << iteration << " -- duration: " << duration << " mu sec"<< std::endl;
	  }
	//
	// count time
	for ( int k = 0 ; k < K ; k++ )
	  {
	    P += old_N[k] - N_[k];
	    //
	    std::cout 
	      << " pi_[" << k <<"] " << exp(ln_pi_[k])
	      << " N_["  << k <<"] " << N_[k]
	      << " alpha_["  << k <<"] " << alpha_[k]
	      << " beta_["  << k <<"] " << beta_[k]
	      << " nu_["  << k <<"] \n" << nu_[k]
	      << "\n m_["<< k <<"] \n" << m_[k]
	      //<< "\n xm_["<< k <<"] \n" << x_mean_[k]
	      << "\n W_["<< k <<"] \n" << W_[k]
	      << "\n S_["<< k <<"] \n" << S_[k]
	      << std::endl;
	  }
	//
	std::cout << "Iteration: " << iteration << " P = " << P
		  << " && difference last iter: " << P - old_P  
		  << std::endl;
      }
    //
    //
    //
    template < int Dim, int K > double
      VBGaussianMixture< Dim, K >::gaussian( const Eigen::Matrix< double, Dim, 1 >&   X, 
					     const Eigen::Matrix< double, Dim, 1 >&   Mu, 
					     const Eigen::Matrix< double, Dim, Dim >& Cov ) const
      {
	//
	//
	double 
	  arg        = ((X-Mu).transpose() * Cov.inverse() * (X-Mu))(0,0)/2.,
	  two_pi_dim = 1.;
	for ( int d = 0 ; d < Dim ; d++ )
	  two_pi_dim *= 2 * M_PI;
	//      
	return exp( - arg ) * sqrt( Cov.determinant() * two_pi_dim); 
      }
    //
    //
    //
    template < int Dim, int K > double
      VBGaussianMixture< Dim, K >::digamma( const double X ) const
      {
	//
	//      
	return gsl_sf_psi(X); 
      }
    //
    //
    //
    template < int Dim, int K > double
      VBGaussianMixture< Dim, K >::ln_Dirichlet_partition_func( std::vector< double > Alpha ) const
      {
	//
	//
	double 
	  alpha_hat   = 0.,
	  Denominator = 0.;
	for ( int k = 0 ; k < K ; k++ )
	  {
	    alpha_hat   += Alpha[k];
	    Denominator += gsl_sf_lngamma( Alpha[k] );
	  }

	//
	//      
	return gsl_sf_lngamma( alpha_hat ) - Denominator; 
      }
    //
    //
    //
    template < int Dim, int K > double
      VBGaussianMixture< Dim, K >::Dirichlet_partition_func( std::vector< double > Alpha ) const
      {
	//
	//
	double 
	  alpha_hat   = 0.,
	  Denominator = 1.;
	for ( int k = 0 ; k < K ; k++ )
	  {
	    alpha_hat   += Alpha[k];
	    Denominator *= gsl_sf_gamma( Alpha[k] );
	  }

	//
	//      
	return gsl_sf_gamma( alpha_hat ) / Denominator; 
      }
    //
    //
    //
    template < int Dim, int K > double
      VBGaussianMixture< Dim, K >::ln_Dirichlet_partition_func( const double Alpha0 ) const
      {
	//
	//
	double 
	  alpha_hat   = 0.,
	  Denominator = 0.;
	for ( int k = 0 ; k < K ; k++ )
	  {
	    alpha_hat   += Alpha0 / static_cast< double >(K);
	    Denominator += gsl_sf_lngamma( Alpha0 );
	  }
	//
	//      
	return gsl_sf_lngamma( alpha_hat ) - Denominator; 
      }
    //
    //
    //
    template < int Dim, int K > double
      VBGaussianMixture< Dim, K >::Dirichlet_partition_func( const double Alpha0 ) const
      {
	//
	//
	double 
	  alpha_hat   = 0.,
	  Denominator = 1.;
	for ( int k = 0 ; k < K ; k++ )
	  {
	    alpha_hat   += Alpha0;
	    Denominator *= gsl_sf_gamma( Alpha0 );
	  }
	//
	//      
	return gsl_sf_gamma( alpha_hat ) / Denominator; 
      }
    //
    //
    //
    template < int Dim, int K > double
      VBGaussianMixture< Dim, K >::ln_Wishart_partition_func( const Eigen::Matrix< double, Dim, Dim > W0,
							      const double Nu0 ) const
      {
	//
	//
	double 
	  arg = 0.,
	  dim = static_cast<double>(Dim);
	//
	arg += Nu0 * dim * ln_2 *0.5;
	arg += dim *(dim-1) * ln_pi * 0.25;
	arg += Nu0 * W0.determinant() * 0.5;
	//
	for ( int d = 0 ; d < Dim ; d++ )
	  arg += gsl_sf_lngamma( 0.5*(Nu0+1-d) );
	//
	//      
	return arg;
      }
    //
    //
    //
    template < int Dim, int K > double
      VBGaussianMixture< Dim, K >::Wishart_entropy( const Eigen::Matrix< double, Dim, Dim > W,
						    const double Nu, const double Ln_lambda ) const
      {
	//
	//
	double 
	  H   = - ln_Wishart_partition_func( W, Nu ),
	  dim = static_cast<double>(Dim);
	//
	H -= (Nu - dim -1) * Ln_lambda * 0.5;
	H += Nu * dim * 0.5;

	//
	//      
	return H;
      }
    //
    //
    //
    template < int Dim, int K > double
      VBGaussianMixture< Dim, K >::responsability( const double Beta, const double Nu,
						   const Eigen::Matrix< double, Dim, 1 >&   X, 
						   const Eigen::Matrix< double, Dim, 1 >&   M, 
						   const Eigen::Matrix< double, Dim, Dim >& W ) const
      {
	//
	//
	double 
	  arg = static_cast< double >(Dim) / Beta;
	arg  += Nu * ((X-M).transpose() * W * (X-M))(0,0);
	//      
	return exp( -0.5*arg) + 1.e-16;
      }
    //
    //
    //
    template < int Dim, int K > double
      VBGaussianMixture< Dim, K >::ln_sum_exp( const std::vector< std::vector< double > > Ln_gamma, 
					       const int X ) const
      {
	//
	// by Tom Minka
	double 
	  max = -700.,
	  Z   = 0.,
	  e_ln_gam[K];
	for ( int k = 0 ; k < K ; k++ )
	  {
	    if ( max < Ln_gamma[k][X] )
	      max = Ln_gamma[k][X];
	  }
	//
	for ( int k = 0 ; k < K ; k++ )
	  Z += exp(Ln_gamma[k][X] - max);

	//  std::cout 
	//    << "max " << max
	//    << " Z " << Z
	//    << " log(Z) + max " << log(Z) + max
	//    << std::endl;

	//
	return log(Z) + max;
      }
  }
}
#endif
