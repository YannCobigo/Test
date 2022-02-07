#ifndef LGSSM_POSTERIOR_H
#define LGSSM_POSTERIOR_H
//
//
//
#include <limits>
#include <vector>
#include <random>
#include <math.h>
#include <chrono>
#include <memory>
// GSL - GNU Scientific Library
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_sf_gamma.h>
//
//
// Eigen
#include <Eigen/Core>
#include <Eigen/Eigen>
//
//
//
#include "Tools.h"
//
//
//
namespace noVB 
{
  namespace LGSSM
  { 
    template< int Dim, int S > class P_qsi;
    template< int Dim, int S > class P_qdch;
    template< int Dim, int S > class P_qgau;

    //
    enum Vpost {QSI,QDCH,QGAU};
    //
    //
    //
    /** \class variational_posterior
     *
     * \brief variational posterior interface application.
     * 
     * All developpments are inspired and adapted from:
     *
     *  @book{Bishop,
     * }
     *
     * - Dim: number of features
     * - n: number of entries
     * - S: number of state dimensions
     *
     *
     */
    template< int Dim, int S >
      class Posterior
    {
    public:
  
      //
      // Functions
      // Process the expectation step
      //virtual ~Posterior(){};
      virtual void Expectation()  = 0;
      virtual void Maximization() = 0;
    };
    //
    //
    //
    /** \class P_qsi
     *
     * \brief Gaussian posterior probabilities
     * 
     * The Gaussian posterior probabilities represent the 
     * probability density of the transmission probability. 
     * 
     * Parameters:
     * - A: transition matrix -- stochastic matrix
     * - $\Gamma$: Distribution covariance matrix
     * - n (number of inputs)
     * - S (number of state dimensions).
     * - $L_{qsi}$: Cost function associated with the class paramteres
     *
     */
    template< int Dim, int S >
      class P_qsi : public Posterior<Dim,S>
      {
      public:
	/** Constructor. */
	explicit P_qsi(){};
	/** Constructor. */
	explicit P_qsi( const std::vector< std::vector< Eigen::Matrix < double, Dim , 1 > > >& );
    
	/** Destructor */
	virtual ~P_qsi(){};
  
	//
	// Functions
	// Process the expectation step
	virtual void Expectation();
	virtual void Maximization();

	//
	// accessors
	const std::vector< std::vector< Eigen::Matrix < double, S , 1 > > >& get_s()   const {return s_;}
	const std::vector< std::vector< Eigen::Matrix < double, S , S > > >& get_ss()  const {return ss_;}
	const std::vector< std::vector< Eigen::Matrix < double, S , S > > >& get_s_s() const {return s_s_;}
	const double                                                         get_L()   const {return L_qsi_;}
	//
	void set( std::shared_ptr< noVB::LGSSM::P_qdch<Dim,S> > Qdch,
		  std::shared_ptr< noVB::LGSSM::P_qgau<Dim,S> > Qgau )
	{qdch_ = Qdch; qgau_ = Qgau;};

      private:
	//
	//
	std::shared_ptr< noVB::LGSSM::P_qdch<Dim,S> > qdch_;
	std::shared_ptr< noVB::LGSSM::P_qgau<Dim,S> > qgau_;


	//
	//
	std::vector< std::vector< Eigen::Matrix < double, Dim , 1 > > > Y_;
	// Size of the data set
	std::size_t n_{0};

	//
	// hidden state
	// <s_{i,t}>
	std::vector< std::vector< Eigen::Matrix < double, S , 1 > > >   s_;
	// <s_{i,t-1} x s_{i,t}>
	std::vector< std::vector< Eigen::Matrix < double, S , S > > >   ss_;
	// <s_{i,t} x s_{i,t}>
	std::vector< std::vector< Eigen::Matrix < double, S , S > > >   s_s_;
	//
	// Dynamic programing
	// Forward:  compute alpha(s_{i,t})
	// alpha = N(s_{t}^{n} | \mu_{t}^{n}, V_{t}^{n})
	std::vector< std::vector< double > >                            alpha_i_t_;
	std::vector< std::vector< Eigen::Matrix < double, S , S > > >   alpha_V_i_t_;
	// scaling
	std::vector< std::vector< double > >                            c_i_t_;
	// Kalman gain matrix
	std::vector< std::vector< Eigen::Matrix < double, S , Dim > > > Kalman_gain_i_t_;
	std::vector< std::vector< Eigen::Matrix < double, S , S > > >   P_i_t_;
	std::vector< std::vector< Eigen::Matrix < double, S , S > > >   J_i_t_;
	// Backward: compute beta(s_{i,t})
	std::vector< std::vector< double > >                            beta_i_t_;
	std::vector< std::vector< Eigen::Matrix < double, S , S > > >   beta_V_i_t_;
	//
	// Transition and covariance matrices
	Eigen::Matrix< double, S, S >                                   A_;
	Eigen::Matrix< double, S, S >                                   Gamma_;
	//
	// Cost function
	double                                                          L_qsi_{1.e-6};
      };
    //
    //
    template< int Dim, int S > 
      P_qsi<Dim,S>::P_qsi(  const std::vector< std::vector< Eigen::Matrix < double, Dim , 1 > > >& Y ):
      Y_{Y}, n_{Y.size()}
    {
      //
      // random engine
      std::random_device rd;
      std::mt19937       generator( rd() );
      std::uniform_real_distribution< double > uniform(0., 1.);
      //
      s_.resize( n_ );
      ss_.resize( n_ );
      s_s_.resize( n_ );
      //
      alpha_i_t_.resize( n_ );
      alpha_V_i_t_.resize( n_ );
      //
      c_i_t_.resize( n_ );
      //
      Kalman_gain_i_t_.resize( n_ );
      P_i_t_.resize( n_ );
      J_i_t_.resize( n_ );
      //
      beta_i_t_.resize( n_ );
      beta_V_i_t_.resize( n_ );  
      //
      for ( int i = 0 ; i < n_ ; i++ )
	{
	  int Ti = Y_[i].size();
	  //
	  s_[i].resize( Ti );
	  ss_[i].resize( Ti );
	  s_s_[i].resize( Ti );
	  //
	  alpha_i_t_[i].resize( Ti );
	  alpha_V_i_t_[i].resize( Ti );
	  //
	  c_i_t_[i].resize( Ti );
	  //
	  Kalman_gain_i_t_[i].resize( Ti );
	  P_i_t_[i].resize( Ti );
	  J_i_t_[i].resize( Ti );
	  //
	  beta_i_t_[i].resize( Ti );
	  beta_V_i_t_[i].resize( Ti );
	  //
	  for ( int t = 0 ; t < Ti ; t++ )
	    {
	      s_[i][t]         = Eigen::Matrix < double, S , 1 >::Zero();
	      ss_[i][t]        = Eigen::Matrix < double, S , S >::Zero(); 
	      s_s_[i][t]       = Eigen::Matrix < double, S , S >::Zero(); 
	      alpha_i_t_[i][t] = 0.;
	      beta_i_t_[i][t]  = 0.;
	      c_i_t_[i][t]     = 0.;
	    }
	}
      //
      Eigen::Matrix< double, S, S > Temp = 0.01 * Eigen::Matrix< double, S, S >::Random();
      Gamma_ = 0.5 * ( Temp + Temp.transpose() ) + Eigen::Matrix< double, S, S >::Identity();
      //
      Eigen::Matrix< double, S, 1 > a1 = 
	NeuroBayes::gaussian_multivariate< S >( Eigen::Matrix< double, S, 1 >::Zero(),
						Gamma_ );
      Eigen::Matrix< double, S, 1 > a2 = 
	NeuroBayes::gaussian_multivariate< S >( Eigen::Matrix< double, S, 1 >::Zero(),
						Gamma_ );
      A_ = a1 * a2.transpose(); 
    }
    //
    //
    template< int Dim, int S > void
      P_qsi<Dim,S>::Expectation()
      {
	//
	// The E step is carried out using a dynamic programming trick which 
	// utilises the conditional independence of future hidden states from 
	// past hidden states given the setting of the current hidden state.
	//

	//
	//
	const std::vector< Eigen::Matrix< double, S, 1 > >& _mu0_ = qdch_->get_mu_0();
	const std::vector< Eigen::Matrix< double, S, S > >& _V0_  = qdch_->get_V_0();
	const Eigen::Matrix< double, Dim, S >&   _C_              = qgau_->get_C();
	const Eigen::Matrix< double, Dim, Dim >& _Sigma_          = qgau_->get_sigma();
	//
	std::vector< std::vector< Eigen::Matrix< double, S, 1 > > > duplicate_s_i_t;
	duplicate_s_i_t.resize( n_ );
	//
	for ( int i = 0 ; i < n_ ; i++ )
	  {
	    //
	    int Ti = Y_[i].size();
	    duplicate_s_i_t[i].resize( Ti );
	    //
	    // alpha calculation
	    // Since alpha(s_{t}) is the posterior probability of s_{t}
	    // given data y_{1:t}, it must sum to one
	    // 
	    // first elements
	    // Convension the first alpha is 1. Each elements will be normalized
	    Eigen::Matrix< double, Dim, Dim > K_gain_part = _C_ * _V0_[i] * _C_.transpose() + _Sigma_;
	    Kalman_gain_i_t_[i][0]  = _V0_[i] * _C_.transpose();
	    Kalman_gain_i_t_[i][0] *= NeuroBayes::inverse( K_gain_part );
	    //
	    // parameters of alpha distribution
	    s_[i][0]  = _mu0_[i];
	    s_[i][0] += Kalman_gain_i_t_[i][0] * (Y_[i][0] - _C_ * _mu0_[i]);
	    duplicate_s_i_t[i][0] = s_[i][0];
	    alpha_V_i_t_[i][0]  = Eigen::Matrix< double,S,S>::Identity() - Kalman_gain_i_t_[i][0] * _C_;
	    alpha_V_i_t_[i][0] *= _V0_[i];
	    //std::cout << "alpha_i_t_["<<i<<"][0] \n" << alpha_i_t_[i][0] << std::endl;
	    // next timepoints
	    for ( int t = 1 ; t < Ti ; t++ )
	      {
		//
		// Covariance
		P_i_t_[i][t-1] = A_ * alpha_V_i_t_[i][t-1] * A_.transpose() + Gamma_;
		// Kalman gain
		Eigen::Matrix< double, Dim, Dim > K_part = _C_*P_i_t_[i][t-1]*_C_.transpose() + _Sigma_;
		Kalman_gain_i_t_[i][t]  = P_i_t_[i][t-1] * _C_.transpose();
		Kalman_gain_i_t_[i][t] *= NeuroBayes::inverse( K_part );
		// alpha parameters
		s_[i][t]  = A_ * s_[i][t-1];
		s_[i][t] += Kalman_gain_i_t_[i][t]*(Y_[i][t] - _C_ * A_ * s_[i][t-1]);
		duplicate_s_i_t[i][t] = s_[i][t];
		alpha_V_i_t_[i][t]  = Eigen::Matrix< double,S,S>::Identity() - Kalman_gain_i_t_[i][t]*_C_;
		alpha_V_i_t_[i][t] *= P_i_t_[i][t-1];
		//
		std::cout 
		  << "Filter: s_["<<i<<"]["<<t<<"] = \n" << s_[i][t]
		  << std::endl;
	      }
	    //
	    // Beta calculation
	    // Convension the last beta is 1.
	    //beta_V_i_t_[i][Ti-1] = alpha_V_i_t_[i][Ti-1];
	    for ( int t = Ti-2 ; t >= 0 ; t-- )
	      {
		//
		J_i_t_[i][t]  = alpha_V_i_t_[i][t] * A_.transpose();
		J_i_t_[i][t] *= NeuroBayes::inverse_def_pos( P_i_t_[i][t] );
		//
		beta_V_i_t_[i][t]  = alpha_V_i_t_[i][t];
		beta_V_i_t_[i][t] += J_i_t_[i][t]*(beta_V_i_t_[i][t+1] - P_i_t_[i][t])*J_i_t_[i][t].transpose();
		//
		//s_[i][t]    += J_i_t_[i][t] * (s_[i][t+1] - s_[i][Ti-1] );
		s_[i][t]    += J_i_t_[i][t] * (s_[i][t+1] - A_ * duplicate_s_i_t[i][t] );
		ss_[i][t+1]  = J_i_t_[i][t] * beta_V_i_t_[i][t+1] + s_[i][t+1] * s_[i][t].transpose();
		s_s_[i][t]   = beta_V_i_t_[i][t] + s_[i][t] * s_[i][t].transpose();
		
		//
		std::cout 
		  << "Smoothing: s_["<<i<<"]["<<t<<"] = \n" << s_[i][t]
		  << std::endl;
	      }
	  }// for ( int i = 0 ; i < n_ ; i++ )

	//
	// Cost function
	L_qsi_ = 0.;
	//
	std::cout
	  << "A_ = \n " << A_
	  << "\n Gamma = \n " <<  Gamma_
	  << std::endl;
	//
	double ln_G = 0.;
	if ( !NeuroBayes::is_positive_definite(Gamma_) )
	  ln_G = NeuroBayes::ln_determinant( NeuroBayes::closest_sym_def_pos(Gamma_) );
	else
	  ln_G = NeuroBayes::ln_determinant( Gamma_ );
	//
	Eigen::Matrix< double, S, S > inv_G = NeuroBayes::inverse( Gamma_ );
	//
	for ( int i = 0 ; i < n_ ; i++ )
	  {
	    int Ti = Y_[i].size();
	    double L_qsi_part = 0.;
	    for ( int t = 1 ; t < Ti ; t++ )
	      {
		Eigen::Matrix < double, S , 1 > Diff = s_[i][t] - A_ * s_[i][t-1];
		L_qsi_part += Diff.transpose() * inv_G * Diff;
		//std::cout
		//  << "s_["<<i<<"]["<<t<<"] = \n" << s_[i][t]
		//  << "\n s_["<<i<<"]["<<t-1<<"] = \n" << s_[i][t-1]
		//  << "\n Diff = " << Diff
		//  << "\n Gamma_.inv() = \n" << NeuroBayes::inverse_def_pos( Gamma_ )
		//  << std::endl;
	      }
	    //
	    L_qsi_ += - 0.5 * ((Ti-1) * ln_G + L_qsi_part );
	  }
	std::cout << "Expectation: L_qsi_ = " << L_qsi_ << std::endl;
	std::cout << "L_qsi_lv_part = " << ln_G << std::endl;
      }
    //
    //
    template< int Dim, int S > void
      P_qsi<Dim,S>::Maximization()
      {
	//
	Eigen::Matrix < double, S , S >	A_part1    = Eigen::Matrix< double, S, S >::Zero();
	Eigen::Matrix < double, S , S >	A_part2    = Eigen::Matrix< double, S, S >::Zero();
	Eigen::Matrix < double, S , S >	Gamma_part = Eigen::Matrix< double, S, S >::Zero();
	//
	Gamma_ = Eigen::Matrix< double, S, S >::Zero();
	//
	for ( int i = 0 ; i < n_ ; i++ )
	  {
	    int Ti = Y_[i].size();
	    //
	    for ( int t = 1 ; t < Ti ; t++ )
	      {
		A_part1 += ss_[i][t];
		A_part2 += s_s_[i][t-1];
	      }
	  }
	//
	//A_ = A_part1 * A_part2.inverse();
	A_ = A_part1 * NeuroBayes::inverse( A_part2 );
	//
	//
	double scaling = 0;
	//
	for ( int i = 0 ; i < n_ ; i++ )
	  {
	    Gamma_part =  Eigen::Matrix< double, S, S >::Zero();
	    int Ti     = Y_[i].size();
	    for ( int t = 1 ; t < Ti ; t++ )
	      {
		Gamma_part += s_s_[i][t];
		Gamma_part -= A_ * ss_[i][t].transpose();
		Gamma_part -= ss_[i][t] * A_.transpose();
		Gamma_part += A_ * s_s_[i][t-1] * A_.transpose();
	      }
	    scaling += static_cast<double>(Ti-1);
	    Gamma_  += Gamma_part;
	  }
	//
	Gamma_ /= scaling;
	//
	// Check if the variance is symetric positive def
//	if ( NeuroBayes::is_positive_definite( Gamma_ ) )
//	  Gamma_ = NeuroBayes::closest_sym_def_pos( Gamma_ );
	//
	std::cout << "Maximization: A_ = \n" << A_  << std::endl;
	std::cout << "Maximization: Gamma_ = \n" << Gamma_  << std::endl;
      }
    //
    //
    //
    /** \class P_qdch
     *
     * \brief Initial state posterior probability distribution
     * 
     * The gaussian posterior probabilities are set for the initial 
     * porbability and the transition matrix.
     *
     * Parameters:
     * - $\mu_{0}$: initial state
     * - $V_{0}$: initial covariance matrix
     * - $L_{qdch}$: Cost function associated with the class paramteres
     *
     * Hyper parameters
     *
     */
    template< int Dim, int S >
      class P_qdch : public Posterior<Dim,S>
      {
      public:
	/** Constructor. */
	explicit P_qdch(){};
	/** Constructor. */
	explicit P_qdch( std::shared_ptr< noVB::LGSSM::P_qsi<Dim,S> >,
			 const std::vector< std::vector< Eigen::Matrix < double, Dim , 1 > > >& );
    
	/** Destructor */
	virtual ~P_qdch(){};
  
	//
	// Functions
	// Process the expectation step
	virtual void Expectation();
	virtual void Maximization();

	//
	// accessors
	const std::vector< Eigen::Matrix< double, S, 1 > >& get_mu_0() const {return mu_0_;}
	const std::vector< Eigen::Matrix< double, S, S > >& get_V_0()  const {return V_0_;}
	const double                                        get_L()    const {return L_qdch_;}
	
  
      private:
	//
	//
	std::shared_ptr< noVB::LGSSM::P_qsi<Dim,S> >                    qsi_;

	//
	//
	std::vector< std::vector< Eigen::Matrix < double, Dim , 1 > > > Y_;
	// Size of the data set
	std::size_t n_{0};

	//
	// Pi and A distribution
	// State probability
	std::vector< Eigen::Matrix< double, S, 1 > >     mu_0_;
	std::vector< Eigen::Matrix< double, S, S > >     V_0_;
	//
	// Cost function
	double                                                          L_qdch_{1.e-06};
      };
    //
    //
    template< int Dim, int S > 
      P_qdch<Dim,S>::P_qdch(       std::shared_ptr< noVB::LGSSM::P_qsi<Dim,S> >                     Qsi,
			     const std::vector< std::vector< Eigen::Matrix < double, Dim , 1 > > >& Y ):
      qsi_{Qsi}, Y_{Y}, n_{Y.size()}
    {
      //
      //
      mu_0_.resize( n_ );
      V_0_.resize( n_ );
      //
      for ( int i = 0 ; i < n_ ; i++ )
      	{
	  Eigen::Matrix< double, S, S > Temp = 0.01 * Eigen::Matrix< double, S, S >::Random();
	  V_0_[i]  = 0.5 * ( Temp + Temp.transpose() ) + Eigen::Matrix< double, S, S >::Identity();
	  //
	  mu_0_[i] = NeuroBayes::gaussian_multivariate< S >( Eigen::Matrix< double, S, 1 >::Zero(),
							     V_0_[i] );
      	}
    }
    //
    //
    template< int Dim, int S > void
      P_qdch<Dim,S>::Expectation()
      {
	//
	//
	const std::vector< std::vector< Eigen::Matrix < double, S , 1 > > > &_s_   = qsi_->get_s();
	//
	L_qdch_ = 0.;
	//
	for ( int i = 0 ; i < n_ ; i++ )
	  {
	    Eigen::Matrix < double, S , 1 > Diff = _s_[i][0] - mu_0_[i];
	    double L_qdch_part = Diff.transpose() * NeuroBayes::inverse( V_0_[i] ) * Diff;
	    //
	    if ( !NeuroBayes::is_positive_definite(V_0_[i]) )
	      L_qdch_ += - 0.5 * (NeuroBayes::ln_determinant( NeuroBayes::closest_sym_def_pos(V_0_[i]) ) + L_qdch_part );
	    else
	      L_qdch_ += - 0.5 * (NeuroBayes::ln_determinant(V_0_[i]) + L_qdch_part );
	    std::cout << "Expectation: L_qdch_ = " << L_qdch_ << std::endl;
	    std::cout << "Expectation: L_qdch_part1 = " << NeuroBayes::ln_determinant( V_0_[i] ) 
		      << std::endl;
	    std::cout << "Expectation: L_qdch_part2 = " << L_qdch_part << std::endl;
	    
	  }
      }
    //
    //
    template< int Dim, int S > void
      P_qdch<Dim,S>::Maximization()
      {
	//
	const std::vector< std::vector< Eigen::Matrix < double, S , 1 > > > &_s_   = qsi_->get_s();
	//const std::vector< std::vector< Eigen::Matrix < double, S , S > > > &_ss_  = qsi_->get_ss();
	const std::vector< std::vector< Eigen::Matrix < double, S , S > > > &_s_s_ = qsi_->get_s_s();
	//
	//
	for ( int i = 0 ; i < n_ ; i++ )
	  {
	    mu_0_[i] = _s_[i][0];
	    V_0_[i]  = _s_s_[i][0] - _s_[i][0] * _s_[i][0].transpose();
	    std::cout << "Maximization: " << std::endl;
	    //for ( int i = 0 ; i < n_ ; i++ )
	    std::cout << "mu0_["<<i<<"] = \n" << mu_0_[i]
		      << "\n V_0_["<<i<<"] = \n" << V_0_[i]
		      << std::endl;
	    //
	    // Check if the variance is symetric positive def
	    //if ( NeuroBayes::is_positive_definite( V_0_[i] ) )
	    //  V_0_[i] = NeuroBayes::closest_sym_def_pos( V_0_[i] );
	  }
      }
    //
    //
    //
    /** \class P_qgau
     *
     * \brief Gaussian posterior probabilities
     * 
     * The Gaussian posterior probabilities represent the 
     * probability density of the emission probability. 
     *
     * Parameters:
     * - $C$: Emission matrix
     * - $\Sigma$: Distribution covariance
     * - $L_{qgau}$: Cost function associated with the class paramteres
     *
     */
    template< int Dim, int S >
      class P_qgau : public Posterior<Dim,S>
      {
      public:
	/** Constructor. */
	explicit P_qgau(){};
	/** Constructor. */
	explicit P_qgau( std::shared_ptr< noVB::LGSSM::P_qsi<Dim,S> >,
			 const std::vector< std::vector< Eigen::Matrix < double, Dim, 1 > > >&,
			 const std::vector< std::vector< Eigen::Matrix < double, 1, 1 > > >& );
    
	/** Destructor */
	virtual ~P_qgau(){};
  
	//
	// Functions
	// Process the expectation step
	virtual void Expectation();
	virtual void Maximization();

	//
	// accessors
	const Eigen::Matrix< double, Dim, S >&      get_C()         const {return C_;};
	const Eigen::Matrix< double, Dim, Dim >&    get_sigma()     const {return sigma_;};
	const std::vector< std::vector< double > >& get_gamma()     const {return gamma_;};
	const std::vector< std::vector< double > >& get_ln_gamma()  const {return ln_gamma_;};
	const double                                get_L()         const {return L_qgau_;}
	//
	void set( std::shared_ptr< noVB::LGSSM::P_qsi<Dim,S> >  Qsi,
		  std::shared_ptr< noVB::LGSSM::P_qdch<Dim,S> > Qdch)
	{qsi_ = Qsi; qdch_ = Qdch;};
  
      private:
	//
	//
	std::shared_ptr< noVB::LGSSM::P_qsi<Dim,S> >                    qsi_;
	std::shared_ptr< noVB::LGSSM::P_qdch<Dim,S> >                   qdch_;
	//
	//
	std::vector< std::vector< Eigen::Matrix < double, Dim , 1 > > > Y_;
	std::vector< std::vector< Eigen::Matrix < double, 1, 1 > > >    Age_;
	// Size of the data set
	std::size_t n_{0};


	//
	// Emission distribution
	// y_{t}^{n} = C x s_{t}^{n} + N(0,Sigma) where precision = Sigma^{-1}
	Eigen::Matrix< double, Dim, S >                                 C_;
	// 
	// Gaussian distribution
	// vectors/matrices
	Eigen::Matrix< double, Dim, Dim >                               sigma_;
	// responsability
	// Gamma
	std::vector< std::vector< double > >                            gamma_;
	// log Gamma 
	std::vector< std::vector< double > >                            ln_gamma_;
	//
	// Cost function
	double                                                          L_qgau_{1.e-06};
      };
    //
    //
    //
    template< int Dim, int S > 
      P_qgau<Dim,S>::P_qgau( std::shared_ptr< noVB::LGSSM::P_qsi<Dim,S> >                           Qsi,
			     const std::vector< std::vector< Eigen::Matrix < double, Dim , 1 > > >& Y,
			     const std::vector< std::vector< Eigen::Matrix < double, 1, 1 > > >&    Age ):
      qsi_{Qsi}, Y_{Y}, Age_{Age}, n_{Y.size()}
    {
      //
      //
      //C_     = 1.e-01 * Eigen::Matrix< double, Dim, S >::Ones();
      Eigen::Matrix< double, Dim, Dim > Temp = 0.01 * Eigen::Matrix< double, Dim, Dim >::Random();
      sigma_ = 0.5 * ( Temp + Temp.transpose() ) + Eigen::Matrix< double, Dim, Dim >::Identity();
      //
      Eigen::Matrix< double, S, S > Temp2 = 0.01 * Eigen::Matrix< double, S, S >::Random();
      Eigen::Matrix< double, S, S > Temp3 = 0.5 * ( Temp2 + Temp2.transpose() ) + Eigen::Matrix< double, S, S >::Identity();
      //
      Eigen::Matrix< double, Dim, 1 > c1 = 
	NeuroBayes::gaussian_multivariate< Dim >( Eigen::Matrix< double, Dim, 1 >::Zero(),
						  sigma_ );
      Eigen::Matrix< double, S, 1 >   c2 = 
	NeuroBayes::gaussian_multivariate< S >( Eigen::Matrix< double, S, 1 >::Zero(),
						Temp3 );
      C_ = c1 * c2.transpose();
    }
    //
    //
    //
    template< int Dim, int S > void
      P_qgau<Dim,S>::Expectation()
      {
	//
	//
	const std::vector< std::vector< Eigen::Matrix < double, S , 1 > > > &_s_   = qsi_->get_s();
	//
	L_qgau_ = 0.;
	//
	//
	double ln_s = 0.;
	if ( !NeuroBayes::is_positive_definite(sigma_) )
	  ln_s = NeuroBayes::ln_determinant( NeuroBayes::closest_sym_def_pos(sigma_) );
	else
	  ln_s = NeuroBayes::ln_determinant( sigma_ );
	//
	Eigen::Matrix< double, Dim, Dim > inv_s =  NeuroBayes::inverse( sigma_ );
	for ( int i = 0 ; i < n_ ; i++ )
	  {
	    int Ti = Y_[i].size();
	    double L_qgau_part = 0.;
	    for ( int t = 0 ; t < Ti ; t++ )
	      {
		Eigen::Matrix < double, Dim , 1 > Diff = Y_[i][t] - C_ * _s_[i][t];
		L_qgau_part += Diff.transpose() * inv_s * Diff;
	      }
	    //
	    L_qgau_ += - 0.5 * (Ti * ln_s + L_qgau_part );
	  }
	std::cout << "Expectation: L_qgau_ = " << L_qgau_ << std::endl;
	std::cout << "Expectation: L_qgau_ln_part = " << ln_s << std::endl;
      }
    //
    //
    template< int Dim, int S > void
      P_qgau<Dim,S>::Maximization()
      {
      //
      //
      const std::vector< std::vector< Eigen::Matrix < double, S , 1 > > > &_s_   = qsi_->get_s();
      const std::vector< std::vector< Eigen::Matrix < double, S , S > > > &_ss_  = qsi_->get_ss();
      const std::vector< std::vector< Eigen::Matrix < double, S , S > > > &_s_s_ = qsi_->get_s_s();

      //
      //
      gamma_.resize(n_);
      ln_gamma_.resize(n_);
      //
      Eigen::Matrix < double, Dim , S >	  C_part1 = Eigen::Matrix< double, Dim, S >::Zero();
      Eigen::Matrix < double, S , S >	  C_part2 = Eigen::Matrix< double, S, S >::Zero();
      Eigen::Matrix < double, Dim , Dim > Sigma   = Eigen::Matrix< double, Dim, Dim >::Zero();
      sigma_ = Eigen::Matrix< double, Dim, Dim >::Zero();
      //
      for ( int i = 0 ; i < n_ ; i++ )
	{
	  int Ti = Y_[i].size();
	  for ( int t = 0 ; t < Ti ; t++ )
	    {
	      C_part1 += Y_[i][t] * _s_[i][t].transpose();
	      C_part2 += _s_s_[i][t];
	    }
	}
      //
      C_ = C_part1 * NeuroBayes::inverse( C_part2 );
      //
      //
      double scaling = 0.;
      for ( int i = 0 ; i < n_ ; i++ )
	{
	  int Ti = Y_[i].size();
	  Sigma  = Eigen::Matrix< double, Dim, Dim >::Zero();
	  //
	  for ( int t = 0 ; t < Ti ; t++ )
	    {
	      Sigma  = Y_[i][t] * Y_[i][t].transpose();
	      Sigma -= C_ * _s_[i][t] * Y_[i][t].transpose();
	      Sigma -= Y_[i][t] * _s_[i][t].transpose() * C_.transpose();
	      Sigma += C_ * _s_s_[i][t] * C_.transpose();
	      //
	      std::cout 
		<< "Y_["<<i<<"]["<<t<<"] = \n" << Y_[i][t] 
		<< "\n C * s["<<i<<"]["<<t<<"] = \n " << C_ * _s_[i][t] 
		<< std::endl;
	    }
	  scaling += static_cast<double>(Ti);
	  sigma_  += Sigma;
	}
      //
      sigma_ /= scaling;
      //
      // Check if the variance is symetric positive def
//      if ( NeuroBayes::is_positive_definite( sigma_ ) )
//	sigma_ = NeuroBayes::closest_sym_def_pos( sigma_ );
      //
      std::cout << "Maximization: C_ = \n" << C_  << std::endl;
      std::cout << "Maximization: sigma_ = \n" << sigma_  << std::endl;
      }
  }
}
#endif
