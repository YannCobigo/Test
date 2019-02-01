#ifndef HMM_VARIATIONALPOSTERIOR_H
#define HMM_VARIATIONALPOSTERIOR_H
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
#define ln_2    0.6931471805599453L
#define ln_2_pi 1.8378770664093453L
#define ln_pi   1.1447298858494002L
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
namespace VB 
{
  namespace HMM
  { 
    template< int Dim, int S > class VP_qsi;
    template< int Dim, int S > class VP_qdch;
    template< int Dim, int S > class VP_qgau;

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
     *  @phdthesis{beal2003,
     *   added-at = {2010-03-25T16:34:19.000+0100},
     *   author = {Beal, Matthew J.},
     *   biburl = {https://www.bibsonomy.org/bibtex/223a0ca246a6d81fe92a70bbcda7dc1fb/3mta3},
     *   file = {beal2003.pdf:Papers/beal2003.pdf:PDF},
     *   interhash = {c7675c921755e23c8cc2377c0c0c387c},
     *   intrahash = {23a0ca246a6d81fe92a70bbcda7dc1fb},
     *   keywords = {Variationalmethods},
     *   school = {Gatsby Computational Neuroscience Unit, University College London},
     *   timestamp = {2010-03-25T16:34:19.000+0100},
     *   title = {Variational Algorithms for Approximate Bayesian Inference},
     *   url = {http://www.cse.buffalo.edu/faculty/mbeal/thesis/index.html},
     *   year = 2003
     * }
     *
     * - Dim: number of features
     * - n: number of entries
     * - S: number of states
     *
     *
     */
    template< int Dim, int S >
      class VariationalPosterior
    {
    public:
  
      //
      // Functions
      // Process the expectation step
      using Var_post = std::tuple<
	VP_qsi<Dim,S>,
	VP_qdch<Dim,S>,
	VP_qgau<Dim,S>
	>;
      virtual void Expectation( const Var_post& )  = 0;
      virtual void Maximization( const Var_post& ) = 0;
    };
    //
    //
    //
    /** \class VP_qsi
     *
     * \brief 
     * 
     * Parameters:
     * - responsability (gamma): 
     *      Matrix[n x S] n (number of inputs) and S (number of states).
     *
     */
    template< int Dim, int S >
      class VP_qsi : public VariationalPosterior<Dim,S>
      {
      public:
	/** Constructor. */
	explicit VP_qsi(){};
	/** Constructor. */
	explicit VP_qsi( const std::vector< std::vector< Eigen::Matrix < double, Dim , 1 > > >& );
    
	/** Destructor */
	virtual ~VP_qsi(){};
  
	//
	// Functions
	// Process the expectation step
	using Var_post = std::tuple<
	  VP_qsi<Dim,S>,
	  VP_qdch<Dim,S>,
	  VP_qgau<Dim,S>
	  >;
	virtual void Expectation( const Var_post& );
	virtual void Maximization( const Var_post& );

	//
	// accessors
	const inline double                                                         get_F()  const {return F_qsi_;}
	const        std::vector< std::vector< Eigen::Matrix < double, S , 1 > > >& get_s()  const {return s_;}
	const        std::vector< std::vector< Eigen::Matrix < double, S , S > > >& get_ss() const {return ss_;}

      private:
	//
	//
	std::vector< std::vector< Eigen::Matrix < double, Dim , 1 > > > Y_;
	// Size of the data set
	std::size_t n_{0};

	//
	// hidden state
	// <s_{i,t}>
	std::vector< std::vector< Eigen::Matrix < double, S , 1 > > > s_;
	// <s_{i,t-1} x s_{i,t}>
	std::vector< std::vector< Eigen::Matrix < double, S , S > > > ss_;
	//
	// Dynamic programing
	// Forward:  compute alpha(s_{i,t})
	std::vector< std::vector< Eigen::Matrix < double, S , 1 > > > alpha_i_t_;
	// Backward: compute beta(s_{i,t})
	std::vector< std::vector< Eigen::Matrix < double, S , 1 > > > beta_i_t_;
	//
	// responsability
	std::vector< std::vector< double > > gamma_;
	// log marginal likelihood lower bound: \qsi component
	double F_qsi_{-1.e06};
      };
    //
    //
    template< int Dim, int S > 
      VP_qsi<Dim,S>::VP_qsi(  const std::vector< std::vector< Eigen::Matrix < double, Dim , 1 > > >& Y ):
      Y_{Y}, n_{Y.size()}
    {
      //
      s_.resize(n_);
      ss_.resize(n_);
      alpha_i_t_.resize(n_);
      beta_i_t_.resize(n_);
      //
      //
      for ( int i = 0 ; i < n_ ; i++ )
	{
	  int Ti = Y_[i].size();
	  //
	  s_[i].resize(Ti);
	  ss_[i].resize(Ti);
	  alpha_i_t_[i].resize(Ti);
	  beta_i_t_[i].resize(Ti);
	  //
	  for ( int t = 0 ; t < Ti ; t++ )
	    {
	      s_[i][t]         = Eigen::Matrix < double, S , 1 >::Zero();
	      ss_[i][t]        = Eigen::Matrix < double, S , S >::Zero();
	      alpha_i_t_[i][t] = Eigen::Matrix < double, S , 1 >::Zero();
	      beta_i_t_[i][t]  = Eigen::Matrix < double, S , 1 >::Zero();
	    }
	}
    }
    //
    //
    template< int Dim, int S > void
      VP_qsi<Dim,S>::Expectation( const Var_post& VP )
      {
	//
	// The E step is carried out using a dynamic programming trick which 
	// utilises the conditional independence of future hidden states from 
	// past hidden states given the setting of the current hidden state.
	//

	//
	//
	F_qsi_ = 0.;
	const VP_qdch<Dim,S>  &qdch  = std::get< QDCH >( VP );
	const VP_qgau<Dim,S>  &qgau  = std::get< QGAU >( VP );
	//
	const Eigen::Matrix< double, S, 1 >                                 &_pi_ = qdch.get_pi();
	const Eigen::Matrix< double, S, S >                                 &_A_  = qdch.get_A();
	const std::vector< std::vector< Eigen::Matrix < double, S , 1 > > > &_N_  = qgau.get_N();
	//
	//
	for ( int i = 0 ; i < n_ ; i++ )
	  {
	    //
	    int Ti = Y_[i].size();
	    std::vector< double > scale(Ti,0.);

	    //
	    // alpha calculation
	    // Since alpha(s_{t}) is the posterior probability of s_{t}
	    // given data y_{1:t}, it must sum to one
	    // 
	    // first elements
	    // Convension the first alpha is 1.
	    scale[0] = static_cast< double >(S);
	    F_qsi_  += log(scale[0]);
	    alpha_i_t_[i][0] = Eigen::Matrix < double, S , 1 >::Ones() / scale[0];
	    // each elements will be normalized to one
	    alpha_i_t_[i][1] = (alpha_i_t_[i][0].transpose() * _pi_)(0,0) * _N_[i][1];
	    scale[1] = alpha_i_t_[i][1].sum();
	    F_qsi_  += log(scale[1]);
	    alpha_i_t_[i][1] /= scale[1];
	    //
	    for ( int t = 2 ; t < Ti ; t++ )
	      {
		// mult with array is a coefficient-wise multiplication
		alpha_i_t_[i][t]  = _N_[i][t].array() * (_A_.transpose() * alpha_i_t_[i][t-1]).array();
		scale[t]          = alpha_i_t_[i][t].sum();
		F_qsi_           += log(scale[t]);
		alpha_i_t_[i][t] /= scale[t];
	      }
	    //
	    // Beta calculation
	    // Convension the last beta is 1.
	    beta_i_t_[i][Ti-1] = Eigen::Matrix < double, S , 1 >::Ones() / scale[Ti-1];
	    //
	    for ( int t = Ti-2 ; t >= 0 ; t-- )
	      {
		beta_i_t_[i][t] = Eigen::Matrix < double, S , 1 >::Zero();
		for ( int s = 0 ; s < S ; s++ )
		  for ( int ss = 0 ; ss < S ; ss++ )
		    beta_i_t_[i][t](s,0) += _A_(s,ss) * _N_[i][t+1](ss,0) * beta_i_t_[i][t+1](ss,0);
		//
		beta_i_t_[i][t] /= scale[t];
	      }
	    //
	    // <s_{i,t}> && <s_{i,t-1} x s_{i,t}>
	    for ( int t = 0 ; t < Ti ; t++ )
	      {
		//
		// <s_{i,t}>
		std::cout << "(" << i << "," << t << ")" << std::endl;
		std::cout << "alpha_i_t_[i][t] \n" << alpha_i_t_[i][t] << std::endl;
		std::cout << "beta_i_t_[i][t] \n" << beta_i_t_[i][t] << std::endl;

		s_[i][t]  = alpha_i_t_[i][t].array() * beta_i_t_[i][t].array();

		std::cout << "s_[i][t] \n" << s_[i][t]<< std::endl;
		std::cout << "alpha_i_t_[i][t] " << alpha_i_t_[i][t] << std::endl;
		std::cout << "beta_i_t_[i][t] " << beta_i_t_[i][t] << std::endl;
		std::cout << "s_[i][t].sum() " << s_[i][t].sum() << std::endl;
		s_[i][t] /= s_[i][t].sum();
		//
		//  <s_{i,t-1} x s_{i,t}>
		if ( t > 0 )
		  {
		    double norm = 0.;
		    for ( int s = 0 ; s < S ; s++ )
		      for ( int ss = 0 ; ss < S ; ss++ )
			{
			  ss_[i][t](s,ss)  = alpha_i_t_[i][t-1](s,0)*_A_(s,ss);
			  ss_[i][t](s,ss) *= beta_i_t_[i][t](ss,0)*_N_[i][t](ss,0);
			  //
			  norm +=  ss_[i][t](s,ss);
			}
		    //
		    ss_[i][t] /= norm;
		  }
	      }
	  }
      }
    //
    //
    template< int Dim, int S > void
      VP_qsi<Dim,S>::Maximization( const Var_post& VP )
      {
      }
    //
    //
    //
    /** \class VP_qdch
     *
     * \brief Dirichlet posterior probabilities
     * 
     * The Dirichlet posterior probabilities are set for the initial 
     * porbability and the transition matrix.
     *
     * Parameters:
     * - 
     *
     * Hyper parameters
     *
     */
    template< int Dim, int S >
      class VP_qdch : public VariationalPosterior<Dim,S>
      {
      public:
	/** Constructor. */
	explicit VP_qdch(){};
	/** Constructor. */
	explicit VP_qdch( const std::vector< std::vector< Eigen::Matrix < double, Dim , 1 > > >& );
    
	/** Destructor */
	virtual ~VP_qdch(){};
  
	//
	// Functions
	// Process the expectation step
	using Var_post = std::tuple<
	  VP_qsi<Dim,S>,
	  VP_qdch<Dim,S>,
	  VP_qgau<Dim,S>
	  >;
	virtual void Expectation( const Var_post& );
	virtual void Maximization( const Var_post& );

	//
	// accessors
	const inline double                        get_F()   const {return F_qdch_;}
	const        Eigen::Matrix< double, S, 1 >& get_pi() const {return posterior_pi_;}
	const        Eigen::Matrix< double, S, S >& get_A()  const {return posterior_A_;}

  
      private:
	//
	// Compute local Kullback-Leibler divergence for Dirichlet distributions
	// KL(Q||P) = \int dX \, Q(X) \ln (Q(X)/P(X))
	double KL_Dirichlet_(const Eigen::Matrix< double, S, 1 >&,
			     const Eigen::Matrix< double, S, 1 >& ) const;

	//
	//
	std::vector< std::vector< Eigen::Matrix < double, Dim , 1 > > > Y_;
	// Size of the data set
	std::size_t n_{0};

	//
	// Pi distribution
	double alpha_pi_0_{1.};
	Eigen::Matrix< double, S, 1 > alpha_pi_;
	// Posterior density of the state
	Eigen::Matrix< double, S, 1 > posterior_pi_;

	//
	// A distribution
	double alpha_A_0_{1.};
	Eigen::Matrix< double, S, S > alpha_A_;
	// Posterior density of the state
	Eigen::Matrix< double, S, S > posterior_A_;

	//
	// log marginal likelihood lower bound: \qsi component
	double F_qdch_{-1.e06};
      };
    //
    //
    template< int Dim, int S > 
      VP_qdch<Dim,S>::VP_qdch(  const std::vector< std::vector< Eigen::Matrix < double, Dim , 1 > > >& Y ):
      Y_{Y}, n_{Y.size()}
    {
      //
      //
      alpha_pi_ = ( alpha_pi_0_ / static_cast< double >(S) ) * Eigen::Matrix< double, S, 1 >::Ones();
      alpha_A_  = ( alpha_A_0_  / static_cast< double >(S) ) * Eigen::Matrix< double, S, S >::Ones();
      //
      // Posterior density of the state
      // We initialize the posterior with a Dirichlet distribution
      std::default_random_engine generator;
      std::gamma_distribution< double > distribution_pi( alpha_pi_0_ / static_cast< double >(S),
							 1.0 );
      std::gamma_distribution< double > distribution_A( alpha_A_0_ / static_cast< double >(S),
							 1.0 );
      //
      posterior_pi_ = Eigen::Matrix< double, S, 1 >::Zero();
      posterior_A_  = Eigen::Matrix< double, S, S >::Zero();
      //
      for ( int s = 0 ; s < S ; s++ ) 
	{
	  posterior_pi_(s,0) = distribution_pi( generator );
	  for ( int ss = 0 ; ss < S ; ss++ )
	    posterior_A_(s,ss) = distribution_A( generator );
	}
      std::cout << "posterior_pi_ = \n" << posterior_pi_ << std::endl;
      std::cout << "posterior_A_ = \n" << posterior_A_ << std::endl;
    }
    //
    //
    template< int Dim, int S > double
      VP_qdch<Dim,S>::KL_Dirichlet_( const Eigen::Matrix< double, S, 1 > &Alpha_Q,
				     const Eigen::Matrix< double, S, 1 > &Alpha_P  ) const 
      {
	//std::cout << Alpha_Q << std::endl;
	double Alpha_Q_sum = Alpha_Q.sum();
	double KL = gsl_sf_lngamma( Alpha_Q_sum ) - gsl_sf_lngamma( Alpha_P.sum() );
	//
	for ( int s = 0 ; s < S ; s++ )
	  {
	    KL -= gsl_sf_lngamma( Alpha_Q(s,0) ) - gsl_sf_lngamma( Alpha_P(s,0) );
	    KL += (Alpha_Q(s,0)-Alpha_P(s,0)) * (gsl_sf_psi(Alpha_Q(s,0))-gsl_sf_psi(Alpha_Q_sum));
	  }

	//
	//
	return KL;
      }
    //
    //
    template< int Dim, int S > void
      VP_qdch<Dim,S>::Expectation( const Var_post& VP )
      {
      }
    //
    //
    template< int Dim, int S > void
      VP_qdch<Dim,S>::Maximization( const Var_post& VP )
      {
	//
	//
	const VP_qsi<Dim,S>  &qsi  = std::get< QSI >( VP );
	//
	const std::vector< std::vector< Eigen::Matrix < double, S , 1 > > > &_s_  = qsi.get_s();
	const std::vector< std::vector< Eigen::Matrix < double, S , S > > > &_ss_ = qsi.get_ss();
	//
	Eigen::Matrix< double, S, 1 > mean_s1 = Eigen::Matrix< double, S, 1 >::Zero();
	Eigen::Matrix< double, S, S > mean_ss = Eigen::Matrix< double, S, S >::Zero();
	//
	for ( int i = 0 ; i < n_ ; i++ )
	  {
	    int Ti = Y_[i].size();
	    mean_s1 += _s_[i][0];
	    std::cout << mean_s1 << std::endl;
	    std::cout << _s_[i][0] << std::endl;
	    //
	    for ( int t = 1 ; t < Ti ; t++ )
	      mean_ss += _ss_[i][t];
	  }

	//
	// Posterior Dirichlet parameters
	//
	double                        prior_pi       = alpha_pi_0_ / static_cast< double >(S);
	double                        prior_A        = alpha_A_0_  / static_cast< double >(S);
	double                        alpha_pi_sum   = 0.;
	Eigen::Matrix< double, S, 1 > alpha_A_sum    = Eigen::Matrix< double, S, 1 >::Zero();
	Eigen::Matrix< double, S, 1 > alpha_pi_prior = prior_pi * Eigen::Matrix< double, S, 1 >::Ones();
	Eigen::Matrix< double, S, 1 > alpha_A_prior  = prior_A  * Eigen::Matrix< double, S, 1 >::Ones();
	
	//
	alpha_pi_    = alpha_pi_prior + mean_s1;
	alpha_pi_sum = alpha_pi_.sum();
	//
	for ( int s = 0 ; s < S ; s++ )
	  {
	    // A
	    for ( int ss = 0 ; ss < S ; ss++ )
	      {
		alpha_A_(s,ss)     = prior_A + mean_ss(s,ss);
		alpha_A_sum(ss,0) += alpha_A_(s,ss);
	      }
	  }

	//
	// log marginal likelihood lower bound
	std::cout << "alpha_pi_ \n" << alpha_pi_ << std::endl;
	std::cout << "alpha_pi_prior \n" << alpha_pi_prior << std::endl;
	double F_pi = - KL_Dirichlet_( alpha_pi_,  alpha_pi_prior );
	double F_A  = 0.;
	//
	for ( int s = 0 ; s < S ; s++ )
	  F_A -= KL_Dirichlet_( alpha_A_.col(s).transpose(),  alpha_A_prior );

	//
	// update the posterior proba density
	for ( int s = 0 ; s < S ; s++ )
	  {
	    // Pi
	    posterior_pi_(s,0) = exp( gsl_sf_psi(alpha_pi_(s,0)) - gsl_sf_psi(alpha_pi_sum) );
	    // A
	    for ( int ss = 0 ; ss < S ; ss++ )
	      {
		posterior_A_(s,ss) = exp( gsl_sf_psi(alpha_A_(s,ss)) - gsl_sf_psi(alpha_A_sum(ss,0)) );
	      }
	  }
	//
	F_qdch_ = F_pi + F_A;
      }
    //
    //
    //
    /** \class VP_qgau
     *
     * \brief Gaussian posterior probabilities
     * 
     * The Gaussian posterior probabilities represent the 
     * probbility density of the emission probability. 
     * The prior probability is Gaussin-Wishart density.
     *
     * Parameters:
     * - 
     *
     * Hyper parameters
     *
     *
     */
    template< int Dim, int S >
      class VP_qgau : public VariationalPosterior<Dim,S>
      {
      public:
	/** Constructor. */
	explicit VP_qgau(){};
	/** Constructor. */
	explicit VP_qgau( const std::vector< std::vector< Eigen::Matrix < double, Dim , 1 > > >& );
    
	/** Destructor */
	virtual ~VP_qgau(){};
  
	//
	// Functions
	// Process the expectation step
	using Var_post = std::tuple<
	  VP_qsi<Dim,S>,
	  VP_qdch<Dim,S>,
	  VP_qgau<Dim,S>  >;
	virtual void Expectation( const Var_post& );
	virtual void Maximization( const Var_post& );

	//
	// accessors
	const inline double                                                         get_F() const {return F_qgau_;}
	const        std::vector< std::vector< Eigen::Matrix < double, S , 1 > > >& get_N() const {return posteriror_N_;}
  
      private:
	//
	//
	std::vector< std::vector< Eigen::Matrix < double, Dim , 1 > > > Y_;
	// Size of the data set
	std::size_t n_{0};

	//
	//
	// Gaussian-Wishart
	//
	// Gaussian
	// scalars
	double beta_0_{1.};
	std::vector< double > beta_{S};
	// vectors
	std::vector< Eigen::Matrix< double, Dim, 1 > > mu_0_{S};
	std::vector< Eigen::Matrix< double, Dim, 1 > > mu_mean_{S};
	//
	// Wishart
	// scalars
	double nu_0_{50.};
	std::vector< double > nu_{S};
	// vectors/matrices
	Eigen::Matrix< double, Dim, Dim >                S_0_inv_{Eigen::Matrix< double, Dim, Dim >::Zero()};
	std::vector< Eigen::Matrix< double, Dim, Dim > > S_mean_inv_{S};
	std::vector< Eigen::Matrix< double, Dim, 1 > >   mu_0_mean_{S};
 

	//
	// Mean N over the posterior proba
	std::vector< std::vector< Eigen::Matrix < double, S , 1 > > > posteriror_N_;
	// log marginal likelihood lower bound: \qgau component
	double F_qgau_{-1.e06};
      };
    //
    //
    //
    template< int Dim, int S > 
      VP_qgau<Dim,S>::VP_qgau(  const std::vector< std::vector< Eigen::Matrix < double, Dim , 1 > > >& Y ):
      Y_{Y}, n_{Y.size()}
    {
      //
      for ( int s = 0 ; s < S ; s++ )
	{
	  // Gaussian part
	  mu_0_[s]       = Eigen::Matrix< double, Dim, 1 >::Zero();
	  mu_mean_[s]    = Eigen::Matrix< double, Dim, 1 >::Zero();
	  beta_[s]       = beta_0_;
	  // Wishart part
	  S_mean_inv_[s] = S_0_inv_;
	  mu_0_mean_[s]  = Eigen::Matrix< double, Dim, 1 >::Zero();
	}
      //
      posteriror_N_.resize(n_);
      for ( int i = 0 ; i < n_ ; i++ )
	{
	  int Ti = Y_[i].size();
	  posteriror_N_[i].resize( Ti );
	  //
	  // 
	  for ( int t = 0 ; t < Ti ; t++ )
	    posteriror_N_[i][t] = (1. / static_cast<double>(S) ) * Eigen::Matrix < double, S , 1 >::Ones();
	}
    }
    //
    //
    template< int Dim, int S > void
      VP_qgau<Dim,S>::Expectation( const Var_post& VP )
      {
	//
	//
	const VP_qsi<Dim,S>  &qsi  = std::get< QSI >( VP );
	//
	const std::vector< std::vector< Eigen::Matrix < double, S , 1 > > > &_s_ = qsi.get_s();
	
	//
	//
	std::vector< double >                            Delta(S,0);
	std::vector< Eigen::Matrix< double, Dim, 1 > >   y_mean( S, Eigen::Matrix< double, Dim, 1 >::Zero() );
	std::vector< Eigen::Matrix< double, Dim, Dim > > W_mean_inv( S, Eigen::Matrix< double, Dim, Dim >::Zero() );
	//
	for ( int s = 0 ; s < S ; s++ )
	  {
	    //
	    // Re-initialise
	    // Gaussian part
	    mu_mean_[s]    = Eigen::Matrix< double, Dim, 1 >::Zero();
	    beta_[s]       = beta_0_;
	    // Wishart part
	    S_mean_inv_[s] = S_0_inv_;
	    mu_0_mean_[s]  = Eigen::Matrix< double, Dim, 1 >::Zero();
      
	    //
	    // Build the means over the measures
	    for ( int i = 0 ; i < n_ ; i++ )
	      {
		int Ti = Y_[i].size();
		for ( int t = 0 ; t < Ti ; t++ )
		  {
		    y_mean[s] += _s_[i][t](s,0) * Y_[i][t];
		    Delta[s]  += _s_[i][t](s,0);
		  }
	      }
	    //
	    for ( int i = 0 ; i < n_ ; i++ )
	      {
		typename std::vector< Eigen::Matrix < double, Dim , 1 > >::const_iterator t;
		int Ti = Y_[i].size();
		for ( int t = 0 ; t < Ti ; t++ )
		  {
		    Eigen::Matrix< double, Dim, 1 > diff_vect = Y_[i][t] - y_mean[s] / (beta_0_ * (beta_[s] - beta_0_));
		    W_mean_inv[s] += _s_[i][t](s,0) * diff_vect * diff_vect.transpose(); 
		  }
	      }

	    //
	    //
	    beta_[s] += Delta[s];
	    nu_[s]   += Delta[s];
	    //
	    mu_mean_[s]     = ( beta_0_ * mu_0_[s] + y_mean[s] ) / beta_[s];
	    mu_0_mean_[s]   = y_mean[s] / ( beta_[s] - beta_0_ );
	    //
	    Eigen::Matrix< double, Dim, 1 > diff_mus = mu_0_[s] - mu_0_mean_[s];
	    S_mean_inv_[s] += beta_0_ * ( beta_[s] - beta_0_ ) * diff_mus * diff_mus.transpose() / beta_[s];
	    S_mean_inv_[s] += W_mean_inv[s];
	  }
      }
    //
    //
    template< int Dim, int S > void
      VP_qgau<Dim,S>::Maximization( const Var_post& VP )
      {
	//posteriror_N_
	std::vector< std::vector< Eigen::Matrix < double, S , 1 > > > ln_posteriror_N( posteriror_N_ );
	double
	  cd1       = - Dim * ln_2_pi,
	  cd2       = 0., // <ln|lambda_j|>
	  cd3       = S_0_inv_.inverse().determinant(), // ln|S_0|
	  cd4       = S * Dim * log(beta_0_), // 
	  cd5       = 1., // ln(prod beta_j)
	  cd6       = 0., //
	  cd7       = 0., // Sum nu_j
	  cd8       = 0., // Sum nu_j * ln|Sj|
	  diff_ln_Z = 0.;
	// log marginal likelihood lower bound
	F_qgau_ = 0.;
	//
	for ( int s = 0 ; s < S ; s++ )
	  {
	    //
	    //
	    cd2  = Dim * ln_2;
	    cd5 *= beta_[s];
	    //
	    // ln|S| = 2 * sum_i ln(Lii)
	    // where S = LL^T
	    Eigen::Matrix< double, Dim, Dim > S_mean = S_mean_inv_[s].inverse();
	    double ln_Sdet = NeuroBayes::ln_determinant( S_mean );
	    //
	    for ( int u = 0 ; u < Dim ; u++ )
	      {
		cd2 += gsl_sf_psi( 0.5*(nu_[s] + 1 - u) );
		//
		diff_ln_Z += gsl_sf_lngamma( 0.5*(nu_[s] + 1 - u) ) - gsl_sf_lngamma( 0.5*(nu_0_ + 1 - u) );
	      }
	    //
	    cd2 += ln_Sdet;
	    //
	    //
	    Eigen::Matrix < double, Dim , 1 > diff_vec_0 = mu_mean_[s] - mu_0_[s];
	    cd6  = nu_[s] * ( Dim - (S_0_inv_*S_mean).trace() ) + (nu_0_ - nu_[s])*cd2;
	    cd6 -= nu_[s] * beta_0_ * (diff_vec_0.transpose() * S_mean * diff_vec_0)(0,0);
	    //
	    cd7 += nu_[s];
	    cd8 += nu_[s] * ln_Sdet;
	    //
	    cd3 = nu_[s] * beta_[s];
	    //
	    for ( int i = 0 ; i < n_ ; i++ )
	      {
		int Ti = Y_[i].size();
		for ( int t = 0 ; t < Ti ; t++ )
		  {
		    //
		    //
		    Eigen::Matrix < double, Dim , 1 > diff_vec = Y_[i][t] - mu_mean_[s];
		    ln_posteriror_N[i][t](s,0)  = cd1 + cd2;
		    ln_posteriror_N[i][t](s,0) -= cd3 * (diff_vec.transpose() * S_mean * diff_vec)(0,0);  
		    ln_posteriror_N[i][t](s,0) *= 0.5;  
		    //
		    posteriror_N_[i][t](s,0)    = exp( ln_posteriror_N[i][t](s,0) );
		  }
	      }
	  }
	//
	diff_ln_Z += 0.5 * Dim * ln_2 * ( cd7 - nu_0_ );
	diff_ln_Z += 0.5 * ( cd8 - S * nu_0_ * cd3 );
	//
	F_qgau_  = cd4 - Dim * log(cd5) + cd6;
	F_qgau_ *= 0.5;
	F_qgau_ += diff_ln_Z;
      }
  }
}
#endif
