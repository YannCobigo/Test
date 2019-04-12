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
      //virtual ~VariationalPosterior(){};
      virtual void Expectation()  = 0;
      virtual void Maximization() = 0;
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
	virtual void Expectation();
	virtual void Maximization();

	//
	// accessors
	const inline double                                                         get_F()  const {return F_qsi_;}
	const        std::vector< std::vector< Eigen::Matrix < double, S , 1 > > >& get_s()  const {return s_;}
	const        std::vector< std::vector< Eigen::Matrix < double, S , S > > >& get_ss() const {return ss_;}
	//
	void set( std::shared_ptr< VB::HMM::VP_qdch<Dim,S> > Qdch,
		  std::shared_ptr< VB::HMM::VP_qgau<Dim,S> > Qgau )
	{qdch_ = Qdch; qgau_ = Qgau;};

      private:
	//
	//
	std::shared_ptr< VB::HMM::VP_qdch<Dim,S> > qdch_;
	std::shared_ptr< VB::HMM::VP_qgau<Dim,S> > qgau_;


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
      VP_qsi<Dim,S>::Expectation()
      {
	//
	// The E step is carried out using a dynamic programming trick which 
	// utilises the conditional independence of future hidden states from 
	// past hidden states given the setting of the current hidden state.
	//

	//
	//
	F_qsi_ = 0.;
	//
	const Eigen::Matrix< double, S, 1 >                                 &_pi_ = qdch_->get_pi();
	const Eigen::Matrix< double, S, S >                                 &_A_  = qdch_->get_A();
	const std::vector< std::vector< Eigen::Matrix < double, S , 1 > > > &_N_  = qgau_->get_N();
	//
	Eigen::Matrix< double, S, 1 > update_posterior_alpha_pi = Eigen::Matrix< double, S, 1 >::Zero();
	Eigen::Matrix< double, S, S > update_posterior_alpha_A  = Eigen::Matrix< double, S, S >::Zero();
	  
	//std::cout << "VP_qsi<Dim,S>::Expectation" << std::endl;
	//std::cout << "_pi_ \n" << _pi_ << std::endl;
	//std::cout << "_A_  \n" << _A_ << std::endl;
	//std::cout << "_N_ 0 \n" << _N_[0][0] << std::endl;
	//std::cout << "_N_ 1 \n" << _N_[1][0] << std::endl;
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
	    // Convension the first alpha is 1. Each elements will be normalized
	    alpha_i_t_[i][0] = _N_[i][0].array() * _pi_.array();
	    scale[0]          = alpha_i_t_[i][0].sum();
	    F_qsi_           += log(scale[0]);
	    alpha_i_t_[i][0] /= scale[0];
	    //std::cout << "alpha_i_t_["<<i<<"][0] \n" << alpha_i_t_[i][0] << std::endl;
	    // next timepoints
	    for ( int t = 1 ; t < Ti ; t++ )
	      {
		// mult with array is a coefficient-wise multiplication
		alpha_i_t_[i][t]  = _N_[i][t].array() * (_A_*alpha_i_t_[i][t-1]).array();
		scale[t]          = alpha_i_t_[i][t].sum();
		F_qsi_           += log(scale[t]);
		alpha_i_t_[i][t] /= scale[t];
		//std::cout << "alpha_i_t_["<<i<<"]["<<t<<"] \n" << alpha_i_t_[i][t] << std::endl;
	      }
	    //
	    // Beta calculation
	    // Convension the last beta is 1.
	    beta_i_t_[i][Ti-1] = Eigen::Matrix < double, S , 1 >::Ones() / static_cast<double>(Ti/*S*/);
	    //
	    for ( int t = Ti-2 ; t >= 0 ; t-- )
	      {
		//
		beta_i_t_[i][t] = Eigen::Matrix < double, S , 1 >::Zero();
		double beta_norm = 0.;
		//
		beta_i_t_[i][t] = _N_[i][t+1].array() *(_A_*beta_i_t_[i][t+1]).array();
		//
		//std::cout << "_A_ \n " << _A_ << std::endl;
		//std::cout << "_N_["<<i<<"][t+1:"<<t+1<<"] \n" << _N_[i][t+1] << std::endl;
		//std::cout << "beta_i_t_["<<i<<"][t+1:"<<t+1<<"] \n" << beta_i_t_[i][t+1] << std::endl;
		//std::cout << "beta_i_t_["<<i<<"][t:"<<t<<"] \n" << beta_i_t_[i][t] << std::endl;
	      }
	    //
	    // <s_{i,t}> && <s_{i,t-1} x s_{i,t}>
	    for ( int t = 0 ; t < Ti ; t++ )
	      {
		//
		// <s_{i,t}>
		s_[i][t]  = alpha_i_t_[i][t].array() * beta_i_t_[i][t].array();
		s_[i][t] /= s_[i][t].sum();

		//std::cout << "alpha_i_t_["<<i<<"]["<<t<<"] \n" << alpha_i_t_[i][t] << std::endl;
		//std::cout << "beta_i_t_["<<i<<"]["<<t<<"] \n" << beta_i_t_[i][t] << std::endl;
		std::cout << "s_["<<i<<"]["<<t<<"] \n" << s_[i][t]<< std::endl;

		//
		//  <s_{i,t-1} x s_{i,t}>
		if ( t > 0 )
		  {
		    Eigen::Matrix < double, S , S > B = Eigen::Matrix < double, S , S >::Zero();
		    double norm = 0.;
		    //
		    B = alpha_i_t_[i][t-1] * (beta_i_t_[i][t].array()*_N_[i][t].array()).matrix().transpose();
		    //
		    for ( int s = 0 ; s < S ; s++ )
		      for ( int ss = 0 ; ss < S ; ss++ )
			{
			  ss_[i][t](s,ss)  = _A_(s,ss) * B(s,ss);
			  //ss_[i][t](s,ss)  = alpha_i_t_[i][t-1](s,0)*_A_(s,ss);
			  //ss_[i][t](s,ss) *= beta_i_t_[i][t](ss,0)*_N_[i][t](ss,0);
			  //
			  norm +=  ss_[i][t](s,ss);
			}
		    //
		    ss_[i][t] /= norm;
		    //std::cout << "ss_["<<i<<"]["<<t<<"] \n" <<  ss_[i][t] << std::endl;
		  }
		//
		update_posterior_alpha_A += ss_[i][t];
	      }

	    //
	    //
	    update_posterior_alpha_pi += s_[i][0];
	  }// for ( int i = 0 ; i < n_ ; i++ )

	//
	// Update of the Posterior values 
	// Dirichlet
	qdch_->update_alpha_pi( update_posterior_alpha_pi );
	qdch_->update_alpha_A( update_posterior_alpha_A );
	//
	//std::cout << "update_posterior_alpha_pi\n" << update_posterior_alpha_pi << std::endl;
      }
    //
    //
    template< int Dim, int S > void
      VP_qsi<Dim,S>::Maximization()
      {}
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
	virtual void Expectation();
	virtual void Maximization();

	//
	// accessors
	const inline double                         get_F()  const {return F_qdch_;}
	const        Eigen::Matrix< double, S, 1 >& get_pi() const {return posterior_pi_;}
	const        Eigen::Matrix< double, S, S >& get_A()  const {return posterior_A_;}
	//
	void update_alpha_pi( const Eigen::Matrix< double, S, 1 > Update )
	{posterior_alpha_pi_ = Update + alpha_pi_;};
	void update_alpha_A( const Eigen::Matrix< double, S, S > Update )
	{posterior_alpha_A_ = Update + alpha_A_;};
	//
	void set( std::shared_ptr< VB::HMM::VP_qsi<Dim,S> > Qsi )
	{qsi_ = Qsi;};

  
      private:
	//
	//
	std::shared_ptr< VB::HMM::VP_qsi<Dim,S> > qsi_;
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
	Eigen::Matrix< double, S, 1 > posterior_alpha_pi_;

	//
	// A distribution
	double alpha_A_0_{1.};
	Eigen::Matrix< double, S, S > alpha_A_;
	// Posterior density of the state
	Eigen::Matrix< double, S, S > posterior_A_;
	Eigen::Matrix< double, S, S > posterior_alpha_A_;

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
      std::gamma_distribution< double > distribution_pi( alpha_pi_0_ / static_cast< double >(S), 1.0 );
      std::gamma_distribution< double > distribution_A( alpha_A_0_ / static_cast< double >(S), 1.0 );
      //
      posterior_pi_       = Eigen::Matrix< double, S, 1 >::Zero();
      posterior_alpha_pi_ = Eigen::Matrix< double, S, 1 >::Zero();
      posterior_A_        = Eigen::Matrix< double, S, S >::Zero();
      //
      //
      double                        norm_alpha_pi = 0.;
      Eigen::Matrix< double, S, 1 > norm_alpha_A  = Eigen::Matrix< double, S, 1 >::Zero();
      // Calculate the full length of input data
      int total_length = 0;
      for ( int i = 0 ; i < n_ ; i++ )
	total_length += Y_[i].size();
      //
      for ( int s = 0 ; s < S ; s++ ) 
	{
	  posterior_alpha_pi_(s,0) = distribution_pi( generator );
	  norm_alpha_pi           += posterior_alpha_pi_(s,0);
	  for ( int ss = 0 ; ss < S ; ss++ )
	    {
	      posterior_alpha_A_(s,ss) = distribution_A( generator );
	      norm_alpha_A(s)         += posterior_alpha_A_(s,ss);
	    }
	}
      // normalization
      posterior_alpha_pi_ /= norm_alpha_pi;
      posterior_alpha_pi_ *= n_;
      posterior_alpha_pi_ += alpha_pi_;
      //
      // Posterior pi
      double sum_post_alpha_pi = posterior_alpha_pi_.sum();
      //
      for ( int s = 0 ; s < S ; s++ )
	{
	  //
	  posterior_pi_(s,0) = exp( gsl_sf_psi(posterior_alpha_pi_(s,0)) - gsl_sf_psi(sum_post_alpha_pi) );
	  //
	  posterior_alpha_A_.row(s) /= norm_alpha_A(s);
	  posterior_alpha_A_.row(s) *= total_length;
	}
      posterior_alpha_A_ += alpha_A_;
      // Poesterior A
      Eigen::Matrix< double, S, 1 > sum_post_alpha_A = posterior_alpha_A_.rowwise().sum();
      for ( int s = 0 ; s < S ; s++ )
	for ( int ss = 0 ; ss < S ; ss++ )
	  posterior_A_(s,ss) = exp( gsl_sf_psi(posterior_alpha_A_(s,ss)) - gsl_sf_psi(sum_post_alpha_A(s,0)) );
      //
      std::cout << "posterior_alpha_pi_ = \n" << posterior_alpha_pi_ << std::endl;
      std::cout << "posterior_alpha_A_ = \n" << posterior_alpha_A_ << std::endl;
      //
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
      VP_qdch<Dim,S>::Expectation()
      {}
    //
    //
    template< int Dim, int S > void
      VP_qdch<Dim,S>::Maximization()
      {
	//
	//
	//const std::vector< std::vector< Eigen::Matrix < double, S , 1 > > > &_s_  = qsi_->get_s();
	const std::vector< std::vector< Eigen::Matrix < double, S , S > > > &_ss_ = qsi_->get_ss();
	//
	//Eigen::Matrix< double, S, 1 > mean_s1 = Eigen::Matrix< double, S, 1 >::Zero();
	Eigen::Matrix< double, S, S > mean_ss = Eigen::Matrix< double, S, S >::Zero();

	//
	// Posterior Dirichlet parameters
	//
	double                        posterior_alpha_pi_sum = 0.;
	Eigen::Matrix< double, S, 1 > posterior_alpha_A_sum  = Eigen::Matrix< double, S, 1 >::Zero();
	
	//
	//posterior_alpha_pi_    = alpha_pi_ + mean_s1;
	posterior_alpha_pi_sum = posterior_alpha_pi_.sum();
	posterior_alpha_A_sum  = posterior_alpha_A_.rowwise().sum();

	//
	// log marginal likelihood lower bound
	double F_pi = - KL_Dirichlet_( posterior_alpha_pi_,  alpha_pi_ );
	double F_A  = 0.;
	//
	for ( int s = 0 ; s < S ; s++ )
	  F_A -= KL_Dirichlet_( posterior_alpha_A_.col(s).transpose(),  alpha_A_.col(s) );

	//
	// update the posterior proba density
	double norm_pi = 0.;
	for ( int s = 0 ; s < S ; s++ )
	  {
	    // Pi
	    posterior_pi_(s,0) = exp( gsl_sf_psi(posterior_alpha_pi_(s,0)) - gsl_sf_psi(posterior_alpha_pi_sum) );
	    // A
	    for ( int ss = 0 ; ss < S ; ss++ )
	      posterior_A_(s,ss) = exp( gsl_sf_psi(posterior_alpha_A_(s,ss)) - gsl_sf_psi(posterior_alpha_A_sum(s,0)) );
	  }
	//
	//
	std::cout << "posterior Pi and A" << std::endl;
	std::cout << "posterior_pi_\n" << posterior_pi_ << std::endl;
	std::cout << "posterior_A_\n" << posterior_A_ << std::endl;
	//std::cout << "posterior_A_ col wise\n" << posterior_A_.colwise().sum() << std::endl;
	std::cout << "posterior_A_ row wise\n" << posterior_A_.rowwise().sum() << std::endl;
	
	// 
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
	virtual void Expectation();
	virtual void Maximization();

	//
	// accessors
	const inline double                                                         get_F() const {return F_qgau_;}
	const        std::vector< std::vector< Eigen::Matrix < double, S , 1 > > >& get_N() const {return posteriror_N_;}
	//
	void set( std::shared_ptr< VB::HMM::VP_qsi<Dim,S> > Qsi )
	{qsi_ = Qsi;};
  
      private:
	//
	//
	std::shared_ptr< VB::HMM::VP_qsi<Dim,S> > qsi_;
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
	std::vector< double > beta_;
	// vectors
	std::vector< Eigen::Matrix< double, Dim, 1 > > mu_0_;
	std::vector< Eigen::Matrix< double, Dim, 1 > > mu_mean_;
	//
	// Wishart
	// scalars
	double nu_0_{Dim * 2.};
	std::vector< double > nu_;
	// vectors/matrices
	Eigen::Matrix< double, Dim, Dim >                S_0_inv_{ 1.e-3 * Eigen::Matrix< double, Dim, Dim >::Identity() };
	std::vector< Eigen::Matrix< double, Dim, Dim > > S_mean_inv_;
	std::vector< Eigen::Matrix< double, Dim, 1 > >   mu_0_mean_;
 

	//
	// Mean N over the posterior proba
	// N( Yit | mu_i,Lambda_i )
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
      //
      beta_.resize(S);
      // vectors
      mu_0_.resize(S);
      mu_mean_.resize(S);
      //
      // Wishart
      // scalars
      nu_.resize(S);
      // vectors/matrices
      S_mean_inv_.resize(S);
      mu_0_mean_.resize(S);
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
      VP_qgau<Dim,S>::Expectation()
      {

	//
	//
	const std::vector< std::vector< Eigen::Matrix < double, S , 1 > > > &_s_ = qsi_->get_s();
	
	//
	//
	std::vector< double >                            Delta( S, 0 );
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
	    nu_[s]         = nu_0_;
	    // 
	    y_mean[s]      = Eigen::Matrix< double, Dim, 1 >::Zero();
	    W_mean_inv[s]  = Eigen::Matrix< double, Dim, Dim >::Zero();

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
	    //
	    beta_[s] += Delta[s];
	    nu_[s]   += Delta[s];
	    //
	    for ( int i = 0 ; i < n_ ; i++ )
	      {
		int Ti = Y_[i].size();
		for ( int t = 0 ; t < Ti ; t++ )
		  {
		    Eigen::Matrix< double, Dim, 1 > diff_vect = Y_[i][t] - y_mean[s] / (beta_0_ * (beta_[s] - beta_0_));
		    W_mean_inv[s] += _s_[i][t](s,0) * diff_vect * diff_vect.transpose(); 
		    //std::cout << "#########################" << std::endl;
		    //std::cout << "W_mean_inv[" << s << "]\n" << W_mean_inv[s] << std::endl;
		    //std::cout << "Y_[i][t] \n" << Y_[i][t] << std::endl;
		    //std::cout << "y_mean[s] \n" << y_mean[s] << std::endl;
		    //std::cout << "(beta_0_ * (beta_[s] - beta_0_) " << (beta_0_ * (beta_[s] - beta_0_)) << std::endl;
		    //std::cout << "s_[i][t](s,0) " << _s_[i][t](s,0) << std::endl;
		    //std::cout << "diff_vect \n" << diff_vect << std::endl;
		    //std::cout << "diff_vect * diff_vect.transpose()  \n" << diff_vect * diff_vect.transpose() << std::endl;
		    //std::cout << "W_mean_inv[" << s << "]\n" << W_mean_inv[s] << std::endl;
		  }
	      }

	    //
	    mu_mean_[s]     = ( beta_0_ * mu_0_[s] + y_mean[s] ) / beta_[s];
	    mu_0_mean_[s]   = y_mean[s] / ( beta_[s] - beta_0_ );
	    //
	    Eigen::Matrix< double, Dim, 1 > diff_mus = mu_0_[s] - mu_0_mean_[s];
	    S_mean_inv_[s] += beta_0_ * ( beta_[s] - beta_0_ ) * diff_mus * diff_mus.transpose() / beta_[s];
	    S_mean_inv_[s] += W_mean_inv[s];
	    //
	    //
	    std::cout << "mu_0_mean_[" << s << "]\n" << mu_0_mean_[s] << std::endl;
	    std::cout << "S_mean_inv_[" << s << "]\n" << S_mean_inv_[s] << std::endl;
	  }
      }
    //
    //
    template< int Dim, int S > void
      VP_qgau<Dim,S>::Maximization()
      {
	//posteriror_N_
	std::vector< std::vector< Eigen::Matrix < double, S , 1 > > > ln_posteriror_N( posteriror_N_ );
	double
	  cd1       = - Dim * ln_2_pi,
	  cd2       = 0., // <ln|lambda_j|>
	  cd3       = 0., 
	  cd4       = S * Dim * log(beta_0_), // 
	  cd5       = 1., // ln(prod beta_j)
	  cd6       = 0., //
	  cd7       = 0., // Sum nu_j
	  cd8       = 0., // Sum nu_j * ln|Sj|
	  cd9       = log( S_0_inv_.inverse().determinant() ), // ln|S_0|
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
	    cd6 += nu_[s] * ( Dim - (S_0_inv_*S_mean).trace() ) + (nu_0_ - nu_[s])*cd2;
	    cd6 -= nu_[s] * beta_0_ * (diff_vec_0.transpose() * S_mean * diff_vec_0)(0,0);
	    //
	    cd7 += nu_[s];
	    cd8 += nu_[s] * ln_Sdet;
	    //
	    cd3 = nu_[s] / beta_[s];
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
		    ln_posteriror_N[i][t](s,0) -= Dim;  

		    ln_posteriror_N[i][t](s,0) *= 0.5;  
		    //
		    posteriror_N_[i][t](s,0)    = exp( ln_posteriror_N[i][t](s,0) );
		  }
	      }
	  }
	//
	//
	if ( true )
	  for ( int i = 0 ; i < n_ ; i++ )
	    {
	      int Ti = Y_[i].size();
	      for ( int t = 0 ; t < Ti ; t++ )
		{
		  posteriror_N_[i][t] /= posteriror_N_[i][t].sum();
		//std::cout << "posteriror_N_[" << i << "][" << t << "] \n" 
		//	    << posteriror_N_[i][t] << std::endl;
		}
	    }
	//
	diff_ln_Z += 0.5 * Dim * ln_2 * ( cd7 - nu_0_ );
	diff_ln_Z += 0.5 * ( cd8 - S * nu_0_ * cd9 );
	//
	F_qgau_  = cd4 - Dim * log(cd5) + cd6;
	F_qgau_ *= 0.5;
	F_qgau_ += diff_ln_Z;
	//
	//std::cout << "F_qgau_ " << F_qgau_ << std::endl;
      }
  }
}
#endif
