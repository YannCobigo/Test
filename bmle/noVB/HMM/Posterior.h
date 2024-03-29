#ifndef HMM_POSTERIOR_H
#define HMM_POSTERIOR_H
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
#include "Algorithms/kmean.h"
//
//
//
namespace noVB 
{
  namespace HMM
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
     * \brief 
     * 
     * Parameters:
     * - responsability (gamma): 
     *      Matrix[n x S] n (number of inputs) and S (number of states).
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
	const        std::vector< std::vector< Eigen::Matrix < double, S , 1 > > >& get_s()  const {return s_;}
	const        std::vector< std::vector< Eigen::Matrix < double, S , S > > >& get_ss() const {return ss_;}
	//
	void set( std::shared_ptr< noVB::HMM::P_qdch<Dim,S> > Qdch,
		  std::shared_ptr< noVB::HMM::P_qgau<Dim,S> > Qgau )
	{qdch_ = Qdch; qgau_ = Qgau;};

      private:
	//
	//
	std::shared_ptr< noVB::HMM::P_qdch<Dim,S> > qdch_;
	std::shared_ptr< noVB::HMM::P_qgau<Dim,S> > qgau_;


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
      alpha_i_t_.resize( n_ );
      beta_i_t_.resize( n_ );
      //
      for ( int i = 0 ; i < n_ ; i++ )
	{
	  int Ti = Y_[i].size();
	  //
	  s_[i].resize( Ti );
	  ss_[i].resize( Ti );
	  alpha_i_t_[i].resize( Ti );
	  beta_i_t_[i].resize( Ti );
	  //
	  for ( int t = 0 ; t < Ti ; t++ )
	    {
	      s_[i][t]         = Eigen::Matrix < double, S , 1 >::Zero();
	      ss_[i][t]        = Eigen::Matrix < double, S , S >::Zero(); 
	      alpha_i_t_[i][t] = Eigen::Matrix < double, S , 1 >::Zero();
	      beta_i_t_[i][t]  = Eigen::Matrix < double, S , 1 >::Zero();
	      //
//	      for ( int s = 0 ; s < S ; s++ )
//		{
//		  s_[i][t](s,0)         = uniform( generator );
//		}
//	      //
//	      s_[i][t] /= s_[i][t].sum();
//	      //std::cout << "s_["<<i<<"]["<<t<<"] = \n" << s_[i][t] << std::endl;
//	      //
//	      // Simplified cross-states calculation to initialize the
//	      // stochastic transition matrix
//	      if ( t > 0 )
//		for ( int s = 0 ; s < S ; s++ )
//		  for ( int ss = 0 ; ss < S ; ss++ )
//		    ss_[i][t](s, ss) = s_[i][t-1](s,0) * s_[i][t](ss,0);		
	    }
	}
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
	const std::vector< Eigen::Matrix< double, S, 1 > >                  &_pi_ = qdch_->get_pi();
	const Eigen::Matrix< double, S, S >                                 &_A_  = qdch_->get_A();
	const std::vector< std::vector< Eigen::Matrix < double, S , 1 > > > &_N_  = qgau_->get_gamma();
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
	    std::cout << "_pi_["<<i<<"] = \n" << _pi_[i] << std::endl;
	    std::cout << "_N_["<<i<<"][0] = \n" << _N_[i][0] << std::endl;
	    alpha_i_t_[i][0]  = _N_[i][0].array() * _pi_[i].array();
	    scale[0]          = alpha_i_t_[i][0].sum();
	    alpha_i_t_[i][0] /= scale[0];
	    //std::cout << "alpha_i_t_["<<i<<"][0] \n" << alpha_i_t_[i][0] << std::endl;
	    // next timepoints
	    for ( int t = 1 ; t < Ti ; t++ )
	      {
		alpha_i_t_[i][t] = Eigen::Matrix< double, S, 1 >::Zero();
		Eigen::Matrix < double, S , 1 >
		  alpha_A = (alpha_i_t_[i][t-1].transpose() * _A_).transpose();
		alpha_i_t_[i][t] = _N_[i][t].array() * alpha_A.array();
		//
		scale[t]          = alpha_i_t_[i][t].sum();
		alpha_i_t_[i][t] /= scale[t];
		//std::cout << "alpha_i_t_["<<i<<"]["<<t<<"] \n" << alpha_i_t_[i][t] << std::endl;
	      }
	    //
	    // Beta calculation
	    // Convension the last beta is 1.
	    //beta_i_t_[i][Ti-1] = Eigen::Matrix < double, S , 1 >::Ones() / static_cast<double>( S );
	    beta_i_t_[i][Ti-1] = Eigen::Matrix < double, S , 1 >::Ones() / scale[Ti-1];
	    //
	    for ( int t = Ti-2 ; t >= 0 ; t-- )
	      {
		//
		beta_i_t_[i][t] = Eigen::Matrix< double, S, 1 >::Zero();
		for ( int s = 0 ; s < S ; s++ )
		  beta_i_t_[i][t] += _N_[i][t+1](s,0) * beta_i_t_[i][t+1](s,0) * _A_.col(s);
		beta_i_t_[i][t] /= scale[t];
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
		std::cout << "Expect. s_["<<i<<"]["<<t<<"] \n" << s_[i][t]<< std::endl;

		//
		//  <s_{i,t-1} x s_{i,t}>
		if ( t > 0 )
		  {
		    //
		    double norm = 0.;
		    for ( int s = 0 ; s < S ; s++ )
		      for ( int ss = 0 ; ss < S ; ss++ )
			{
			  ss_[i][t](s,ss)  = alpha_i_t_[i][t-1](s,0)*_A_(s,ss);
			  ss_[i][t](s,ss) *= _N_[i][t](ss,0)*beta_i_t_[i][t](ss,0);
			  norm            += ss_[i][t](s,ss);
			}
		    //
		    ss_[i][t] /= norm;
		    //ss_[i][t] /= s_[i][t].sum();
		    //std::cout << "ss_["<<i<<"]["<<t<<"] \n" <<  ss_[i][t] << std::endl;
		  }
	      }
	  }// for ( int i = 0 ; i < n_ ; i++ )
      }
    //
    //
    template< int Dim, int S > void
      P_qsi<Dim,S>::Maximization()
      {}
    //
    //
    //
    /** \class P_qdch
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
      class P_qdch : public Posterior<Dim,S>
      {
      public:
	/** Constructor. */
	explicit P_qdch(){};
	/** Constructor. */
	explicit P_qdch(       std::shared_ptr< noVB::HMM::P_qsi<Dim,S> >,
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
	const std::vector< Eigen::Matrix< double, S, 1 > >& get_pi() const {return posterior_pi_;}
	const Eigen::Matrix< double, S, S >&                get_A()  const {return posterior_A_;}
	
  
      private:
	//
	//
	std::shared_ptr< noVB::HMM::P_qsi<Dim,S> > qsi_;

	//
	//
	std::vector< std::vector< Eigen::Matrix < double, Dim , 1 > > > Y_;
	// Size of the data set
	std::size_t n_{0};

	//
	// Pi and A distribution
	// State probability
	std::vector< Eigen::Matrix< double, S, 1 > > posterior_pi_;
	// Transition matrix
	Eigen::Matrix< double, S, S >                posterior_A_;
      };
    //
    //
    template< int Dim, int S > 
      P_qdch<Dim,S>::P_qdch(       std::shared_ptr< noVB::HMM::P_qsi<Dim,S> >                       Qsi,
			     const std::vector< std::vector< Eigen::Matrix < double, Dim , 1 > > >& Y ):
      qsi_{Qsi}, Y_{Y}, n_{Y.size()}
    {
      //
      // random engine
      std::random_device rd;
      std::mt19937       generator( rd() );
      std::uniform_real_distribution< double > uniform(0., 1.);
      //
      //
      posterior_pi_.resize( n_ );
      //
      for ( int i = 0 ; i < n_ ; i++ )
      	{
	  //
	  for ( int s = 0 ; s < S ; s++ )
	    posterior_pi_[i](s,0) =  uniform( generator );
	  //
	  posterior_pi_[i] /= posterior_pi_[i].sum();
	  //
	  std::cout << "init posterior_pi_["<<i<<"] = \n" << posterior_pi_[i] << std::endl;
      	}
      //
      posterior_A_ = Eigen::Matrix< double, S, S >::Random();
      for ( int s = 0 ; s < S ; s++ )
	{
	  for ( int c = 0 ; c < S ; c++ )
	    if ( posterior_A_(s,c) < 0 )
	      posterior_A_(s,c) = - posterior_A_(s,c);
	  double norm = posterior_A_.row(s).sum();
	  posterior_A_.row(s) /= norm;
	}
      std::cout << "init A = \n" << posterior_A_ << std::endl;
    }
    //
    //
    template< int Dim, int S > void
      P_qdch<Dim,S>::Expectation()
      {}
    //
    //
    template< int Dim, int S > void
      P_qdch<Dim,S>::Maximization()
      {
	//
	const std::vector< std::vector< Eigen::Matrix < double, S , 1 > > > &_s_  = qsi_->get_s();
	const std::vector< std::vector< Eigen::Matrix < double, S , S > > > &_ss_ = qsi_->get_ss();
	posterior_A_  = Eigen::Matrix< double, S, S >::Zero();
	//
	posterior_pi_.resize( n_ );
	for ( int i = 0 ; i < n_ ; i++ )
	  posterior_pi_[i] = Eigen::Matrix< double, S, 1 >::Zero();
	//
	//
	for ( int s = 0 ; s < S ; s++ ) 
	  {
	    for ( int ss = 0 ; ss < S ; ss++ ) 
	      {
		double norm = 0.;
		for ( int i = 0 ; i < n_ ; i++ )
		  {
		    int Ti = Y_[i].size();
		    for ( int t = 1 ; t < Ti ; t++ )
		      {
			posterior_A_(s, ss) += _ss_[i][t](s,ss);
			norm += _s_[i][t-1](s,0);
		      }
		    // State probability
		    if ( s == 0 && ss == 0 )
		      posterior_pi_[i] += _s_[i][0];
		  }
		//
		posterior_A_(s, ss) /= norm;
	      }
	  }
	std::cout << "transition matrix: \n" << posterior_A_ << std::endl;
	//normalize the first state probability
	for ( int i = 0 ; i < n_ ; i++ )
	  std::cout << "pi_["<<i<<"] = \n" << posterior_pi_[i] << std::endl;
      }
    //
    //
    //
    /** \class P_qgau
     *
     * \brief Gaussian posterior probabilities
     * 
     * The Gaussian posterior probabilities represent the 
     * probbility density of the emission probability. 
     *
     * Parameters:
     * - 
     *
     */
    template< int Dim, int S >
      class P_qgau : public Posterior<Dim,S>
      {
      public:
	/** Constructor. */
	explicit P_qgau(){};
	/** Constructor. */
	explicit P_qgau( std::shared_ptr< noVB::HMM::P_qsi<Dim,S> >,
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
	const std::vector< std::vector< Eigen::Matrix < double, S , 1 > > >& get_gamma()    const {return gamma_;};
	const std::vector< std::vector< Eigen::Matrix < double, S , 1 > > >& get_ln_gamma() const {return ln_gamma_;};
	//
	void set( std::shared_ptr< noVB::HMM::P_qsi<Dim,S> >  Qsi,
		  std::shared_ptr< noVB::HMM::P_qdch<Dim,S> > Qdch)
	{qsi_ = Qsi; qdch_ = Qdch;};
  
      private:
	//
	//
	std::shared_ptr< noVB::HMM::P_qsi<Dim,S> >  qsi_;
	std::shared_ptr< noVB::HMM::P_qdch<Dim,S> > qdch_;
	//
	//
	std::vector< std::vector< Eigen::Matrix < double, Dim , 1 > > > Y_;
	std::vector< std::vector< Eigen::Matrix < double, 1, 1 > > >    Age_;
	// Size of the data set
	std::size_t n_{0};

	//
	//
	// Gaussian
	// vectors
	std::vector< Eigen::Matrix< double, Dim, 1 > >                mu_s_;
	// vectors/matrices
	std::vector< Eigen::Matrix< double, Dim, Dim > >              precision_;
	// responsability
	// Gamma
	std::vector< std::vector< Eigen::Matrix < double, S , 1 > > > gamma_;
	// log Gamma 
	std::vector< std::vector< Eigen::Matrix < double, S , 1 > > > ln_gamma_;

      };
    //
    //
    //
    template< int Dim, int S > 
      P_qgau<Dim,S>::P_qgau( std::shared_ptr< noVB::HMM::P_qsi<Dim,S> >                             Qsi,
			     const std::vector< std::vector< Eigen::Matrix < double, Dim , 1 > > >& Y,
			     const std::vector< std::vector< Eigen::Matrix < double, 1, 1 > > >&    Age ):
      qsi_{Qsi}, Y_{Y}, Age_{Age}, n_{Y.size()}
    {
      //
      //
      mu_s_.resize(S);
      precision_.resize(S);
      gamma_.resize(n_);
      ln_gamma_.resize(n_);
      //
      // mean and covariance for each state
      if ( false )
	{
	  for ( int s = 0 ; s < S ; s++ )
	    {
	      Eigen::Matrix< double, Dim, Dim > sigma = 0.01 * Eigen::Matrix< double, Dim, Dim >::Random();
	      sigma = 0.5 * ( sigma + sigma.transpose() ) + Eigen::Matrix< double, Dim, Dim >::Identity();
	      //
	      precision_[s] =  NeuroBayes::inverse_def_pos( sigma );
	      std::cout << "sigma = \n " << sigma << std::endl;
	      //
	      mu_s_[s] = NeuroBayes::gaussian_multivariate< Dim >( Eigen::Matrix< double, Dim, 1 >::Zero(),
								   sigma );
	      //
	      std::cout 
		<< "init mu_s_["<<s<<"] = \n" << mu_s_[s]
		<< "\n init precision_["<<s<<"] = \n" << precision_[s]
		<< std::endl;
	    }
	}
      else
	{
	  //
	  // Create one vector with all teh entries
	  std::vector< Eigen::Matrix < double, Dim , 1 > > all_measures;
	  // load all the measures in one container
	  for ( std::size_t n = 0 ; n < n_ ; n++ )
	    {
	      std::size_t Ti = Y_[n].size();
	      for ( std::size_t t = 0 ; t < Ti ; t++ )
		all_measures.push_back( Y_[n][t] );
	    }
	  
	  //
	  // Run the k-mean clustering
	  NeuroBayes::kmean< Dim, S > clustering( all_measures );
	  clustering.ExpectationMaximization();
	  // return the centroids and covariance
	  for ( int s = 0 ; s < S ; s++ )
	    {
	      mu_s_[s]      = clustering.get_mu( s );
	      precision_[s] = NeuroBayes::inverse_def_pos( clustering.get_covariance(s) );
	      //
	      std::cout 
		<< "init mu_s_["<<s<<"] = \n" << mu_s_[s]
		<< "\n init precision_["<<s<<"] = \n" << precision_[s]
		<< std::endl;
	    }
	}
      //
      // responsability
      for ( int i = 0 ; i < n_ ; i++ )
	{
	  int Ti = Y_[i].size();
	  gamma_[i].resize(Ti);
	  ln_gamma_[i].resize(Ti);
	  for ( int t = 0 ; t < Ti ; t++ )
	    for ( int s = 0 ; s < S ; s++ )
	      {
		gamma_[i][t](s,0)    = (S / static_cast<double>(n_)  );
		ln_gamma_[i][t](s,0) = std::log(S / static_cast<double>(n_) );
	      }
	}
    }
    //
    //
    //
    template< int Dim, int S > void
      P_qgau<Dim,S>::Expectation()
      {}
    //
    //
    template< int Dim, int S > void
      P_qgau<Dim,S>::Maximization()
      {
      //
      //
      const std::vector< std::vector< Eigen::Matrix < double, S , 1 > > > &_s_  = qsi_->get_s();

      //
      //
      mu_s_.resize(S);
      precision_.resize(S);
      gamma_.resize(n_);
      ln_gamma_.resize(n_);
      //
      std::vector< Eigen::Matrix< double, Dim, Dim > > Cov(S);
      //
      for ( int s = 0 ; s < S ; s++ )
	{
	  //
	  precision_[s] = Eigen::Matrix< double, Dim, Dim >::Zero();
	  Cov[s]        = Eigen::Matrix< double, Dim, Dim >::Zero();
	  //
	  mu_s_[s]      = Eigen::Matrix< double, Dim, 1 >::Zero();
	  double norm   = 0.;
	  for ( int i = 0 ; i < n_ ; i++ )
	    {
	      int    Ti     = Y_[i].size();
	      for ( int t = 0 ; t < Ti ; t++ )
		{
		  mu_s_[s] += _s_[i][t](s,0) * Y_[i][t];
		  norm     += _s_[i][t](s,0);
		}
	    }
	  //
	  mu_s_[s] /= norm;
	  // Variance
	  for ( int i = 0 ; i < n_ ; i++ )
	    {
	      int    Ti     = Y_[i].size();
	      for ( int t = 0 ; t < Ti ; t++ )
		Cov[s] += _s_[i][t](s,0) * (Y_[i][t]-mu_s_[s]) * ((Y_[i][t]-mu_s_[s]).transpose());
	    }
	  //
	  Cov[s] /= norm;
	  Cov[s] = NeuroBayes::closest_sym_def_pos( Cov[s] );
	  //Cov[s] += 0.01 * Eigen::Matrix< double, Dim, Dim >::Identity();
	  precision_[s] = NeuroBayes::inverse_def_pos( Cov[s] );
	  std::cout << "mu_["<<s<<"] = \n" << mu_s_[s]  << std::endl;
	  std::cout << "Cov["<<s<<"] = \n" << Cov[s]  << std::endl;
	  std::cout << "precision_["<<s<<"] = \n" << precision_[s]  << std::endl;
	  //
	  // responsability
	  for ( int i = 0 ; i < n_ ; i++ )
	    {
	      int Ti = Y_[i].size();
	      gamma_[i].resize(Ti);
	      ln_gamma_[i].resize(Ti);
	      for ( int t = 0 ; t < Ti ; t++ )
		{
		  gamma_[i][t](s,0)    = NeuroBayes::gaussian<Dim>( Y_[i][t], mu_s_[s], precision_[s] );
		  ln_gamma_[i][t](s,0) = NeuroBayes::log_gaussian<Dim>( Y_[i][t], mu_s_[s], precision_[s] );
		}
	    }
	}
      }
  }
}
#endif
