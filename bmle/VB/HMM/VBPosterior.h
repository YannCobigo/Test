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
#define ln_2    0.69314718055994529L
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
 * All developpments are inspired from:
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
  explicit VP_qsi( const std::vector< std::list< Eigen::Matrix < double, Dim , 1 > > >& );
    
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
  const inline double                        get_F()              const {return F_qsi_;}
  const std::vector< std::vector< double > > get_responsability() const {return gamma_;}
  
 private:
  //
  //
  std::vector< std::list< Eigen::Matrix < double, Dim , 1 > > > Y_;
  // Dimension reduction
  std::vector< int > k_;
  // Size of the data set
  std::size_t n_{0};

  //
  // responsability
  std::vector< std::vector< double > > gamma_;
  // log marginal likelihood lower bound: \qsi component
  double F_qsi_{-1.e06};
};
//
//
template< int Dim, int S > 
  VP_qsi<Dim,S>::VP_qsi(  const std::vector< std::list< Eigen::Matrix < double, Dim , 1 > > >& Y ):
Y_{Y}, n_{Y.size()}
{
//  //
//  // normalization
//  double inv_S = 1. / S_;
//  for ( int s = 0 ; s < S_ ; s++ )
//    for ( int i = 0 ; i < n_ ; i++ )
//      gamma_[s][i] = inv_S;
}
//
//
template< int Dim, int S > void
  VP_qsi<Dim,S>::Expectation( const Var_post& VP )
{
//  //
//  //
//  const VP_hyper<Dim,S>  &hyper  = std::get< HYPER >( VP );
//    
//  //
//  //
//  S_ = qlambs.get_cov_lamb().size();
//  k_.resize( S_ );
//  gamma_.resize( S_ );
//  double Z = 0.; // partition function
//  //
//  Eigen::Matrix< double, Dim, Dim > inv_psi = hyper.get_inv_psi();
//  //
//  for ( int s = 0 ; s < S_ ; s++ )
//    {
//    }
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
  explicit VP_qdch( const std::vector< std::list< Eigen::Matrix < double, Dim , 1 > > >& );
    
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
  const inline double get_F() const {return F_qdch_;}

  
 private:
  //
  //
  std::vector< std::list< Eigen::Matrix < double, Dim , 1 > > > Y_;
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
  Eigen::Matrix< double, S, 1 > alpha_A_;

  //
  // log marginal likelihood lower bound: \qsi component
  double F_qdch_{-1.e06};
};
//
//
template< int Dim, int S > 
  VP_qdch<Dim,S>::VP_qdch(  const std::vector< std::list< Eigen::Matrix < double, Dim , 1 > > >& Y ):
Y_{Y}, n_{Y.size()}
{
  //
  //
  alpha_pi_ = ( alpha_pi_0_ / static_cast< double >(S) ) * Eigen::Matrix< double, S, 1 >::Ones();
  alpha_A_  = ( alpha_A_0_  / static_cast< double >(S) ) * Eigen::Matrix< double, S, 1 >::Ones();
  //
  // Posterior density of the state
  // Initialize posterior pi wit a !!!dirichlet distribution!!!
  // Or at leat !!! normalize !!!
  posterior_pi_ = Eigen::Matrix< double, S, 1 >::Random();
}
//
//
template< int Dim, int S > void
  VP_qdch<Dim,S>::Expectation( const Var_post& VP )
{
  //
  // Posterior Dirichlet parameters
  double prior_pi = alpha_pi_0_ / static_cast< double >(S);
  double prior_A  = alpha_A_0_  / static_cast< double >(S);
  double alpha_pi_sum = 0.;
  //
  for ( int s = 0 ; s < S ; s++ )
    {
      // Pi
      alpha_pi_(s,0) = prior_pi + /*update from qsi*/ 0.;
      alpha_pi_sum  += alpha_pi_(s,0);
      // A
      alpha_A_(s,0)  = prior_A  + /*update from qsi*/ 0.;
    }
  // update the posterior proba density
  // and the log marginal likelihood lower bound
  double F_pi = gsl_sf_lngamma(alpha_pi_sum) - gsl_sf_lngamma(alpha_pi_0_);
  double F_A  = 0.;
  for ( int s = 0 ; s < S ; s++ )
    {
      // Pi
      posterior_pi_(s,0) = exp( gsl_sf_psi(alpha_pi_(s,0)) - gsl_sf_psi(alpha_pi_sum) );
      // A
      //
      // log marginal likelihood lower bound
      // Pi
      F_pi += (alpha_pi_(s,0) - prior_pi)*( gsl_sf_psi(alpha_pi_(s,0)) - gsl_sf_psi(alpha_pi_sum) );
      F_pi -= gsl_sf_lngamma(alpha_pi_(s,0));
      // A
    }
  //
  F_qdch_ = ( F_pi + S*gsl_sf_lngamma(prior_pi) ) + F_A;
}
//
//
template< int Dim, int S > void
  VP_qdch<Dim,S>::Maximization( const Var_post& VP )
{
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
  explicit VP_qgau( const std::vector< std::list< Eigen::Matrix < double, Dim , 1 > > >& );
    
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
  const inline double get_F() const {return F_qgau_;}
  
 private:
  //
  //
  std::vector< std::list< Eigen::Matrix < double, Dim , 1 > > > Y_;
  // Dimension reduction
  std::vector< int > k_;
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
  std::vector< std::vector< std::list< Eigen::Matrix < double, Dim , 1 > > > > posteriror_N_{S};
  // log marginal likelihood lower bound: \qsi component
  double F_qgau_{-1.e06};
};
//
//
//
template< int Dim, int S > 
  VP_qgau<Dim,S>::VP_qgau(  const std::vector< std::list< Eigen::Matrix < double, Dim , 1 > > >& Y ):
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
}
//
//
template< int Dim, int S > void
  VP_qgau<Dim,S>::Expectation( const Var_post& VP )
{
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
	  typename std::list< Eigen::Matrix < double, Dim , 1 > >::const_iterator t;
	  for ( t = Y_[i].begin() ; t != Y_[i].end() ; t++ )
	    {
	      y_mean[s] += /*<delta>*/ 1. * (*t); //!!!
	      Delta[s]  += /*<delta>*/ 1.; //!!!
	    }
	}
      //
      for ( int i = 0 ; i < n_ ; i++ )
	{
	  typename std::list< Eigen::Matrix < double, Dim , 1 > >::const_iterator t;
	  for ( t = Y_[i].begin() ; t != Y_[i].end() ; t++ )
	    {
	      Eigen::Matrix< double, Dim, 1 > diff_vect = (*t) -  y_mean[s] / (beta_0_ * (beta_[s] - beta_0_));
	      W_mean_inv[s] += /*<delta>*/ 1. * diff_vect * diff_vect.transpose(); // !!!
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
  std::vector< std::vector< std::list< Eigen::Matrix < double, Dim , 1 > > > > log_posteriror_N_{S};
  double c1 = - 0.5 * Dim * ln_2_pi;
  for ( int s = 0 ; s < S ; s++ )
    {
      double cs = Dim * ln_2;
      //
      // ln|S| = 2 * sum_i ln(Lii)
      // where S = LL^T
      double lnSigmadet = 0;
      Eigen::LLT< Eigen::MatrixXd > lltOf( S_mean_inv_[s].inverse() );
      Eigen::MatrixXd L = lltOf.matrixL(); 
      //
      for ( int u = 0 ; u < Dim ; u++ )
	{
	  cs += gsl_sf_psi( 0.5*(nu_[s] + 1 - u) );
	  lnSigmadet += log( L(u,u) );
	}
      //
      cs += 2. * lnSigmadet;
      //
      cs *= 0.5;

      //
      //
      for ( int i = 0 ; i < n_ ; i++ )
	{
	  typename std::list< Eigen::Matrix < double, Dim , 1 > >::const_iterator t;
	  for ( t = Y_[i].begin() ; t != Y_[i].end() ; t++ )
	    {
	      //!!!
	    }
	}
    }
}
#endif
