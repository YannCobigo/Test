#ifndef VARIATIONALPOSTERIOR_H
#define VARIATIONALPOSTERIOR_H
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
//#include <cmath.h>
// Eigen
#include <Eigen/Core>
#include <Eigen/Eigen>
//
//
//
//#include "MAC_tools.h"
template< int Dim > class VP_hyper;  //: public VariationalPosterior<Dim>
template< int Dim > class VP_qlambs; //: public VariationalPosterior<Dim>
template< int Dim > class VP_qxisi;  //: public VariationalPosterior<Dim>
template< int Dim > class VP_qsi;    //: public VariationalPosterior<Dim>
template< int Dim > class VP_qnu;    //: public VariationalPosterior<Dim>
//
enum Vpost {HYPER, QLAMBS, QXISI, QNU, QSI};
//
//
//
/** \class variational_posterior
 *
 * \brief variational posterior interface application.
 * 
 * All developpments come from:
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
 * 
 * Update:
 * - k: is the dimension reduction
 * - S: number of clusters
 *
 *
 */
template< int Dim >
class VariationalPosterior
{
 public:
  
  //
  // Functions
  // Process the expectation step
  using Var_post = std::tuple<
    VP_hyper<Dim>,
    VP_qlambs<Dim>,
    VP_qxisi<Dim>,
    VP_qnu<Dim>,
    VP_qsi<Dim>  >;
  virtual void Expectation( const Var_post& ) = 0;
};
//
//
//
/** \class VP_qlambs
 *
 * \brief The variational posterior over the centres and factor loadings
 * 
 * The variational posterior over the centres and factor loadings 
 * of each analyser is obtained by taking functional derivatives 
 * with respect to q(\tild(\Lambda)):
 *
 */
template< int Dim >
class VP_qlambs : public VariationalPosterior<Dim>
{
 public:
  /** Constructor. */
  explicit VP_qlambs(){};
  /** Constructor. */
  explicit VP_qlambs( const std::vector< Eigen::Matrix < double, Dim , 1 > >& );
    
  /** Destructor */
  virtual ~VP_qlambs(){};
  
  //
  // Functions
  // Process the expectation step
  using Var_post = std::tuple<
    VP_hyper<Dim>,
    VP_qlambs<Dim>,
    VP_qxisi<Dim>,
    VP_qnu<Dim>,
    VP_qsi<Dim>  >;
  virtual void Expectation( const Var_post& );

  //
  // accessors
  inline const int get_n() const {return n_;}
  const std::vector< std::vector< Eigen::MatrixXd > >&    get_cov_lamb()  const {return cov_lamb_;}
  const std::vector< Eigen::Matrix< double, Dim, Dim > >& get_cov_mu()    const {return cov_mu_;}
  const std::vector< Eigen::MatrixXd >&                   get_mean_lamb() const {return mean_lamb_;}
  const std::vector< Eigen::Matrix< double, Dim, 1 > >&   get_mean_mu()   const {return mean_mu_;}

 private:
  //
  //
  std::vector< Eigen::Matrix < double, Dim , 1 > > Y_;

  // Dimension reduction
  std::vector< int > k_;
  // Number of clusters
  int S_{1};
  // Size of the data set
  std::size_t n_{0};

  //
  // One element for each cluster S_
  std::vector< Eigen::MatrixXd >                   mean_lamb_;
  std::vector< Eigen::Matrix< double, Dim, 1 > >   mean_mu_;
  std::vector< std::vector< Eigen::MatrixXd > >    cov_lamb_;
  std::vector< Eigen::Matrix< double, Dim, Dim > > cov_mu_;
};
//
//
//
template< int Dim > 
VP_qlambs<Dim>::VP_qlambs( const std::vector< Eigen::Matrix < double, Dim , 1 > >& Y ):
Y_{Y}, n_{Y.size()}
{
  //
  // One per cluster
  cov_lamb_.resize(S_);
  cov_mu_.resize(S_);
  mean_lamb_.resize(S_);
  mean_mu_.resize(S_);
  k_.resize(S_);
  // one per dimension q
  for ( int s = 0 ; s < S_ ; s++ )
    {
      k_[s] = Dim;
      cov_lamb_[s].resize(k_[s]);
      //
      for ( int q = 0 ; q < Dim ; q++ )
	cov_lamb_[s][q] = Eigen::MatrixXd::Identity( k_[s], k_[s] );
      cov_mu_[s]        = Eigen::Matrix< double, Dim, Dim >::Random();
      // mu
      mean_lamb_[s]     = Eigen::MatrixXd::Random( Dim, k_[s] );
      mean_mu_[s]       = Eigen::Matrix< double, Dim, 1 >::Random();
    }
}
//
//
//
template< int Dim > void
VP_qlambs<Dim>::Expectation( const Var_post& VP )
{
  //
  //
  const VP_hyper<Dim>  &hyper  = std::get< HYPER >( VP );
  const VP_qxisi<Dim>  &qxisi  = std::get< QXISI >( VP );
  const VP_qnu<Dim>    &qnu    = std::get< QNU >  ( VP );
  const VP_qsi<Dim>    &qsi    = std::get< QSI >  ( VP );
    
  //
  //
  S_ = cov_lamb_.size();
  k_.resize( S_ );
  //
  Eigen::Matrix< double, Dim, Dim >   inv_psi = hyper.get_inv_psi();
  Eigen::Matrix < double, Dim , 1 >   mu_star = hyper.get_mu_star();
  Eigen::Matrix < double, Dim , Dim > nu_star = hyper.get_nu_star();
  //
  for ( int s = 0 ; s < S_ ; s++ )
    {
      //
      //
      k_[s] = cov_lamb_[s][0].cols();
      std::vector< double > gamma_s = ( qsi.get_responsability() )[s];
      //
      std::vector< Eigen::MatrixXd > xs     = ( qxisi.get_mean() )[s];
      Eigen::MatrixXd                cov_xs = ( qxisi.get_cov() )[s];
      Eigen::MatrixXd                nu     = ( qnu.get_mean() )[s];
      
      //
      // global
      double                            qs          = 0.;
      Eigen::Matrix < double, Dim , 1 > qsy         = Eigen::Matrix < double, Dim, 1 >::Zero();
      Eigen::MatrixXd                   qsXXT       = Eigen::MatrixXd::Zero( k_[s], k_[s] );
      std::vector< Eigen::MatrixXd >    qsYX(Dim);
      for ( int q = 0 ; q < Dim ; q++ )
	qsYX[q] = Eigen::MatrixXd::Zero( k_[s], 1 );
      //
      for ( int i = 0 ; i < n_ ; i++ )
	{
	  qs    += gamma_s[i];
	  qsy   += gamma_s[i] * Y_[i];
	  qsXXT += gamma_s[i] * xs[i]*xs[i].transpose();
	  //
	  for ( int q = 0 ; q < Dim ; q++ )
	    qsYX[q] += gamma_s[i] * Y_[i](q,0) * xs[i];
	}
      //
      qsXXT += qs * cov_xs;

      //
      // {mean,cov}_mu_ then {mean,cov}_lamb_
      double          inv_cov_mu_qq  = 0.;
      Eigen::MatrixXd inv_cov_lamb_q = Eigen::MatrixXd::Zero(k_[s], k_[s]);
      for ( int q = 0 ; q < Dim ; q++ )
	{
	  // {mean,cov}_mu_
	  inv_cov_mu_qq    = nu_star(q,q) + inv_psi(q,q) * qs;
	  mean_mu_[s](q,0) = inv_cov_mu_qq * ( inv_psi(q,q) * qsy(q,0) + mu_star(q,0)*nu_star(q,q) );
	  cov_mu_[s](q,q)  = 1. / inv_cov_mu_qq;
	  // {mean,cov}_lamb_
	  inv_cov_lamb_q       = nu + inv_psi(q,q) * qsXXT;
	  mean_lamb_[s].row(q) = (inv_psi(q,q) * inv_cov_lamb_q * qsYX[q]).transpose();
	  cov_lamb_[s][q]      = inv_cov_lamb_q.inverse();
	}
    }
}
//
//
//
/** \class VP_qxisi
 *
 * \brief variational posterior for the hidden factors $x_{i}$.
 * 
 * The variational posterior for the hidden factors $x_{i}$, 
 * conditioned on the indicator variable $s_{i}$, is given
 * by taking functional derivatives with respect to 
 * $q(x_{i} | s_{i})$.
 *
 */
template< int Dim >
class VP_qxisi : public VariationalPosterior<Dim>
{
 public:
  /** Constructor. */
  explicit VP_qxisi(){};
  /** Constructor. */
  explicit VP_qxisi( const std::vector< Eigen::Matrix < double, Dim , 1 >  >& );
    
  /** Destructor */
  virtual ~VP_qxisi(){};
  
  //
  // Functions
  // Process the expectation step
  using Var_post = std::tuple<
    VP_hyper<Dim>,
    VP_qlambs<Dim>,
    VP_qxisi<Dim>,
    VP_qnu<Dim>,
    VP_qsi<Dim>  >;
  virtual void Expectation( const Var_post& );

  //
  // accessors
  inline const int get_n() const {return n_;}
  inline const int get_S() const {return S_;}
  const std::vector< int > get_k() const {return k_;};
  //
  const std::vector< Eigen::MatrixXd >                get_cov()  const {return cov_;}
  const std::vector< std::vector< Eigen::MatrixXd > > get_mean() const {return mean_;}
  const double                                        get_F()    const {return F_xisi_;}

 private:
  //
  //
  std::vector< Eigen::Matrix < double, Dim , 1 > > Y_;

  // Dimension reduction
  std::vector< int > k_;
  // Number of clusters
  int S_{1};
  // Size of the data set
  std::size_t n_{0};

  //
  // In each cluster s, the covariance of x_i is the same
  std::vector< Eigen::MatrixXd >                cov_;
  std::vector< std::vector< Eigen::MatrixXd > > mean_;
  // log marginal likelihood lower bound: \xisi component
  double                                        F_xisi_{0.};

};
//
//
//
template< int Dim > 
VP_qxisi<Dim>::VP_qxisi( const std::vector< Eigen::Matrix < double, Dim , 1 > >& Y ):
Y_{Y}, n_{Y.size()}
{}
//
//
//
template< int Dim > void
VP_qxisi<Dim>::Expectation( const Var_post& VP )
{
  //
  //
  const VP_hyper<Dim>  &hyper  = std::get< HYPER > ( VP );
  const VP_qlambs<Dim> &qlambs = std::get< QLAMBS >( VP );
  const VP_qsi<Dim>    &qsi    = std::get< QSI >   ( VP );
    
  //
  //
  S_ = qlambs.get_cov_lamb().size();
  cov_.resize( S_ );
  mean_.resize( S_ );
  k_.resize( S_ );
  F_xisi_ = 0.;
  
  //
  Eigen::Matrix< double, Dim, Dim > inv_psi = hyper.get_inv_psi();
  //
  for ( int s = 0 ; s < S_ ; s++ )
    {
      // reduced number of features
      k_[s] = qlambs.get_cov_lamb()[s][0].cols();
      //
      std::vector< double >             gamma    = ( qsi.get_responsability() )[s];
      Eigen::MatrixXd                   lamb     = (qlambs.get_mean_lamb())[s];
      std::vector< Eigen::MatrixXd >    cov_lamb = ( qlambs.get_cov_lamb() )[s];
      Eigen::Matrix < double, Dim , 1 > mu       = (qlambs.get_mean_mu())[s];
      Eigen::MatrixXd                   I        = Eigen::MatrixXd::Identity(k_[s],k_[s]);
      Eigen::MatrixXd                   inv_cov  = I;
      //
      inv_cov += lamb.transpose() * inv_psi * lamb;
      for ( int q = 0 ; q < Dim ; q++ )
	inv_cov += inv_psi(q,q) * cov_lamb[q];
      //
      cov_[s] = inv_cov.inverse();
      //
      // ln|Sigma| = 2 * sum_i ln(Lii)
      // where Sigma = LL^T
      double lnSigmadet = 0;
      Eigen::LLT< Eigen::MatrixXd > lltOf( cov_[s] );
      Eigen::MatrixXd L = lltOf.matrixL(); 
      for ( int k = 0 ; k < k_[s] ; k++ )
	lnSigmadet += log( L(k,k) );
      lnSigmadet *= 2.;
      //
      //
      mean_[s].resize( n_ );
      double N_s = 0.;
      for ( int i = 0 ; i < n_ ; i++ )
	{
	  mean_[s][i] = cov_[s] * lamb.transpose() * inv_psi * ( Y_[i] - mu );
	  // lower bound
	  N_s     += gamma[i];
	  F_xisi_ += gamma[i] * ( I-cov_[s]-mean_[s][i]*mean_[s][i].transpose() ).trace();
	}
      // lower bound
      F_xisi_ += N_s * lnSigmadet;
    }
  // lower bound
  F_xisi_ *= -0.5;
}
//
//
//
/** \class VP_qnu
 *
 * \brief variational posterior for the hidden factors $x_{i}$.
 * 
 * The variational posterior in the precision parameter for the l-th column 
 * of the s-th factor loading matrix $\Lambda^{s}$
 *
 */
template< int Dim >
class VP_qnu : public VariationalPosterior<Dim>
{
 public:
  /** Constructor. */
  explicit VP_qnu(){};
  /** Constructor. */
  explicit VP_qnu( const std::vector< Eigen::Matrix < double, Dim , 1 >  >& );
    
  /** Destructor */
  virtual ~VP_qnu(){};
  
  //
  // Functions
  // Process the expectation step
  using Var_post = std::tuple<
    VP_hyper<Dim>,
    VP_qlambs<Dim>,
    VP_qxisi<Dim>,
    VP_qnu<Dim>,
    VP_qsi<Dim>  >;
  virtual void Expectation( const Var_post& );

  //
  // accessors
  const std::vector< Eigen::MatrixXd > get_mean() const {return mean_;}
  const double                         get_F()    const {return F_nu_;}

 private:
  //
  //
  std::vector< Eigen::Matrix < double, Dim , 1 > > Y_;

  // Dimension reduction
  std::vector< int > k_;
  // Number of clusters
  int S_{1};
  // Size of the data set
  std::size_t n_{0};

  //
  // In each cluster s, 
  std::vector< Eigen::MatrixXd > mean_;
  // a_star b_star
  std::vector< std::vector<double> > a_start_;
  std::vector< std::vector<double> > b_start_;
  // prior a_star b_star
  std::vector< double >             prior_a_start_;
  std::vector< double >             prior_b_start_;
  // log marginal likelihood lower bound: \nu component
  double                             F_nu_{0.};
};
//
//
//
template< int Dim > 
VP_qnu<Dim>::VP_qnu( const std::vector< Eigen::Matrix < double, Dim , 1 > >& Y ):
Y_{Y}, n_{Y.size()}
{
  a_start_.resize(S_);
  b_start_.resize(S_);
  prior_a_start_.resize(S_);
  prior_b_start_.resize(S_);
  k_.resize(S_);
  mean_.resize(S_);
  //
  for ( int s = 0 ; s < S_ ; s++ )
    {
      k_[s]    = Dim;
      mean_[s] = Eigen::MatrixXd::Zero( k_[s], k_[s] );
      //
      a_start_[s].resize(k_[s]);
      b_start_[s].resize(k_[s]);
      //
      prior_a_start_[s] = 1.;
      prior_b_start_[s] = 1.;
      //
      for ( int l = 0 ; l < k_[s] ; l++ )
	{
	  a_start_[s][l] = 1.;
	  b_start_[s][l] = 1.;
	  //
	  mean_[s](l,l)  = a_start_[s][l] / b_start_[s][l];
	}
    }
}
//
//
//
template< int Dim > void
VP_qnu<Dim>::Expectation( const Var_post& VP )
{
  //
  //
  const VP_qlambs<Dim> &qlambs = std::get< QLAMBS >( VP );
    
  //
  //
  S_ = qlambs.get_cov_lamb().size();
  a_start_.resize(S_);
  b_start_.resize(S_);
  prior_a_start_.resize(S_);
  prior_b_start_.resize(S_);
  k_.resize( S_ );
  mean_.resize(S_);
  //
  F_nu_ = 0.;
  double post_a = 0., post_b = 0;
  //
  for ( int s = 0 ; s < S_ ; s++ )
    {
      // reduced number of features
      k_[s]    = qlambs.get_cov_lamb()[s][0].cols();
      mean_[s] = Eigen::MatrixXd::Zero( k_[s], k_[s] );
      //
      Eigen::MatrixXd                     lamb     = ( qlambs.get_mean_lamb() )[s];
      std::vector< Eigen::MatrixXd >      cov_lamb = ( qlambs.get_cov_lamb() )[s];
      //
      std::vector< Eigen::MatrixXd >      A(Dim);
      for ( int q = 0 ; q < Dim ; q++ )
	A[q] = cov_lamb[q] + lamb.row(q)*lamb.row(q).transpose();
      //
      a_start_[s].resize(k_[s]);
      b_start_[s].resize(k_[s]);
      // always the same prior for all the cluster
      prior_a_start_[s] = 1.;
      prior_b_start_[s] = 1.;
      //
      for ( int l = 0 ; l < k_[s] ; l++ )
	{
	  post_a += a_start_[s][l] = 1. + 0.5*static_cast<double>(Dim);
	  post_b += b_start_[s][l] = 1.;
	  for ( int q = 0 ; q < Dim ; q++ )
	    post_b += b_start_[s][l] += A[q](l,l);
	  //
	  mean_[s](l,l)  = a_start_[s][l] / b_start_[s][l];
	}
      // log marginal likelihood lower bound
      F_nu_ += post_a * log(post_b) - prior_a_start_[s] * log(prior_b_start_[s]);
      F_nu_ -= gsl_sf_lngamma(post_a) - gsl_sf_lngamma(prior_a_start_[s]);
      F_nu_ += (post_a-prior_a_start_[s])*( gsl_sf_psi(post_a) - log(post_b));
      F_nu_ -= post_a * ( 1. - prior_b_start_[s] / post_b );
    }
}
//
//
//
/** \class VP_qsi
 *
 * \brief 
 * 
 * Parameters:
 * - responsability (gamma): 
 *      Matrix[n x S] n (number of inputs) and S (number of clusters).
 *
 */
template< int Dim >
class VP_qsi : public VariationalPosterior<Dim>
{
 public:
  /** Constructor. */
  explicit VP_qsi(){};
  /** Constructor. */
  explicit VP_qsi( const std::vector< Eigen::Matrix < double, Dim , 1 > >& );
    
  /** Destructor */
  virtual ~VP_qsi(){};
  
  //
  // Functions
  // Process the expectation step
  using Var_post = std::tuple<
    VP_hyper<Dim>,
    VP_qlambs<Dim>,
    VP_qxisi<Dim>,
    VP_qnu<Dim>,
    VP_qsi<Dim>  >;
  virtual void Expectation( const Var_post& );

  //
  // accessors
  inline const int get_n() const {return n_;}
  const std::vector< std::vector< double > > get_responsability() const {return gamma_;}
  
 private:
  //
  //
  std::vector< Eigen::Matrix < double, Dim , 1 > > Y_;
  // Dimension reduction
  std::vector< int > k_;
  // Number of clusters
  int S_{1};
  // Size of the data set
  std::size_t n_{0};

  //
  // responsability
  std::vector< std::vector< double > > gamma_;
};
//
//
//
template< int Dim > 
VP_qsi<Dim>::VP_qsi(  const std::vector< Eigen::Matrix < double, Dim , 1 > >& Y):
Y_{Y}, n_{Y.size()}
{
  //
  // normalization
  double inv_S = 1. / S_;
  for ( int s = 0 ; s < S_ ; s++ )
    for ( int i = 0 ; i < n_ ; i++ )
      gamma_[s][i] = inv_S;
}
//
//
//
template< int Dim > void
VP_qsi<Dim>::Expectation( const Var_post& VP )
{
  //
  //
  const VP_hyper<Dim>  &hyper  = std::get< HYPER >( VP );
  const VP_qlambs<Dim> &qlambs = std::get< QLAMBS >( VP );
  const VP_qxisi<Dim>  &qxisi  = std::get< QXISI > ( VP );
    
  //
  //
  S_ = qlambs.get_cov_lamb().size();
  k_.resize( S_ );
  gamma_.resize( S_ );
  double Z = 0.; // partition function
  //
  Eigen::Matrix< double, Dim, Dim > inv_psi = hyper.get_inv_psi();
  //
  for ( int s = 0 ; s < S_ ; s++ )
    {
      //
      gamma_[s].resize(n_);
      // reduced number of features
      k_[s] = qlambs.get_cov_lamb()[s][0].cols();
      //
      Eigen::MatrixXd                     lamb     = ( qlambs.get_mean_lamb() )[s];
      Eigen::Matrix < double, Dim , 1 >   mu       = ( qlambs.get_mean_mu() )[s];
      std::vector< Eigen::MatrixXd >      cov_lamb = ( qlambs.get_cov_lamb() )[s];
      Eigen::Matrix < double, Dim , Dim > cov_mu   = ( qlambs.get_cov_mu() )[s];
      //
      std::vector< Eigen::MatrixXd > xs     = ( qxisi.get_mean() )[s];
      Eigen::MatrixXd                cov_xs = ( qxisi.get_cov() )[s];
      //
      Eigen::MatrixXd T_lamb = lamb.transpose() * inv_psi * lamb; // [kxk]
      double T_mu = (mu.transpose() * inv_psi * mu)(0,0);         // [1x1]
      for ( int q = 0 ; q < Dim ; q++ )
	{
	  T_lamb += inv_psi(q,q) * cov_lamb[q];
	  T_mu   += inv_psi(q,q) * cov_mu(q,q);
	}
      //
      // ln|Sigma| = 2 * sum_i ln(Lii)
      // where Sigma = LL^T
      double lnSigmadet = 0;
      Eigen::LLT< Eigen::MatrixXd > lltOf( cov_xs );
      Eigen::MatrixXd L = lltOf.matrixL(); 
      for ( int k = 0 ; k < k_[s] ; k++ )
	lnSigmadet += log( L(k,k) );
      lnSigmadet *= 2.;
      //
      // Responsability
      const double digamma = hyper.get_digamma(s);
      for ( int i = 0 ; i < n_ ; i++ )
	{
	  //
	  //
	  Eigen::MatrixXd TT = (lamb*xs[i]).transpose();
	  TT *= inv_psi * mu;
	  // log of the responsability
	  gamma_[s][i]  = - lnSigmadet;
	  gamma_[s][i] += (Y_[i].transpose() * inv_psi * (Y_[i] - 2*(lamb*xs[i]+mu)))(0,0);
	  gamma_[s][i] += 2 * (TT)(0,0);
	    //	  gamma_[s][i] += 2 * ( (lamb*xs[i]).transpose() * inv_psi * mu )(0,0);
	  gamma_[s][i] += (T_lamb * cov_xs).trace() + (xs[i].transpose() * T_lamb * xs[i])(0,0);
	  gamma_[s][i] += T_mu;
	  gamma_[s][i] *= -0.5;
	  //
	  gamma_[s][i] += digamma;
	  //
	  //
	  gamma_[s][i] = exp( gamma_[s][i] );
	  Z           += gamma_[s][i];
	}
    }

  //
  // normalization
  double inv_Z = 1. / Z;
  for ( int s = 0 ; s < S_ ; s++ )
    for ( int i = 0 ; i < n_ ; i++ )
      gamma_[s][i] *= inv_Z;
}
//
//
//
/** \class VP_hyper
 *
 * \brief The variational posterior over the centres and factor loadings
 * 
 * The variational posterior over the centres and factor loadings 
 * of each analyser is obtained by taking functional derivatives 
 * with respect to q(\tild(\Lambda)):
 *
 */
template< int Dim >
class VP_hyper : public VariationalPosterior<Dim>
{
 public:
  /** Constructor. */
  explicit VP_hyper(){};
  /** Constructor. */
  explicit VP_hyper( const std::vector< Eigen::Matrix < double, Dim , 1 > >& );
    
  /** Destructor */
  virtual ~VP_hyper(){};
  
  //
  // Functions
  // Process the expectation step
  using Var_post = std::tuple<
    VP_hyper<Dim>,
    VP_qlambs<Dim>,
    VP_qxisi<Dim>,
    VP_qnu<Dim>,
    VP_qsi<Dim> >;
  virtual void Expectation( const Var_post& );

  //
  // accessors
  inline const int                            get_n()              const {return n_;}
  const Eigen::Matrix< double, Dim, Dim >&    get_inv_psi()        const {return inv_psi_;}
  const double                                get_digamma( int s ) const {return gsl_sf_psi(alpham_[s])-gsl_sf_psi(hat_alpha_);}
  const double                                get_F()              const {return F_pi_;}
  //
  const Eigen::Matrix < double, Dim , 1 >&    get_mu_star()        const {return mu_star_;}
  const Eigen::Matrix < double, Dim , Dim >&  get_nu_star()        const {return nu_star_;}

  
 private:
  // Dimension reduction
  int k_{1};
  // Number of clusters
  int S_{1};
  // Size of the data set
  std::size_t n_{0};
  //
  //
  std::vector< Eigen::Matrix < double, Dim , 1 > > Y_;

  //
  // sensor of noise
  Eigen::Matrix< double, Dim, Dim > inv_psi_;
  //
  // S alpha*m_s
  double                             alpha_star_{1.};
  double                             hat_alpha_{0.};
  double                             hat_prior_alpha_{0.};
  //
  std::vector< double >              alpham_;
  std::vector< double >              prior_alpham_;
  // log marginal likelihood lower bound: \pi component
  double                             F_pi_{0.};
  // prior over the centres of each of the factor analysers
  Eigen::Matrix < double, Dim , 1 >   mu_star_;
  Eigen::Matrix < double, Dim , Dim > nu_star_;
};
//
//
//
template< int Dim > 
VP_hyper<Dim>::VP_hyper( const std::vector< Eigen::Matrix < double, Dim , 1 > >& Y ):
Y_{Y}, n_{Y.size()}
{
  //
  //
  inv_psi_ = Eigen::Matrix< double, Dim, Dim >::Identity();
  // alpha 
  alpham_.resize( S_ );
  prior_alpham_.resize( S_ );
  for ( int s = 0 ; s < S_ ; s++ )
    hat_alpha_ += prior_alpham_[s] = alpham_[s] = alpha_star_ / static_cast< double >( S_ );
  hat_prior_alpha_ = hat_alpha_;
  //
  // {mu,nu}_star
  mu_star_ = Eigen::Matrix < double, Dim , 1 >::Zero();
  nu_star_ = Eigen::Matrix < double, Dim , Dim >::Zero();
  //Eigen::MatrixXd
  //  YY = Eigen::MatrixXd::Zero(Dim,n_);
  for ( int i = 0 ; i < n_ ; i++ )
    {
      mu_star_ += Y_[i];
      //YY.col(i) = Y_[i];
    }
  //
  mu_star_ /= static_cast<double>(n_);
  nu_star_  = 100. * Eigen::Matrix< double, Dim, Dim >::Identity();
}
//
//
//
template< int Dim > void
VP_hyper<Dim>::Expectation( const Var_post& VP )
{
  //
  //
  const VP_qlambs<Dim> &qlambs = std::get< QLAMBS >( VP );
  const VP_qxisi<Dim>  &qxisi  = std::get< QXISI > ( VP );
  const VP_qsi<Dim>    &qsi    = std::get< QSI >   ( VP );
    
  //
  //
  S_ = qlambs.get_cov_lamb().size();

  //
  // Hyperparameters
  //
  // inv psi: Sensor of noise
  inv_psi_ = Eigen::Matrix< double, Dim, Dim >::Zero();
  // alpha 
  alpham_.resize( S_ );
  prior_alpham_.resize( S_ );
  hat_alpha_ = 0.;
  //
  F_pi_ = 0.;
  //
  mu_star_ = Eigen::Matrix < double, Dim , 1 >::Zero();
  nu_star_ = Eigen::Matrix < double, Dim , Dim >::Zero();
  //
  // clusters
  for ( int s = 0 ; s < S_ ; s++ )
    {
      //
      //
      std::vector< double >               gamma    = ( qsi.get_responsability() )[s];
      Eigen::MatrixXd                     lamb     = ( qlambs.get_mean_lamb() )[s];
      Eigen::Matrix < double, Dim , 1 >   mu       = ( qlambs.get_mean_mu() )[s];
      std::vector< Eigen::MatrixXd >      cov_lamb = ( qlambs.get_cov_lamb() )[s];
      Eigen::Matrix < double, Dim , Dim > cov_mu   = ( qlambs.get_cov_mu() )[s];
      //
      std::vector< Eigen::MatrixXd > xs     = ( qxisi.get_mean() )[s];
      Eigen::MatrixXd                cov_xs = ( qxisi.get_cov() )[s];
      // alpha
      alpham_[s] = prior_alpham_[s] = alpha_star_ / static_cast< double >( S_ );
      //
      // inputs
      for ( int i = 0 ; i < n_ ; i++ )
	{
	  //
	  // invpsi
	  Eigen::MatrixXd T_xi   = cov_xs + xs[i]*xs[i].transpose();
	  Eigen::MatrixXd T_lamb = lamb * T_xi * lamb.transpose();
	  Eigen::MatrixXd T_mu   = cov_mu + mu*mu.transpose();
	  //
	  for ( int q = 0 ; q < Dim ; q++ )
	    {
	      inv_psi_(q,q) += Y_[i](q,0)*(Y_[i] - 2*(lamb*xs[i]+mu))(q,0);
	      inv_psi_(q,q) += T_lamb(q,q) + (T_xi*cov_lamb[q]).trace();
	      inv_psi_(q,q) += T_mu(q,q);
	      inv_psi_(q,q) += 2*(lamb*xs[i]*mu.transpose())(q,q);
	      inv_psi_(q,q) *= gamma[i];
	    }
	  // alpha
	  alpham_[s] += gamma[i];
	}
      //
      // alpha
      hat_alpha_       += alpham_[s];
      hat_prior_alpha_ += prior_alpham_[s];
      // log marginal likelihood lower bound
      F_pi_ -= gsl_sf_lngamma(alpham_[s]) - gsl_sf_lngamma(prior_alpham_[s]);
      //
      //
      mu_star_ += mu;
    }
  // log marginal likelihood lower bound
  F_pi_ += gsl_sf_lngamma(hat_alpha_) - gsl_sf_lngamma(hat_prior_alpha_);
  for ( int s = 0 ; s < S_ ; s++ )
    F_pi_ += (alpham_[s]-prior_alpham_[s])*( gsl_sf_psi(alpham_[s]) - gsl_sf_psi(hat_alpha_) );
  //
  // invpsi
  inv_psi_ /= n_;
  // {mu,nu}_star
  mu_star_ /= static_cast<double>(S_);
  for ( int s = 0 ; s < S_ ; s++ )
    {
      Eigen::Matrix < double, Dim , 1 > mu = ( qlambs.get_mean_mu() )[s];
      nu_star_ += (mu - mu_star_) * (mu - mu_star_).transpose();
    }
  nu_star_ /=  static_cast<double>(S_);
}
#endif
