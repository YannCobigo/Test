#ifndef REML_RUNIN_H
#define REML_RUNIN_H
//
#include "Exception.h"
#include "Optimizer.h"

//
//
//
namespace NeuroBayes
{
  /** \class REML_run_in
   *
   * \brief Newton-REML_run_in algorithm
   * 
   */
  template< class Algo, int DimY >
  class REML_run_in : public Optimizer
  {
  public:
    /** Constructor. */
    REML_run_in();
    
    //
    // Public functions
    virtual void update();
    virtual bool converged();

    //
    // Should be VIRTUAL !!

    //
    //
    void init_covariance( const int, const int,
			  const int,	     
			  const Eigen::MatrixXd&, 
			  const Eigen::MatrixXd&, 
			  const Eigen::MatrixXd& );
    //
    //
    void set_response( const Eigen::MatrixXd& Y ){Y_ = Y;};
    //
    //
    const Eigen::MatrixXd& get_beta() const {return beta_hat_;}
    const Eigen::MatrixXd  get_var_beta() const 
    { return ( X_.transpose() * Sigma_inverse_ * X_).inverse();}
    const Eigen::MatrixXd  get_u() const 
    { return G_ * Z_.transpose() * Sigma_inverse_ * (Y_ - X_ * beta_hat_);}
    const Eigen::MatrixXd  get_var_u() const 
    { 
      Eigen::MatrixXd P = inverse_def_pos( X_.transpose() * Sigma_inverse_ * X_ );
      P = Sigma_inverse_ * X_ * P * X_.transpose() * Sigma_inverse_;
      P = Sigma_inverse_ - P;
      //
      return G_ * Z_.transpose() * P * Z_ * G_;
    }
    
  private:		  
//    //			  
//    // Algorithm
//    Algo algo_;

    //
    // private functions
    void   init_covariance();
    bool   is_positive_def( const Eigen::MatrixXd& ) const;
    double cost_function() const;
    double cost_function( const Eigen::MatrixXd&, 
			  const Eigen::MatrixXd&, 
			  const Eigen::MatrixXd&) const;

    //
    // Dim free response elements
    int n_{0};
    // Dimension for random variables
    int D_r_{0};
    //
    int num_subjects_{0};
    
    //
    // Design Matrices
    Eigen::MatrixXd X_;
    Eigen::MatrixXd Y_;
    Eigen::MatrixXd Z_;
    

    //
    // Parameters
    //

    //
    // Beta
    Eigen::MatrixXd                                   beta_hat_;
    // Cost function
    std::vector< double >                             L_;
    std::vector< double >                             delta_L_;
    // convergence criteria
    double                                            epsilon_{1.e-1};
    int                                               num_max_iterations_{10};
    int                                               iteration_{0};
    int                                               window_{20};
    bool                                              interrupted_{false};

    //
    // Covariance parameters
    // covariances element
    Eigen::Matrix< double, DimY, DimY >                sigma_;
    // Covariances
    Eigen::MatrixXd                                    Sigma_;
    Eigen::MatrixXd                                    Sigma_inverse_;
    std::vector< Eigen::MatrixXd >                     Sigdot_mapping_;
    // All covariance elements
    Eigen::MatrixXd                                    kappa_;
    // 
    // Covariance R
    // covariances element
    Eigen::Matrix< double, DimY, DimY >                r_;
    // Covariances
    Eigen::MatrixXd                                    R_;
    // mappings
    std::vector< Eigen::Matrix< double, DimY, DimY > > rdot_mapping_;
    std::vector< Eigen::MatrixXd >                     Rdot_mapping_;
    // 
    // Covariance G
    // covariances element
    Eigen::MatrixXd                                    g_;
    // Covariances
    Eigen::MatrixXd                                    G_;
    // mappings
    std::vector< Eigen::MatrixXd >                     gdot_mapping_;
    std::vector< Eigen::MatrixXd >                     Gdot_mapping_;
  };

  //
  //
  template< class Algo, int DimY >
  REML_run_in<Algo,DimY>::REML_run_in()
    {
      //
      // First cost function
      delta_L_.push_back( 1.e+03 );
    };

  //
  //
  template< class Algo, int DimY > void
    REML_run_in<Algo,DimY>::init_covariance( const int N, const int S,
					   const int D_r,
					   const Eigen::MatrixXd& X,
					   const Eigen::MatrixXd& Z,
					   const Eigen::MatrixXd& Y )
    {
      //
      // Design matrices
      X_            = X;
      Z_            = Z;
      Y_            = Y;
      D_r_          = D_r;
      num_subjects_ = S;
      // Number of free elements
      n_ = N / DimY;
      //
      beta_hat_ = Eigen::MatrixXd::Zero( X.cols(), 1);
      kappa_    = Eigen::MatrixXd::Zero( 2, 1);

      bool reasonable = false;
      while( !reasonable )
	{
	  //
	  // reset the containers
	  rdot_mapping_.clear();
	  Rdot_mapping_.clear();
	  gdot_mapping_.clear();
	  Gdot_mapping_.clear();
	  Sigdot_mapping_.clear();

	  //
	  // random generator
	  std::default_random_engine generator;
	  std::uniform_real_distribution< double > distribution( 1., 10. );

	  //
	  // R
	  // Random symmetric positive definit 
	  double a = distribution( generator );
	  r_       = Eigen::Matrix< double, DimY, DimY >::Zero();
	  r_ <<
	     2.*a, -a,
	    -a,     2.*a;
	  // mapping
	  // derivative mapping
	  Eigen::Matrix< double, DimY, DimY > U = Eigen::Matrix< double, DimY, DimY >::Zero();
	  U <<
	    2., -1.,
	    -1., 2.;
	  rdot_mapping_.push_back( U );
	  // Covariance: Kronecker R_ x In_
	  Eigen::MatrixXd In = Eigen::MatrixXd::Identity( n_, n_ );
	  R_ = Eigen::kroneckerProduct( In, r_ );
	  for ( auto rdot : rdot_mapping_ )
	    {
	      Rdot_mapping_.push_back( Eigen::kroneckerProduct(In, rdot) );
	      Sigdot_mapping_.push_back( Eigen::kroneckerProduct(In, rdot) );
	    }
	  //
	  int ii = 0;
	  kappa_(ii++) = a;
	  
	  
	  //
	  // Covariance: Kronecker G_ x In_
	  g_ = Eigen::Matrix< double, 1, 1 >::Random();
	  g_ = g_ * g_;
	  // mapping
	  Eigen::MatrixXd Im = Eigen::MatrixXd::Identity( S, S );
	  // derivative mapping
	  Eigen::MatrixXd V  = Eigen::Matrix< double, 1, 1 >::Ones();
	  gdot_mapping_.push_back( V );
	  Gdot_mapping_.push_back( Z * Eigen::kroneckerProduct(Im, V) * Z.transpose() );
	  Sigdot_mapping_.push_back( Z * Eigen::kroneckerProduct(Im, V) * Z.transpose() );
	  //
	  kappa_(ii++) = g_(0,0);
	  //
	  G_ = Eigen::kroneckerProduct( Im, g_ );
	  
	  
	  //
	  // Covariance 
	  Sigma_         = R_ + Z_ * G_ * Z_.transpose();
	  Sigma_inverse_ = inverse_def_pos( Sigma_ );
	  // 
	  beta_hat_      = inverse_def_pos(X_.transpose() * Sigma_inverse_ * X_);
	  beta_hat_     *= X_.transpose() * Sigma_inverse_ * Y_;
      
	  //
	  //
	  double L   = cost_function( beta_hat_, Sigma_, Sigma_inverse_ );
	  if ( L > 0 )
	    {
	      reasonable = true;
	      L_.push_back( L + 100. );
	    }
	  else
	    reasonable = false;
	}

      //
      //
      if ( false )
	{
	  //
	  // R
	  for ( auto Rdot : Rdot_mapping_ )
	    std::cout << "Rdot \n" << Rdot << std::endl;
	  for ( auto rdot : rdot_mapping_ )
	    std::cout << "rdot \n" << rdot << std::endl;
	  // G
	  for ( auto gdot : gdot_mapping_ )
	    std::cout << "gdot \n" << gdot << std::endl;
	  for ( auto Gdot : Gdot_mapping_ )
	    std::cout << "Gdot \n" << Gdot << std::endl;
	  // 
	  std::cout << "R_ \n" << R_ << std::endl;
	  std::cout << "G_ \n" << G_ << std::endl;
	  // Sigma
	  std::cout << "Sigma_ \n" << Sigma_ << std::endl;
	}
    };

  //
  //
  template< class Algo, int DimY > void
    REML_run_in<Algo,DimY>::init_covariance()
    {
      //
      // Covariance: Kronecker R_ x In_
      Eigen::MatrixXd A = Eigen::MatrixXd::Random( D_r_, D_r_ );
      g_                = A * A.transpose();
      //
      Eigen::MatrixXd In = Eigen::MatrixXd::Identity( n_, n_ );
      R_ = Eigen::kroneckerProduct( In, r_ );
      //
      int ii = 0;
      for ( int i = 0 ; i < DimY ; i++ )
	for ( int j = i ; j < DimY ; j++ )
	  kappa_(ii++) = r_(i,j);


      //
      // Covariance: Kronecker G_ x In_
      Eigen::MatrixXd B = Eigen::MatrixXd::Random( D_r_, D_r_ );
      g_                = B * B.transpose();
      // mapping
      Eigen::MatrixXd Im = Eigen::MatrixXd::Identity( num_subjects_, num_subjects_ );
      for ( int i = 0 ; i < D_r_ ; i++ )
	for ( int j = i ; j < D_r_ ; j++ )
	  kappa_(ii++) = g_(i,j);
      //
      G_ = Eigen::kroneckerProduct( Im, g_ );


      //
      // Covariance 
      Sigma_         = R_ + Z_ * G_ * Z_.transpose();
      Sigma_inverse_ = inverse_def_pos( Sigma_ );
      //
      beta_hat_ = (X_.transpose()*Sigma_inverse_* X_).inverse() * X_.transpose() * Sigma_inverse_ * Y_;


      //
      //
      if ( false )
	{
	  //
	  // R
	  for ( auto Rdot : Rdot_mapping_ )
	    std::cout << "Rdot \n" << Rdot << std::endl;
	  for ( auto rdot : rdot_mapping_ )
	    std::cout << "rdot \n" << rdot << std::endl;
	  // G
	  for ( auto gdot : gdot_mapping_ )
	    std::cout << "gdot \n" << gdot << std::endl;
	  for ( auto Gdot : Gdot_mapping_ )
	    std::cout << "Gdot \n" << Gdot << std::endl;
	  // 
	  std::cout << "R_ \n" << R_ << std::endl;
	  std::cout << "G_ \n" << G_ << std::endl;
	  // Sigma
	  std::cout << "Sigma_ \n" << Sigma_ << std::endl;
	}
    };

  //
  //
  template< class Algo, int DimY > void
    REML_run_in<Algo,DimY>::update()
    {
      try 
	{
	  //
	  //
	  int number_covariance_param = Sigdot_mapping_.size();
	  Eigen::MatrixXd 
	    grad_L      = Eigen::MatrixXd::Zero( 2, 1 ),
	    H           = Eigen::MatrixXd::Zero( 2, 2 ),
	    // Xbeta - Y
	    e           = X_ * beta_hat_ - Y_;
	  Eigen::MatrixXd new_kappa;
	  Eigen::MatrixXd 
	    R             = R_,
	    G             = G_,          
	    Sigma         = Sigma_,        
	    Sigma_inverse = Sigma_inverse_,
	    //
	    beta_hat      = beta_hat_;

	  //
	  Eigen::MatrixXd 
	    SigSig,
	    SigSigSig,
	    XTSigX_inv = X_ * inverse_def_pos(X_.transpose() * Sigma_inverse_ * X_) * X_.transpose();
	  
	  //
	  // Set the gradiant and the Hessian
	  int ii = 0;
	  for ( auto Sigdot : Sigdot_mapping_ )
	    {
	      //
	      // Compute the gradiant
	      SigSig = Sigma_inverse_*Sigdot*Sigma_inverse_;
	      grad_L(ii,0)  =  ( Sigma_inverse_ * Sigdot ).trace();
	      grad_L(ii,0) -=  (e.transpose() * SigSig * e)(0,0);
	      // REML
	      grad_L(ii,0) -=  (XTSigX_inv * SigSig).trace();

	      //
	      // Compute the Hessian
	      int jj = 0;
	      for ( auto Ssigdot : Sigdot_mapping_ )
		{
		  SigSigSig   = SigSig * Ssigdot;
		  H(ii,jj)    = - SigSigSig.trace();
		  H(ii,jj) +=   2 * (e.transpose() * SigSigSig * Sigma_inverse_ * e)(0,0);
	      	  // REML
		  H(ii,jj) +=   2 * (XTSigX_inv * SigSigSig * Sigma_inverse_).trace();
	      	  H(ii,jj) -=   (XTSigX_inv * Sigdot * XTSigX_inv * Sigma_inverse_ * Ssigdot * Sigma_inverse_).trace();
	      	  jj++;
		}
	      //
	      ii++; 
	    }
	  //
	  if ( false )
	    {
	      std::cout
		<< "grad_L = \n" << grad_L
		<< "\n H = \n" << H 
		<< std::endl;
	    }

	  //
	  // Reasonable update
	  Algo algo_;
	  iteration_        = 0;
	  bool   reasonable = false;
	  double old_L      = 1.+03;
	  while ( !reasonable )
	    {
	      //
	      // Minimization
	      // Create symmetric pos. def. covariance
	      //			  
	      // Algorithm
	      bool is_positive = false;
	      while ( !is_positive )
		{
		  //
		  // Apply the algorithm
		  algo_.set_matrices( kappa_, grad_L, H );
		  algo_.update();
		  new_kappa = algo_.get_parameters();
		  
		  //
		  // Update the matrices
		  // r and g
		  r_ <<
		    2.*new_kappa(0), -new_kappa(0),
		    -new_kappa(0),     2.*new_kappa(0);
		  g_(0,0) = new_kappa(1);
		  // Are r and g pos. def.
		  is_positive = is_positive_def( r_ ) & is_positive_def( g_ );
		}


	      //
	      // R, G and Sigma
	      Eigen::MatrixXd 
		In = Eigen::MatrixXd::Identity( n_, n_ ),
		Im = Eigen::MatrixXd::Identity( num_subjects_, num_subjects_ );
	      //
	      R             = Eigen::kroneckerProduct( In, r_ );
	      G             = Eigen::kroneckerProduct( Im, g_ );
	      Sigma         = R + Z_ * G * Z_.transpose();
	      Sigma_inverse = inverse_def_pos( Sigma );
	      // 
	      beta_hat      = inverse_def_pos(X_.transpose() * Sigma_inverse * X_);
	      beta_hat     *= X_.transpose() * Sigma_inverse * Y_;

	      //
	      //
	      double L = cost_function( beta_hat, Sigma, Sigma_inverse );
	      if ( /*L < 0 ||*/ std::isnan(L) )
		{
		  reasonable = false;
		  if ( algo_.get_learning_rate() < std::numeric_limits< double >::epsilon() )
		    {
		      // get out of the loop
		      reasonable   = true;
		      interrupted_ = true;
		    }
		  else
		    old_L = L;
		}
	      else
		reasonable = true;
	      //
	      //std::cout << "Cost func: " << L << std::endl;
	      //std::cout << "Learning rate: " << algo_.get_learning_rate() << std::endl;
	    }
	  
	  //
	  // update 
	  // kappa
	  if( is_positive_def( r_ ) & is_positive_def( g_ ) )
	    {
	      kappa_         = new_kappa;
	      R_             = R;            
	      G_             = G;            
	      Sigma_         = Sigma;        
	      Sigma_inverse_ = Sigma_inverse;
	      // 
	      beta_hat_      = beta_hat;
	    }
	  
	  //
	  //
	  if ( false )
	    {
	      std::cout 
		<< "r_: \n" 
		<< r_
		<< std::endl;
	      std::cout 
		<< "g_: \n" 
		<< g_
		<< std::endl;
	      std::cout 
		<< "Sigma: \n" 
		<< Sigma_
		<< std::endl;
	      std::cout 
		<< "beta_hat: \n" 
		<< beta_hat_
		<< std::endl;
	      std::cout 
		<< "Sigma(beta_hat): \n" 
		<< (X_.transpose() * Sigma_.inverse() * X_).inverse()
		<< std::endl;
	      std::cout 
		<< "u: \n" 
		<< G_ * Z_.transpose() * Sigma_.inverse() * (Y_ - X_ * beta_hat_)
		<< std::endl;
	      std::cout 
		<< "Sigma(u): \n" 
		<< "something"
		<< std::endl;
	    }
	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	  exit( -1 );
	}
    };

  //
  //
  template< class Algo, int DimY > bool
    REML_run_in<Algo,DimY>::converged()
    {
      try 
	{
	  //
	  // Cost function
	  double 
	    L_prev = L_.back(),
	    L      = cost_function();
	  //
	  if ( false )
	    std::cout 
	      << "L_prev " << L_prev
	      << " ~ L " << L 
	      << " ~ (L - L_prev) " << L - L_prev 
	      << std::endl;
	  //
	  L_.push_back(L);
	  delta_L_.push_back( L - L_prev );
	  //
	  //
	  if ( !interrupted_ )
	    if ( L_.size() > window_ )
	      if (std::all_of( delta_L_.end() - window_, delta_L_.end(), 
			       [&](double dl){return fabs(dl) <  epsilon_ ;} ) ||
		  std::any_of( delta_L_.end() - window_, delta_L_.end(),
			       [&](double dl){return dl >  epsilon_ ;} ))
		return true;
	      else
		return false;
	    else
	      return false;
	  else
	    return true;
	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	  exit( -1 );
	}
    };
  //
  //
  template< class Algo, int DimY > bool
    REML_run_in<Algo,DimY>::is_positive_def( const Eigen::MatrixXd& A ) const 
    {
      try 
	{
	  //
	  // Check the matrix is a scalar
	  int cols = A.cols();
	  if ( cols == 1 )
	    return ( A(0,0) > 0 ? true : false );
	  else
	    {
	      //
	      // compute its eigen values and check that eigenvalues()(0)>0 
	      // && eigenvalues()(0)/eigenvalues()(n-1) > machine_precision.
	      Eigen::SelfAdjointEigenSolver< Eigen::MatrixXd > es;
	      es.compute(A);
	      if ( es.eigenvalues()(0) > 0 )
		{
		  bool positive = true;
		  for ( int c = 1 ; c < cols ; c++ )
		    if ( es.eigenvalues()(0) / es.eigenvalues()(c) < std::numeric_limits< double >::epsilon() )
		      positive = false;
		  //
		  return positive;
		}
	      else
		return false;
	    }
	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	  exit( -1 );
	}
    };
  //
  //
  template< class Algo, int DimY > double
    REML_run_in<Algo,DimY>::cost_function() const 
    {
      try 
	{
	  //
	  // L = -2 * ln(P)
	  double L = ln_determinant( Sigma_ );
	  L += ( (X_ * beta_hat_ - Y_).transpose() * Sigma_inverse_ * (X_ * beta_hat_ - Y_) )(0,0);
	  L += Y_.cols() * ln_2_pi;

	  if ( false )
	    {
	      //std::cout 
	      //	<< "e = \n" << X_ * beta_hat_ - Y_ 
	      //	<< std::endl;
	      std::cout 
		<< "exp(L1) = " <<  Sigma_.determinant()
		<< " L1 = " << ln_determinant( Sigma_ )
		<< " ~ L2 = " << (X_ * beta_hat_ - Y_).transpose() * Sigma_inverse_ * (X_ * beta_hat_ - Y_)
		<< std::endl;
	    }

	  //
	  //
	  return L;
	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	  exit( -1 );
	}
    };
  //
  //
  template< class Algo, int DimY > double
    REML_run_in<Algo,DimY>::cost_function( const Eigen::MatrixXd& Beta,
					   const Eigen::MatrixXd& Sigma,
					   const Eigen::MatrixXd& Sigma_inv  ) const 
    {
      try 
	{
	  //
	  // L = -2 * ln(P)
	  double L = ln_determinant( Sigma );
	  L += ( (X_ * Beta - Y_).transpose() * Sigma_inv * (X_ * Beta - Y_) )(0,0);
	  L += Y_.cols() * ln_2_pi;

	  //
	  //
	  return L;
	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	  exit( -1 );
	}
    };
}
#endif
