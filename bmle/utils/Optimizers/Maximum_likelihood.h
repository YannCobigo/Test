#ifndef MAXIMUM_LIKELIHOOD_H
#define MAXIMUM_LIKELIHOOD_H
//
#include "Optimizer.h"

//
//
//
namespace NeuroBayes
{
  /** \class Maximum_likelihood
   *
   * \brief Newton-Maximum_likelihood algorithm
   * 
   */
  template< class Algo, int DimY >
  class Maximum_likelihood : public Optimizer
  {
  public:
    /** Constructor. */
    Maximum_likelihood();
    
    //
    //
    virtual void update(){};

    //
    //
    void init_covariance( const int, const int,
			  const int,	     
			  const Eigen::MatrixXd, 
			  const Eigen::MatrixXd );
    
  private:		  
    //			  
    // Algorithm
    Algo algo_;
    //
    // Dim free response elements
    int n_{0};
    
    //
    // Design Matrices
    Eigen::MatrixXd X_;
    Eigen::MatrixXd Z_;
    

    //
    // Parameters
    //

    //
    // Covariance parameters
    // covariances element
    Eigen::Matrix< double, DimY, DimY > sigma_;
    // Covariances
    Eigen::MatrixXd Sigma_;
    // 
    // Covariance R
    // covariances element
    Eigen::Matrix< double, DimY, DimY > r_;
    // Covariances
    Eigen::MatrixXd R_;
    // mappings
    std::vector< std::vector<int> >                    r_mapping_;
    std::vector< Eigen::Matrix< double, DimY, DimY > > rdot_mapping_;
    std::vector< Eigen::MatrixXd >                     Rdot_mapping_;
    // 
    // Covariance G
    // covariances element
    Eigen::MatrixXd g_;
    // Covariances
    Eigen::MatrixXd G_;
    // mappings
    std::vector< std::vector<int> >                    g_mapping_;
    std::vector< Eigen::MatrixXd >                     gdot_mapping_;
    std::vector< Eigen::MatrixXd >                     Gdot_mapping_;
  };

  //
  //
  template< class Algo, int DimY >
  Maximum_likelihood<Algo,DimY>::Maximum_likelihood()
    {
      // Random symmetric positive definit 
      Eigen::MatrixXd A = Eigen::MatrixXd::Random( DimY, DimY );
      r_                = A * A.transpose();
      // mapping
      for ( int i = 0 ; i < DimY ; i++ )
	for ( int j = i ; j < DimY ; j++ )
	  {
	    // parameters mapping
	    r_mapping_.push_back( {i,j} );
	    // derivative mapping
	    Eigen::Matrix< double, DimY, DimY > U = Eigen::Matrix< double, DimY, DimY >::Zero();
	    U(i,j) = U(j,i) = 1.;
	    rdot_mapping_.push_back( U );
	  }
    };

  //
  //
  template< class Algo, int DimY > void
    Maximum_likelihood<Algo,DimY>::init_covariance( const int N, const int S,
						    const int D_r,
						    const Eigen::MatrixXd X,
						    const Eigen::MatrixXd Z )
    {
      //
      // Design matrices
      X_ = X;
      Z_ = Z;
      // Number of free elements
      n_ = N / DimY;



      //
      // Covariance: Kronecker R_ x In_
      Eigen::MatrixXd In = Eigen::MatrixXd::Identity( n_, n_ );
      R_ = Eigen::kroneckerProduct( In, r_ );
      for ( auto rdot : rdot_mapping_ )
	Rdot_mapping_.push_back( Eigen::kroneckerProduct(In, rdot) );



      //
      // Covariance: Kronecker G_ x In_
      Eigen::MatrixXd B = Eigen::MatrixXd::Random( D_r, D_r );
      g_                = B * B.transpose();
      // mapping
      Eigen::MatrixXd Im = Eigen::MatrixXd::Identity( S, S );
      for ( int i = 0 ; i < D_r ; i++ )
	for ( int j = i ; j < D_r ; j++ )
	  {
	    // parameters mapping
	    g_mapping_.push_back( {i,j} );
	    // derivative mapping
	    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(D_r,D_r);
	    V(i,j) = V(j,i) = 1.;
	    gdot_mapping_.push_back( V );
	    Gdot_mapping_.push_back( Eigen::kroneckerProduct(Im, V) );
	  }
      //
      G_ = Eigen::kroneckerProduct( Im, g_ );

      //
      // Covariance 
      Sigma_ = R_ + Z * G_ * Z_.transpose();
    };
}
#endif
