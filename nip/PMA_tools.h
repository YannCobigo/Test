#ifndef NIPPMA_TOOLS_H
#define NIPPMA_TOOLS_H
//
//
//
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <math.h>  
#include <map>
#include <list>
// Egen
#include <Eigen/Core>
#include <Eigen/Eigen>
//
//
//
#include "NipException.h"
//
//
//
namespace MAC_nip
{
  // Penalization
  enum Normalization {NORMALIZE, STANDARDIZE, DEMEAN};
  /** \class NipPMA_tools
   *
   * \brief PMA_tools: linear algebra tools
   *
   * "A penalized matrix decomposition, with applications to sparse principal components and canonical correlation analysis" (PMID:19377034)
   * 
   */
  class NipPMA_tools
  {
  public:
    NipPMA_tools(){};
    //
    // Is finite checks on the nan in the input vector
    static void is_finite( Eigen::MatrixXd& Xu )
    {
      if ( ((Xu - Xu).array() != (Xu - Xu).array()).all() )
	{
	  std::cout << "is_finite\n" << Xu << std::endl;
	}
    }
    
    //
    // Soft thresholding
    static double soft_threshold( const double A, const double C )
    {
      //
      // Tests
      // C must be strictly positive value
      
      // 
      double abs_a = ( A > 0 ? A : -A );
      //
      //
      return ( A > 0 ? 1.:-1 ) * ( abs_a > C ? abs_a - C : 0. );
    }
    
    //
    // Dichotomy search
    static double dichotomy_search( const Eigen::MatrixXd& Xu, const double L1norm_u,
				    const double C, const int Niter = 1000 )
    {
      //
      // Tests
      if ( Xu.lpNorm< 2 >() < 1.e-16 )
	{
	  return 0.;
	}
      else
	{
	  //
	  //
	  double
	    delta_1 = 0,
	    delta_2 = C;
	  Eigen::MatrixXd soft_u = Eigen::MatrixXd::Zero( Xu.rows(), Xu.cols() );
	  
	  //
	  //
	  int count = 0;
	  while( delta_2 - delta_1 > 1.e-6 /*&& ++count < Niter*/ )
	    {
	      for ( int i = 0 ; i < soft_u.rows() ; i++ )
		soft_u(i,0) = soft_threshold( Xu(i,0), (delta_1 + delta_2) / 2. );
	      //
	      soft_u /= soft_u.lpNorm< 2 >();
	      if ( soft_u.lpNorm< 1 >() < L1norm_u )
		delta_2 = (delta_1 + delta_2) / 2.;
	      else
		delta_1 = (delta_1 + delta_2) / 2.;
	    }
	  
	  //
	  //
	  //std::cout << "delta delta " << delta_1  << " " <<  delta_2 << std::endl;
	  return (delta_1 + delta_2) / 2.;
	}
    }
    
    //
    // Normalize, column-wise, the correlation matrix
    static Eigen::MatrixXd normalize( const Eigen::MatrixXd& X, Normalization N )
      {
	//
	// Test
	// X.rows() > 1
	
	//
	//
	Eigen::MatrixXd X_normalized;
	//
	switch( N )
	  {
	  case STANDARDIZE:
	    {std::cout << "Inside: " << X.trace() << std::endl;
	      Eigen::MatrixXd demeaned = normalize(X,DEMEAN);
	      Eigen::VectorXd std = (demeaned.colwise().norm() / sqrt( static_cast< double >(X.rows() - 1)));
	      X_normalized = demeaned.array().rowwise() / std.transpose().array();
	      std::cout << "Inside: " << X_normalized.trace() << std::endl;
	      break;
	    }
	  case DEMEAN:
	    {
	      Eigen::MatrixXd Ones  = Eigen::MatrixXd::Ones(X.rows(),1);
	      Eigen::MatrixXd Means = X.colwise().sum() / static_cast< double >(X.rows());
	      X_normalized = X - Ones*Means;
	      break;
	    }
	  case NORMALIZE:
	    {
	      Eigen::VectorXd col_max = X.colwise().maxCoeff();
	      Eigen::VectorXd col_min = X.colwise().minCoeff();
	      //
	      Eigen::VectorXd max_min = col_max - col_min;
	      Eigen::MatrixXd Ones    = Eigen::MatrixXd::Ones(X.cols(),1);
	      //
	      X_normalized = X - Ones*col_min.transpose();
	      X_normalized = X_normalized.array().rowwise() / max_min.transpose().array();
	      break;
	    }
	  default:
	    {
	      std::cout << "Raise exception" << std::endl;
	      break;
	    }
	  }
  
	//
	//
	return X_normalized;
      }
  };
}
#endif
