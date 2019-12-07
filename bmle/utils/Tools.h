#ifndef TOOLS_H
#define TOOLS_H
//
#include <limits>       // std::numeric_limits
//
//
// Eigen
#include <Eigen/Core>
#include <Eigen/Eigen>
#include <Eigen/Sparse>
//
#define ln_2    0.69314718055994529L
#define ln_2_pi 1.8378770664093453L
#define ln_pi   1.1447298858494002L
#define pi_2    6.28318530718L
#define pi      3.14159265359L

//
//
// When we reach numerical limits
namespace NeuroStat
{
  enum TimeTransformation {NONE, DEMEAN, NORMALIZE, STANDARDIZE, LOAD};
}
//
//
// When we reach numerical limits
namespace NeuroBayes
{

  //
  // Linear Algebra
  //

  //
  //
  // Numerical inversion
  Eigen::MatrixXd inverse( const Eigen::MatrixXd& Ill_matrix )
    {
      //
      //
      int 
	mat_rows      = Ill_matrix.rows(),
	mat_cols      = Ill_matrix.cols();
      Eigen::MatrixXd I = Eigen::MatrixXd::Identity( mat_rows, mat_cols );

      //
      //
      return Ill_matrix.partialPivLu().solve(I);
    }
  //
  //
  // Inversion for definit positive matrix
  // The input must be a definit positive matrix: covariance
  Eigen::MatrixXd inverse_def_pos( const Eigen::MatrixXd& Def_pos_matrix )
    {
      int 
	mat_rows      = Def_pos_matrix.rows(),
	mat_cols      = Def_pos_matrix.cols();
      Eigen::MatrixXd I = Eigen::MatrixXd::Identity( mat_rows, mat_cols );
      //
      Eigen::JacobiSVD<Eigen::MatrixXd> svd( Def_pos_matrix, Eigen::ComputeThinU | Eigen::ComputeThinV );
      //      std::cout << "eigen svd=" << svd.singularValues() << std::endl;
      Eigen::MatrixXd singular_values = svd.singularValues();
      for ( int eigen_val = 0 ; eigen_val < mat_rows ; eigen_val++ )
	if ( singular_values(eigen_val,0) < 1.e+03 * std::numeric_limits<double>::min() )
	  singular_values(eigen_val,0) = 1.e+03 * std::numeric_limits<double>::min();

      Eigen::MatrixXd fixed_matrix =
	svd.matrixU()*singular_values.asDiagonal()*svd.matrixV().transpose();
//      Eigen::MatrixXd diff = fixed_matrix - Def_pos_matrix;
//      std::cout << "diff:\n" << diff.array().abs().sum() << std::endl;
//      std::cout << "fixed_matrix:\n" << fixed_matrix << std::endl;

      //
      // Inverse
      //return inverse( fixed_matrix );
      return fixed_matrix.llt().solve(I);
    }
  //
  //
  // Logarithm determinant
  // We use a cholesky decomposition
  // ln|S| = 2 * sum_i ln(Lii)
  // where S = LL^T
  // If it is a Cholesky decomposition we should be sure 
  // the matrix is positive definite
  double ln_determinant( const Eigen::MatrixXd& S )
    {
      //
      // result
      double lnSdet = 0;
      // They are supposed to be squarred matrices
      int    dim    = S.cols();
      // Cholesky decomposition
      Eigen::LLT< Eigen::MatrixXd > lltOf( S );
      Eigen::MatrixXd L = lltOf.matrixL(); 
      //
      for ( int u = 0 ; u < dim ; u++ )
	lnSdet += log( L(u,u) );

      //
      //
      return 2. * lnSdet;
    }
  //
  //
  // Logarithm normal (Gaussian)
  template < int Dim > double
    log_gaussian( const Eigen::Matrix< double, Dim, 1 >&   Y, 
		  const Eigen::Matrix< double, Dim, 1 >&   Mu, 
		  const Eigen::Matrix< double, Dim, Dim >& Precision )
    {
      double ln_N = - 0.5 * Dim * ln_2_pi;
      ln_N += 0.5 * ln_determinant( Precision );
      ln_N -= ( (Y-Mu).transpose() * Precision - (Y-Mu) )(0,0);
      //
      return ln_N;
    }
  //
  //  Normal (Gaussian)
  template < int Dim > double
    gaussian( const Eigen::Matrix< double, Dim, 1 >&   Y, 
	      const Eigen::Matrix< double, Dim, 1 >&   Mu, 
	      const Eigen::Matrix< double, Dim, Dim >& Precision )
    {
      double dim_2pi = 1.;
      for ( int d = 0 ; d < Dim ; d++ )
	dim_2pi *= pi_2;
      //
      double N = sqrt( Precision.determinant() / dim_2pi ) ;
      N       *= exp( ((Y-Mu).transpose() * Precision - (Y-Mu))(0,0) );
      //
      return N;
    }
}
#endif
