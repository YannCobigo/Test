#ifndef TOOLS_H
#define TOOLS_H
// Eigen
#include <Eigen/Core>
#include <Eigen/Eigen>
#include <Eigen/Sparse>
//
//
//
namespace NeuroBayes
{
//  //
//  //
//  //
//  int power( double val, unsigned n )
//  {
//    return ( n == 0 ? 1 : val * power( val, n-1 ) );
//  }
//  
//  template< int n >
//    struct TMP_power
//    {
//      enum{value *= TMP_power<n-1>::value};
//    };
//  template<  >
//    struct TMP_power<0>
//    {
//      enum{value = 1};
//    };

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
      //std::cout << "YOYOYO " <<  mat_rows << " " << mat_cols << std::endl;
      //std::cout << "Ill_matrix\n" << Ill_matrix << std::endl;
      //std::cout << "YAYAYAYA" << std::endl;
      //std::cout << "Ill_matrix.partialPivLu().solve(I)\n" << Ill_matrix.partialPivLu().solve(I) << std::endl;
      return Ill_matrix.partialPivLu().solve(I);
    }
  //
  //
  // Inversion for definit positive matrix
  // The input must be a definit positive matrix: covariance
  Eigen::MatrixXd inverse_def_pos( const Eigen::MatrixXd& Def_pos_matrix )
    {
      Eigen::JacobiSVD<Eigen::MatrixXd> svd( Def_pos_matrix, Eigen::ComputeThinU | Eigen::ComputeThinV );
      //      std::cout << "eigen svd=" << svd.singularValues() << std::endl;
      Eigen::MatrixXd singular_values = svd.singularValues();
      for ( int eigen_val = 0 ; eigen_val < singular_values.rows() ; eigen_val++ )
	if ( singular_values(eigen_val,0) < 1.e-16 )
	  singular_values(eigen_val,0) = 1.e-16;

      Eigen::MatrixXd fixed_matrix =
	svd.matrixU()*singular_values.asDiagonal()*svd.matrixV().transpose();
//      Eigen::MatrixXd diff = fixed_matrix - Def_pos_matrix;
//      std::cout << "diff:\n" << diff.array().abs().sum() << std::endl;
//      std::cout << "fixed_matrix:\n" << fixed_matrix << std::endl;

      //
      //
      return inverse(fixed_matrix);
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
}
#endif
