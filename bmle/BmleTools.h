#ifndef BMLETOOLS_H
#define BMLETOOLS_H
// Eigen
#include <Eigen/Core>
#include <Eigen/Eigen>
#include <Eigen/Sparse>
//
//
//
namespace MAC_bmle
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
  //
  Eigen::MatrixXd inverse( const Eigen::MatrixXd& Ill_matrix )
    {
      Eigen::JacobiSVD<Eigen::MatrixXd> svd( Ill_matrix, Eigen::ComputeThinU | Eigen::ComputeThinV );
      //      std::cout << "eigen svd=" << svd.singularValues() << std::endl;
      Eigen::MatrixXd singular_values = svd.singularValues();
      for ( int eigen_val = 0 ; eigen_val < singular_values.rows() ; eigen_val++ )
	if ( singular_values(eigen_val,0) < 1.e-16 )
	  singular_values(eigen_val,0) = 1.e-16;

      Eigen::MatrixXd fixed_matrix =
	svd.matrixU()*singular_values.asDiagonal()*svd.matrixV().transpose();
//      Eigen::MatrixXd diff = fixed_matrix - Ill_matrix;
//      std::cout << "diff:\n" << diff.array().abs().sum() << std::endl;
//      std::cout << "fixed_matrix:\n" << fixed_matrix << std::endl;

      //
      //
      return fixed_matrix.inverse();
    }
}
#endif
