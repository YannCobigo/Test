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
  // Check a file exist
  inline bool file_exists ( const std::string& Name )
  {
    std::ifstream f( Name.c_str() );
    return f.good();
  }


  //
  // Linear Algebra
  //

  //
  //
  //
  bool is_positive_definite( const Eigen::MatrixXd& Check_pos_def )
  {
    //
    // compute the Cholesky decomposition of Check_Pos_def
    Eigen::LLT<Eigen::MatrixXd> lltOfA( Check_pos_def );
    
    //
    //
    if(lltOfA.info() == Eigen::NumericalIssue)
      return false;
    else
      return true;
  }

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
  // Find the closet symmetric positive definit matrix
  // from a hill posed compuited covariance matrix
  // - Replace $X$ with the closest symmetric matrix, $Y = (X+X^{T})/2.
  // - Take an eigendecomposition $Y = PDP^{T}$, 
  //   and form the diagonal matrix $D^{+} = max(D,0)$ (elementwise maximum).
  // - The closest symmetric positive semidefinite matrix to $X$ is $Z = PD^{+}P^{T}$.
  // - The closest positive definite matrix to $X$ does not exist; 
  //   any matrix of the form $Z + εI$ is positive definite for $ε > 0$. 
  //   There is no minimum.
  Eigen::MatrixXd closest_sym_def_pos_depreciated( const Eigen::MatrixXd& Hill_matrix )
    {
      //
      //
      int 
	mat_rows      = Hill_matrix.rows(),
	mat_cols      = Hill_matrix.cols();
      //
      Eigen::MatrixXd I = Eigen::MatrixXd::Identity( mat_rows, mat_cols );
      Eigen::MatrixXd Y = (Hill_matrix + Hill_matrix.transpose()) / 2.;
      //
      Eigen::MatrixXd fixed_matrix = 1.e-06 * I;
      //
//      if( mat_rows > 1 )
//	{
	  //
	  Eigen::JacobiSVD<Eigen::MatrixXd> svd( Y, Eigen::ComputeThinU | Eigen::ComputeThinV );
	  std::cout << "eigen svd=" << svd.singularValues() << std::endl;
	  Eigen::MatrixXd singular_values = svd.singularValues();
	  for ( int eigen_val = 0 ; eigen_val < mat_rows ; eigen_val++ )
	    if ( singular_values(eigen_val,0) < 1.e+03 * std::numeric_limits<double>::min() )
	      singular_values(eigen_val,0) = 1.e+03 * std::numeric_limits<double>::min();
	  //
	  fixed_matrix +=
	    svd.matrixU()*singular_values.asDiagonal()*svd.matrixV().transpose();
	  Eigen::MatrixXd diff = fixed_matrix - Hill_matrix;
	  std::cout << "diff:\n" << diff.array().abs().sum() << std::endl;
	  std::cout << "Hill_matrix:\n" << Hill_matrix << std::endl;
	  std::cout << "fixed_matrix:\n" << fixed_matrix << std::endl;
//	}
//      else
//	if ( Hill_matrix(0,0 ) < 0. )
//	  fixed_matrix = 1.e-06 * I;
      //
      // 
      return fixed_matrix;
    }
  //
  //
  // Higham NJ. Computing a nearest symmetric positive semidefinite matrix. Linear Algebra and its Applications. 1988 May;103(C):103-118.
  Eigen::MatrixXd closest_sym_def_pos( const Eigen::MatrixXd& Hill_matrix )
    {
      //
      //
      const int 
	mat_rows      = Hill_matrix.rows(),
	mat_cols      = Hill_matrix.cols();
      //
      Eigen::MatrixXd I            = Eigen::MatrixXd::Identity( mat_rows, mat_cols );
      Eigen::MatrixXd fixed_matrix = Hill_matrix;
      //std::cout << "fixed_matrix \n" << fixed_matrix << std::endl;
      
      //
      //
      int k = 1;
      Eigen::MatrixXd delta = Eigen::MatrixXd::Zero( mat_rows, mat_rows );
      Eigen::MatrixXd wAw   = Eigen::MatrixXd::Zero( mat_rows, mat_rows );
      Eigen::MatrixXd A     = Eigen::MatrixXd::Zero( mat_rows, mat_rows );
      Eigen::MatrixXd xk    = Eigen::MatrixXd::Zero( mat_rows, mat_rows );
      Eigen::MatrixXd w     = I;
      Eigen::MatrixXd rk    = fixed_matrix;
      //
      while( /*! is_positive_definite(fixed_matrix) && */ k < 10 )
	{
	  //
	  // W is the matrix used for the norm (assumed to be Identity matrix here)
	  // the algorithm should work for any diagonal W
	  //std::cout << "k: " << k << std::endl;
	  rk = fixed_matrix - delta;
	  //std::cout << "rk \n" << rk << std::endl;
	  for ( int i = 0 ; i < mat_rows ; i++ )
	    for ( int j = 0 ; j < mat_cols ; j++ )
	      w(i,j)  = sqrt( I(i,j) );
	  //std::cout << "w \n" << w << std::endl;
	  wAw = w * rk * w;
	  //std::cout << "wAw \n" << wAw << std::endl;
	  //Eigen::JacobiSVD<Eigen::MatrixXd> svd( wAw, Eigen::ComputeThinU | Eigen::ComputeThinV );
	  Eigen::EigenSolver< Eigen::MatrixXd > es( wAw );
	  Eigen::MatrixXd D = es.eigenvalues().real().asDiagonal();
	  //std::cout << "eigen values = \n" << es.eigenvalues() << std::endl;
	  for ( int eigen_val = 0 ; eigen_val < mat_rows ; eigen_val++ )
	    if ( es.eigenvalues().real()[eigen_val] < 1.e+03 * std::numeric_limits<double>::min() )
	      D(eigen_val,eigen_val) = 1.e+03 * std::numeric_limits<double>::min();
	  // Reconstruction
	  Eigen::MatrixXd V = es.eigenvectors().real();
	  A  = V * D * V.transpose();
	  //std::cout << "A \n" << A << std::endl;
	  fixed_matrix = xk = w.inverse() * A * w.inverse();
	  //std::cout << "xk \n" << xk << std::endl;
	  //
	  delta = xk - rk;
	  //std::cout << "delta \n" << delta << std::endl;
	  //
	  //
	  for ( int i = 0 ; i < mat_rows ; i++ )
	    for ( int j = 0 ; j < mat_cols ; j++ )
	      if ( I(i,j) > 0 )
	    fixed_matrix(i,j) = I(i,j);
	  //std::cout << "end loop fixed_matrix \n" << fixed_matrix << std::endl;
	  //
	  k++;
	}

      //
      //
      return fixed_matrix;
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
      //
      double result = 0.;
      //
      // Check the matrix S is not diagonal
      if( S.isDiagonal(0.0001) )
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
	  result = 2. * lnSdet;
	}
      else
	{
	  int    size    = S.cols();
	  double exp_res = 1.;
	  for ( int i = 0 ; i < size ; i++ )
	    exp_res *= S(i,i);
	  //
	  result = log( exp_res );
	}

      //
      //
      return result;
    }
  //
  //
  // Logarithm normal (Gaussian)
  template < int Dim > double
    log_gaussian( const Eigen::Matrix< double, Dim, 1 >&   Y, 
		  const Eigen::Matrix< double, Dim, 1 >&   Mu, 
		  const Eigen::Matrix< double, Dim, Dim >& Precision )
    {
      double ln_N = - Dim * ln_2_pi;
      ln_N += ln_determinant( Precision );
      ln_N -= ( (Y-Mu).transpose() * Precision * (Y-Mu) )(0,0);
      //
      return 0.5*ln_N;
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
      N       *= exp( -0.5*((Y-Mu).transpose() * Precision * (Y-Mu))(0,0) );
      //
      return N;
    }
  //
  //
  template < int Dim >
    Eigen::Matrix< double, Dim, 1 >
    gaussian_multivariate( const Eigen::Matrix< double, Dim, 1 >&   Mu, 
			   const Eigen::Matrix< double, Dim, Dim >& Covariance )
    {
      // random seed
      std::random_device rd;
      std::mt19937                       generator( rd() );
      std::normal_distribution< double > normal_dist(0.0,1.0);
      // Vector of multivariate gaussians
      Eigen::Matrix< double, Dim, 1 > Gaussian_multi_variate;
      
      //
      // Cholesky decomposition
      Eigen::LLT< Eigen::MatrixXd > lltOf( Covariance );
      Eigen::MatrixXd L = lltOf.matrixL(); 
      
      //
      // Sampling
      Eigen::Matrix< double, Dim, 1 > z;
      for ( int d = 0 ; d < Dim ; d++ )
	z(d,0) = normal_dist( generator );
      //
      Gaussian_multi_variate = Mu + L*z;
      
      //
      //
      return Gaussian_multi_variate;
    }
}
#endif
