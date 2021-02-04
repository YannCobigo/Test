//#include "QuickView.h"
#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <fstream>
#include <random>
// Eigen
#include <Eigen/Core>
#include <Eigen/Eigen>
//
//
template < int Dim >
Eigen::Matrix< double, Dim, 1 >
gaussian_multivariate( const Eigen::Matrix< double, Dim, 1 >&   Mu, 
		         const Eigen::Matrix< double, Dim, Dim >& Precision )
{
  // random seed
  std::random_device rd;
  std::mt19937                       generator( rd() );
  std::normal_distribution< double > normal_dist(0.0,1.0);
  // Vector of multivariate gaussians
  Eigen::Matrix< double, Dim, 1 > Gaussian_multi_variate;
  
  //
  // 
  Eigen::Matrix< double, Dim, Dim > covariance = Precision.inverse();
  // Cholesky decomposition
  Eigen::LLT< Eigen::MatrixXd > lltOf( covariance );
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
//
//
template < int Dim > double
gaussian( const Eigen::Matrix< double, Dim, 1 >&   Y, 
	  const Eigen::Matrix< double, Dim, 1 >&   Mu, 
	  const Eigen::Matrix< double, Dim, Dim >& Precision )
{
  double pi_2    = 6.28318530718L;
  double dim_2pi = 1.;
  for ( int d = 0 ; d < Dim ; d++ )
    dim_2pi *= pi_2;
  //
  double N = sqrt( Precision.determinant() / dim_2pi ) ;
  N       *= exp( 0.5*((Y-Mu).transpose() * Precision - (Y-Mu))(0,0) );
  //
  return N;
}
//
//
//
int main(int argc, char const *argv[])
{
  //
  const int Dim = 2;
  const int S   = 5;
  const int n   = 100000;
  //
  //
  std::default_random_engine generator;
  std::vector< Eigen::Matrix< double, Dim, 1   >  > gauss_mu;
  std::vector< Eigen::Matrix< double, Dim, Dim >  > gauss_precision;
  gauss_mu.resize(S);
  gauss_precision.resize(S);
  //
  //
  gauss_mu[0] << 2.08793, 3.84315;
  gauss_precision[0] <<
    0.628375,  0.0284717,
    0.0284717, 1.04547;
  //  
  gauss_mu[1] << 2.02825, 3.36715;
  gauss_precision[1] <<
    2.15677, -0.248381,
    -0.248381, 0.74615;
  //  
  gauss_mu[2] << 1.80548, 3.79898;
  gauss_precision[2] <<
    0.02, -1.17803e-29,
    -1.17803e-29, 0.02;
  //  
  gauss_mu[3] << 0.970906, 1.95106;
  gauss_precision[3] <<
    1.22557, 0.0462203,
    0.0462203,  0.924656;
  //  
  gauss_mu[4] << 3.04923, 6.04412;
  gauss_precision[4] <<
    0.960252, 0.0451433,
    0.0451433, 0.973272;
  //
  std::uniform_real_distribution< double > uniform(0.0,1.0);
  std::uniform_int_distribution< int >     distribution(0,9);
  //
  Eigen::Matrix< double, S, S > A;
  A <<
   0.381777, 7.64569e-05, 6.58505e-05,    0.047073,     0.55298,
   0.536631, 9.50688e-05, 8.39298e-05, 9.43919e-05,    0.444954,
 0.00898837,  0.00898837,  0.00898837,  0.00898837,  0.00898837,
4.61155e-05,    0.290607, 4.00501e-05,     0.70059, 4.54945e-05,
    0.11201, 2.15755e-05, 1.86753e-05, 2.23028e-05,    0.883878;
  //
  Eigen::Matrix< double, S, 1 > Pi;
  Pi << 5.81206e-05, 0.282143, 5.02148e-05, 0.455448, 0.248458;

  //
  // Test case
  // posteriror_N_[0][6] 
  Eigen::Matrix< double, S, 1 > State;
  State << 0.0104799, 0.024551, 5.07425e-27, 5.63176e-06, 0.964963;


  //
  //
  Eigen::Matrix< double, S, 1 > new_State = State.transpose() * A;
  new_State /= new_State.sum();
  std::vector< double >         cumul_new_state( S );
  std::cout << "new state: \n" << new_State << std::endl;
  //
  for ( int s = 0 ; s < S ; s++ )
    {
      if ( s == 0 )
	cumul_new_state[s] = new_State(s,0);
      else
	cumul_new_state[s] = cumul_new_state[s-1] + new_State(s,0);
      std::cout << "cumul_new_state["<<s<<"] = " << cumul_new_state[s] << std::endl;
    }
  //
  int s         = 0;
  double choose = 0.;
  std::vector< Eigen::Matrix< double, Dim, 1 > > samples(n);
  for ( int i = 0 ; i < n ; i++ )
    {
      //
      choose = uniform( generator );
      //
      if ( false )
	{
	  if ( choose < new_State(0,0) )
	    s = 0;
	  else if ( choose < new_State(0,0)+new_State(1,0) && choose >= new_State(0,0) )
	    s = 1;
	  else if ( choose < new_State(0,0)+new_State(1,0)+new_State(2,0) && choose >= new_State(0,0)+new_State(1,0) )
	    s = 2;
	  else if ( choose < new_State(0,0)+new_State(1,0)+new_State(2,0)+new_State(3,0) && choose >= new_State(0,0)+new_State(1,0)+new_State(2,0) )
	    s = 3;
	  else if ( choose >= new_State(0,0)+new_State(1,0)+new_State(2,0)+new_State(3,0) )
	    s = 4;
	}
      else
	{
	  s = 0;
	  while ( choose > cumul_new_state[s] && s < S )
	    s++;
	}
      //std::cout << "choose["<<s<<"]: " << choose << std::endl;
      //
      //
      samples[i] = gaussian_multivariate< Dim >( gauss_mu[s], gauss_precision[s] );
    }

  //
  // Moments
  Eigen::Matrix< double, Dim, 1 >   new_state_val = Eigen::Matrix< double, Dim, 1 >::Zero();
  Eigen::Matrix< double, Dim, Dim > new_state_var = Eigen::Matrix< double, Dim, Dim >::Zero();
  // mean
  for ( auto vec : samples )
    new_state_val += vec;
  new_state_val /= static_cast<double>(n);
  // variance
  for ( auto vec : samples )
    new_state_var += (vec-new_state_val) * (vec-new_state_val).transpose();
  new_state_var /= static_cast<double>(n-1);

  //
  //
  std::cout << "Simulated value: \n" << new_state_val << std::endl;
  std::cout << "Simulated variance: \n" << new_state_var << std::endl;

  //
  //
  return EXIT_SUCCESS;
}
