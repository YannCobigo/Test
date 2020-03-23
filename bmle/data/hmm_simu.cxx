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
  const int Dim = 4;
  const int S   = 3;
  const int n   = 100;
  //
  //
  std::default_random_engine generator;
  std::vector< std::vector< std::normal_distribution< double > > > gauss_sd;
  //
  gauss_sd.resize( S );
  for ( int s = 0 ; s < S ; s++ )
    {
      gauss_sd[s].resize( Dim );
      for ( int d = 0 ; d < Dim ; d++ )
	gauss_sd[s][d] = std::normal_distribution< double >( static_cast< double >( (s+1) * (d+1) ),
							     1.0 );
    }
  //
  std::uniform_real_distribution< double > uniform(0.0,1.0);
  std::uniform_int_distribution< int >     uniform_1_10(1,10);
  std::uniform_int_distribution< int >     distribution(0,9);
  //
  Eigen::Matrix< double, S, S > A;
  A <<
    0.70, 0.20, 0.10,
    0.05, 0.50, 0.45,
    0.02, 0.03, 0.95;
  Eigen::Matrix< double, S, 1 > Pi;
  Pi << 0.50, 0.30, 0.20;
  //
  //
  int
    Ti     = 0,
    old_Ti = 0;
  
  double choose = 0.;
  //
  for ( int i = 0 ; i < n ; i++ )
    {
      //
      //
      std::cout << "subject: " << i << std::endl;
      //
      choose = uniform( generator );
      std::vector< std::vector< double > > Y;
      std::vector< int >                   T;
      std::string line = "S = ";
      int s = 0;
      //
      while ( Ti < 3 || Ti == old_Ti )
	Ti = uniform_1_10( generator );
      old_Ti = Ti;
      std::cout << "Ti = " << Ti << std::endl;
      //
      // first measure
      T.push_back( 0 );
      Y.resize(Ti);
      if ( 0. < choose && choose <= Pi(0,0) )
	s = 0;
      else if ( Pi(0,0) < choose && choose <= Pi(0,0)+Pi(1,0) )
	s = 1;
      else 
	s = 2;
      //
      Y[0].resize( Dim );
      for ( int d = 0 ; d < Dim ; d++ )
	Y[0][d] = gauss_sd[s][d]( generator);
      //
      line += std::to_string( s ) + ",";
      
      //
      //
      for ( int t = 1 ; t < Ti ; t++ )
	{
	  choose = uniform( generator );
	  T.push_back( t );
	  //
	  if ( 0. < choose && choose <= A(s,0) )
	    s = 0;
	  else if ( A(s,0) < choose && choose <= A(s,0)+A(s,1) )
	    s = 1;
	  else 
	    s = 2;
	  //
	  Y[t].resize( Dim );
	  for ( int d = 0 ; d < Dim ; d++ )
	    Y[t][d] = gauss_sd[s][d]( generator );
	  //
	  //
	  line += std::to_string( s ) + ",";
	}

      //
      //
      for ( int d = 0 ; d < Dim ; d++ )
	{
	  for ( auto y : Y )
	    std::cout << y[d] << ",";
	  std::cout << std::endl;
	}
      for ( double t : T )
	std::cout << t << ",";
      std::cout << std::endl;
      std::cout << line << std::endl;
    }
  

  //
  //
  return EXIT_SUCCESS;
}
