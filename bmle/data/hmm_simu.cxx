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
  const int Dim = 1;
  const int S   = 3;
  const int n   = 100;
  //
  //
  std::default_random_engine generator;
  std::normal_distribution< double > gauss_11( 1.0, 1.0 );
  std::normal_distribution< double > gauss_12( 1.0, 1.0 );
  std::normal_distribution< double > gauss_21( 3.0, 1.0 );
  std::normal_distribution< double > gauss_22( 3.0, 1.0 );
  std::normal_distribution< double > gauss_31( 5.0, 1.0 );
  std::normal_distribution< double > gauss_32( 5.0, 1.0 );
  std::normal_distribution< double > sigma( 6.0, 1.0 );
  std::uniform_real_distribution< double > uniform(0.0,1.0);
  std::uniform_int_distribution< int >     uniform_1_10(1,10);
  std::uniform_int_distribution< int >     distribution(0,9);
  //
  Eigen::Matrix< double, S, S > A;
  A <<
    0.80, 0.15, 0.05,
    0.05, 0.75, 0.20,
    0.02, 0.05, 0.93;
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
      std::vector< double > Y;
      std::vector< int >    T;
      std::string line = "S = ";
      int s = 0;
      //
      while ( Ti < 3 || Ti == old_Ti )
	Ti = uniform_1_10( generator );
      old_Ti = Ti;
      std::cout << "Ti = " << Ti << std::endl;
      //
      // first measure
      //
      if ( 0. < choose && choose <= Pi(0,0) )
	{
	  double val = gauss_11( generator);
	  Y.push_back( val );
	  T.push_back( 0 );
	  s = 0;
	}
      else if ( Pi(0,0) < choose && choose <= Pi(0,0)+Pi(1,0) )
	{
	  double val = gauss_21( generator);
	  Y.push_back( val );
	  T.push_back( 0 );
	  s = 1;
	}
      else 
	{
	  double val = gauss_31( generator);
	  Y.push_back( val );
	  T.push_back( 0 );
	  s = 2;
	}
      //
      line += std::to_string( s ) + ",";
      
      //
      //
      for ( int t = 1 ; t < Ti ; t++ )
	{
	  choose = uniform( generator );
	  if ( 0. < choose && choose <= A(s,0) )
	    {
	      double val = gauss_11( generator);
	      Y.push_back( val );
	      T.push_back( t );
	      s = 0;
	    }
	  else if ( A(s,0) < choose && choose <= A(s,0)+A(s,1) )
	    {
	      double val = gauss_21( generator);
	      Y.push_back( val );
	      T.push_back( t );
	      s = 1;
	    }
	  else 
	    {
	      double val = gauss_31( generator);
	      Y.push_back( val );
	      T.push_back( t );
	      s = 2;
	    }
	  //
	  line += std::to_string( s ) + ",";
	}

      //
      //
      for ( double y : Y )
	std::cout << y << ",";
      std::cout << std::endl;
      for ( double t : T )
	std::cout << t << ",";
      std::cout << std::endl;
      std::cout << line << std::endl;
    }
  

  //
  //
  return EXIT_SUCCESS;
}

