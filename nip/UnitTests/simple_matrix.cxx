#include <iostream>
#include <cstdlib>
#include <string>
#include <cstdlib>
#include <fstream>
#include <sstream>  
#include <regex>  
#include <random>
// Eigen
#include <Eigen/Core>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Sparse>
//
//
#include "../PMA.h"
#include "../PMA_tools.h"
#include "../PMD.h"
#include "../SPC.h"
//
//
int main(int argc, char const *argv[]){

  //
  // test PMD
  //

  //
  //
  Eigen::MatrixXd A = Eigen::MatrixXd(3,4);
  A <<
     1, 10, 1, -2,
    -3,  1, 3,  3,
     7,  2, 3,  4;

  Eigen::MatrixXd
    UU = Eigen::MatrixXd::Random(A.rows(),1),
    VV = Eigen::MatrixXd::Random(A.cols(),1);
  
  //
  UU /= UU.lpNorm< 2 >(); VV /= VV.lpNorm< 2 >();


  //
  //
  std::cout << "A:\n" << A << std::endl;
  std::cout << "A norm:\n" << MAC_nip::NipPMA_tools::normalize( A, MAC_nip::STANDARDIZE ) << std::endl;
  std::cout << "ATA:\n" << A.transpose() * A  << std::endl;
  std::cout << "ATa norm:\n" << MAC_nip::NipPMA_tools::normalize( A, MAC_nip::STANDARDIZE ).transpose() *  MAC_nip::NipPMA_tools::normalize( A, MAC_nip::STANDARDIZE )<< std::endl;
  std::cout << "UU:\n" << UU << std::endl;
  std::cout << "VV:\n" << VV << std::endl;

  //
  // K is the supposed rank of the matrix A
  Eigen::MatrixXd AAA = A;//A.transpose() * A;
  std::size_t K = AAA.cols();
  Spectra matrix_spetrum( K );
  // initialize the spectra
  // ToDo: the first vector should be the SVD highest eigen vector
  for ( int k = 0 ; k < K ; k++ )
    {
      // Coefficient
      std::get< coeff_k >( matrix_spetrum[k] ) = 0.;
      // vectors
      std::get< Uk >( matrix_spetrum[k] ) = Eigen::MatrixXd::Random( AAA.rows(), 1 );
      std::get< Vk >( matrix_spetrum[k] ) = Eigen::MatrixXd::Random( AAA.cols(), 1 );
      // normalization
      std::get< Uk >( matrix_spetrum[k] ) /= std::get< Uk >( matrix_spetrum[k] ).lpNorm< 2 >();
      std::get< Vk >( matrix_spetrum[k] ) /= std::get< Vk >( matrix_spetrum[k] ).lpNorm< 2 >();
    }
  //
  //
  MAC_nip::NipPMD pmd;
  pmd.K_factors( AAA, matrix_spetrum, L1, L1 );


  //
  // Test on CCA with small matrices
  //

  //
  // Initialization
  int
    N = 100, // number of samples
    p = 4,  // Dimention Xa
    q = 3;  // Dimention Xb
  // Random information
  std::default_random_engine generator;
  std::normal_distribution< double >
    rd_a(0.0,1.0),
    rd_xi0(0.0,0.02), rd_xi1(0.0,0.04),rd_xi2(0.0,0.03);
  // Correlated matrices
  Eigen::MatrixXd Xa = Eigen::MatrixXd::Zero( N, p );
  Eigen::MatrixXd Xb = Eigen::MatrixXd::Zero( N, q );
  // Matrices initialization
  for ( int n = 0 ; n < N ; n++ )
    {
      // A
      for ( int a = 0 ; a  < 4 ; a++ )
	Xa( n, a ) = rd_a( generator );
      // B
      Xb( n, 0) =  Xa( n, 2 ) + rd_xi0(generator);
      Xb( n, 1) =  Xa( n, 0 ) + rd_xi1(generator);
      Xb( n, 2) = -Xa( n, 3 ) + rd_xi2(generator);
    }

  //
  // Correlation matrix
  Eigen::MatrixXd Z = Xa.transpose() * Xb;
  // K is the min between Xa and Xb
  std::size_t K_cca = (p > q ? q : p);
  Spectra matrix_spetrum_cca( K_cca );
  // initialize the spectra
  // ToDo: the first vector should be the SVD highest eigen vector
  for ( int k = 0 ; k < K_cca ; k++ )
    {
      // Coefficient
      std::get< coeff_k >( matrix_spetrum_cca[k] ) = 0.;
      // vectors
      std::get< Uk >( matrix_spetrum_cca[k] ) = Eigen::MatrixXd::Random( p, 1 );
      std::get< Vk >( matrix_spetrum_cca[k] ) = Eigen::MatrixXd::Random( q, 1 );
      // normalization
      std::get< Uk >( matrix_spetrum_cca[k] ) /= std::get< Uk >( matrix_spetrum_cca[k] ).lpNorm< 2 >();
      std::get< Vk >( matrix_spetrum_cca[k] ) /= std::get< Vk >( matrix_spetrum_cca[k] ).lpNorm< 2 >();
    }
  //
  std::cout << "Z:\n" << Z << std::endl;
  std::cout << "Z_norm:\n" << MAC_nip::NipPMA_tools::normalize( Z, MAC_nip::STANDARDIZE ) << std::endl;

  //
  // CCA through PMD
  MAC_nip::NipPMD pmd_cca;
  pmd_cca.K_factors( MAC_nip::NipPMA_tools::normalize( Z, MAC_nip::STANDARDIZE ),
		     matrix_spetrum_cca, L1, L1 );

  
  
  //
  //
  return EXIT_SUCCESS;
}
