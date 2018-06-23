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
#include "../PMA_cross_validation.h"
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
  pmd.K_factors( AAA, matrix_spetrum, L1, L1, true );


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
		     matrix_spetrum_cca, L1, L1, true );

  

  //
  // Test on SPC with small matrices
  //

  std::cout << "###### PCA #########" << std::endl;
  
  //
  //
  Eigen::MatrixXd B = Eigen::MatrixXd(7,6);
  B <<
    1, 10, 1, -2, 21, 102,
    2, 11, 3, -3, 23, 98,
    0.5, 9, 2, -4, 20, 120,
    4, 13, 3, -1, 18, 112,
    7,  2, 3, -4, 7, -105,
    15,  2, 1, -1, 4, -100,
    12, 1, 3, -2, 5, -88;
  std::cout << B << std::endl;
  //
  //Spectrum
  std::size_t Kb = B.cols();
  Spectra matrix_spetrum_spc( Kb );
  // initialize the spectra
  // ToDo: the first vector should be the SVD highest eigen vector
  for ( int k = 0 ; k < Kb ; k++ )
    {
      // Coefficient
      std::get< coeff_k >( matrix_spetrum_spc[k] ) = 0.;
      // vectors
      std::get< Uk >( matrix_spetrum_spc[k] ) = Eigen::MatrixXd::Random( Kb, 1 );
      std::get< Vk >( matrix_spetrum_spc[k] ) = Eigen::MatrixXd::Random( Kb, 1 );
      // normalization
      std::get< Uk >( matrix_spetrum_spc[k] ) /= std::get< Uk >( matrix_spetrum_spc[k] ).lpNorm< 2 >();
      std::get< Vk >( matrix_spetrum_spc[k] ) /= std::get< Vk >( matrix_spetrum_spc[k] ).lpNorm< 2 >();
    }
  //
  //
  MAC_nip::NipSPC spc;
  std::cout << MAC_nip::NipPMA_tools::normalize( B.transpose() * B,
						   MAC_nip::STANDARDIZE )
	    << std::endl;
  // SPC PCA
  spc.set_cs(1.,1.);
  spc.K_factors( MAC_nip::NipPMA_tools::normalize( B.transpose() * B,
						   MAC_nip::STANDARDIZE ),
		 matrix_spetrum_spc, L1, L1, true );

  //
  //
  std::cout << "Projection on the first sparse principal component: \n"
	    << MAC_nip::NipPMA_tools::normalize( B.transpose() * B,
						 MAC_nip::STANDARDIZE ) * std::get< Vk >( matrix_spetrum_spc[0] )
	    << std::endl;
  std::cout << "Projection on the first sparse principal component: \n"
	    << std::get< Vk >( matrix_spetrum_spc[0] ).transpose() * MAC_nip::NipPMA_tools::normalize( B.transpose() * B,
												       MAC_nip::STANDARDIZE )
	    << std::endl;
  // SVD
  Eigen::JacobiSVD< Eigen::MatrixXd >
    svd_V( MAC_nip::NipPMA_tools::normalize( B.transpose() * B,
					     MAC_nip::STANDARDIZE ),
	   Eigen::ComputeThinV);
  //
  std::cout << "Projection on the first SVD principal component: \n"
	    << MAC_nip::NipPMA_tools::normalize( B.transpose() * B,
						 MAC_nip::STANDARDIZE ) * svd_V.matrixV().col( 0 )
	    << std::endl;
  

  
  //
  //
  return EXIT_SUCCESS;
}
