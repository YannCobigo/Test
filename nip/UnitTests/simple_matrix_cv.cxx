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
#include "../PMD_cross_validation.h"
#include "../SPC_cross_validation.h"
//
//
int main(int argc, char const *argv[]){


  //
  // Test on CCA with small matrices
  //

  //
  // Initialization
  int
    N = 34, // number of samples
    p = 10,  // Dimention Xa
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
      for ( int a = 0 ; a  < p ; a++ )
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
//  MAC_nip::NipPMD pmd_cca;
//  pmd_cca.K_factors( MAC_nip::NipPMA_tools::normalize( Z, MAC_nip::STANDARDIZE ),
//		     matrix_spetrum_cca, L1, L1 );

  //
  // Cross validation
  std::cout  << std::endl;
  //
  std::shared_ptr< Spectra > spetrum = std::make_shared< Spectra >( matrix_spetrum_cca );
  MAC_nip::Nip_PMD_cross_validation< /* K-folds = */ 3, /* CPU */ 8 > pmd_cv( std::make_shared< const Eigen::MatrixXd >( Xa ),
									      std::make_shared< const Eigen::MatrixXd >( Xb ) );
  pmd_cv.validation( spetrum );
  
  
  //
  //
  return EXIT_SUCCESS;
}
