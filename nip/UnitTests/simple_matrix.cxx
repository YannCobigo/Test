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
//using Spectra = std::vector< std::tuple< double, Eigen::MatrixXd, Eigen::MatrixXd> >;
//
//
#include "../PMA.h"
#include "../PMD.h"
#include "../SPC.h"
//
//
int main(int argc, char const *argv[]){

  //
  // test PMD
  Eigen::MatrixXd A = Eigen::MatrixXd(3,4);
  A << 1, 10, 1, -2, -3, 1, 3, 3, 7, 2,3,4;

//  A <<
//    -1.21016,  0.248824,  0.458866,
//    -1.26964,  0.222352,   1.17882,
//    2.20899, -0.489931, -0.193671;

  
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
  // K is the supposed ran of the matrix A
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
  //PCA_K_factors( normalize(AAA,STANDARDIZE), matrix_spetrum, L1, L1 );


  
  //
  //
  return EXIT_SUCCESS;
}
