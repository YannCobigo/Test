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
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIterator.h>
#include <itkNiftiImageIO.h>
#include <itkOrientImageFilter.h>
#include <itkSpatialOrientation.h>

//#include "EM.h"
//#include "VBGaussianMixture.h"
#include "VBHMM.h"
#include "MakeITKImage.h"
#include "IO/Load_csv.h"
//
//
//
int main(int argc, char const *argv[])
{
  //
  // model
  const int Dim = 4;
  const int S   = 2;

  //
  // Load the test dataset
  //NeuroBayes::Load_csv reader("../data/hhm_2d.csv");
  //NeuroBayes::Load_csv reader("../data/hhm_3d.csv");
  //NeuroBayes::Load_csv reader("../data/ADNI_clusters.csv");
  //
  // Size of the sequence can be different for each entry (subject).
//  std::vector< std::vector< Eigen::Matrix< double, Dim+1, 1 > > >
//    HMM_intensity_age = reader.get_VB_HMM_date< Dim+1 >();
  std::vector< std::vector< Eigen::Matrix< double, Dim, 1 > > >
    HMM_intensity;
  std::vector< std::vector< Eigen::Matrix< double, 1, 1 > > >
    HMM_age;
  //
//  HMM_intensity.resize( HMM_intensity_age.size() );
//  HMM_age.resize( HMM_intensity_age.size() );
//  int subject = 0;
//  //
//  for ( auto sub : HMM_intensity_age )
//    {
//      HMM_intensity[subject].resize( HMM_intensity_age[subject].size() );
//      HMM_age[subject].resize( HMM_intensity_age[subject].size() );
//      //std::cout << "Subject " << subject << std::endl;
//      int timepoint = 0;
//      for ( auto tp : sub )
//	{
//	  //std::cout << tp << std::endl;
//	  int d = 0;
//	  for ( d = 0 ; d < Dim ; d++ )
//	    HMM_intensity[subject][timepoint](d,0) = tp(d,0);
//	  HMM_age[subject][timepoint++] << tp(d,0);
//	}
//      subject++;
//    }
  //
  HMM_intensity.resize( 2 /*29*/  );
  HMM_age.resize( 2 /*29*/ );
  //
  //
  HMM_intensity[0].resize(5);
  HMM_intensity[0][0] << 0.0888585,0.239285,0.670569,0.00128727;
  HMM_intensity[0][1] << 0.00744202,0.0229673,0.969501,8.91496E-05;
  HMM_intensity[0][2] << 0.00359251,0.0118994,0.984486,2.18637E-05;
  HMM_intensity[0][3] << 0.00343513,0.0133097,0.983228,2.7527E-05;
  HMM_intensity[0][4] << 0.0495984,0.055406,0.893359,0.00163616;
  HMM_intensity[1].resize(5);
  HMM_intensity[1][0] << 0.157144,0.765944,0.0747845,0.0021276;
  HMM_intensity[1][1] << 0.00350017,0.987859,0.00854729,9.37015E-05;
  HMM_intensity[1][2] << 0.00240256,0.987611,0.00993491,5.19396E-05;
  HMM_intensity[1][3] << 0.00162485,0.868757,0.129544,7.38363E-05;
  HMM_intensity[1][4] << 0.0303165,0.718971,0.247024,0.00368934;
  //
  HMM_age[0].resize(5);
  HMM_age[0][0] << 68;
  HMM_age[0][1] << 69;
  HMM_age[0][2] << 70;
  HMM_age[0][3] << 71;
  HMM_age[0][4] << 72;
  HMM_age[1].resize(5);
  HMM_age[1][0] << 76;
  HMM_age[1][1] << 78;
  HMM_age[1][2] << 81;
  HMM_age[1][3] << 82;
  HMM_age[1][4] << 83;

  //
  //
  VB::HMM::Hidden_Markov_Model < /*Dim*/ Dim, /*number_of_states*/ S > VBHMM_intensity_age( HMM_intensity, HMM_age );
  //
  VBHMM_intensity_age.ExpectationMaximization();

  //
  //
  return EXIT_SUCCESS;
}

