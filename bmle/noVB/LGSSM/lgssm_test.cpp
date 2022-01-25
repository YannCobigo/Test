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
#include "LGSSM.h"
#include "MakeITKImage.h"
#include "IO/Load_csv.h"
//
//
//
int main(int argc, char const *argv[])
{
  //
  // model
  const int Dim = 1;
  const int S   = 5;

  //
  // Load the test dataset
  //NeuroBayes::Load_csv reader("../data/hhm.csv");
  NeuroBayes::Load_csv reader("../data/ADNI_clusters.csv");
  //
  // Size of the sequence can be different for each entry (subject).
  std::vector< std::vector< Eigen::Matrix< double, Dim+1, 1 > > >
    LGSSM_intensity_age = reader.get_VB_LGSSM_date< Dim+1 >();
  std::vector< std::vector< Eigen::Matrix< double, Dim, 1 > > >
    LGSSM_intensity;
  std::vector< std::vector< Eigen::Matrix< double, 1, 1 > > >
    LGSSM_age;
  //
  LGSSM_intensity.resize( LGSSM_intensity_age.size() );
  LGSSM_age.resize( LGSSM_intensity_age.size() );
  int subject = 0;
  for ( auto sub : LGSSM_intensity_age )
    {
      LGSSM_intensity[subject].resize( LGSSM_intensity_age[subject].size() );
      LGSSM_age[subject].resize( LGSSM_intensity_age[subject].size() );
      //std::cout << "Subject " << subject << std::endl;
      int timepoint = 0;
      for ( auto tp : sub )
	{
	  //std::cout << tp << std::endl;
	  int d = 0;
	  for ( d = 0 ; d < Dim ; d++ )
	    LGSSM_intensity[subject][timepoint](d,0) = tp(d,0);
	  LGSSM_age[subject][timepoint++] << tp(d,0);
	}
      subject++;
    }

  //
  //
  noVB::LGSSM::Linear_Gaussian_State_Space_Model < /*Dim*/ Dim, /*number_of_states*/ S > LGSSM_intens_age( LGSSM_intensity, LGSSM_age );
  //
  LGSSM_intens_age.ExpectationMaximization();

  //
  //
  return EXIT_SUCCESS;
}

