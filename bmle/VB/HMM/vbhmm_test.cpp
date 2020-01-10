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
  const int Dim = 1;
  const int S   = 3;

  //
  // Load the test dataset
  //NeuroBayes::Load_csv reader("../data/Sim.train.data.seq.len.mu.08.csv");
  // medium 1 2 5
  //NeuroBayes::Load_csv reader("../data/Sim.train.data.seq.len.mu.08_medium1.csv");
  //NeuroBayes::Load_csv reader("../data/new_sim.txt");
  NeuroBayes::Load_csv reader("../data/hhm.csv");
  // 20 cases
  //NeuroBayes::Load_csv reader("../data/Sim.train.data.seq.len.mu.08_short.csv");
  //two dim NeuroBayes::Load_csv reader("../data/Sim.train.data.seq.len.mu.08_2.csv");

  //
  // Size of the sequence can be different for each entry (subject).
  std::vector< std::vector< Eigen::Matrix< double, Dim+1, 1 > > >
    HMM_intensity_age = reader.get_VB_HMM_date< Dim+1 >();
  std::vector< std::vector< Eigen::Matrix< double, Dim, 1 > > >
    HMM_intensity;
  std::vector< std::vector< Eigen::Matrix< double, Dim, 1 > > >
    HMM_age;
  //
  HMM_intensity.resize( HMM_intensity_age.size() );
  HMM_age.resize( HMM_intensity_age.size() );
  int subject = 0;
  //
  for ( auto sub : HMM_intensity_age )
    {
      HMM_intensity[subject].resize( HMM_intensity_age[subject].size() );
      HMM_age[subject].resize( HMM_intensity_age[subject].size() );
      //std::cout << "Subject " << subject << std::endl;
      int timepoint = 0;
      for ( auto tp : sub )
	{
	  //std::cout << tp << std::endl;
	  HMM_intensity[subject][timepoint] << tp(0,0);
	  HMM_age[subject][timepoint++] << tp(1,0);
	}
      subject++;
    }

  //
  //
  VB::HMM::Hidden_Markov_Model < /*Dim*/ Dim, /*number_of_states*/ S > VBHMM_intensity_age( HMM_intensity, HMM_age );
  //
  VBHMM_intensity_age.ExpectationMaximization();

  //
  //
  return EXIT_SUCCESS;
}

