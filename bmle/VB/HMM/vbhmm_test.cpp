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
#include "Load_csv.h"
//
//
//
int main(int argc, char const *argv[])
{
  //
  // model
  const int Dim = 1;
  const int S   = 2;

  //
  // Load the test dataset
  NeuroBayes::Load_csv reader("../data/Sim.train.data.seq.len.mu.08_short.csv");
  // two dim NeuroBayes::Load_csv reader("../data/Sim.train.data.seq.len.mu.08_2.csv");

  //
  // Size of the sequence can be different for each entry (subject).
  std::vector< std::vector< Eigen::Matrix< double, /*Dim*/ Dim, 1 > > >
    HMM_intensity = reader.get_VB_HMM_date< Dim, S >();
  //
  for ( auto sub : HMM_intensity )
    {
      std::cout << "Subject " << std::endl;
      for ( auto tp : sub )
	std::cout << tp << std::endl;
    }

  //
  //
  VB::HMM::Hidden_Markov_Model < /*Dim*/ Dim, /*number_of_states*/ S > VBHMM_intensity( HMM_intensity );
  //
  VBHMM_intensity.ExpectationMaximization();

  //
  //
  return EXIT_SUCCESS;
}

