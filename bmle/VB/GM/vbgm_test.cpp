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
#include "VBGM.h"
#include "MakeITKImage.h"

int main(int argc, char const *argv[])
{
  //
  //
  std::default_random_engine generator;
  std::normal_distribution< double > gauss_11(5.0,1.0);
  std::normal_distribution< double > gauss_12(5.0,1.0);
  std::normal_distribution< double > gauss_21(10.0,1.0);
  std::normal_distribution< double > gauss_22(10.0,1.0);
  std::uniform_real_distribution< double > uniform(0.0,1.0);

  //
  //
  std::list< Eigen::Matrix< double, 2, 1 > > X_intensity;
  std::list< Eigen::Matrix< double, 3, 1 > > X_pos;
  //
  for ( int i = 0 ; i < 100 ; i++ )
    {
      double mixture = uniform( generator );
      Eigen::Matrix< double, 2, 1 > mixture_gauss = Eigen::Matrix< double, 2, 1 >::Zero();
      if ( mixture < 0.7 )
	{
	  mixture_gauss(0,0) = gauss_11( generator );
	  mixture_gauss(1,0) = gauss_12( generator );
	}
      else
	{
	  mixture_gauss(0,0) = gauss_21( generator );
	  mixture_gauss(1,0) = gauss_22( generator );
	}
      //
      X_intensity.push_back( mixture_gauss );
      //std::cout << mixture_gauss(0,0) << "," << mixture_gauss(1,0) << std::endl;
    }
  


  //
  //
  const int K = 6;
  const int K_clus = 10;
//  EM< /*Dim*/ 1, K > GaussianMixture_intensity;
//  GaussianMixture_intensity.ExpectationMaximization( X_intensity );
//  EM< /*Dim*/ 3, /*K_gaussians*/ K_clus > GaussianMixture;
//  //
//  MAC::MakeITKImage output;
//  output = MAC::MakeITKImage( K_clus, "Clusters_probabilities.nii.gz", reader_CM[0] );
//
  VB::GM::VBGaussianMixture < /*Dim*/ 2, /*K_gaussians*/ K > VBGaussianMixture_intensity( X_intensity );
  VBGaussianMixture_intensity.ExpectationMaximization();
//  //
//  VBGaussianMixture < /*Dim*/ 3, /*K_gaussians*/ K_clus > VBGaussianMixture;
//  //VBGaussianMixture.ExpectationMaximization( X_pos );


  //
  //
  return EXIT_SUCCESS;
}

