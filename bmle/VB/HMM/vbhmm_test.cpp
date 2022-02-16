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
  const int S   = 5;

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
  //
  // SIMULATION
  //
  //
  std::default_random_engine generator;
  std::uniform_int_distribution<int> TP( 3, 7 );
  std::uniform_int_distribution<int> AGE( 45, 65 );
  // 0 - CDR = 0
  // 1 - CDR = 0.5
  // 2 - CDR = 1+
  std::uniform_int_distribution<int> SICK( 0, 2 );
  // 0 - AD
  // 1 - BV
  // 2 - nfvPPA
  // 3 - svPPA
  std::uniform_int_distribution<int> FLAVOR( 0, 3 );
  std::map< int, std::string > Dx;
  Dx[0] = "AD";
  Dx[1] = "BV";
  Dx[2] = "nfvPPA";
  Dx[3] = "svPPA";
  //
  std::vector< std::vector< double > > CN(4);
  CN[0] = {0.28, 0.10};
  CN[1] = {0.53, 0.12};
  CN[2] = {0.18, 0,11};
  CN[3] = {0.01, 0.02};
  Eigen::Matrix< double, 4, 1 > mu_cn   = Eigen::Matrix< double, 4, 1 >::Identity();
  Eigen::Matrix< double, 4, 4 > sig_var = Eigen::Matrix< double, 4, 4 >::Identity();
  mu_cn(0,0) = 0.28;
  mu_cn(1,0) = 0.53;
  mu_cn(2,0) = 0.18;
  mu_cn(3,0) = 0.01;
  sig_var(0,0) = 0.10*0.10 / 2.;
  sig_var(1,1) = 0.12*0.12 / 2.;
  sig_var(2,2) = 0.11*0.11 / 2.;
  sig_var(3,3) = 0.02*0.02 / 2.;
  Eigen::Matrix< double, 4, 4 > sigma_cn = 0.0001 * Eigen::Matrix< double, 4, 4 >::Random();
  sigma_cn = 0.5 * ( sigma_cn + sigma_cn.transpose() ) + sig_var;
  std::cout << "sigma_cn = \n " << sigma_cn << std::endl;
  //
  //
  // Simulation
  const int N = 100;
  HMM_intensity.resize( N /*29*/  );
  HMM_age.resize( N /*29*/ );
  //
  // 
  for ( int i = 0 ; i < N ; i++ )
    {
      //
      //
      int
	Ti     = TP( generator ),
	age    = AGE( generator ),
	status = SICK( generator );
      //
      HMM_intensity[i].resize(Ti);
      HMM_age[i].resize(Ti);
      //
      std::cout
	<< "Subject " << i
	<< " " << age << " years old, status: " << status
	<< std::endl;
      //
      if ( status == 0 )
	{
	  //
	  // Cognitive Normal
	  for ( int t = 0 ; t < Ti ; t++ )
	    {
	      if ( Dim == 4 )
		{
		  //
		  // generate the distribution
		  Eigen::Matrix< double, Dim, 1 > mu = NeuroBayes::gaussian_multivariate< Dim >( (t == 0 ? mu_cn : HMM_intensity[i][t-1] ),
												 sigma_cn );
//		  Eigen::Matrix< double, Dim, 1 > mu = NeuroBayes::gaussian_multivariate< Dim >( ( t == 0 ? mu_cn : (HMM_intensity[i][t-1].array().exp()).matrix() ),
//												 sigma_cn );
		  //
		  // make positive and normalize
		  double norm = 0.;
		  //
		  for ( int d = 0 ; d < Dim ; d++ )
		    {
		      if ( mu(d,0) < 0. )
			mu(d,0) = -mu(d,0);
		      norm += mu(d,0);
		    }
		  mu /= norm;
		  std::cout << "\t tp "<<t<<" mu = \n" << mu << std::endl;
		  //
		  HMM_age[i][t]       << static_cast< double >( age++ );
		  HMM_intensity[i][t] = mu;
		  //HMM_intensity[i][t] = ( mu.array().log() ).matrix();
		  //HMM_intensity[i][t] = ( mu );
		}
	    }	  
	}
      //
      // Cognitive weakness
      else
	{
	  //
	  // Which flavor
	  int flavor = FLAVOR(generator);
	  std::cout
	    << "The patient has " << Dx[ flavor ] 
	    << std::endl;
	  //
	  for ( int t = 0 ; t < Ti ; t++ )
	    if ( Dim == 4 )
	      {
		//
		//
		Eigen::Matrix< double, Dim, 1 > mu_dx = mu_cn;
		Eigen::Matrix< double, Dim, 1 > mu;
		//
		if ( t == 0 )
		  {
		    // MCI
		    if ( status == 1 )
		      mu_dx( flavor, 0 ) += 2 * CN[flavor][1];
		    else
		      mu_dx( flavor, 0 ) += 6 * CN[flavor][1];
		    //
		    mu = NeuroBayes::gaussian_multivariate< Dim >( mu_dx ,
								   sigma_cn );
		  }
		else
		  {
		    mu_dx               = HMM_intensity[i][t-1];
		    //mu_dx               = HMM_intensity[i][t-1].array().exp().matrix();
		    mu_dx( flavor, 0 ) += 3 * CN[flavor][1];
		    //
		    mu = NeuroBayes::gaussian_multivariate< Dim >( mu_dx ,
								   sigma_cn );
		  }
		//
		// make positive and normalize
		double norm = 0.;
		//
		for ( int d = 0 ; d < Dim ; d++ )
		  {
		    if ( mu(d,0) < 0. )
		      mu(d,0) = -mu(d,0);
		    norm += mu(d,0);
		  }
		mu /= norm;
		std::cout << "\t tp "<<t<<" mu = \n" << mu << std::endl;
		//
		HMM_age[i][t]       << static_cast< double >( age++ );
		HMM_intensity[i][t] = mu;
		//HMM_intensity[i][t] = ( mu.array().log() ).matrix();
		//HMM_intensity[i][t] = ( mu );
	      }
	}
    }
  //
  //
  // REAL
  //
  //
//  HMM_intensity.resize( 3/*29*/  );
//  HMM_age.resize( 3 /*29*/ );
//  //
//  //
//  HMM_intensity[0].resize(5);
//  HMM_intensity[0][0] << 0.0888585,0.239285,0.670569,0.00128727;
//  HMM_intensity[0][1] << 0.00744202,0.0229673,0.969501,8.91496E-05;
//  HMM_intensity[0][2] << 0.00359251,0.0118994,0.984486,2.18637E-05;
//  HMM_intensity[0][3] << 0.00343513,0.0133097,0.983228,2.7527E-05;
//  HMM_intensity[0][4] << 0.0495984,0.055406,0.893359,0.00163616;
//  HMM_intensity[1].resize(5);
//  HMM_intensity[1][0] << 0.157144,0.765944,0.0747845,0.0021276;
//  HMM_intensity[1][1] << 0.00350017,0.987859,0.00854729,9.37015E-05;
//  HMM_intensity[1][2] << 0.00240256,0.987611,0.00993491,5.19396E-05;
//  HMM_intensity[1][3] << 0.00162485,0.868757,0.129544,7.38363E-05;
//  HMM_intensity[1][4] << 0.0303165,0.718971,0.247024,0.00368934;
//  HMM_intensity[2].resize(4);
//  HMM_intensity[2][0] << 0.191585,0.720989,0.0847059,0.00272051;
//  HMM_intensity[2][1] << 0.281924,0.54754,0.167426,0.00310993;
//  HMM_intensity[2][2] << 0.238532,0.615785,0.142882,0.00279994;
//  HMM_intensity[2][3] << 0.215355,0.67287,0.109304,0.00247006;
//  //
//  HMM_age[0].resize(5);
//  HMM_age[0][0] << 68;
//  HMM_age[0][1] << 69;
//  HMM_age[0][2] << 70;
//  HMM_age[0][3] << 71;
//  HMM_age[0][4] << 72;
//  HMM_age[1].resize(5);
//  HMM_age[1][0] << 76;
//  HMM_age[1][1] << 78;
//  HMM_age[1][2] << 81;
//  HMM_age[1][3] << 82;
//  HMM_age[1][4] << 83;
//  HMM_age[2].resize(4);
//  HMM_age[2][0] << 41;
//  HMM_age[2][1] << 42;
//  HMM_age[2][2] << 43;
//  HMM_age[2][3] << 45;
  //
  //
  VB::HMM::Hidden_Markov_Model < /*Dim*/ Dim, /*number_of_states*/ S > VBHMM_intensity_age( HMM_intensity, HMM_age );
  //
  VBHMM_intensity_age.ExpectationMaximization();

  //
  //
  return EXIT_SUCCESS;
}

