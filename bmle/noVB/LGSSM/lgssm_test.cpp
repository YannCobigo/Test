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
#include "Tools.h"
//
//
//
int main(int argc, char const *argv[])
{
  //
  // model
  const int Dim = 4;
  const int S   = 4;

  //
  // Load the test dataset
  //NeuroBayes::Load_csv reader("../data/hhm.csv");
  //NeuroBayes::Load_csv reader("../data/ADNI_clusters.csv");
  //
  // Size of the sequence can be different for each entry (subject).
  //std::vector< std::vector< Eigen::Matrix< double, Dim+1, 1 > > >
  //  LGSSM_intensity_age = reader.get_VB_LGSSM_date< Dim+1 >();
  std::vector< std::vector< Eigen::Matrix< double, Dim, 1 > > >
    LGSSM_intensity;
  std::vector< std::vector< Eigen::Matrix< double, 1, 1 > > >
    LGSSM_age;
//  //
//  // Simulation
//  const int N = 100;
//  LGSSM_intensity.resize( N /*29*/  );
//  LGSSM_age.resize( N /*29*/ );
//  //
//  // 
//  for ( int n = 0 ; n < N ; n++ )
//    {
//      int T = 10;
//      LGSSM_intensity[n].resize(T);
//      LGSSM_age[n].resize(T);
//      for ( int t = 0 ; t < T ; t++ )
//	{
//	  Eigen::Matrix< double, S, S > V  = 1.0e-01 * Eigen::Matrix< double, S, S >::Identity();
//	  Eigen::Matrix< double, S, 1 > mu = 
//	    NeuroBayes::gaussian_multivariate< S >( (1+n + n*t) * Eigen::Matrix< double, S, 1 >::Ones(),
//						    V );
//	  //
//	  //
//	  for ( int d = 0 ; d < S ; d++ )
//	    LGSSM_intensity[n][t](d,0) =  mu(d,0);
//	  LGSSM_age[n][t] << 0.;
//	  //
//	  std::cout << "Real,"<<n<<","<<t<<"," << 1+n + n*t << std::endl;
//	}
//    }
  //
  //
  LGSSM_intensity.resize( 3 /*29*/  );
  LGSSM_intensity[0].resize(5);
  LGSSM_intensity[0][0] << 0.0888585,0.239285,0.670569,0.00128727;
  LGSSM_intensity[0][1] << 0.00744202,0.0229673,0.969501,8.91496E-05;
  LGSSM_intensity[0][2] << 0.00359251,0.0118994,0.984486,2.18637E-05;
  LGSSM_intensity[0][3] << 0.00343513,0.0133097,0.983228,2.7527E-05;
  LGSSM_intensity[0][4] << 0.0495984,0.055406,0.893359,0.00163616;
  LGSSM_intensity[1].resize(5);
  LGSSM_intensity[1][0] << 0.157144,0.765944,0.0747845,0.0021276;
  LGSSM_intensity[1][1] << 0.00350017,0.987859,0.00854729,9.37015E-05;
  LGSSM_intensity[1][2] << 0.00240256,0.987611,0.00993491,5.19396E-05;
  LGSSM_intensity[1][3] << 0.00162485,0.868757,0.129544,7.38363E-05;
  LGSSM_intensity[1][4] << 0.0303165,0.718971,0.247024,0.00368934;
//  LGSSM_intensity[2].resize(1);
//  LGSSM_intensity[2][0] << 0.00955131,0.718149,0.00343095,0.268869;
//  LGSSM_intensity[3].resize(1);
//  LGSSM_intensity[3][0] << 0.324358,0.551575,0.120014,0.00405294;
//  LGSSM_intensity[4].resize(1);
//  LGSSM_intensity[4][0] << 0.295926,0.574033,0.126111,0.00392977;
  LGSSM_intensity[2].resize(4);
  LGSSM_intensity[2][0] << 0.191585,0.720989,0.0847059,0.00272051;
  LGSSM_intensity[2][1] << 0.281924,0.54754,0.167426,0.00310993;
  LGSSM_intensity[2][2] << 0.238532,0.615785,0.142882,0.00279994;
  LGSSM_intensity[2][3] << 0.215355,0.67287,0.109304,0.00247006;
//  LGSSM_intensity[6].resize(2);
//  LGSSM_intensity[6][0] << 0.222687,0.66818,0.105976,0.00315745;
//  LGSSM_intensity[6][1] << 0.00395141,0.724707,0.271204,0.000137096;
//  LGSSM_intensity[7].resize(1);
//  LGSSM_intensity[7][0] << 0.295926,0.574033,0.126111,0.00392977;
//  LGSSM_intensity[8].resize(3);
//  LGSSM_intensity[8][0] << 0.251824,0.635765,0.106653,0.00575759;
//  LGSSM_intensity[8][1] << 0.143318,0.789453,0.0593005,0.00792879;
//  LGSSM_intensity[8][2] << 0.0167064,0.921917,0.00813137,0.0532457;
//  LGSSM_intensity[9].resize(2);
//  LGSSM_intensity[9][0] << 0.237294,0.63782,0.121552,0.00333464;
//  LGSSM_intensity[9][1] << 0.455978,0.456958,0.0828703,0.0041939;
//  LGSSM_intensity[10].resize(2);
//  LGSSM_intensity[10][0] << 0.0403983,0.129326,0.017083,0.813193;
//  LGSSM_intensity[10][1] << 0.12638,0.330314,0.10596,0.437346;
//  LGSSM_intensity[11].resize(2);
//  LGSSM_intensity[11][0] << 0.0489377,0.907317,0.0163763,0.0273693;
//  LGSSM_intensity[11][1] << 0.00774167,0.759759,0.0141832,0.218317;
//  LGSSM_intensity[12].resize(1);
//  LGSSM_intensity[12][0] << 0.199872,0.782806,0.0108439,0.00647876;
//  LGSSM_intensity[13].resize(1);
//  LGSSM_intensity[13][0] << 0.278474,0.593185,0.124642,0.00369803;
//  LGSSM_intensity[14].resize(1);
//  LGSSM_intensity[14][0] << 0.295501,0.574645,0.12593,0.00392413;
//  LGSSM_intensity[15].resize(1);
//  LGSSM_intensity[15][0] << 0.295922,0.574027,0.12611,0.00394145;
//  LGSSM_intensity[16].resize(1);
//  LGSSM_intensity[16][0] << 0.294016,0.636798,0.00279379,0.0663929;
//  LGSSM_intensity[17].resize(1);
//  LGSSM_intensity[17][0] << 0.745286,0.229082,0.0245643,0.0010685;
//  LGSSM_intensity[18].resize(1);
//  LGSSM_intensity[18][0] << 0.53571778,0.198621,0.264371,0.00129022;
//  LGSSM_intensity[19].resize(1);
//  LGSSM_intensity[19][0] << 0.256981,0.473892,0.265565,0.00356194;
//  LGSSM_intensity[20].resize(1);
//  LGSSM_intensity[20][0] << 0.295926,0.574033,0.126111,0.00392977;
//  LGSSM_intensity[21].resize(1);
//  LGSSM_intensity[21][0] << 0.289539,0.582488,0.123389,0.00458363;
//  LGSSM_intensity[22].resize(1);
//  LGSSM_intensity[22][0] << 0.295924,0.57403,0.126111,0.00393561;
//  LGSSM_intensity[23].resize(1);
//  LGSSM_intensity[23][0] << 0.315497,0.562436,0.118378,0.0036888;
//  LGSSM_intensity[24].resize(2);
//  LGSSM_intensity[24][0] << 0.291032,0.580908,0.124026,0.00403535;
//  LGSSM_intensity[24][1] << 0.295906,0.573996,0.126103,0.00399442;
//  LGSSM_intensity[25].resize(3);
//  LGSSM_intensity[25][0] << 0.293985,0.576826,0.125284,0.003904;
//  LGSSM_intensity[25][1] << 0.29338,0.577698,0.125026,0.00389596;
//  LGSSM_intensity[25][2] << 0.262453,0.613474,0.117471,0.00660233;
//  LGSSM_intensity[26].resize(2);
//  LGSSM_intensity[26][0] << 0.222687,0.66818,0.105976,0.00315745;
//  LGSSM_intensity[26][1] << 0.00395141,0.724707,0.271204,0.000137096;
//  LGSSM_intensity[27].resize(1);
//  LGSSM_intensity[27][0] << 0.253406,0.6262,0.107824,0.0125707;
//  LGSSM_intensity[28].resize(1);
//  LGSSM_intensity[28][0] << 0.167321,0.75983,0.0706441,0.00220463;
  //
  //LGSSM_age[0].resize(5);
  //LGSSM_age[0][0] << 68;
  //LGSSM_age[0][1] << 69;
  //LGSSM_age[0][2] << 70;
  //LGSSM_age[0][3] << 71;
  //LGSSM_age[0][4] << 72;
 //LGSSM_age[0].resize(5);
 //LGSSM_age[0][0] << 76;
 //LGSSM_age[0][1] << 78;
 //LGSSM_age[0][2] << 81;
 //LGSSM_age[0][3] << 82;
 //LGSSM_age[0][4] << 83;
//  LGSSM_age[2].resize(1);
//  LGSSM_age[2][0] << 51;
//  LGSSM_age[3].resize(1);
//  LGSSM_age[3][0] << 44;
//  LGSSM_age[4].resize(1);
//  LGSSM_age[4][0] << 64;
//  LGSSM_age[5].resize(4);
//  LGSSM_age[5][0] << 41;
//  LGSSM_age[5][1] << 42;
//  LGSSM_age[5][2] << 43;
//  LGSSM_age[5][3] << 45;
//  LGSSM_age[6].resize(2);
//  LGSSM_age[6][0] << 60;
//  LGSSM_age[6][1] << 62;
//  LGSSM_age[7].resize(1);
//  LGSSM_age[7][0] << 41;
//  LGSSM_age[8].resize(3);
//  LGSSM_age[8][0] << 57;
//  LGSSM_age[8][1] << 58;
//  LGSSM_age[8][2] << 59;
//  LGSSM_age[9].resize(2);
//  LGSSM_age[9][0] << 51;
//  LGSSM_age[9][1] << 52;
//  LGSSM_age[10].resize(2);
//  LGSSM_age[10][0] << 38;
//  LGSSM_age[10][1] << 39;
//  LGSSM_age[11].resize(2);
//  LGSSM_age[11][0] << 54;
//  LGSSM_age[11][1] << 55;
//  LGSSM_age[12].resize(1);
//  LGSSM_age[12][0] << 71;
//  LGSSM_age[13].resize(1);
//  LGSSM_age[13][0] << 31;
//  LGSSM_age[14].resize(1);
//  LGSSM_age[14][0] << 39;
//  LGSSM_age[15].resize(1);
//  LGSSM_age[15][0] << 54;
//  LGSSM_age[16].resize(1);
//  LGSSM_age[16][0] << 40;
//  LGSSM_age[17].resize(1);
//  LGSSM_age[17][0] << 25;
//  LGSSM_age[18].resize(1);
//  LGSSM_age[18][0] << 34;
//  LGSSM_age[19].resize(1);
//  LGSSM_age[19][0] << 63;
//  LGSSM_age[20].resize(1);
//  LGSSM_age[20][0] << 45;
//  LGSSM_age[21].resize(1);
//  LGSSM_age[21][0] << 42;
//  LGSSM_age[22].resize(1);
//  LGSSM_age[22][0] << 64;
//  LGSSM_age[23].resize(1);
//  LGSSM_age[23][0] << 46;
//  LGSSM_age[24].resize(2);
//  LGSSM_age[24][0] << 57;
//  LGSSM_age[24][1] << 58;
//  LGSSM_age[25].resize(3);
//  LGSSM_age[25][0] << 54;
//  LGSSM_age[25][1] << 55;
//  LGSSM_age[25][2] << 57;
//  LGSSM_age[26].resize(2);
//  LGSSM_age[26][0] << 55;
//  LGSSM_age[26][1] << 57;
//  LGSSM_age[27].resize(1);
//  LGSSM_age[27][0] << 47;
//  LGSSM_age[28].resize(1);
//  LGSSM_age[28][0] << 73;
  //
  //
  noVB::LGSSM::Linear_Gaussian_State_Space_Model < /*Dim*/ Dim, /*number_of_states*/ S > LGSSM_intens_age( LGSSM_intensity, LGSSM_age );
  //
  LGSSM_intens_age.ExpectationMaximization();

  //
  //
  return EXIT_SUCCESS;
}
