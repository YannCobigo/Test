#include <iostream>
#include <cstdlib>
#include <string>
#include <cstdlib>
#include <fstream>
#include <sstream>  
#include <regex>  
#include <random>
//
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIterator.h>
#include <itkNiftiImageIO.h>
#include <itkOrientImageFilter.h>
#include <itkSpatialOrientation.h>
#include "itkImageDuplicator.h"
// Eigen
#include <Eigen/Core>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Sparse>
//
// Penalized Matrix Analysis
//
using Spectra = std::vector< std::tuple< double, Eigen::MatrixXd, Eigen::MatrixXd> >;
//
// Penalization
enum Penality { L1, L2, FUSION };
enum Normalization {NORMALIZE, STANDARDIZE, DEMEAN};
enum Spectrum {coeff_k = 0, Uk = 1, Vk = 2};
// Soft thresholding
double soft_threshold( const double A, const double C )
{
  //
  // Tests
  // C must be strictly positive value

  // 
  double abs_a = ( A > 0 ? A : -A );

  //
  //
  return ( A > 0 ? 1.:-1 ) * ( abs_a > C ? abs_a - C : 0. );
}
//
// Dichotomy search
double dichotomy_search( const Eigen::MatrixXd& Xu, const double L1norm_u,
			 const double C, const int Niter = 1000 )
{
  //
  // Tests

  //
  //
  double
    delta_1 = 0,
    delta_2 = C;
  Eigen::MatrixXd soft_u = Eigen::MatrixXd::Zero( Xu.rows(), Xu.cols() );

  //
  //
  int count = 0;
  while( delta_2 - delta_1 > 1.e-06 /*&& ++count < Niter*/ )
    {
      for ( int i = 0 ; i < soft_u.rows() ; i++ )
	soft_u(i,0) = soft_threshold( Xu(i,0), (delta_1 + delta_2) / 2. );
      //
      soft_u /= soft_u.lpNorm< 2 >();
      if ( soft_u.lpNorm< 1 >() < L1norm_u )
	delta_2 = (delta_1 + delta_2) / 2.;
      else
	delta_1 = (delta_1 + delta_2) / 2.;
    }

  //
  //
  //std::cout << "delta delta " << delta_2 - delta_1 << std::endl;
  return (delta_1 + delta_2) / 2.;
}
//
// Normalize, column-wise, the correlation matrix

Eigen::MatrixXd normalize( const Eigen::MatrixXd& X, Normalization N )
{
  //
  // Test
  // X.rows() > 1
  
  //
  //
  Eigen::MatrixXd X_normalized;
  //
  switch( N )
    {
    case STANDARDIZE:
      {
	Eigen::MatrixXd demeaned = normalize(X,DEMEAN);
	Eigen::VectorXd std = (demeaned.colwise().norm() / sqrt( static_cast< double >(X.rows() - 1)));
	X_normalized = demeaned.array().rowwise() / std.transpose().array();
	break;
      }
    case DEMEAN:
      {
	Eigen::MatrixXd Ones  = Eigen::MatrixXd::Ones(X.rows(),1);
	Eigen::MatrixXd Means = X.colwise().sum() / static_cast< double >(X.rows());
	X_normalized = X - Ones*Means;
	break;
      }
    case NORMALIZE:
      {
	Eigen::VectorXd col_max = X.colwise().maxCoeff();
	Eigen::VectorXd col_min = X.colwise().minCoeff();
	//
	Eigen::VectorXd max_min = col_max - col_min;
	Eigen::MatrixXd Ones    = Eigen::MatrixXd::Ones(X.cols(),1);
	//
	X_normalized = X - Ones*col_min.transpose();
	X_normalized = X_normalized.array().rowwise() / max_min.transpose().array();
	break;
      }
    default:
      {
	std::cout << "Raise exception" << std::endl;
	break;
      }
    }
  
  //
  //
  return X_normalized;
}

//
// Algorithm 1: computation of a single factor PMD model
// If X = Y^T Z, Y and Z are standardized matrices (column-wise)
// It is suggested to start the algorithm with V as the first SVD vector.
//
// Algo1 include algorithm 3 and 4
//
// Pu and Pv are the penalities associated, repectively, to U and V.
// 
// The algorithm output the the rank coefficient d_{k}
double PMD_single_factor( const Eigen::MatrixXd& X,
			  Eigen::MatrixXd& U, const Penality Pu, 
			  Eigen::MatrixXd& V, const Penality Pv,
			  const int Niter = 1000 )
{
  //
  // Tests
  if ( U.rows() != X.rows() || V.rows() != X.cols() )
    std::cout << "Raise exception" << std::endl;

  //
  // initialization
  int
    xl = X.rows(), xc = X.cols();
  double
    c1 = 1., //sqrt( static_cast< double >(xl) ),
    c2 = 1., //sqrt( static_cast< double >(xc) ),
    Ul1Norm = U.lpNorm< 1 >(),
    Vl1Norm = V.lpNorm< 1 >(),
    delta_1 = 0., delta_2 = 0.;
  Eigen::MatrixXd
    v_old = Eigen::MatrixXd::Random( xc, 1 ),
    Xv, XTu;
  //std::cout << "v_old \n" << v_old << std::endl;
  
  //
  switch ( Pv )
    {
    case L1:
      {
	// Algorithm 3: computtion of a single factor PMD(L1,L1) model
	// ToDo: make sure v is L2-norm 1.
	int count = 0;
	while ( ( V - v_old ).lpNorm< 1 >() > 1.e-06 && ++count < Niter )
	  {
	    //
	    v_old = V;
	    //
	    // Update v
	    Xv      = X * V;
	    Ul1Norm = U.lpNorm< 1 >();
	    //
	    if ( Ul1Norm > c1 )
	      delta_1 = dichotomy_search( Xv, Ul1Norm, c1 );
	    else
	      delta_1 = 0.;
	    std::cout << "delta_1 " << delta_1 << std::endl;
	    //
	    for ( int l = 0 ; l < xl ; l++ )
	      U(l,0) = soft_threshold( Xv(l,0), delta_1 );
	    U /= U.lpNorm< 2 >();
	    //
	    // Update u
	    XTu     = X.transpose() * U;
	    Vl1Norm = V.lpNorm< 1 >();
	    //
	    if ( Vl1Norm > c2 )
	      delta_2 = dichotomy_search( XTu, Vl1Norm, c2 );
	    else
	      delta_2 = 0.;
	    std::cout << "delta_2 " << delta_2 << std::endl;
	    //
	    for ( int c = 0 ; c < xc ; c++ )
	      V(c,0) = soft_threshold( XTu(c,0), delta_2 );
	    V /= V.lpNorm< 2 >();
	    //  std::cout << "v_old \n" << v_old << std::endl;
	    //  std::cout << "V \n" << V << std::endl;
	    //  std::cout << "V-vold " << ( V - v_old ).lpNorm< 1 >()  << std::endl;
	  }
	//
	break;
      }
    case FUSION:
      // Algorithm 4: computtion of a single factor PMD(L1,L1) model
      // ToDo: make sure v is L2-norm 1.
    default:
      {
	std::cout << "Raise exception" << std::endl;
      }
    }

  //
  // output
  // coefficient associated with the rank decomposition level of X (input)
  // compute the rank coefficient
  return (U.transpose() * X * V)(0,0);
}

//
// Algorithm 1bis: computation of a single factor PMD model adapted to PCA
// If X = Y^T Z, Y and Z are standardized matrices (column-wise)
// It is suggested to start the algorithm with V as the first SVD vector.
//
// Algo1 include algorithm 3 and 4
//
// Uprev is the set of orthogonal U_{k-1}
// Pu and Pv are the penalities associated, repectively, to U and V.
// 
// The algorithm output the the rank coefficient d_{k}
double PCA_single_factor( const Eigen::MatrixXd& X, const Eigen::MatrixXd& Uprev,
			  Eigen::MatrixXd& U, const Penality Pu, 
			  Eigen::MatrixXd& V, const Penality Pv,
			  const int Niter = 1000 )
{
  //
  // Tests
  if ( U.rows() != X.rows() || V.rows() != X.cols() )
    std::cout << "Raise exception" << std::endl;

  std::cout << "Size Uprev: " << Uprev.cols()  << std::endl;
  
  //
  // initialization
  int
    xl = X.rows(), xc = X.cols();
  double
    c1 = sqrt( static_cast< double >(xl) ),
    c2 = sqrt( static_cast< double >(xc) ),
    Vl1Norm = V.lpNorm< 1 >(),
    delta_1 = 0., delta_2 = 0.;
  Eigen::MatrixXd
    v_old = Eigen::MatrixXd::Random( xc, 1 ),
    Xv    = X * V, XTu = X.transpose() * U;
  //std::cout << "v_old \n" << v_old << std::endl;
  
  //
  switch ( Pv )
    {
    case L1:
      {
	// Algorithm 3: computtion of a single factor PMD(L1,L1) model
	// ToDo: make sure v is L2-norm 1.
	int count = 0;
	while ( ( V - v_old ).lpNorm< 1 >() > 1.e-06 && ++count < Niter )
	  {
	    //
	    v_old = V;
	    //
	    // Update v
	    Xv = X * V;
	    if ( Uprev.cols() > 0 )
	      {
//		U  = Xv;
//		U /= X.lpNorm< 2 >();
		Eigen::MatrixXd I = Eigen::MatrixXd::Identity(Uprev.rows(), Uprev.rows());
		U  = ( I - Uprev * Uprev.transpose() ) * Xv;
		U /= U.lpNorm< 2 >();
	      }
	    else
	      {
		U  = Xv;
		U /= Xv.lpNorm< 2 >();
	      }
	    //
	    // Update u
	    XTu     = X.transpose() * U;
	    Vl1Norm = V.lpNorm< 1 >();
	    if ( Vl1Norm > c2 )
	      delta_2 = dichotomy_search( XTu, Vl1Norm, c2 );
	    else
	      delta_2 = 0.;
	    std::cout << "delta_2 " << delta_2 << std::endl;
	    //
	    for ( int c = 0 ; c < xc ; c++ )
	      V(c,0) = soft_threshold( XTu(c,0), delta_2 );
	    V /= V.lpNorm< 2 >();
	  }
	//
	break;
      }
    case FUSION:
      // Algorithm 4: computtion of a single factor PMD(L1,L1) model
      // ToDo: make sure v is L2-norm 1.
    default:
      {
	std::cout << "Raise exception" << std::endl;
	break;
      }
    }

  //
  // compute the rank coefficient
  // coefficient associated with the rank decomposition level of X (input)
  return (U.transpose() * X * V)(0,0);
}

//
// Algorithm 2: computation of K factor of PMD
// the input matrix X is at max rank K.
void PMD_K_factors( const Eigen::MatrixXd& X,
		    Spectra& Matrix_spetrum,
		    Penality Pu, Penality Pv )
{
  //
  //
  std::size_t K = Matrix_spetrum.size();
 
  //
  // algorithm
  Eigen::MatrixXd XX = X;
  for ( int k = 0 ; k < K ; k++ )
    {
      std::get< coeff_k >( Matrix_spetrum[k] ) =
	PMD_single_factor( XX,
			   std::get< Uk >(Matrix_spetrum[k]), Pu,
			   std::get< Vk >(Matrix_spetrum[k]), Pv );

      //
      // Update the rest of the matrix
      XX -= std::get< coeff_k >( Matrix_spetrum[k] )
      	* std::get< Uk >(Matrix_spetrum[k])
      	* std::get< Vk >(Matrix_spetrum[k]).transpose();
      //std:: cout << XX << std::endl;
    }
  //
  Eigen::MatrixXd Sol = Eigen::MatrixXd::Zero(X.rows(),X.cols());
  for ( int k = 0 ; k < K ; k++ )
    {
      std::cout << "dk[" << k << "] = " << std::get< coeff_k >( Matrix_spetrum[k] ) << std::endl;
      std::cout << "uk[" << k << "] = \n" << std::get< Uk >(Matrix_spetrum[k]) << std::endl;
      std::cout << "vk[" << k << "] = \n" << std::get< Vk >(Matrix_spetrum[k]) << std::endl;
      Sol += std::get< coeff_k >( Matrix_spetrum[k] )
	* std::get< Uk >(Matrix_spetrum[k])
	* std::get< Vk >(Matrix_spetrum[k]).transpose();
      std::cout << "Sol: \n" << Sol << std::endl;
    }
}

//
// Algorithm 2: computation of K factor of PMD
// the input matrix X is at max rank K.
void PCA_K_factors( const Eigen::MatrixXd& X,
		    Spectra& Matrix_spetrum,
		    Penality Pu, Penality Pv)
{
  //
  //
  std::size_t K = Matrix_spetrum.size();

  //
  // algorithm
  Eigen::MatrixXd XX = X;
  for ( int k = 0 ; k < K ; k++ )
    {
      //
      // Build the orthogonal spectrum Uprev
      Eigen::MatrixXd Uprev;
      if ( k > 0 )
	{
	  Uprev = Eigen::MatrixXd::Zero( std::get< Uk >( Matrix_spetrum[0] ).rows(), k );
	  for ( int i = 0 ; i < k ; i++ )
	    Uprev.col(i) = std::get< Uk >( Matrix_spetrum[i] );
	  std::cout << "Uprev\n" << Uprev << std::endl;
	}

      //
      // Sparse Principal Componant
      std::get< coeff_k >( Matrix_spetrum[k] ) =
	PCA_single_factor( XX, Uprev /*replace with Uprev*/,
			   std::get< Uk >(Matrix_spetrum[k]), Pu,
			   std::get< Vk >(Matrix_spetrum[k]), Pv );      //

      //
      // Update the rest of the matrix
      XX -= std::get< coeff_k >( Matrix_spetrum[k] )
      	* std::get< Uk >(Matrix_spetrum[k])
      	* std::get< Vk >(Matrix_spetrum[k]).transpose();
      //std:: cout << XX << std::endl;
    }
  //
  Eigen::MatrixXd Sol = Eigen::MatrixXd::Zero(X.rows(),X.cols());
  Eigen::MatrixXd D   = Eigen::MatrixXd::Zero(X.rows(),X.cols());
  Eigen::MatrixXd I   = Eigen::MatrixXd::Identity(X.rows(),X.cols());
  Eigen::MatrixXd P   = Eigen::MatrixXd::Zero(std::get< Uk >( Matrix_spetrum[0] ).rows(), K );
  Eigen::VectorXd diagonal = Eigen::VectorXd::Zero(X.rows());
  for ( int k = 0 ; k < K ; k++ )
    {
      std::cout << "dk[" << k << "] = " << std::get< coeff_k >( Matrix_spetrum[k] ) << std::endl;
      std::cout << "uk[" << k << "] = \n" << std::get< Uk >(Matrix_spetrum[k]) << std::endl;
      std::cout << "vk[" << k << "] = \n" << std::get< Vk >(Matrix_spetrum[k]) << std::endl;
      //
      P.col(k)    = std::get< Uk >(Matrix_spetrum[k]);
      diagonal(k) = std::get< coeff_k >( Matrix_spetrum[k] );
      if ( k > 0 )
	std::cout << "uk[" << 0 << "] . uk[" << k << "] = "
		  << std::get< Uk >(Matrix_spetrum[0]).transpose() * std::get< Uk >(Matrix_spetrum[k])
		  << std::endl;
    }
  D = diagonal.asDiagonal();
  std::cout << "D:\n" << D << std::endl;
  std::cout << "P:\n" << P << std::endl;
  std::cout << "Sol:\n: " << P * D * P.inverse() << std::endl;
  //
  for ( int k = 0 ; k < K ; k++ )
    {
      std::cout << "dk[" << k << "] = " << std::get< coeff_k >( Matrix_spetrum[k] ) << std::endl;
      std::cout << "uk[" << k << "] = \n" << std::get< Uk >(Matrix_spetrum[k]) << std::endl;
      std::cout << "vk[" << k << "] = \n" << std::get< Vk >(Matrix_spetrum[k]) << std::endl;
      Sol += std::get< coeff_k >( Matrix_spetrum[k] )
	* std::get< Uk >(Matrix_spetrum[k])
	* std::get< Vk >(Matrix_spetrum[k]).transpose();
      std::cout << "Sol: \n" << Sol << std::endl;
    }
}

//
//
std::vector< std::string > split( const std::string& Input, char Delim ) 
{
  std::stringstream ss(Input);
  std::string item;
  std::vector<std::string> elems;
  while ( std::getline(ss, item, Delim) ) 
    {
      //elems.push_back(item);
      elems.push_back( std::move(item) ); // if C++11 (based on comment from @mchiasson)
    }
  //
  return elems;
}

itk::ImageIOBase::Pointer getImageIO(std::string input){
  itk::ImageIOBase::Pointer imageIO = itk::ImageIOFactory::CreateImageIO(input.c_str(), itk::ImageIOFactory::ReadMode);

  imageIO->SetFileName(input);
  imageIO->ReadImageInformation();

  return imageIO;
}

itk::ImageIOBase::IOComponentType component_type(itk::ImageIOBase::Pointer imageIO){
  return imageIO->GetComponentType();
}

itk::ImageIOBase::IOPixelType pixel_type(itk::ImageIOBase::Pointer imageIO){
  return imageIO->GetPixelType();
}

size_t num_dimensions(itk::ImageIOBase::Pointer imageIO){
  return imageIO->GetNumberOfDimensions();
}

int main(int argc, char const *argv[]){
  //
  // 
  itk::ImageIOBase::Pointer mask  = NULL;
  itk::ImageIOBase::Pointer atlas = NULL;
  std::string output_name         = "";

  //
  //
  if( argc == 4 )
    {
      output_name = argv[1];
      atlas = getImageIO( argv[2] );
      mask  = getImageIO( argv[3] );
    }
  else
    {
      std::cerr << "mk_test requires: \"output_suffix\" \"Atlas.nii.gz\" \"file_model.nii.gz\" " << std::endl;
      return EXIT_FAILURE;
    }
    

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
  std::cout << A << std::endl;
  std::cout << normalize(A,STANDARDIZE) << std::endl;
  std::cout << A.transpose() * A  << std::endl;
  std::cout << normalize(A.transpose() * A,STANDARDIZE) << std::endl;
  std::cout << UU << std::endl;
  std::cout << VV << std::endl;
  //
  //  double d = PMD_single_factor(A, UU, L1, VV, L1 );
  //  std::cout << d * UU * VV.transpose() << std::endl;
  //

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
  PMD_K_factors( AAA, matrix_spetrum, L1, L1 );
  //PCA_K_factors( normalize(AAA,STANDARDIZE), matrix_spetrum, L1, L1 );


  
  //
  // Random information
  std::default_random_engine generator;
  std::normal_distribution< double >
    rd_element(0.0,1.0),
    rd_noise(0.0,0.05);

  
  //
  // reader
  typedef itk::Image< float, 3 > Image;
  typedef itk::ImageFileReader< Image >  Image_reader;
  typedef itk::ImageDuplicator< Image > DuplicatorType;
  typedef itk::Image< char, 3 > Atlas;
  typedef itk::ImageFileReader< Atlas >  Atlas_reader;

  //
  // Image
  Image_reader::Pointer reader_mask = Image_reader::New();
  reader_mask->SetFileName( mask->GetFileName() );
  reader_mask->Update();
  Atlas_reader::Pointer reader_atlas = Atlas_reader::New();
  reader_atlas->SetFileName( atlas->GetFileName() );
  reader_atlas->Update();

  //
  // Region to explore
  Image::RegionType region;
  //
  Image::Pointer   image_in = reader_mask->GetOutput();
  Image::SizeType  img_size = image_in->GetLargestPossibleRegion().GetSize();
  Image::IndexType start    = {0, 0, 0};
  std::cout << img_size << std::endl;
  //
  region.SetSize( img_size );
  region.SetIndex( start );

  //
  // Explanatory variables
  int
    KK = 1,  // number of factor
    n = 10, // number of subjects
    p = 4,  // neuropsy metrics
    q = img_size[0] * img_size[1] * img_size[2];
  //
  Eigen::MatrixXd Z = Eigen::MatrixXd::Zero( n, p );
  Eigen::MatrixXd X = Eigen::MatrixXd::Zero( n, q );
  //
  Eigen::MatrixXd u = Eigen::MatrixXd::Ones( p, KK );
  Eigen::MatrixXd v = Eigen::MatrixXd::Ones( q, KK );
  //
  u /= static_cast< double >(p);
  v /= static_cast< double >(q);

  //
  // l1 l2 norms
  Eigen::MatrixXd AA = Eigen::MatrixXd::Ones( 3, 1 );
  AA *= 2.;
  std::cout << AA << std::endl;
  std::cout << AA.lpNorm<1>() << std::endl;
  std::cout << AA.lpNorm<2>() << std::endl;

  //
  // CSV output
  std::ofstream
    X_out("X.csv"), Z_out("Z.csv");

  std::string X_output, Z_output;

  //
  // loop over subjects
  for ( int i = 0 ; i < n ; i++ )
    {
      // 
      // Output
      X_output += std::to_string(i) + ",";
      Z_output += std::to_string(i) + ",";
      //
      DuplicatorType::Pointer duplicator = DuplicatorType::New();
      duplicator->SetInputImage( reader_mask->GetOutput() );
      duplicator->Update();
      //
      Image::Pointer image_out = duplicator->GetOutput();

      //
      itk::ImageRegionIterator<Image>  imageIterator_mask( reader_mask->GetOutput(), region );
      //
      while( !imageIterator_mask.IsAtEnd() )
	{
	  int atlas_val = static_cast< int >( reader_atlas->GetOutput()->GetPixel( imageIterator_mask.GetIndex() ) );
	  if ( atlas_val > 0 )
	    {
	      //
	      // Voxel index
	      auto vox_idx = imageIterator_mask.GetIndex();
	      int idx = vox_idx[0] + img_size[0] * vox_idx[1] + img_size[0] * img_size[1] * vox_idx[2];
	      //
	      // Value of the voxel
	      double element = rd_element(generator);
	      //
	      if ( atlas_val == 31 )
		{
		  Z( i, 0 ) += element + rd_noise(generator);
		  Z( i, 1 ) += element + rd_noise(generator);
		  Z( i, 2 ) += element + rd_noise(generator);
		}
	      else if ( atlas_val == 19 )
		{
		  Z( i, 1 ) += element + rd_noise(generator);
		  Z( i, 2 ) += -1. * element + rd_noise(generator);
		}
	      else if ( atlas_val == 1 )
		{
		  Z( 0, 2 ) += 2. * element + rd_noise(generator);
		}
	      else
		{
		  Z( i, 0 ) += rd_noise(generator);
		  Z( i, 1 ) += rd_noise(generator);
		  Z( i, 2 ) += rd_noise(generator);
		  Z( i, 3 ) += rd_noise(generator);
		}
	      //
	      double vox_val  = element + rd_noise(generator);
	      X(i,idx) = vox_val;
	      image_out->SetPixel( imageIterator_mask.GetIndex(), vox_val );
	      //
	      X_output += std::to_string( vox_val ) + ",";
	    }
	  else
	    image_out->SetPixel( imageIterator_mask.GetIndex(), 0.0 );
	  //
	  ++imageIterator_mask;
	}

      for ( int pp = 0 ; pp < p ; pp++ )
	Z_output += std::to_string( Z(i,pp) ) + ",";
  
//      //
//      // Writer
//      itk::NiftiImageIO::Pointer nifti_io = itk::NiftiImageIO::New();
//      nifti_io->SetPixelType( pixel_type(mask) );
//      //
//      itk::ImageFileWriter< Image >::Pointer writer = itk::ImageFileWriter< Image >::New();
//      writer->SetFileName( output_name + "_" + std::to_string(i) + ".nii.gz" );
//      writer->SetInput( image_out );
//      writer->SetImageIO( nifti_io );
//      writer->Update();

      //
      // next subject
      X_output += "\n";
      Z_output += "\n";
    }

  //
  //
  std::cout << "X: [" << X.rows() << "," << X.cols() << "]" << std::endl;

//  //
//  // Write the output
//  X_out << X_output;
//  Z_out << Z_output;
//  X_out.close();
//  Z_out.close();

  
  //
  //
  return EXIT_SUCCESS;
}
