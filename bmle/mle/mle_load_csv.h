#ifndef MLELOADCSV_H
#define MLELOADCSV_H
//
//
//
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <algorithm>
#include <vector>
#include <list>
#include <map>
#include <set>
#include <math.h>
//#include <cmath.h>
// Eigen
#include <Eigen/Core>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/KroneckerProduct>
//
// ITK
//
#include <itkImageFileReader.h>
#include <itkSpatialOrientationAdapter.h>
#include "itkChangeInformationImageFilter.h"
using MaskType   = itk::Image< unsigned char, 3 >;
using MaskType4D = itk::Image< unsigned char, 4 >;
//
//
//
#include "Exception.h"
#include "MakeITKImage.h"
#include "mle_subject.h"
// Tools
#include "Tools.h"
//
//
//
namespace NeuroBayes
{
  /** \class MleLoadCSV
   *
   * \brief 
   * 
   */
  template< class Optimizer, int DimY, int D_f >
    class MleLoadCSV
  {
  public:
    /** Constructor. */
    explicit MleLoadCSV( // dataset 
			const std::string&, 
			// input and output dir
			const std::string&, const std::string&, 
			 // Age Dns: demean, normalize, standardize
			 const NeuroStat::TimeTransformation,
			 // Prediction
			 const std::string& );
    
    /** Destructor */
    virtual ~MleLoadCSV(){};

    Optimizer optim_;

    //
    // This function will load all the patients images into a 4D image.
    void build_groups_design_matrices();
    // Expectation maximization algorithm
    void Expectation_Maximization( MaskType::IndexType );
    // Write the output
    void write_subjects_solutions( );
    // multi-threading
    void operator ()( const MaskType::IndexType idx )
    {
      if ( prediction_ )
	{
	  std::cout << "Prediction treatment for parameters: " 
		    << idx;
	}
      else
	{
	  std::cout << "treatment for parameters: " 
		    << idx;
	  Expectation_Maximization( idx );
	}
    };


  private:
    //
    // Functions
    //

    // Cumulative centered normal cumulative distribution function
    // https://en.wikipedia.org/wiki/Error_function
    double Normal_CFD_( const double value ) const
    { return 0.5 * erfc( - value * M_SQRT1_2 ); };

    //
    // Members
    //
    
    //
    // Prediction calculation
    bool prediction_{false};

    //
    // CSV file
    std::ifstream csv_file_;
    // input directory
    std::string   input_dir_;
    // output directory
    std::string   output_dir_;
    // Statistic transformation of ages
    std::string   age_statistics_tranformation_;
    //
    // Arrange pidns into groups
    std::set< int > groups_;
    std::vector< std::map< std::string /*pidn*/, MleSubject< DimY, D_f > > > group_pind_{10};
    // Number of subjects per group
    std::vector< int > group_num_subjects_{0,0,0,0,0,0,0,0,0,0};
    //
    // Measures grouped in vector of 3D image
    using Image4DType  = itk::Image< double, 4 >;
    using Reader4DType = itk::ImageFileReader< Image4DType >;
    using Image3DType  = itk::Image< double, 3 >;
    using Reader3DType = itk::ImageFileReader< Image3DType >;
    std::vector< Reader3DType::Pointer > Y_;

    // number of PIDN
    long unsigned int num_subjects_{0};
    // number of 3D images = number of time points (TP)
    long unsigned int num_3D_images_{0};
    // number of covariates
    int num_covariates_{0};

    //
    // Design matrices
    //

    // X
    Eigen::MatrixXd X_;
    // Z
    Eigen::MatrixXd Z_;

    //
    // Constrast matrix
    //

    // Contrast groupe for level one and two
    Eigen::MatrixXd constrast_;
    Eigen::MatrixXd constrast_l2_;

    
    //
    // Records
    //

    //
    // Contrast vectors
    
    //
    // Level 1
    //
    // Posterior Probability Maps
    NeuroBayes::NeuroBayesMakeITKImage PPM_;
    // Posterior t-maps
    NeuroBayes::NeuroBayesMakeITKImage post_T_maps_;
    // Posterior groups parameters
    NeuroBayes::NeuroBayesMakeITKImage post_groups_param_;
    //
    // level 2
    // Posterior Probability Maps
    NeuroBayes::NeuroBayesMakeITKImage PPM_l2_;
    // Posterior t-maps
    NeuroBayes::NeuroBayesMakeITKImage post_T_maps_l2_;
    // Posterior groups parameters
    NeuroBayes::NeuroBayesMakeITKImage post_groups_param_l2_;
    // Posterior groups variance
    NeuroBayes::NeuroBayesMakeITKImage post_groups_cov_l2_;
    // 
    // R-square
    NeuroBayes::NeuroBayesMakeITKImage R_sqr_l2_;
  };
  //
  //
  //
  template< class Optimizer, int DimY, int D_f >
    MleLoadCSV< Optimizer, DimY, D_f >::MleLoadCSV( const std::string& CSV_file,
						    const std::string& Input_dir,
						    const std::string& Output_dir,
						    // Age Dns: demean, normalize, standardize
						    const NeuroStat::TimeTransformation Dns,
						    const std::string& Inv_cov_error):
    csv_file_{ CSV_file.c_str() }, input_dir_{ Input_dir }, output_dir_{ Output_dir }
  {
    try
      {
	//
	//
	double 
	  mean_age = 0.;
	int
	  max_age  = 0,
	  min_age  = 70000;
	std::list< int > age_stat;
	std::string line;
	//skip the first line
	std::getline(csv_file_, line);
	//
	// then loop
	while( std::getline(csv_file_, line) )
	  {
	    std::stringstream  lineStream( line );
	    std::string        cell;
	    std::cout << "ligne: " << line << std::endl;

	    //
	    // Get the PIDN
	    std::getline(lineStream, cell, ',');
	    const std::string PIDN = cell;
	    // Get the group
	    std::getline(lineStream, cell, ',');
	    const int group = std::stoi( cell );
	    if ( group == 0 )
	      //
	      // The groups must be labeled 1, 2, 3, ...
	      throw NeuroBayes::NeuroBayesException( __FILE__, __LINE__,
				   "Select another group name than 0 (e.g. 1, 2, ...). 0 is reserved.",
				   ITK_LOCATION );
	    // Get the age
	    std::getline(lineStream, cell, ',');
	    int age = std::stoi( cell );
	    age_stat.push_back(age);
	    mean_age += static_cast< double >( age );
	    if ( age < min_age )
	      min_age = age;
	    if ( age > max_age )
	      max_age = age;
	    // Get the image
	    std::string image;
	    std::getline(lineStream, image, ',');
	    // Covariates
	    std::list< double > covariates;
	    while( std::getline(lineStream, cell, ',') )
	      covariates.push_back( std::stof(cell) );
	    num_covariates_ = covariates.size();

	    //
	    // check we have less than 10 groups
	    if( group > 10 )
	      throw NeuroBayes::NeuroBayesException( __FILE__, __LINE__,
				   "The CSV file should have less than 10 gourps.",
				   ITK_LOCATION );
	    // If the PIDN does not yet exist
	    if ( group_pind_[ group ].find( PIDN ) == group_pind_[ group ].end() )
	      {
		std::cout << PIDN << " " << group << std::endl;
		groups_.insert( group );
		group_pind_[ group ][PIDN] = MleSubject< DimY, D_f >( PIDN, group, Output_dir );
		group_num_subjects_[ group ]++;
		num_subjects_++;
	      }
	    //
	    group_pind_[ group ][ PIDN ].add_tp( age, covariates, image );
	    num_3D_images_++;
	  }
	//

	// 
	// Design Matrix for every subject
	//

	//
	// mean age
	mean_age /= static_cast< double >( num_3D_images_ );
	//
	Y_.resize( num_3D_images_ );
	int sub_image{0};
	age_statistics_tranformation_ = Output_dir + "/age_transformation.txt";
	//
	switch ( Dns )
	  {
	  case NeuroStat::TimeTransformation::NONE:
	  case NeuroStat::TimeTransformation::DEMEAN:
	    {
	      std::cout 
		<< "Age will be demeaned with mean age: " 
		<< mean_age << " or 0 infunction of the option you have chosen.\n"
		<< std::endl;
	      // record the transformation
	      std::ofstream fout( age_statistics_tranformation_ );
	      fout << Dns << " " 
		   << ( Dns == NeuroStat::TimeTransformation::NONE ? 0 : mean_age )
		   << std::endl;
	      fout.close();
	      //
	      for ( auto g : groups_ )
		for ( auto& s : group_pind_[g] )
		  {
		    //
		    s.second.build_design_matrices( (Dns == NeuroStat::TimeTransformation::NONE ? 
						     0 : mean_age) );
		    // Create the vector of 3D measurements image
		    for ( auto image : s.second.get_age_images() )
		      Y_[ sub_image++ ] = image.second;
		  }
	      //
	      break;
	    }
	  case NeuroStat::TimeTransformation::NORMALIZE:
	    {
	      std::cout 
		<< "Age will be normalized between: (" 
		<< min_age << "," 
		<< max_age << ").\n" 
		<< std::endl;
	      //
	      double 
		C1 = static_cast< double >(min_age),
		C2 = static_cast< double >(max_age - min_age);
	      //
	      switch ( D_f )
		{
		case 2:
		  {
		    Eigen::Matrix< double, 2, 2 > M;
		    M << 1., -C1/C2, 0., 1./C2;
		    std::cout 
		      <<  "C1 = min_age = " << min_age
		      <<  ", C2 = (max_age - min_age) = " << max_age - min_age
		      <<  ". change of variable is u = (t-C1)/C2. \n"
		      <<  "y = a0 + a1xt + epsilon\n"
		      <<  "  = b0 + b2xu + epsilon\n"
		      <<  "a = M x b. Where M = \n"
		      <<  M
		      << std::endl;
		    //
		    break;
		  }
		case 3:
		  {
		    Eigen::Matrix< double, 3, 3 > M;
		    M << 1., -C1/C2, C1*C1/C2/C2, 0., 1./C2, -2*C1/C2/C2, 0., 0., 1./C2/C2;
		    std::cout 
		      <<  "C1 = min_age = " << min_age
		      <<  ", C2 = (max_age - min_age) = " << max_age - min_age
		      <<  ". change of variable is u = (t-C1)/C2. \n"
		      <<  "y = a0 + a1xt + a1xt^2 + epsilon\n"
		      <<  "  = b0 + b2xu + b2xu^2 + epsilon\n"
		      <<  "a = M x b. Where M = \n"
		      <<  M
		      << std::endl;
		    //
		    break;
		  }
		default:
		  {
		    std::string mess = "Case D_f is 4 or more, has not yet been developped.";
		    throw NeuroBayes::NeuroBayesException( __FILE__, __LINE__,
							   mess.c_str(),
							   ITK_LOCATION );
		  }
		}
	      // record the transformation
	      std::ofstream fout( age_statistics_tranformation_ );
	      fout << Dns << " " 
		   << C1 << " " << C2
		   << std::endl;
	      fout.close();
	      //
	      for ( auto g : groups_ )
		for ( auto& s : group_pind_[g] )
		  {
		    s.second.build_design_matrices( C1, C2 );
		    // Create the vector of 3D measurements image
		    for ( auto image : s.second.get_age_images() )
		      Y_[ sub_image++ ] = image.second;
		  }
	      //
	      break;
	    }
	  case NeuroStat::TimeTransformation::STANDARDIZE:
	    {
	      //
	      // Standard deviation
	      double accum = 0.0;
	      std::for_each ( std::begin( age_stat ), std::end( age_stat ), 
			      [&](const double d) {accum += (d - mean_age) * (d - mean_age);} );
	      //
	      double stdev = sqrt( accum / static_cast< double >(age_stat.size()-1) );
	      //
	      //
	      std::cout 
		<< "Age will be standardized with: (mu = " 
		<< mean_age << ", std = " 
		<< stdev << ").\n" 
		<< std::endl;
	      //
	      double 
		C1 = mean_age,
		C2 = stdev;
	      //
	      switch ( DimY )
		{
		case 2:
		  {
		    Eigen::Matrix< double, 2, 2 > M;
		    M << 1., -C1/C2, 0., 1./C2;
		    std::cout 
		      <<  "C1 = mu = " << C1
		      <<  ", C2 = stdev = " << stdev
		      <<  ". change of variable is u = (t-C1)/C2. \n"
		      <<  "y = a0 + a1xt + epsilon\n"
		      <<  "  = b0 + b2xu + epsilon\n"
		      <<  "a = M x b. Where M = \n"
		      <<  M
		      << std::endl;
		    //
		    break;
		  }
		case 3:
		  {
		    Eigen::Matrix< double, 3, 3 > M;
		    M << 1., -C1/C2, C1*C1/C2/C2, 0., 1./C2, -2*C1/C2/C2, 0., 0., 1./C2/C2;
		    std::cout 
		      <<  "C1 = mu = " << C1
		      <<  ", C2 = stdev = " << stdev
		      <<  ". change of variable is u = (t-C1)/C2. \n"
		      <<  "y = a0 + a1xt + a1xt^2 + epsilon\n"
		      <<  "  = b0 + b2xu + b2xu^2 + epsilon\n"
		      <<  "a = M x b. Where M = \n"
		      <<  M
		      << std::endl;
		    //
		    break;
		  }
		default:
		  {
		    std::string mess = "Case DimY is 4 or more, has not yet been developped.";
		    throw NeuroBayes::NeuroBayesException( __FILE__, __LINE__,
							   mess.c_str(),
							   ITK_LOCATION );
		  }
		}
	      // record the transformation
	      std::ofstream fout( age_statistics_tranformation_ );
	      fout << Dns << " " 
		   << C1 << " " << C2
		   << std::endl;
	      fout.close();
	      //
	      for ( auto g : groups_ )
		for ( auto& s : group_pind_[g] )
		  {
		    s.second.build_design_matrices( C1, C2 );
		    // Create the vector of 3D measurements image
		    for ( auto image : s.second.get_age_images() )
		      Y_[ sub_image++ ] = image.second;
		  }
	      //
	      break;
	    }
	  case NeuroStat::TimeTransformation::LOAD:
	    {
	      // record the transformation
	      std::ifstream fin( age_statistics_tranformation_ );
	      std::stringstream stat_string;
	      // Check the file exist
	      if ( fin )
		stat_string << fin.rdbuf();
	      else
		{
		  std::string mess = "The statistic transformation file does not exist.\n";
		  mess += "Please, check on: " + age_statistics_tranformation_;
		  throw NeuroBayes::NeuroBayesException( __FILE__, __LINE__,
							 mess.c_str(),
							 ITK_LOCATION );
		}
	      //
	      std::string str_time_transfo;
	      int time_transfo = 0;
	      //
	      stat_string >> str_time_transfo;
	      time_transfo = std::stoi( str_time_transfo );
	      std::cout << "Transforamtion: " << time_transfo << std::endl;
	      //
	      //
	      switch ( time_transfo )
		{
		case NeuroStat::TimeTransformation::NONE:
		case NeuroStat::TimeTransformation::DEMEAN:
		  {
		    //
		    std::string    str_C1;
		    stat_string >> str_C1;
		    //
		    double             C1 = std::stod( str_C1 );
		    std::cout << "C1 " << C1 << std::endl;
		    //
		    for ( auto g : groups_ )
		      for ( auto& s : group_pind_[g] )
			{
			  s.second.build_design_matrices( C1 );
			  // Create the vector of 3D measurements image
			  for ( auto image : s.second.get_age_images() )
			    Y_[ sub_image++ ] = image.second;
			}
		    //
		    break;
		  }
		case NeuroStat::TimeTransformation::NORMALIZE:
		case NeuroStat::TimeTransformation::STANDARDIZE:
		  {
		    std::string 
		      str_C1,
		      str_C2;
		    stat_string >> str_C1 >> str_C2;
		    //
		    double          
		      C1 = std::stod( str_C1 ),
		      C2 = std::stod( str_C2 );
		    std::cout << "C1 " << C1 << std::endl;
		    std::cout << "C2 " << C2 << std::endl;
		    //
		    for ( auto g : groups_ )
		      for ( auto& s : group_pind_[g] )
			{
			  s.second.build_design_matrices( C1, C2 );
			  // Create the vector of 3D measurements image
			  for ( auto image : s.second.get_age_images() )
			    Y_[ sub_image++ ] = image.second;
			}
		    //
		    break;
		  }
		default:
		  {
		    std::string mess = "The statistic transformation requiered is unknown.\n";
		    mess += "Please, check on: " + age_statistics_tranformation_;
		    throw NeuroBayes::NeuroBayesException( __FILE__, __LINE__,
							   mess.c_str(),
							   ITK_LOCATION );
		  }
		}
	      
	      //
	      //
	      fin.close();
	      //
	      break;
	    }
	  }
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
	exit( -1 );
      }
  }
  //
  //
  //
  template< class Optimizer, int DimY, int D_f > void
    MleLoadCSV< Optimizer, DimY, D_f >::build_groups_design_matrices()
  {
    try
      {
	//
	// Build X1 and X2 design matrices
	//
	
	//
	//
	// Number of random parameters
	int D_r = D_f + num_covariates_;
	int
	  // X
	  X_lines = num_3D_images_,
	  X_cols  = D_f,
	  // Z
	  Z_lines = num_3D_images_,
	  Z_cols  = group_num_subjects_[1] * D_r;
	//
	X_ = Eigen::MatrixXd::Zero( X_lines, X_cols );
	Z_ = Eigen::MatrixXd::Zero( Z_lines, Z_cols );
	//
	int line_x = 0, col_x = 0;
	int line_z = 0, col_z = 0;
	int current_gr = ( *groups_.begin() ), increme_dist_x1 = 0, increme_dist_x2 = 0;
	for ( auto g : groups_ )
	  for ( auto subject : group_pind_[g] )
	    {
	      //
	      // we change group
	      if ( current_gr != g )
		{
		  //// Add the ID matrix for the fixed parameters
		  ////X2_.block( line_z, DimY * (num_covariates_ + 1) + increme_dist_x2,
		  ////	     D_f, D_f ) = Eigen::MatrixXd::Identity( D_f, D_f );
		  //line_z += D_f;
		  ////
		  //increme_dist_x1 += group_num_subjects_[current_gr] * DimY + D_f;
		  //increme_dist_x2 += DimY * (num_covariates_ + 1) + D_f;
		  //current_gr       = g;
		}
	      //
	      // X and Z designs
	      int
		sub_line_x = subject.second.get_fixed_matrix().rows(),
		sub_col_x  = subject.second.get_fixed_matrix().cols(),
		sub_line_z = subject.second.get_random_matrix().rows(),
		sub_col_z  = subject.second.get_random_matrix().cols();
	      X_.block( line_x, col_x, sub_line_x, sub_col_x ) = subject.second.get_fixed_matrix();
	      Z_.block( line_z, col_z, sub_line_z, sub_col_z ) = subject.second.get_random_matrix();
	      //
	      line_x += sub_line_x;
	      line_z += sub_line_z;
	      col_z  += sub_col_z;
	    }
	//
	//
	if ( true )
	  {
	    std::cout << "Design matrices X and Z" << std::endl;
	    std::cout << X_ << std::endl;
	    std::cout << Z_ << std::endl;
	  }


	//
	// Init the Covariance matrix
	optim_.init_covariance( num_3D_images_, group_pind_[1].size(), D_r, X_, Z_ );
	

//	//
//	// Contrast output
//	std::string
//	  // level 1
//	  sPPM = output_dir_ + "/" + "PPM.nii.gz",
//	  sPtM = output_dir_ + "/" + "Posterior_t_maps.nii.gz",
//	  sPgP = output_dir_ + "/" + "Post_groups_param.nii.gz",
//	  // level 2
//	  sPPMl2 = output_dir_ + "/" + "PPM_l2.nii.gz",
//	  sPtMl2 = output_dir_ + "/" + "Posterior_t_maps_l2.nii.gz",
//	  sPgPl2 = output_dir_ + "/" + "Post_groups_param_l2.nii.gz",
//	  sPgCl2 = output_dir_ + "/" + "Post_groups_cov_l2.nii.gz",
//	  // R^{2} level 2
//	  sR_2   = output_dir_ + "/" + "Post_R_square_l2.nii.gz",
//	  // Output for prediction
//	  piel2  = output_dir_ + "/" + "Prediction_inverse_error_l2.nii.gz";
//	
//	
//	// level 1
//	PPM_               = NeuroBayes::NeuroBayesMakeITKImage( constrast_.cols(), sPPM, Y_[0] );
//	post_T_maps_       = NeuroBayes::NeuroBayesMakeITKImage( constrast_.cols(), sPtM, Y_[0] );
//	post_groups_param_ = NeuroBayes::NeuroBayesMakeITKImage( constrast_.cols(), sPgP, Y_[0] );
//	// level 2
//	PPM_l2_               = NeuroBayes::NeuroBayesMakeITKImage( constrast_l2_.cols(), sPPMl2, Y_[0] );
//	post_T_maps_l2_       = NeuroBayes::NeuroBayesMakeITKImage( constrast_l2_.cols(), sPtMl2, Y_[0] );
//	post_groups_param_l2_ = NeuroBayes::NeuroBayesMakeITKImage( constrast_l2_.cols(), sPgPl2, Y_[0] );
//	// we only save the variance for each parameter of each group
//	post_groups_cov_l2_   = NeuroBayes::NeuroBayesMakeITKImage( constrast_l2_.rows(), sPgCl2, Y_[0] );
//	// r-squared
//	R_sqr_l2_             = NeuroBayes::NeuroBayesMakeITKImage( 2, sR_2, Y_[0] );
//	// output for predictive model
//	Prediction_inverse_error_l2_  = NeuroBayes::NeuroBayesMakeITKImage( C_eps_num_block_diag, piel2, Y_[0] );
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
	exit( -1 );
      }
  }
  //
  //
  //
  template< class Optimizer, int DimY, int D_f > void
    MleLoadCSV< Optimizer, DimY, D_f >::Expectation_Maximization( MaskType::IndexType Idx )
    {
      try
	{
	//
	// Measure
	//
	std::cout << Idx << std::endl;
	
	//
	// measured data Y
	Eigen::MatrixXd Y = Eigen::MatrixXd::Zero( num_3D_images_, 1 );
	// First lines are set to the measure
	for ( int img = 0 ; img < num_3D_images_ ; img++ )
	  Y( img, 0 ) = Y_[ img ]->GetOutput()->GetPixel( Idx );
	//
	if ( true )
	  std::cout << "response: " << Y.rows() << "\n" << Y << std::endl;
	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	  exit( -1 );
	}
    }
  //
  //
  //
  template< class Optimizer, int DimY, int D_f > void
    MleLoadCSV< Optimizer, DimY, D_f >::write_subjects_solutions( )
  {
    try
      {
//	    //
//	    std::cout << "Global solutions" << std::endl;
//	    // level 1
//	    PPM_.write();
//	    // Posterior t-maps
//	    post_T_maps_.write();
//	    // Posterior groups parameters
//	    post_groups_param_.write();
//	    // level 2
//	    PPM_l2_.write();
//	    // Posterior t-maps
//	    post_T_maps_l2_.write();
//	    // Posterior groups parameters
//	    post_groups_param_l2_.write();
//	    // Posterior groups variance
//	    post_groups_cov_l2_.write();
//	    // R-square
//	    R_sqr_l2_.write();
//	    // Output for prediction
//	    Prediction_inverse_error_l2_.write();
//	//
//	//
//	std::cout << "Subjects solutions" << std::endl;
//	for ( auto g : groups_ )
//	  for ( auto subject : group_pind_[g] )
//	    subject.second.write_solution();
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
	exit( -1 );
      }
  }
}
#endif
