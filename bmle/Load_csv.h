#ifndef BMLELOADCSV_H
#define BMLELOADCSV_H
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
// Eigen
#include <Eigen/Core>
#include <Eigen/Eigen>
#include <Eigen/Sparse>
//
// ITK
//
#include <itkImageFileReader.h>
#include <itkSpatialOrientationAdapter.h>
#include "itkChangeInformationImageFilter.h"
using MaskType = itk::Image< unsigned char, 3 >;
//
//
//
#include "BmleException.h"
#include "BmleTools.h"
#include "Subject.h"
//
//
//
namespace MAC_bmle
{
  /** \class BmleLoadCSV
   *
   * \brief 
   * 
   */
  template< int D_r, int D_f >
    class BmleLoadCSV
  {
  public:
    /** Constructor. */
    explicit BmleLoadCSV( const std::string& );
    
    /** Destructor */
    virtual ~BmleLoadCSV() {};


    //
    // This function will load all the patients images into a 4D image.
    void build_groups_design_matrices();
    // Expectation maximization algorithm
    void Expectation_Maximization( MaskType::IndexType );
    
  private:
    //
    // Functions
    //

    // This function will load all the patients images into a 4D image.
    void image_concat();
    // Thermodynamic free energy
    double F_( const Eigen::MatrixXd& , const Eigen::MatrixXd& ,
	       const Eigen::MatrixXd& , const Eigen::MatrixXd& ) const;
    // Thermodynamic free energy
    void Lambda_init_( const Eigen::MatrixXd& ) const;

    //
    // Members
    //
    
    //
    // CSV file
    std::ifstream csv_file_;
    //
    // Arrange pidns inti groups
    std::set< int > groups_;
    std::vector< std::map< int /*pidn*/, BmleSubject< D_r, D_f > > > group_pind_{10};
    // Number of subjects per group
    std::vector< int > group_num_subjects_{0,0,0,0,0,0,0,0,0,0};
    //
    // Measures grouped in vector of 3D image
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

    // X1
    Eigen::MatrixXd X1_;
    // X2
    Eigen::MatrixXd X2_;
    // X augmented
    Eigen::MatrixXd X_;

    //
    // Covariance base matrices
    //

    // Base matrices
    // 1 /*C_eps_1_base*/+ D_r /*C_eps_2_base*/+ 1 /*fixed effect*/
    std::map< int /*group*/, std::vector< Eigen::MatrixXd > > Q_k_;
    // Covariance matrix theta level two
    Eigen::MatrixXd C_theta_;
  };
  //
  //
  //
  template< int D_r, int D_f >
    BmleLoadCSV< D_r, D_f >::BmleLoadCSV( const std::string& CSV_file ):
    csv_file_{ CSV_file.c_str() }
  {
    try
      {
	//
	//
	double mean_age = 0.;
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
	    const int PIDN = std::stoi( cell );
	    // Get the group
	    std::getline(lineStream, cell, ',');
	    const int group = std::stoi( cell );
	    if ( group == 0 )
	      throw BmleException( __FILE__, __LINE__,
				   "Select another group name than 0 (e.g. 1, 2, ...). 0 is reserved.",
				   ITK_LOCATION );
	    // Get the age
	    std::getline(lineStream, cell, ',');
	    int age = std::stoi( cell );
	    mean_age += static_cast< double >( age );
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
	      throw BmleException( __FILE__, __LINE__,
				   "The CSV file should have less than 10 gourps.",
				   ITK_LOCATION );
	    // If the PIDN does not yet exist
	    if ( group_pind_[ group ].find( PIDN ) == group_pind_[ group ].end() )
	      {
		groups_.insert( group );
		group_pind_[ group ][PIDN] = BmleSubject< D_r, D_f >( PIDN, group );
		group_num_subjects_[ group ]++;
		num_subjects_++;
	      }
	    //
	    group_pind_[ group ][ PIDN ].add_tp( age, covariates, image );
	    num_3D_images_++;
	  }

	// 
	// Design Matrix for every subject
	//

	//
	// mean age
	mean_age /= static_cast< double >( num_3D_images_ );
	std::cout << "mean age: " << mean_age << std::endl;
	//
	Y_.resize( num_3D_images_ );
	int sub_image{0};
	for ( auto g : groups_ )
	  for ( auto& s : group_pind_[g] )
	    {
	      s.second.build_design_matrices( mean_age );
	      // Create the vector of 3D measurements image
	      for ( auto image : s.second.get_age_images() )
		Y_[ sub_image++ ] = image.second;
	    }
	//
	// Create the 4D measurements image
	//image_concat();
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
	exit( -1 );
      }


    //
    //
    if ( false )
      for ( auto g : groups_ )
	for ( auto s : group_pind_[g] )
	  s.second.print();
  }
  //
  //
  //
  template< int D_r, int D_f > void
    BmleLoadCSV< D_r, D_f >::image_concat()
  {
    try
      {
//	//
//	// ITK image types
//	using Image3DType = itk::Image< double, 3 >;
//	using Reader3D    = itk::ImageFileReader< Image3DType >;
//	// 
//	using Iterator3D = itk::ImageRegionConstIterator< Image3DType >;
//	using Iterator4D = itk::ImageRegionIterator< Image4DType >;
//
//	//
//	// Create the 4D image of measures
//	//
//
//	//
//	// Set the measurment 4D image
//	Y_ = Image4DType::New();
//	//
//	Image4DType::RegionType region;
//	Image4DType::IndexType  start = { 0, 0, 0, 0 };
//	//
//	// Take the dimension of the first subject image:
//	Reader3D::Pointer subject_image_reader_ptr =
//	  group_pind_[ (*groups_.begin()) ].begin()->second.get_age_images().begin()->second;
//	//
//	Image3DType::Pointer  raw_subject_image_ptr = subject_image_reader_ptr->GetOutput();
//	Image3DType::SizeType size = raw_subject_image_ptr->GetLargestPossibleRegion().GetSize();
//	Image4DType::SizeType size_4D{ size[0], size[1], size[2], num_3D_images_ };
//	//
//	region.SetSize( size_4D );
//	region.SetIndex( start );
//	//
//	Y_->SetRegions( region );
//	Y_->Allocate();
//	//
//	// ITK orientation, most likely does not match our orientation
//	// We have to reset the orientation
//	using FilterType = itk::ChangeInformationImageFilter< Image4DType >;
//	FilterType::Pointer filter = FilterType::New();
//	// Origin
//	Image3DType::PointType orig_3d = raw_subject_image_ptr->GetOrigin();
//	Image4DType::PointType origin;
//	origin[0] = orig_3d[0]; origin[1] = orig_3d[1]; origin[2] = orig_3d[2]; origin[3] = 0.;
//	// Spacing 
//	Image3DType::SpacingType spacing_3d = raw_subject_image_ptr->GetSpacing();
//	Image4DType::SpacingType spacing;
//	spacing[0] = spacing_3d[0]; spacing[1] = spacing_3d[1]; spacing[2] = spacing_3d[2]; spacing[3] = 1.;
//	// Direction
//	Image3DType::DirectionType direction_3d = raw_subject_image_ptr->GetDirection();
//	Image4DType::DirectionType direction;
//	direction[0][0] = direction_3d[0][0]; direction[0][1] = direction_3d[0][1]; direction[0][2] = direction_3d[0][2]; 
//	direction[1][0] = direction_3d[1][0]; direction[1][1] = direction_3d[1][1]; direction[1][2] = direction_3d[1][2]; 
//	direction[2][0] = direction_3d[2][0]; direction[2][1] = direction_3d[2][1]; direction[2][2] = direction_3d[2][2];
//	direction[3][3] = 1.; // 
//	//
//	filter->SetOutputSpacing( spacing );
//	filter->ChangeSpacingOn();
//	filter->SetOutputOrigin( origin );
//	filter->ChangeOriginOn();
//	filter->SetOutputDirection( direction );
//	filter->ChangeDirectionOn();
//	//
//	//
//	Iterator4D it4( Y_, Y_->GetBufferedRegion() );
//	it4.GoToBegin();
//	//
//	for ( auto group : groups_ )
//	  for ( auto subject : group_pind_[group] )
//	    for ( auto image : subject.second.get_age_images() )
//	      {
//		std::cout << image.second->GetFileName() << std::endl;
//		Image3DType::RegionType region = image.second->GetOutput()->GetBufferedRegion();
//		Iterator3D it3( image.second->GetOutput(), region );
//		it3.GoToBegin();
//		while( !it3.IsAtEnd() )
//		  {
//		    it4.Set( it3.Get() );
//		    ++it3; ++it4;
//		  }
//	      }
//
//	//
//	// Writer
//	std::cout << "Writing the 4d measure image." << std::endl;
//	//
//	filter->SetInput( Y_ );
//	itk::NiftiImageIO::Pointer nifti_io = itk::NiftiImageIO::New();
//	//
//	itk::ImageFileWriter< Image4DType >::Pointer writer = itk::ImageFileWriter< Image4DType >::New();
//	writer->SetFileName( "measures_4D.nii.gz" );
//	writer->SetInput( filter->GetOutput() );
//	writer->SetImageIO( nifti_io );
//	writer->Update();
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
  template< int D_r, int D_f > void
    BmleLoadCSV< D_r, D_f >::build_groups_design_matrices()
  {
    try
      {
	//
	// Build X1 and X2 design matrices
	//

	//
	//
	int
	  X1_lines = num_3D_images_,
	  X1_cols  = num_subjects_ * D_r + groups_.size() * D_f,
	  X2_lines = num_subjects_ * D_r + groups_.size() * D_f ,
	  X2_cols  = groups_.size() * ( D_r * (num_covariates_ + 1) + D_f );
	//
	X1_ = Eigen::MatrixXd::Zero( X1_lines, X1_cols );
	X2_ = Eigen::MatrixXd::Zero( X2_lines, X2_cols );
	//
	int line_x1 = 0, col_x1 = 0;
	int line_x2 = 0, col_x2 = 0;
	int current_gr = ( *groups_.begin() ), increme_dist_x1 = 0, increme_dist_x2 = 0;
	for ( auto g : groups_ )
	  for ( auto subject : group_pind_[g] )
	    {
	      //
	      // we change group
	      if ( current_gr != g )
		{
		  // Add the ID matrix for the fixed parameters
		  X2_.block( line_x2, D_r * (num_covariates_ + 1) + increme_dist_x2,
			     D_f, D_f ) = Eigen::MatrixXd::Identity( D_f, D_f );
		  line_x2 += D_f;
		  //
		  increme_dist_x1 += group_num_subjects_[current_gr] * D_r + D_f;
		  increme_dist_x2 += D_r * (num_covariates_ + 1) + D_f;
		  col_x1          += D_f;
		  current_gr       = g;
		}
	      //
	      // X1 design
	      int
		sub_line_x1 = subject.second.get_random_matrix().rows(),
		sub_col_x1  = subject.second.get_random_matrix().cols(),
		sub_line_x1_fixed = subject.second.get_fixed_matrix().rows(),
		sub_col_x1_fixed  = subject.second.get_fixed_matrix().cols();
	      X1_.block( line_x1, col_x1, sub_line_x1, sub_col_x1 ) = subject.second.get_random_matrix();
	      X1_.block( line_x1, increme_dist_x1 + group_num_subjects_[g]  * D_r,
			 sub_line_x1_fixed, sub_col_x1_fixed ) = subject.second.get_fixed_matrix();
	      //
	      line_x1 += sub_line_x1;
	      col_x1  += sub_col_x1;
	      //
	      // X2 design
	      int
		sub_line_x2 = subject.second.get_X2_matrix().rows(),
		sub_col_x2  = subject.second.get_X2_matrix().cols();
	      X2_.block( line_x2, increme_dist_x2, sub_line_x2, sub_col_x2 ) = subject.second.get_X2_matrix();
	      //
	      line_x2 += sub_line_x2;
	    }
	// And the last Id matrix of the X2 design
	X2_.block( line_x2, D_r * (num_covariates_ + 1) + increme_dist_x2,
		   D_f, D_f ) = Eigen::MatrixXd::Identity( D_f, D_f );
	//
	//
	if ( false )
	  {
	    std::cout << X1_ << std::endl;
	    std::cout << X2_ << std::endl;
	  }

	//
	// Build X augmented
	//

	//
	// Before keeping going, we check the X1_cols == X2_lines
	if ( X1_cols != X2_lines )
	  {
	    std::string mess = std::string("Number of lines first design (X1) must be equal to the ");
	    mess += std::string("number of columns second design (X2).");
	    //
	    throw BmleException( __FILE__, __LINE__,
				 mess.c_str(),
				 ITK_LOCATION );
	  }
	
	//
	//
	int
	  X_lines = X1_cols + X2_cols + X1_lines,
	  X_cols  = X1_cols + X2_cols;
	//
	X_ = Eigen::MatrixXd::Zero( X_lines, X_cols );
	// lines 
	X_.block( 0, 0, X1_lines, X1_cols )                 = X1_;
	X_.block( X1_lines, 0, X1_cols, X1_cols )           = Eigen::MatrixXd::Identity( X1_cols, X1_cols );
	X_.block( X1_lines + X1_cols, 0, X2_cols, X1_cols ) = Eigen::MatrixXd::Zero( X2_cols, X1_cols );
	// Columns
	X_.block( 0, X1_cols, X1_lines, X2_cols )                 = X1_ * X2_;
	X_.block( X1_lines, X1_cols, X1_cols, X2_cols )           = Eigen::MatrixXd::Zero( X1_cols, X2_cols );
	X_.block( X1_lines + X1_cols, X1_cols, X2_cols, X2_cols ) = Eigen::MatrixXd::Identity( X2_cols, X2_cols );
	//
	if ( false )
	  {
	    std::cout << X_ << std::endl;
	  }

	//
	// Building the covariance base matrices & covariante matrices
	//

	//
	// Based matrix for a subject
	std::vector< Eigen::MatrixXd > q( D_r );
	for ( int i = 0 ; i < D_r ; i++ )
	  {
	    q[i] = Eigen::MatrixXd::Zero( D_r, D_r );
	    q[i](i,i) = 1.;
	  }

	//
	// Based matrix per group
	std::map< int, std::vector< Eigen::MatrixXd > > Q_group;
	for ( auto g : groups_ )
	  {
	    //
	    // For each group we create D_r based matrix + 1 matrix for fixed effects over
	    // D_r dimensions
	    if ( Q_group.find( g ) == Q_group.end() )
		Q_group[g] = std::vector< Eigen::MatrixXd >( D_r + 1 );
	    // fixed effect matrix
	    Q_group[g][D_r] = Eigen::MatrixXd::Zero( group_num_subjects_[g] * D_r + D_f,
						     group_num_subjects_[g] * D_r + D_f );
	    // We create each D_r dimensions
	    for ( int d = 0 ; d < D_r ; d++ )
	      {
		Q_group[g][d] = Eigen::MatrixXd::Zero( group_num_subjects_[g] * D_r + D_f,
						       group_num_subjects_[g] * D_r + D_f );
		//
		int linco = 0;
		for ( int s = 0 ; s < group_num_subjects_[g] ; s++ )
		  {
		    Q_group[g][d].block( linco, linco, D_r, D_r ) = q[d];
		    linco += D_r;
		  }
		// Add the fixed effect
		Q_group[g][D_r].block( linco, linco,
				       D_f, D_f ) = Eigen::MatrixXd::Identity( D_f, D_f );
		if ( false )
		  {
		    std::cout << "Q_group[" << g << "][" << d << "] = \n"
			      << Q_group[g][d] << std::endl;
		    std::cout << "Fixed part: " << std::endl;
		    std::cout << "Q_group[" << g << "][D_r] = \n"
			      << Q_group[g][D_r] << std::endl;
		  }
	      }
	  }

	
	//
	// Building the starting block covariance matrix
	// 

	//
	// Global dimension of the base matrix
	int
	  C_theta_dim = groups_.size() * ( D_r * (num_covariates_+1) + D_f ) /*C_theta dimension*/,
	  Q_k_linco = num_3D_images_ /*C_eps_1_base*/
	  + num_subjects_ * D_r + groups_.size() * D_f /*C_eps_2_base*/
	  + C_theta_dim;
	int linco = 0;
	//
	// C_eps_1_base
	Q_k_[0]    = std::vector< Eigen::MatrixXd >( 1 );
	Q_k_[0][0] = Eigen::MatrixXd::Zero( Q_k_linco, Q_k_linco );
	Q_k_[0][0].block( 0, 0,
			  num_3D_images_,  num_3D_images_ ) = Eigen::MatrixXd::Identity(num_3D_images_,
											num_3D_images_);
	//
	// C_eps_2_base and fixed effects
	linco = num_3D_images_;
	for ( auto g : groups_ )
	  {
	    // check for a new group
	    if ( Q_k_.find( g ) == Q_k_.end() )
		Q_k_[g] = std::vector< Eigen::MatrixXd >( D_r + 1 );
	    // fixed effect matrix
	    Q_k_[g][D_r]  = Eigen::MatrixXd::Zero( Q_k_linco, Q_k_linco );
	    int sub_linco = Q_group[g][0].rows();
	    for ( int d_r = 0 ; d_r < D_r ; d_r++ )
	      {
		Q_k_[g][d_r] = Eigen::MatrixXd::Zero( Q_k_linco, Q_k_linco );
		Q_k_[g][d_r ].block( linco, linco,
				     sub_linco, sub_linco) = Q_group[g][d_r];
	      }
	    // fixed effects
	    Q_k_[g][D_r ].block( linco + sub_linco - D_f, linco + sub_linco - D_f,
				 D_f, D_f ) = 1.e-16 * Eigen::MatrixXd::Identity(D_f, D_f);
	    //
	    linco += sub_linco;
	  }
	//
	// Covariance matrix theta level two
	C_theta_ = Eigen::MatrixXd::Zero( Q_k_linco, Q_k_linco );
	C_theta_.block( linco, linco,
			C_theta_dim,  C_theta_dim ) = 1.e+16 * Eigen::MatrixXd::Identity( C_theta_dim,
											  C_theta_dim );
	//
	//
	if ( false )
	  {
	    std::cout << "Q_k_[0][0] = \n"
		      << Q_k_[0][0] << "\n\n\n"
		      << std::endl;
	    for ( auto g : groups_ )
	      for ( int k = 0 ; k <  D_r + 1 ; k++ )
		std::cout << "Q_k_[" << g << "][" << k << "] = \n"
			  << Q_k_[g][k] << "\n\n\n"
			  << std::endl;
	    std::cout << "C_theta_ = \n"
		      << C_theta_ << "\n\n\n"
		      << std::endl;
	  }

	//
	//
	if ( X_.rows() != Q_k_linco )
	  {
	    //std::cout << "[X_] = " << X_.rows() << "x" << X_.cols() << std::endl;
	    //std::cout << "[Q_k_] = " << Q_k_linco << "x" << Q_k_linco << std::endl;
	    std::string mess = std::string("Dimensions of the covriance matrix and ");
	    mess += std::string("design matrix must comply:");
	    //
	    throw BmleException( __FILE__, __LINE__,
				 mess.c_str(),
				 ITK_LOCATION );
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
  template< int D_r, int D_f > void
    BmleLoadCSV< D_r, D_f >::Expectation_Maximization( MaskType::IndexType Idx )
  {
    try
      {
	//
	// Measure
	//
	std::cout << Idx << std::endl;
	
	//
	// Augmented measured data Y
	Eigen::MatrixXd Y = Eigen::MatrixXd::Zero( X_.rows(), 1 );
	// First lines are set to the measure
	// other lines are set to 0. 
	for ( int img = 0 ; img < num_3D_images_ ; img++ )
	  {
	    // std::cout << Idx << " " << Y_[ img ]->GetOutput()->GetPixel( Idx )
	    //           << std::endl;
	    Y( img, 0 ) = Y_[ img ]->GetOutput()->GetPixel( Idx );
	  }
	//
	if ( false )
	  std::cout << Y << std::endl;

	//
	// Covariance
	//
	
	//
	// Hyper-parameters
	// 1 /*C_eps_1*/+ groups_.size * D_r /*C_eps_2*/+ 1 /*fixed effect*/
	std::map< int /*group*/, std::vector< double > > lambda_k;
	// C_eps_1 lambda
	lambda_k[0]    = std::vector< double >( 1 );
	lambda_k[0][0] = 1.e+0 /*log( 1.e+8 )*/;
	// C_eps_2_ and fixed effects lambda
	for ( auto g : groups_ )
	  {
	    if ( lambda_k.find( g ) == lambda_k.end() )
	      lambda_k[g] = std::vector< double >( D_r + (D_f > 0  ? 1 : 0) );
	    //
	    for ( int d_r = 0 ; d_r < D_r ; d_r++ )
	      {
		lambda_k[g][d_r] = 1.e+0;
	      }
	    // fixed effects
	    if ( D_f > 0 )
	      lambda_k[g][D_r ] = 0.; // Always enforce this value to be 0
	  }

	//
	// Covariance
	Eigen::MatrixXd Cov_eps = C_theta_;
	Cov_eps +=  exp( lambda_k[0][0] ) * Q_k_[0][0];
	for ( auto g : groups_ )
	  for ( int k = 0 ; k < D_r + (D_f > 0  ? 1 : 0) ; k++ )
	    Cov_eps +=  exp( lambda_k[g][k] ) * Q_k_[g][k];
	//
	Eigen::MatrixXd inv_Cov_eps = Cov_eps.inverse();

	//
	//
	// posterior
	// Eigen::MatrixXd cov_theta_Y = MAC_bmle::inverse( X_.transpose() * inv_Cov_eps * X_ );
	Eigen::MatrixXd cov_theta_Y = ( X_.transpose() * inv_Cov_eps * X_ ).inverse();
	Eigen::MatrixXd eta_theta_Y = cov_theta_Y * X_.transpose() * inv_Cov_eps * Y;
	
	double
	  F_old = 1.,
	  F     = 1.,
	  delta_F  = 100.;
	int
	  n = 0,
	  N = 10;
	
	//
	while( n < N  )
	  {
	    F_old = F;

	    //
	    // Expectaction step
	    //	    cov_theta_Y = MAC_bmle::inverse( X_.transpose() * inv_Cov_eps * X_ );
	    cov_theta_Y = ( X_.transpose() * inv_Cov_eps * X_ ).inverse();
	    eta_theta_Y = cov_theta_Y * X_.transpose() * inv_Cov_eps * Y;
//	    std::cout << eta_theta_Y << std::endl;

	    //
	    // Maximization step
	    int hyper_dim = D_r + (D_f > 0  ? 1 : 0);
	    Eigen::MatrixXd P  = inv_Cov_eps - inv_Cov_eps * X_ * cov_theta_Y * X_.transpose() * inv_Cov_eps;
	    // Fisher Information matrix
	    Eigen::MatrixXd H = Eigen::MatrixXd::Zero( groups_.size() * hyper_dim + 1,
						       groups_.size() * hyper_dim + 1 );
	    // Fisher gradient
	    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero( groups_.size() * hyper_dim + 1, 1 );
	    int count_group = 0;
	    // groupe 0 C_eps_1
	    grad(0,0)  = - (Y.transpose() * P.transpose() * Q_k_[0][0] * P * Y)(0,0);
	    grad(0,0) += (P*Q_k_[0][0]).trace();
	    grad(0,0) *= - exp(lambda_k[0][0]) / 2.;
	    H(0,0) = exp(lambda_k[0][0] + lambda_k[0][0]) * ( P*Q_k_[0][0]*P*Q_k_[0][0] ).trace() / 2.;
	    //
	    for ( auto g : groups_ )
	      {
		for ( int i = 0 ; i < hyper_dim ; i++ )
		  {
		    // 
		    grad(i + count_group * hyper_dim + 1,0)  = - (Y.transpose() * P.transpose() * Q_k_[g][i] * P * Y)(0,0);
		    grad(i + count_group * hyper_dim + 1,0) += (P*Q_k_[g][i]).trace();
		    grad(i + count_group * hyper_dim + 1,0) *= - exp(lambda_k[g][i]) / 2.;
		    for ( int j = 0 ; j < D_r + (D_f > 0  ? 1 : 0); j++ )
		      H( i + count_group * hyper_dim + 1,j + count_group * hyper_dim + 1 ) =
			exp(lambda_k[g][i] + lambda_k[g][j]) * ( P*Q_k_[g][i]*P*Q_k_[g][j] ).trace() / 2.;
		  }
		// next group
		count_group++;
	      }
//	std::cout << std::endl;
//	std::cout << std::endl;
//	std::cout << std::endl;
//	std::cout << std::endl;
//	std::cout << std::endl;
//	std::cout << std::endl;
////	std::cout << P  << std::endl;
//	std::cout <<  grad << std::endl;
//	std::cout << H  << std::endl;
	    //
	    // Lambda update
	Eigen::MatrixXd delta_lambda = MAC_bmle::inverse( H - Eigen::MatrixXd::Identity( H.rows(), H.cols() ) / 32.) * grad;
	//Eigen::MatrixXd delta_lambda = 0.1 * grad;
	//	std::cout << delta_lambda << std::endl;
	    count_group = 0;
	    for ( auto g : groups_ )
	      {
		for ( int k = 1 ; k < hyper_dim ; k++ )
		  {
		    lambda_k[g][k] += delta_lambda( k + count_group * hyper_dim, 0 );
		    // std::cout << "lambda_k[" << g << "][" << k << "] = " << lambda_k[g][k] << " " << exp(lambda_k[g][k])<< std::endl;
		  }
		if ( D_f > 0 )
		  lambda_k[g][D_r] += 0.;
		//
		count_group++;
	      }
	    // Update of the covariance matrix
	    Cov_eps = C_theta_;
	    Cov_eps +=  exp( lambda_k[0][0] ) * Q_k_[0][0];
	    for ( auto g : groups_ )
	      for ( int k = 1 ; k < hyper_dim ; k++ )
		{
		  Cov_eps +=  exp( lambda_k[g][k] ) * Q_k_[g][k];
		}
//	    std::cout << Cov_eps << std::endl;
	    //
	    inv_Cov_eps =  MAC_bmle::inverse( Cov_eps );
	    //std::cout << inv_Cov_eps << std::endl;
	    //
	    // Free energy
	    F = F_( Y, inv_Cov_eps, eta_theta_Y, cov_theta_Y );
	    delta_F = F - F_old;
	    if ( fabs( delta_F ) < 1.e-8 )
	      n++;
	    else
	      n = 0;
	    std::cout << "n = " << n << " - F = " << F << " delta_F = " << fabs( delta_F )
		      << std::endl;
	  }

	//
	//
	int eta_theta_Y_2_theta_Y_dim = X2_.cols();
	int eta_theta_Y_2_eps_Y_dim   = cov_theta_Y.rows() - eta_theta_Y_2_theta_Y_dim;
	//
	Eigen::MatrixXd eta_theta_Y_2_eps_Y   = eta_theta_Y.block( 0, 0,
								   eta_theta_Y_2_eps_Y_dim, 1 );
	Eigen::MatrixXd eta_theta_Y_2_theta_Y = eta_theta_Y.block( eta_theta_Y_2_eps_Y_dim, 0,
								   eta_theta_Y_2_theta_Y_dim, 1 );
	//
	std::cout << eta_theta_Y_2_eps_Y_dim << " " << eta_theta_Y_2_theta_Y_dim << std::endl;
	
	std::cout << eta_theta_Y << std::endl;
	std::cout << eta_theta_Y_2_theta_Y << std::endl;
	std::cout  << std::endl;
	std::cout << X2_ * eta_theta_Y_2_theta_Y + eta_theta_Y_2_eps_Y << std::endl;
	std::cout << cov_theta_Y << std::endl;
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
  template< int D_r, int D_f > double
    BmleLoadCSV< D_r, D_f >::F_( const Eigen::MatrixXd& Augmented_Y,
				 const Eigen::MatrixXd& Inv_Cov_eps,
				 const Eigen::MatrixXd& Eta_theta_Y,
				 const Eigen::MatrixXd& Cov_theta_Y ) const
  {
    try
      {
	//
	// residual
	Eigen::MatrixXd r = Augmented_Y - X_ * Eta_theta_Y;

	//
	// Terms of free energy
	//

	//
	// log of determinants
	double F_1 = 0, F_4 = 0 ;
	//
	for ( int linco = 0 ; linco < Inv_Cov_eps.rows() ; linco++ )
	  F_1 += log( Inv_Cov_eps(linco,linco) );
	//
	for ( int linco = 0 ; linco < Cov_theta_Y.rows() ; linco++ )
	  F_4 += log( Cov_theta_Y(linco,linco) );
	
	double
	  F_2 = - (r.transpose() * Inv_Cov_eps * r).trace(), // tr added for compilation reason
	  F_3 = - ( Cov_theta_Y * X_.transpose() * Inv_Cov_eps * X_ ).trace();
	std::cout << "F_1 = " << F_1<< std::endl;
	std::cout << "F_2 = " << F_2<< std::endl;
	std::cout << "F_3 = " << F_3<< std::endl;
	std::cout << "F_4 = " << F_4<< std::endl;
	//
	//
	return ( F_1 + F_2 + F_3 + F_4 ) / 2.;
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
  template< int D_r, int D_f > void
    BmleLoadCSV< D_r, D_f >::Lambda_init_( const Eigen::MatrixXd& Augmented_Y ) const
  {
    try
      {
//	//
//	// residual
//	Eigen::MatrixXd r = Augmented_Y/* - X_ * Eta_theta_Y*/;
//
//	int
//	  C_theta_dim = groups_.size() * ( D_r * (num_covariates_+1) + D_f ) /*C_theta dimension*/,
//	  Q_k_linco = num_3D_images_ /*C_eps_1_base*/
//	  + num_subjects_ * D_r + groups_.size() * D_f /*C_eps_2_base*/
//	  + C_theta_dim;
//
//
//	std::vector< double > lambda_k( 1 /*C_eps_1*/+ D_r /*C_eps_2*/+ 1 /*fixed effect*/);
//	for ( int k = 0 ; k < lambda_k.size() - 1 ; k++ )
//	  {
//	    Eigen::MatrixXd inv_Q_k = Eigen::MatrixXd::Zero( Q_k_linco, Q_k_linco );
//	    for ( int q = 0 ; q < Q_k_linco ; q++ )
//	      if( Q_k_[k](q,q) != 0. )
//		inv_Q_k(q,q) = 1. / Q_k_[k](q,q);
//	    //
//	    // lambda numerator
//	    double l_num = (r.transpose() * inv_Q_k * r)(0,0);
//	    //
//	    // lambda denominator
//	    Eigen::MatrixXd R = Eigen::MatrixXd::Identity( X_.rows(), Q_k_linco );
//	    R -= X_ * ( X_.transpose() * inv_Q_k * X_ ).inverse() * X_.transpose() * inv_Q_k;
//	    //
//	    double l_denom = R.trace();
//
//	    std::cout << l_num << std::endl;
//	    std::cout << l_denom << std::endl;
//	    std::cout << "lamba[" << k << "] = " << l_num / l_denom << std::endl;
//	  }
//	lambda_k[ 0 ]       = log( 1.e+9 ); 
//	lambda_k[ D_r + 1 ] = 0.; // Always enforce this value to be 0
//
//
//
//
//
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
	exit( -1 );
      }
  }
}
#endif
