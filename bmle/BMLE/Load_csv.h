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
#include "Tools.h"
#include "Subject.h"
//
//
//
namespace NeuroBayes
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
    explicit BmleLoadCSV( const std::string&, const std::string&, 
			  // Age demeaning
			  const bool,
			  // Prediction
			  const std::string& );
    
    /** Destructor */
    virtual ~BmleLoadCSV() {};


    //
    // This function will load all the patients images into a 4D image.
    void build_groups_design_matrices();
    // Expectation maximization algorithm
    void Expectation_Maximization( MaskType::IndexType );
    // Prediction algorithm
    void Prediction( MaskType::IndexType );
    // Write the output
    void write_subjects_solutions( );
    // multi-threading
    void operator ()( const MaskType::IndexType idx )
    {
      if ( prediction_ )
	{
	  std::cout << "Prediction treatment for parameters: " 
		    << idx;
	  Prediction( idx );
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

    // Thermodynamic free energy
    double F_( const Eigen::MatrixXd& , const Eigen::MatrixXd& ,
	       const Eigen::MatrixXd& , const Eigen::MatrixXd& ) const;
    // Thermodynamic free energy
    void lambda_regulation_( std::map< int /*group*/, std::vector< double > >& );
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
    // output directory
    std::string   output_dir_;
    //
    // Arrange pidns into groups
    std::set< int > groups_;
    std::vector< std::map< std::string /*pidn*/, BmleSubject< D_r, D_f > > > group_pind_{10};
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

    //
    // Constrast matrix
    //

    // Contrast groupe for level one and two
    Eigen::MatrixXd constrast_;
    Eigen::MatrixXd constrast_l2_;

    //
    // Prediction matrices
    // Inverse covariance error
    // C_{epsilon}^{-1}
    Reader4DType::Pointer Prediction_inverse_error_;
    
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
    //
    // Output for prediction
    // model output for Inverse covariance error
    NeuroBayes::NeuroBayesMakeITKImage Prediction_inverse_error_l2_;
  };
  //
  //
  //
  template< int D_r, int D_f >
    BmleLoadCSV< D_r, D_f >::BmleLoadCSV( const std::string& CSV_file,
					  const std::string& Output_dir,
					  const bool Demeaning,
					  const std::string& Inv_cov_error):
    csv_file_{ CSV_file.c_str() }, output_dir_{ Output_dir }
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
	      throw NeuroBayes::NeuroBayesException( __FILE__, __LINE__,
				   "The CSV file should have less than 10 gourps.",
				   ITK_LOCATION );
	    // If the PIDN does not yet exist
	    if ( group_pind_[ group ].find( PIDN ) == group_pind_[ group ].end() )
	      {
		std::cout << PIDN << " " << group << std::endl;
		groups_.insert( group );
		group_pind_[ group ][PIDN] = BmleSubject< D_r, D_f >( PIDN, group, Output_dir );
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
	if ( Demeaning )
	  std::cout << "mean age: " << mean_age << std::endl;
	else
	  std::cout << "No demeaning " << std::endl;
	//
	Y_.resize( num_3D_images_ );
	int sub_image{0};
	for ( auto g : groups_ )
	  for ( auto& s : group_pind_[g] )
	    {
	      s.second.build_design_matrices( (Demeaning ? mean_age : 0) );
	      // Create the vector of 3D measurements image
	      for ( auto image : s.second.get_age_images() )
		Y_[ sub_image++ ] = image.second;
	    }

	//
	// Prediction
	if ( access( Inv_cov_error.c_str(), F_OK ) != -1 )
	  {
	    //
	    // switch into prediction mode
	    prediction_ = true;
	    std::cout << "Prediction map calculation using:" << std::endl;
	    std::cout << "\t - " << Inv_cov_error  << std::endl;

	    //
	    // Posterior 
	    //auto image_ptr = itk::ImageIOFactory::CreateImageIO( Post_theta.c_str(),
	    //						       itk::ImageIOFactory::ReadMode );
	    //image_ptr->SetFileName( Post_theta );
	    //image_ptr->ReadImageInformation();
	    //
	    Prediction_inverse_error_ = Reader4DType::New();
	    Prediction_inverse_error_->SetFileName( Inv_cov_error );
	    Prediction_inverse_error_->Update();
	    
	    //
	    // load the posterior maps already processed
	    for ( auto g : groups_ )
	      for ( auto& s : group_pind_[g] )
		s.second.load_model_matrices();
	  }
	else
	  std::cout << "Posterior map calculation" << std::endl;
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
	    throw NeuroBayes::NeuroBayesException( __FILE__, __LINE__,
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
	    std::cout << "Augmented model:" << std::endl;
	    std::cout << X_ << std::endl;
	  }

	//
	// Building the covariance base matrices & covariante matrices
	//

	//
	// Based matrix for a subject
	// D_r qi matrices (D_r x D_r)
	// 
	// | 0           |
	// |  0          |
	// |    ...      |
	// |      1      | ith element
	// |        ...  |
	// |           0 |
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
	int C_eps_num_block_diag = 2 + groups_.size() * (D_r+D_f);
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
	    throw NeuroBayes::NeuroBayesException( __FILE__, __LINE__,
				 mess.c_str(),
				 ITK_LOCATION );
	  }

	
	//
	// Build the contrast matrix
	//

	//
	// We build the weight of the cohort and the global contrasts
	int
	  swap_row = 0,
	  previouse_grp_size = 0;
	//
	Eigen::MatrixXd G                = - Eigen::MatrixXd::Identity( groups_.size(), groups_.size() );
	Eigen::MatrixXd all_contrasts    =   Eigen::MatrixXd::Zero( groups_.size(), groups_.size() * groups_.size() );
	Eigen::MatrixXd all_contrasts_l2 =   Eigen::MatrixXd::Zero( groups_.size(), groups_.size() * groups_.size() );
	Eigen::MatrixXd cohort           =   Eigen::MatrixXd::Zero( num_subjects_, groups_.size() );
	// First line is Ones
	G.block(0,0,1,groups_.size()) =   Eigen::MatrixXd::Ones( 1, groups_.size() );
	//
	for ( auto g : groups_ )
	  {
	    Eigen::MatrixXd tempo_G = G;
	    tempo_G.row(swap_row).swap( tempo_G.row(0) );
	    all_contrasts.block( 0, 0 + groups_.size() * swap_row,
				 groups_.size(), groups_.size() ) = tempo_G;
	    
	    cohort.block( previouse_grp_size , swap_row, group_num_subjects_[ g ], 1 ) =
	      Eigen::MatrixXd::Ones( group_num_subjects_[ g ], 1 ) / group_num_subjects_[ g ];
	    previouse_grp_size += group_num_subjects_[ g ];
	    
	    //
	    //
	    swap_row++;
	  }
	// Copy the constrast for the level 2
	all_contrasts_l2 = all_contrasts;
	//
	std::cout << "all_contrast For each of the parameters: \n" << all_contrasts << std::endl << std::endl;
	//
	// Cohort distribution
	// All contrasts between groups
	Eigen::MatrixXd all_cont_grps = Eigen::MatrixXd::Zero( num_subjects_,
							       groups_.size() * groups_.size() );
	//
	int
	  current_row = 0,
	  past_rows   = 0;
	//
	for ( auto g : groups_ )
	  {
	    for ( int c = 0 ; c < all_contrasts.cols() ; c++ )
	      all_cont_grps.block( past_rows, c,
				   group_num_subjects_[ g ], 1 ) =
		all_contrasts( current_row, c ) * Eigen::MatrixXd::Ones( group_num_subjects_[ g ], 1 ) / group_num_subjects_[ g ];
	    // next row
	    past_rows += group_num_subjects_[ g ];
	    current_row++;
	  }
	//
	// Global contrast
	Eigen::MatrixXd Id_Dr = Eigen::MatrixXd::Identity( D_r, D_r );
	//  
	constrast_    = Eigen::kroneckerProduct( all_cont_grps, Id_Dr );
	constrast_l2_ = Eigen::kroneckerProduct( all_contrasts_l2, 
						 Eigen::MatrixXd::Identity( D_r * (num_covariates_+1),
									    D_r * (num_covariates_+1) ) );
	//
	if ( true )
	  {
	    std::cout << "constrast_ \n" << constrast_  << std::endl;
	    std::cout << "constrast_l2 \n" << constrast_l2_  << std::endl;
	  }
	//
	// Contrast output
	std::string
	  // level 1
	  sPPM = output_dir_ + "/" + "PPM.nii.gz",
	  sPtM = output_dir_ + "/" + "Posterior_t_maps.nii.gz",
	  sPgP = output_dir_ + "/" + "Post_groups_param.nii.gz",
	  // level 2
	  sPPMl2 = output_dir_ + "/" + "PPM_l2.nii.gz",
	  sPtMl2 = output_dir_ + "/" + "Posterior_t_maps_l2.nii.gz",
	  sPgPl2 = output_dir_ + "/" + "Post_groups_param_l2.nii.gz",
	  sPgCl2 = output_dir_ + "/" + "Post_groups_cov_l2.nii.gz",
	  // R^{2} level 2
	  sR_2   = output_dir_ + "/" + "Post_R_square_l2.nii.gz",
	  // Output for prediction
	  piel2  = output_dir_ + "/" + "Prediction_inverse_error_l2.nii.gz";
	
	
	// level 1
	PPM_               = NeuroBayes::NeuroBayesMakeITKImage( constrast_.cols(), sPPM, Y_[0] );
	post_T_maps_       = NeuroBayes::NeuroBayesMakeITKImage( constrast_.cols(), sPtM, Y_[0] );
	post_groups_param_ = NeuroBayes::NeuroBayesMakeITKImage( constrast_.cols(), sPgP, Y_[0] );
	// level 2
	PPM_l2_               = NeuroBayes::NeuroBayesMakeITKImage( constrast_l2_.cols(), sPPMl2, Y_[0] );
	post_T_maps_l2_       = NeuroBayes::NeuroBayesMakeITKImage( constrast_l2_.cols(), sPtMl2, Y_[0] );
	post_groups_param_l2_ = NeuroBayes::NeuroBayesMakeITKImage( constrast_l2_.cols(), sPgPl2, Y_[0] );
	// we only save the variance for each parameter of each group
	post_groups_cov_l2_   = NeuroBayes::NeuroBayesMakeITKImage( constrast_l2_.rows(), sPgCl2, Y_[0] );
	// r-squared
	R_sqr_l2_             = NeuroBayes::NeuroBayesMakeITKImage( 2, sR_2, Y_[0] );
	// output for predictive model
	Prediction_inverse_error_l2_  = NeuroBayes::NeuroBayesMakeITKImage( C_eps_num_block_diag, piel2, Y_[0] );
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
	Eigen::MatrixXd  Y  = Eigen::MatrixXd::Zero( X_.rows(), 1 );
	Eigen::MatrixXd _Y_ = Eigen::MatrixXd::Zero( num_3D_images_, 1 );
	// First lines are set to the measure
	// other lines are set to 0. 
	for ( int img = 0 ; img < num_3D_images_ ; img++ )
	  {
	    // std::cout << Idx << " " << Y_[ img ]->GetOutput()->GetPixel( Idx )
	    //           << std::endl;
	    Y( img, 0 ) = Y_[ img ]->GetOutput()->GetPixel( Idx );
	    // pure raw measure vector
	    _Y_( img, 0 ) = Y_[ img ]->GetOutput()->GetPixel( Idx );
	  }
	//
	if ( false )
	  std::cout << "response: " << Y.rows() << "\n" << Y << std::endl;

	//
	// Covariance
	//
	
	//
	// Hyper-parameters
	// 1 /*C_eps_1*/+ groups_.size * D_r /*C_eps_2*/+ 1 /*fixed effect*/
	std::map< int /*group*/, std::vector< double > > lambda_k;
	// C_eps_1 lambda
	lambda_k[0]    = std::vector< double >( 1 );
	lambda_k[0][0] = .01 /*log( 1.e+8 )*/;
	// C_eps_2_ and fixed effects lambda
	for ( auto g : groups_ )
	  {
	    if ( lambda_k.find( g ) == lambda_k.end() )
	      lambda_k[g] = std::vector< double >( D_r + (D_f > 0  ? 1 : 0) );
	    //
	    for ( int d_r = 0 ; d_r < D_r ; d_r++ )
	      {
		lambda_k[g][d_r] = .02;
	      }
	    // fixed effects
	    if ( D_f > 0 )
	      lambda_k[g][D_r ] = 0.; // Always enforce this value to be 0
	  }

	//
	// Covariance
	// Cov_eps = C_theta_ + C_eps_2 + C_eps_1
	Eigen::MatrixXd Cov_eps = C_theta_;
	Cov_eps +=  exp( lambda_k[0][0] ) * Q_k_[0][0];
	for ( auto g : groups_ )
	  for ( int k = 0 ; k < D_r + (D_f > 0  ? 1 : 0) ; k++ )
	    {
	      Cov_eps +=  exp( lambda_k[g][k] ) * Q_k_[g][k];
	      //std::cout << "lambda_k__[" << g << "][" << k << "] = " << lambda_k[g][k] << std::endl;
	      //std::cout << "Q_k__[" << g << "][" << k << "] = \n" << Q_k_[g][k] << std::endl;
	    }
	//std::cout << "Cov_eps__ = \n" << Cov_eps << std::endl;

	//
	Eigen::MatrixXd inv_Cov_eps = NeuroBayes::inverse( Cov_eps );
	// Eigen::MatrixXd inv_Cov_eps = Cov_eps.inverse();

	//
	//
	// posterior
	Eigen::MatrixXd cov_theta_Y = NeuroBayes::inverse( X_.transpose() * inv_Cov_eps * X_ );
	//Eigen::MatrixXd cov_theta_Y = ( X_.transpose() * inv_Cov_eps * X_ ).inverse();
	Eigen::MatrixXd eta_theta_Y = cov_theta_Y * X_.transpose() * inv_Cov_eps * Y;
	// R-square calculation
	Eigen::MatrixXd M_eta_theta = Eigen::MatrixXd::Zero( X_.rows(), X_.rows() );
	//Eigen::MatrixXd Id_m        = Eigen::MatrixXd::Identity( M_eta_theta.rows(), M_eta_theta.cols() ); 
	Eigen::MatrixXd Id_m        = Eigen::MatrixXd::Identity( num_3D_images_, num_3D_images_ ); 
	Eigen::MatrixXd L_m         = Eigen::MatrixXd::Ones(num_3D_images_, 1 ); 
	//
	Eigen::MatrixXd H_m         = Eigen::MatrixXd::Zero(num_3D_images_, num_3D_images_ ); ; //= L * (L.transpose() * L).inverse() * L.transpose();
	
	
	bool Fisher_H = true;
	double
	  F_old = 1.,
	  F     = 1.,
	  delta_F  = 100.;
	int
	  n  = 0, it = 0,
	  N  = 10,         // N regulate the EM loop to be sure converge smootly
	  NN = 1000,       // failed convergence criterias
	  early_stop = 800;
	//
	double 
	  learning_rate_  = 1.e-04,//1.e-02,
	  convergence_    = 1.e-06,
	  new_convergence = 1.e-16,
	  epsilon         = 1.e-16;
	std::list< double > best_convergence;
	//
	// Fisher strategy
	if ( Fisher_H )
	  {
	    early_stop      = 100;
	    learning_rate_  = 1.e-01;
	  }
	
	//
	while( n < N && it++ < NN )
	  {
	    if( !isnan(F) )
	      F_old = F;

	    //
	    // Expectaction step
	    cov_theta_Y = NeuroBayes::inverse( X_.transpose() * inv_Cov_eps * X_ );
	    //cov_theta_Y = ( X_.transpose() * inv_Cov_eps * X_ ).inverse();
	    eta_theta_Y = cov_theta_Y * X_.transpose() * inv_Cov_eps * Y;
	    //std::cout << "eta_theta_Y = \n" << eta_theta_Y << std::endl;
	    //std::cout << "cov_theta_Y = \n" << cov_theta_Y << std::endl;

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
	    grad(0,0) *= - exp(lambda_k[0][0]) * 0.5 ;
	    if ( Fisher_H )
	      H(0,0) = exp( 2 * lambda_k[0][0] ) * ( P*Q_k_[0][0]*P*Q_k_[0][0] ).trace() * 0.5;
	    //std::cout << "Q_k_g0_[0][0] = \n" << Q_k_[0][0] << std::endl;

	    //
	    for ( auto g : groups_ )
	      {
		for ( int i = 0 ; i < hyper_dim ; i++ )
		  {
		    // 
		    grad(i + count_group * hyper_dim + 1,0)  = - (Y.transpose() * P.transpose() * Q_k_[g][i] * P * Y)(0,0);
		    grad(i + count_group * hyper_dim + 1,0) += (P*Q_k_[g][i]).trace();
		    grad(i + count_group * hyper_dim + 1,0) *= - exp(lambda_k[g][i]) * 0.5;
		    //std::cout << "Q_k_g_[" << g << "][" << i << "] = \n" << Q_k_[g][i] << std::endl;
		    if ( Fisher_H )
		      for ( int j = 0 ; j < D_r + (D_f > 0  ? 1 : 0); j++ )
			H( i + count_group * hyper_dim + 1, j + count_group * hyper_dim + 1 ) = 
			  exp(lambda_k[g][i] + lambda_k[g][j]) * ( P*Q_k_[g][i]*P*Q_k_[g][j] ).trace() * 0.5;
		  }
		// next group
		count_group++;
	      }
	    //  comment
	    //std::cout << P  << std::endl;
	    //std::cout <<  grad << std::endl;
	    //std::cout << H  << std::endl;
	    ///  comment
	    //
	    // Lambda update
	    // | add a learning rate
	    // 
	    Eigen::MatrixXd delta_lambda;
	    if ( Fisher_H )
	      {
		// delta_lambda = NeuroBayes::inverse( H ) * grad;
		//// delta_lambda = NeuroBayes::inverse( H - Eigen::MatrixXd::Ones( H.rows(), H.cols() ) / 32. ) * grad;
		// delta_lambda = -learning_rate_ * NeuroBayes::inverse( H - 1.e-16 * Eigen::MatrixXd::Identity( H.rows(), H.cols() ) ) * grad;
		delta_lambda = learning_rate_ * NeuroBayes::inverse( H - Eigen::MatrixXd::Ones( H.rows(), H.cols() ) / 32. ) * grad;
		//std::cout << NeuroBayes::inverse( H - 1.e-16 * Eigen::MatrixXd::Identity( H.rows(), H.cols() ) )  << std::endl;
	      }
	    else
	      delta_lambda = learning_rate_ * grad;
	    //std::cout << delta_lambda << std::endl;
	    //std::cout << std::endl;
	    //std::cout << grad << std::endl;
	    lambda_k[0][0] += delta_lambda( 0, 0 );
	    //std::cout << "lambda_k[0][0] = " << lambda_k[0][0] << " " << exp(lambda_k[0][0])<< std::endl;
	    //
	    count_group = 0;
	    for ( auto g : groups_ )
	      {
		for ( int k = 0 ; k < hyper_dim ; k++ )
		  {
		    lambda_k[g][k] += delta_lambda( 1 + k + count_group * hyper_dim, 0 );
		    //std::cout << "lambda_k[" << g << "][" << k << "] = " << lambda_k[g][k] << " " << exp(lambda_k[g][k])<< std::endl;
		  }
		if ( D_f > 0 )
		  lambda_k[g][D_r] = 0.;
		//
		count_group++;
	      }
	    // Lambda regulation
	    lambda_regulation_( lambda_k );
	    // Update of the covariance matrix
	    Cov_eps = C_theta_;
	    Cov_eps +=  exp( lambda_k[0][0] ) * Q_k_[0][0];
	    for ( auto g : groups_ )
	      for ( int k = 0 ; k < hyper_dim ; k++ )
		{
		  Cov_eps +=  exp( lambda_k[g][k] ) * Q_k_[g][k];
		  //std::cout << "lambda_k[" << g << "][" << k << "] = " << lambda_k[g][k] << std::endl;
		  //std::cout << "Q_k_[" << g << "][" << k << "] = \n" << Q_k_[g][k] << std::endl;
		}
	    //std::cout << "Cov_eps_g_ = \n" << Cov_eps << std::endl;
	    //
	    inv_Cov_eps =  NeuroBayes::inverse( Cov_eps );
	    //inv_Cov_eps = Cov_eps.inverse();
	    //std::cout << "inv_Cov_eps_g_ = \n" << inv_Cov_eps << std::endl;
	    //
	    // Free energy
	    F = F_( Y, inv_Cov_eps, eta_theta_Y, cov_theta_Y );
	    //
	    if( !isnan(F) )
	      delta_F = F - F_old;
	    
	    double grad_level = 0.;
	    for ( int r = 1 /* we don't want the first element 0 */ ; r < grad.rows() ; r++ )
	      grad_level += grad( r, 0 );
	    
	    //std::cout << "mark_out,"
	    //	      << it << ","
	    //	      << Idx[0] << "_"<< Idx[1] << "_"<< Idx[2] << ","
	    //	      << n << ","
	    //	      << F << ","
	    //	      << F_old << "," 
	    //	      << delta_F << "," 
	    //	      << fabs( delta_F /F_old ) << "," 
	    //	      <<  grad_level << std::endl;

	    //
	    //
	    double abs_deltaF_F = fabs( delta_F /F_old );
	    if ( abs_deltaF_F < 1. && abs_deltaF_F > convergence_)
	      best_convergence.push_back( abs_deltaF_F );
	    //
	    //if ( fabs( grad_level ) < 5.e-03  )
	    //if ( fabs( F ) < 5.e-30  )
	    if ( abs_deltaF_F < convergence_ )
	      n++;
	    else
	      n = 0;
	    // Algo must converge fast
	    // If at 100 iterations we still did not converge, we create a new threshold
	    // based on the best values. Best value for the regular gradiant descent (800)
	    if ( it == early_stop )
	      {
		best_convergence.sort();
		auto fit = best_convergence.begin();
		new_convergence  = *(++fit);
		new_convergence += new_convergence/10.;
		//std::cout << "mark_best,"
		//	  << new_convergence << ","
		//	  << *(++fit) << ","
		//	  << *(++fit) << ",\n";
	      }
	    //
	    if ( it > early_stop && abs_deltaF_F < new_convergence )
	      n = N;
	  } // while( n < N && it++ < NN )

	//
	//
	int eta_theta_Y_2_theta_Y_dim = X2_.cols();
	int eta_theta_Y_2_eps_Y_dim   = cov_theta_Y.rows() - eta_theta_Y_2_theta_Y_dim;
	//
	Eigen::MatrixXd eta_theta_Y_2_eps_Y   = eta_theta_Y.block( 0, 0,
								   eta_theta_Y_2_eps_Y_dim, 1 );
	Eigen::MatrixXd eta_theta_Y_2_theta_Y = eta_theta_Y.block( eta_theta_Y_2_eps_Y_dim, 0,
								   eta_theta_Y_2_theta_Y_dim, 1 );
	//std::cout << "eta_theta_Y_2_eps_Y_dim: " << eta_theta_Y_2_eps_Y_dim << std::endl;
	//std::cout << "eta_theta_Y:\n" << eta_theta_Y << std::endl;
	//std::cout << "cov_theta_Y:\n" << cov_theta_Y << std::endl;
	//
	// Solution
	//

	//
	//
	Eigen::MatrixXd parameters = X2_ * eta_theta_Y_2_theta_Y + eta_theta_Y_2_eps_Y;
	Eigen::MatrixXd param_cov  = cov_theta_Y.block( 0, 0,
							parameters.rows(), parameters.rows() );
	Eigen::MatrixXd cov_theta_Y_l2  = cov_theta_Y.block( parameters.rows(), parameters.rows(),
							     eta_theta_Y_2_theta_Y.rows(), eta_theta_Y_2_theta_Y.rows() );

	//std::cout << "parameters" << "\n" << parameters << std::endl;
	//std::cout << "param_cov"  << "\n" << param_cov  << std::endl;
	//std::cout << "cov_theta_Y"  << "\n" <<  cov_theta_Y << std::endl;

	//std::cout << "inv_Cov_eps"  << "\n" <<  inv_Cov_eps << std::endl;

	//
	// R-squared
	//old M_eta_theta = X_ * cov_theta_Y * X_.transpose() * inv_Cov_eps;
	//old H_m         = L_m * (L_m.transpose() * L_m).inverse() * L_m.transpose();
	//old //
	//old double R_sqr = 1.;
	//old R_sqr       -=  ((Y.transpose() * (Id_m - M_eta_theta) * Y))(0,0) / ((Y.transpose() * (Id_m - H_m) * Y))(0,0);
	//old //std::cout << "R-sqr: " << R_sqr << std::endl;
	//old R_sqr_l2_.set_val( 0, Idx, R_sqr );

	// test 2
	//M_eta_theta = inv_Cov_eps * X_ * cov_theta_Y * X_.transpose() * inv_Cov_eps;
	//H_m         = L_m * (L_m.transpose() * L_m).inverse() * L_m.transpose();
	////
	////double R_sqr = (( X_ * eta_theta_Y ).transpose() * (Id_m - H_m) * X_ * eta_theta_Y )(0,0) / (Y.transpose() * (Id_m - H_m) * Y)(0,0);
	//Eigen::MatrixXd X_x_eta_theta_Y  = X_ * eta_theta_Y;
	//double R_sqr = (X_x_eta_theta_Y.transpose() * (Id_m - H_m) * X_x_eta_theta_Y )(0,0) / (Y.transpose() * (Id_m - H_m) * Y)(0,0);
	//std::cout << "R-sqr: " << R_sqr << std::endl;
	//R_sqr_l2_.set_val( 0, Idx, R_sqr );

	// test 3
	//
	Eigen::MatrixXd _X_  = Eigen::MatrixXd::Zero( X1_.rows(),  X1_.cols() + X2_.cols() );
	_X_.block( 0, 0, X1_.rows(),  X1_.cols() ) = X1_;
	_X_.block( 0, X1_.cols(), X1_.rows(),  X2_.cols() ) = X1_ * X2_;
	//
	H_m  = L_m * NeuroBayes::inverse(L_m.transpose() * L_m) * L_m.transpose();
	// H_m  = L_m * (L_m.transpose() * L_m).inverse() * L_m.transpose();
	//
	Eigen::MatrixXd X_x_eta_theta_Y  = _X_ * eta_theta_Y;
	double R_sqr = (X_x_eta_theta_Y.transpose() * (Id_m - H_m) * X_x_eta_theta_Y )(0,0) / (_Y_.transpose() * (Id_m - H_m) * _Y_)(0,0);
	R_sqr_l2_.set_val( 0, Idx, R_sqr );


	//
	// t-test
	if ( D_f == 0 )
	  {
	    // level 1
	    for ( int col = 0 ; col < constrast_.cols() ; col++ )
	      {
		Eigen::MatrixXd C = constrast_.block( 0, col,
						      constrast_.rows(), 1 );
		double T = ( C.transpose() * parameters )(0,0);
		// Record the parameters
		post_groups_param_.set_val( col, Idx, T );
		// T-score
		T /= sqrt( (C.transpose() * param_cov * C)(0,0) );
		// Record T map and PPM
		post_T_maps_.set_val( col, Idx, T );
		PPM_.set_val( col, Idx, Normal_CFD_(T) );
		//std::cout << "T = " << T << std::endl;
	      }
	    // level 2
	    for ( int col = 0 ; col < constrast_l2_.cols() ; col++ )
	      {
		Eigen::MatrixXd C = constrast_l2_.block( 0, col,
							 constrast_l2_.rows(), 1 );
		double T = ( C.transpose() * eta_theta_Y_2_theta_Y )(0,0);
		// Record the parameters
		post_groups_param_l2_.set_val( col, Idx, T );
		//std::cout << "COV \n" << cov_theta_Y_l2 << std::endl;
		//std::cout << "C \n" << C << std::endl;
		//std::cout << "eta \n" << eta_theta_Y_2_theta_Y << std::endl;
 		// T-score
		T /= sqrt( (C.transpose() * cov_theta_Y_l2 * C)(0,0) );
		// Record T map and PPM
		post_T_maps_l2_.set_val( col, Idx, T );
		PPM_l2_.set_val( col, Idx, Normal_CFD_(T) );
	      }
	    // Variance
	    for ( int row = 0 ; row < constrast_l2_.rows() ; row++ )
	      post_groups_cov_l2_.set_val( row, Idx, cov_theta_Y_l2(row,row) );
	  }

	  	
	int increme_subject = 0;
	for ( auto g : groups_ )
	  for ( auto subject : group_pind_[g] )
	    {
	      subject.second.set_fit( Idx,
				      parameters.block( increme_subject, 0, D_r, 1 ),
				      param_cov.block( increme_subject, increme_subject, 
						       D_r, D_r ) );
	      increme_subject += D_r;
	    }
	
	//std::cout << eta_theta_Y_2_eps_Y_dim << " " << eta_theta_Y_2_theta_Y_dim << std::endl;
	
	//std::cout << eta_theta_Y << std::endl;
	//std::cout << eta_theta_Y_2_theta_Y << std::endl;
	//std::cout  << std::endl;
	//std::cout  << std::endl;
	//std::cout << X2_ * eta_theta_Y_2_theta_Y + eta_theta_Y_2_eps_Y << std::endl;
	//std::cout  << std::endl;
	//std::cout << cov_theta_Y << std::endl;

	//
	// Prediction output
	// Inverse covariance error
	int 
	  index_pred_inv_error = 0,
	  pos_pred_inv_error   = 0;
	// C_eps_1_base: all images have the same error
	// we only save the first one
	Prediction_inverse_error_l2_.set_val( pos_pred_inv_error++, Idx, 
					      inv_Cov_eps(index_pred_inv_error,
							  index_pred_inv_error) );
	index_pred_inv_error += num_3D_images_;
	// C_eps_2: just the first subject covariates for each groups
	// Others subjects have the same covariance
	for ( auto g : groups_ )
	  {
	    // Dr
	    // We save only he D_r first parameters
	    // Others are the same
	    for ( int dr = 0 ; dr < D_r ; dr++ )
	      Prediction_inverse_error_l2_.set_val( pos_pred_inv_error++, Idx, 
						    inv_Cov_eps(index_pred_inv_error+dr,
								index_pred_inv_error+dr) );
	    // Df
	    if ( D_f > 0 )
	      for ( int df = 0 ; df < D_f ; df++ )
		Prediction_inverse_error_l2_.set_val( pos_pred_inv_error++, Idx, 
						      inv_Cov_eps(index_pred_inv_error+D_r+df,
								  index_pred_inv_error+D_r+df) );
	    //
	    index_pred_inv_error += group_num_subjects_[g]*D_r + D_f;
	  }
	// C_theta_2
	// all must be 1.e+16
	Prediction_inverse_error_l2_.set_val( pos_pred_inv_error, Idx, 
					      inv_Cov_eps(index_pred_inv_error,
							  index_pred_inv_error) );
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
    BmleLoadCSV< D_r, D_f >::Prediction( MaskType::IndexType Idx )
  {
    try
      {
	//
	// Inverse covariance value:
	MaskType4D::IndexType Idx_inv_cov = { Idx[0], Idx[1], Idx[2], 0 };

	for ( auto g : groups_ )
	  for ( auto s : group_pind_[g] )
	    s.second.prediction( Idx,
				 Prediction_inverse_error_->GetOutput()->GetPixel(Idx_inv_cov) );
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
	//std::cout << "residual = " << r.norm() << std::endl;

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
	// using cholesky decomposition
	int Cov_theta_Y_rows = Cov_theta_Y.rows();
	if ( true )
	  {
	    // ln |A| = 2 * sum_i ln(L_ii); where A=LL^{T}
	    // compute the Cholesky decomposition of A
	    Eigen::LLT< Eigen::MatrixXd > lltOf( Cov_theta_Y ); 
	    // retrieve factor L in the decomposition
	    Eigen::MatrixXd Lchol = lltOf.matrixL();
	    for ( int linco = 0 ; linco < Cov_theta_Y_rows ; linco++ )
	      {
		//std::cout << "Lchol(linco,linco) " << Lchol(linco,linco) << " ** log " << log( Lchol(linco,linco) )<< std::endl;
		F_4 += (Lchol(linco,linco) < 1.e-16 ? -37.: log( Lchol(linco,linco) ));
	      }
	    //
	    F_4 *= 2.;
	  }
	else
	  F_4 = NeuroBayes::ln_determinant( Cov_theta_Y );
	
	double
	  F_2 = - (r.transpose() * Inv_Cov_eps * r).trace(), // tr added for compilation reason
	  F_3 = - ( Cov_theta_Y * X_.transpose() * Inv_Cov_eps * X_ ).trace();
	std::cout << "mark_F1," << F_1 << "," << F_2 << "," << F_3 << "," << F_4  << "," << F_1 + F_2 + F_3 + F_4 << std::endl;
	//std::cout << "Inv_Cov_eps = " << Inv_Cov_eps << std::endl;
	//
	//
	//std::cout << "F = " << ( F_1 + F_2 + F_3 + F_4 ) * 0.5 << std::endl;
	return ( F_1 + F_2 + F_3 + F_4 ) * 0.5;
	//return F_2 * 0.5;
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
    BmleLoadCSV< D_r, D_f >::lambda_regulation_( std::map< int /*group*/, std::vector< double > >& Lambda ) 
    {    
      try
	{
	  for ( auto g : Lambda )
	    for ( auto& lambda_k_g_k : Lambda[g.first] )
	      {
		if ( lambda_k_g_k > 32 )
		  lambda_k_g_k = 16.;
		if ( lambda_k_g_k < -32 )
		  lambda_k_g_k = -16.;
		//std::cout << "lambda_k[" << g.first << "] = " << lambda_k_g_k << " " << exp(lambda_k_g_k)<< std::endl;
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
    BmleLoadCSV< D_r, D_f >::write_subjects_solutions( )
  {
    try
      {
	if ( !prediction_ )
	  {
	    //
	    std::cout << "Global solutions" << std::endl;
	    // level 1
	    PPM_.write();
	    // Posterior t-maps
	    post_T_maps_.write();
	    // Posterior groups parameters
	    post_groups_param_.write();
	    // level 2
	    PPM_l2_.write();
	    // Posterior t-maps
	    post_T_maps_l2_.write();
	    // Posterior groups parameters
	    post_groups_param_l2_.write();
	    // Posterior groups variance
	    post_groups_cov_l2_.write();
	    // R-square
	    R_sqr_l2_.write();
	    // Output for prediction
	    Prediction_inverse_error_l2_.write();
	  }
	//
	//
	std::cout << "Subjects solutions" << std::endl;
	for ( auto g : groups_ )
	  for ( auto subject : group_pind_[g] )
	    subject.second.write_solution();
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
	exit( -1 );
      }
  }
}
#endif
