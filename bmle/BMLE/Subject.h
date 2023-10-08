#ifndef BMLESUBJECT_H
#define BMLESUBJECT_H
//
//
//
#include <iostream>
#include <fstream>
#include <math.h>  
#include <map>
#include <list>
#include <memory>
// Egen
#include <Eigen/Core>
#include <Eigen/Eigen>
//
// ITK
//
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIterator.h>
#include <itkNiftiImageIO.h>
#include <itkOrientImageFilter.h>
#include <itkSpatialOrientation.h>
//
//
#define inv_sqrt_2         0.70710678118654746
#define inv_two_pi_squared 0.3989422804014327L
//
//
//
#include "Exception.h"
#include "Tools.h"
//
//
//
//
//
//
namespace NeuroBayes
{
  /** \class BmleSubject
   *
   * \brief 
   * D_r: number of random degres of the model.
   * D_f: number of fixed degres of the model.
   * 
   */
  template< int D_r, int D_f >
    class BmleSubject
  {
    //
    // Some typedef
    using Image3DType = itk::Image< double, 3 >;
    using Reader3D    = itk::ImageFileReader< Image3DType >;
    using MaskType    = itk::Image< unsigned char, 3 >;
 
  public:
    /** Constructor. */
  BmleSubject():
    PIDN_{""}, group_{0} {};
    //
    explicit BmleSubject( const std::string, const int, const std::string& );
    
    /** Destructor */
    virtual ~BmleSubject(){};

    //
    // Accessors
    inline const std::string get_PIDN() const { return PIDN_ ;}
    //
    inline const std::map< int, Reader3D::Pointer >&
      get_age_images() const { return age_ITK_images_ ;}
    //
    const Eigen::MatrixXd& get_random_matrix() const {return X_1_rand_;}
    const Eigen::MatrixXd& get_fixed_matrix() const {return X_1_fixed_;}
    const Eigen::MatrixXd& get_X2_matrix() const {return X_2_;}

    //
    // Write the fitted solution to the output image pointer
    void set_fit( const MaskType::IndexType, const Eigen::MatrixXd , const Eigen::MatrixXd );
    //
    // Write the output matrix: fitted parameters and the covariance matrix
    void write_solution();

    //
    // Add time point
    void add_tp( const int, const std::list< double >&, const std::string& );
    //
    // Convariates' model
    // for none and demean
    void build_design_matrices( const double );
    // for normailization and standardization
    void build_design_matrices( const double, const double );
    // Print
    void print() const;

    //
    // Prediction
    // load the model posterior parameters and covariance
    void load_model_matrices();
    // Process the prediction
    void prediction( const MaskType::IndexType, 
		     const double );
    //
    // W-score
    // load the fitted posterior parameters
    void load_fitted_parameters( const std::string, const int );
    // Process the longitudinal w-score
    void w_score( const MaskType::IndexType, 
		  const double );
      

  private:
    //
    // private member function
    //

    //
    // Add time point
    void create_theta_images();
    //
    // output directory
    std::string output_dir_;
    //
    // Statistics: age transformation
    double 
      C1_{0.},
      C2_{0.};


    //
    // Subject parameters
    //
    
    // Identification number
    std::string PIDN_;
    // Group for multi-group comparison (controls, MCI, FTD, ...)
    // It can only take 1, 2, ... value
    int group_;
    // 
    // Age covariate map
    std::map< int, std::list< double > > age_covariates_;
    //
    // Age image maps
    // age-image name
    std::map< int, std::string >       age_images_; 
    // age-ITK image
    std::map< int, Reader3D::Pointer > age_ITK_images_; 
    //
    // Number of time points
    int time_points_{0};

    //
    // Prediction
    bool prediction_{false};
    // W-score
    bool w_score_{false};

    //
    // Model parameters
    //

    //
    // Level 1
    // Random matrix
    Eigen::MatrixXd X_1_rand_;
    // Fixed matrix
    Eigen::MatrixXd X_1_fixed_;
    // Second level design matrix, Matrix of covariates
    Eigen::MatrixXd X_2_;
    //
    // Prediction
    // Posterior parameters
    Eigen::Matrix< double, D_r, 1 >    theta_y_;
    // Posterior covariance
    Eigen::Matrix< double, D_r, D_r >  cov_theta_y_;
    //
    // Random effect results
    NeuroBayes::NeuroBayesMakeITKImage Random_effect_ITK_model_;
    // Random effect results
    NeuroBayes::NeuroBayesMakeITKImage Random_effect_ITK_variance_;
    //
    // Prediction
    NeuroBayes::NeuroBayesMakeITKImage Probability_prediction_map_;
    // error function
    NeuroBayes::NeuroBayesMakeITKImage Error_prediction_map_;
    //
    // Longitudinal W-score for each groups
    // L2 posterior fitted parameters
    NeuroBayes::NeuroBayesMakeITKImage l2_posterior_parameters_map_;
    // Number of posterior fitted parameters 
    long unsigned int                  number_fitted_parameters_{0};
    // Number of covariates
    int                                number_covariates_{0};
    // Number of parameters per group in the posterior fitted process
    int                                number_param_per_group_{0};
    // Number of groups in the posterior fitted proces
    int                                number_groups_{0};
    // w-score
    NeuroBayes::NeuroBayesMakeITKImage groups_w_map_;
  };

  //
  //
  //
  template < int D_r, int D_f >
    NeuroBayes::BmleSubject< D_r, D_f >::BmleSubject( const std::string Pidn,
						      const int Group,
						      const std::string& Output_dir ):
    PIDN_{Pidn}, group_{Group}, output_dir_{Output_dir}
  {
    /* 
       g(t, \theta_{i}^{(1)}) = \sum_{d=1}^{D+1} \theta_{i,d}^{(1)} t^{d-1}
    */
  }
  //
  //
  //
  template < int D_r, int D_f > void
    NeuroBayes::BmleSubject< D_r, D_f >::build_design_matrices( const double Age_mean )
    {
      try
	{
	  std::cout << "DEMEAN AGE: " << Age_mean << std::endl;
	  C1_ = Age_mean;
	  //
	  // Design matrix level 1
	  //

	  //
	  //
	  int num_lignes    = age_images_.size();
	  //
	  X_1_rand_.resize(  num_lignes, D_r );
	  X_1_fixed_.resize( num_lignes, D_f );
	  X_1_rand_  = Eigen::MatrixXd::Zero( num_lignes, D_r );
	  X_1_fixed_ = Eigen::MatrixXd::Zero( num_lignes, D_f );
	  // record ages
	  std::vector< double > ages;
	  for ( auto age : age_images_ )
	    ages.push_back( age.first );
	  // random part of the design matrix
	  for ( int l = 0 ; l < num_lignes ; l++ )
	    {
	      for ( int c = 0 ; c <  D_r ; c++ )
		X_1_rand_(l,c) = pow( ages[l] - Age_mean, c );
	      // fixed part of the design matrix
	      for ( int c = 0 ; c <  D_f ; c++ )
		X_1_fixed_(l,c) = pow( ages[l] - Age_mean, D_r + c );
	    }

	  std::cout << "Random and fixed design matrices:" << std::endl;
	  std::cout << X_1_rand_ << std::endl;
	  std::cout << X_1_fixed_ << std::endl;
	  std::cout << std::endl;
	
	  //
	  // Design matrix level 2
	  //

	
	  //
	  // Initialize the covariate matrix
	  // and the random parameters
	  std::map< int, std::list< double > >::const_iterator age_cov_it = age_covariates_.begin();
	  //
	  X_2_.resize( D_r, ((*age_cov_it).second.size() + 1) * D_r );
	  X_2_ = Eigen::MatrixXd::Zero( D_r, ((*age_cov_it).second.size() + 1)* D_r  );
	  //
	  //
	  int line = 0;
	  int col  = 0;
	  X_2_.block< D_r, D_r >( 0, 0 ) = Eigen::MatrixXd::Identity( D_r, D_r );
	  // covariates
	  for ( auto cov : (*age_cov_it).second )
	    X_2_.block< D_r, D_r >( 0, ++col * D_r ) = cov * Eigen::MatrixXd::Identity( D_r, D_r );
      
	  std::cout << X_2_ << std::endl;

	  //
	  // Prediction
	  //
	
	  // Posterior parameters
	  theta_y_ = Eigen::Matrix< double, D_r, 1 >::Zero( D_r );
	  // Posterior covariance
	  cov_theta_y_ = Eigen::Matrix< double, D_r, D_r >::Zero( D_r, D_r );
	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	  return exit( -1 );
	}
    }
  //
  //     norm       -- Std
  // C1: min           mu (mean)
  // C2: (max -min)    sigma (standard dev)
  //
  template < int D_r, int D_f > void
    NeuroBayes::BmleSubject< D_r, D_f >::build_design_matrices( const double C1,
								const double C2 )
    {
      try
	{
	  std::cout 
	    << "NORM/STD: (" << C1 << "," << C2 << ")." << std::endl;
	  //
	  C1_ = C1;
	  C2_ = C2;
	  //
	  // Design matrix level 1
	  //

	  //
	  //
	  int num_lignes    = age_images_.size();
	  //
	  X_1_rand_.resize(  num_lignes, D_r );
	  X_1_fixed_.resize( num_lignes, D_f );
	  X_1_rand_  = Eigen::MatrixXd::Zero( num_lignes, D_r );
	  X_1_fixed_ = Eigen::MatrixXd::Zero( num_lignes, D_f );
	  // record ages
	  std::vector< double > ages;
	  for ( auto age : age_images_ )
	    ages.push_back( age.first );
	  // random part of the design matrix
	  for ( int l = 0 ; l < num_lignes ; l++ )
	    {
	      for ( int c = 0 ; c <  D_r ; c++ )
		X_1_rand_(l,c) = pow( (ages[l] - C1) / C2, c );
	      // fixed part of the design matrix
	      for ( int c = 0 ; c <  D_f ; c++ )
		X_1_fixed_(l,c) = pow( (ages[l] - C1) / C2, D_r + c );
	    }

	  std::cout << "Random and fixed design matrices:" << std::endl;
	  std::cout << X_1_rand_ << std::endl;
	  std::cout << X_1_fixed_ << std::endl;
	  std::cout << std::endl;
	
	  //
	  // Design matrix level 2
	  //

	
	  //
	  // Initialize the covariate matrix
	  // and the random parameters
	  std::map< int, std::list< double > >::const_iterator age_cov_it = age_covariates_.begin();
	  //
	  X_2_.resize( D_r, ((*age_cov_it).second.size() + 1) * D_r );
	  X_2_ = Eigen::MatrixXd::Zero( D_r, ((*age_cov_it).second.size() + 1)* D_r  );
	  //
	  //
	  int line = 0;
	  int col  = 0;
	  X_2_.block< D_r, D_r >( 0, 0 ) = Eigen::MatrixXd::Identity( D_r, D_r );
	  // covariates
	  for ( auto cov : (*age_cov_it).second )
	    X_2_.block< D_r, D_r >( 0, ++col * D_r ) = cov * Eigen::MatrixXd::Identity( D_r, D_r );
      
	  std::cout << X_2_ << std::endl;

	  //
	  // Prediction
	  //
	
	  // Posterior parameters
	  theta_y_ = Eigen::Matrix< double, D_r, 1 >::Zero( D_r );
	  // Posterior covariance
	  cov_theta_y_ = Eigen::Matrix< double, D_r, D_r >::Zero( D_r, D_r );
	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	  return exit( -1 );
	}
    }
  //
  //
  //
  template < int D_r, int D_f > void
    NeuroBayes::BmleSubject< D_r, D_f >::add_tp( const int                  Age,
						 const std::list< double >& Covariates,
						 const std::string&         Image )
    {
      try
	{
	  if ( age_covariates_.find( Age ) == age_covariates_.end() )
	    {
	      age_covariates_[ Age ] = Covariates;
	      age_images_[ Age ]     = Image;
	      //
	      // load the ITK images
	      if ( file_exists(Image) )
		{
		  //
		  // load the image ITK pointer
		  auto image_ptr = itk::ImageIOFactory::CreateImageIO( Image.c_str(),
								       itk::ImageIOFactory::ReadMode );
		  image_ptr->SetFileName( Image );
		  image_ptr->ReadImageInformation();
		  // Read the ITK image
		  age_ITK_images_[ Age ] = Reader3D::New();
		  age_ITK_images_[ Age ]->SetFileName( image_ptr->GetFileName() );
		  age_ITK_images_[ Age ]->Update();
		  // create the result image, only one time
		  if ( age_ITK_images_.size() < 2 )
		    create_theta_images();
		}
	      else
		{
		  std::string mess = "Image " + Image + " does not exists.";
		  throw NeuroBayes::NeuroBayesException( __FILE__, __LINE__,
							 mess.c_str(),
							 ITK_LOCATION );
		}
	      //
	      time_points_++;
	    }
	  else
	    {
	      std::string mess = "Age " + std::to_string(Age) + " is already entered for the patient ";
	      mess += PIDN_ + ".";
	      //
	      throw NeuroBayes::NeuroBayesException( __FILE__, __LINE__,
						     mess.c_str(),
						     ITK_LOCATION );
	    }
	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	  return exit( -1 );
	}
    }
  //
  //
  //
  template < int D_r, int D_f > void
    NeuroBayes::BmleSubject< D_r, D_f >::set_fit( const MaskType::IndexType Idx, 
						  const Eigen::MatrixXd Model_fit, 
						  const Eigen::MatrixXd Cov_fit )
    {
      //
      // ToDo: I would like to write the goodness of the score (r-square ...)
      //
      // copy Eigen Matrix information into a vector
      // We only record the diagonal sup of the covariance.
      std::vector< double > model( D_r ), cov( D_r * (D_r + 1) / 2 );
      int current_mat_coeff = 0;
      for ( int d ; d < D_r ; d++ )
	{
	  model[d] = Model_fit(d,0);
	  Random_effect_ITK_model_.set_val( d, Idx, Model_fit(d,0) );
	  for ( int c = d ; c < D_r ; c++)
	    {
	      cov[d]  = Cov_fit(d,c);
	      Random_effect_ITK_variance_.set_val( current_mat_coeff++, Idx, Cov_fit(d,c) );
	    }
	}
    }
  //
  //
  //
  template < int D_r, int D_f > void
    NeuroBayes::BmleSubject< D_r, D_f >::create_theta_images()
    {
      //std::cout << "We create output only one time" << std::endl;
      // Model output
      std::string output_model = output_dir_ + "/" + "model_" 
	+ PIDN_ + "_" + std::to_string( group_ )
	+ ".nii.gz";
      Random_effect_ITK_model_ = NeuroBayes::NeuroBayesMakeITKImage( D_r,
								     output_model,
								     age_ITK_images_.begin()->second);
      // Variance output
      // We only record the diagonal sup elements
      //
      // | 1 2 3 |
      // | . 4 5 |
      // | . . 6 |
      std::string output_var = output_dir_ + "/" + "var_" 
	+ PIDN_ + "_" + std::to_string( group_ )
	+ ".nii.gz";
      Random_effect_ITK_variance_ = NeuroBayes::NeuroBayesMakeITKImage( D_r * (D_r + 1) / 2 /*we make sure it is a int*/,
									output_var,
									age_ITK_images_.begin()->second );
    }
  //
  //
  //
  template < int D_r, int D_f > void
    NeuroBayes::BmleSubject< D_r, D_f >::write_solution()
    {
      std::cout 
	<< "prediction_ " << prediction_
	<< " w_score_ " << w_score_
	<< " iamge name: " << groups_w_map_.get_name() 
	<< std::endl;
      if ( prediction_ )
	{
	  Probability_prediction_map_.write();
	  Error_prediction_map_.write();
	}
      else if ( w_score_ )
	{
	  groups_w_map_.write();
	}
      else
	{
	  Random_effect_ITK_model_.write();
	  Random_effect_ITK_variance_.write();
	}
    }
  //
  //
  //
  template < int D_r, int D_f > void
    NeuroBayes::BmleSubject< D_r, D_f >::print() const
    {
      std::cout << "PIDN: " << PIDN_ << std::endl;
      std::cout << "Group: " << group_ << std::endl;
      std::cout << "Number of time points: " << time_points_ << std::endl;
      //
      std::cout << "Age and covariates: " << std::endl;
      if ( !age_covariates_.empty() )
	for ( auto age_cov : age_covariates_ )
	  {
	    std::cout << "At age " << age_cov.first << " covariates were:";
	    for( auto c : age_cov.second )
	      std::cout << " " << c;
	    std::cout << std::endl;
	  }
      else
	std::cout << "No age and covariates recorded." << std::endl;
      //
      std::cout << "Age and imagess: " << std::endl;
      if ( !age_images_.empty() )
	for ( auto age_img : age_images_ )
	  std::cout << "At age " << age_img.first << " iamge was: "
		    << age_img.second << std::endl;
      else
	std::cout << "No age and images recorded." << std::endl;
    }
  //
  //
  //
  template < int D_r, int D_f > void
    NeuroBayes::BmleSubject< D_r, D_f >::load_model_matrices() 
    {
      try
	{
	  //
	  // posterior maps are loaded if the predictions is called
	  prediction_ = true;
	  std::cout << "Loading model matrices for: " << PIDN_ 
		    << ", Group: " << group_ 
		    << ". Number of new time points: " << time_points_ << std::endl;

	  //
	  // Creating Output prediction image
	  std::string prediction = output_dir_ + "/" + "prediction_" 
	    + PIDN_ + "_" + std::to_string( group_ )
	    + ".nii.gz";
	  std::string error_prediction = output_dir_ + "/" + "erf_prediction_" 
	    + PIDN_ + "_" + std::to_string( group_ )
	    + ".nii.gz";
	  //
	  Probability_prediction_map_ = NeuroBayes::NeuroBayesMakeITKImage( time_points_,
									    prediction,
									    age_ITK_images_.begin()->second );
	  Error_prediction_map_ = NeuroBayes::NeuroBayesMakeITKImage( time_points_,
								      error_prediction,
								      age_ITK_images_.begin()->second );

	  //
	  // Load the matrices
	  //

	  //
	  // Model output
	  std::string output_model = output_dir_ + "/" + "model_" 
	    + PIDN_ + "_" + std::to_string( group_ )
	    + ".nii.gz";
	  if ( access( output_model.c_str(), F_OK ) != -1 )
	    Random_effect_ITK_model_ = NeuroBayes::NeuroBayesMakeITKImage( D_r, output_model );
	  else
	    {
	      std::string mess = "The posterior parameters have not been generated for ";
	      mess += PIDN_ + ".\n";
	      mess += "Looking for: " + output_model;
	      //
	      throw NeuroBayes::NeuroBayesException( __FILE__, __LINE__,
						     mess.c_str(),
						     ITK_LOCATION );
	    }
      
	  //
	  // Variance output
	  // We only record the diagonal sup elements
	  //
	  // | 1 2 3 |
	  // | . 4 5 |
	  // | . . 6 |
	  std::string output_var = output_dir_ + "/" + "var_" 
	    + PIDN_ + "_" + std::to_string( group_ )
	    + ".nii.gz";
	  if ( access( output_var.c_str(), F_OK ) != -1 )
	    Random_effect_ITK_variance_ = NeuroBayes::NeuroBayesMakeITKImage( D_r * (D_r + 1) / 2, output_var );
	  else
	    {
	      std::string mess = "The posterior parameters have not been generated for ";
	      mess += PIDN_ + ".";
	      mess += "Looking for: " + output_model;
	      //
	      throw NeuroBayes::NeuroBayesException( __FILE__, __LINE__,
						     mess.c_str(),
						     ITK_LOCATION );
	    }
	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	  return exit( -1 );
	}
    }
  //
  //
  //
  template < int D_r, int D_f > void
    NeuroBayes::BmleSubject< D_r, D_f >::prediction( const MaskType::IndexType Idx, 
						     const double Inv_C_eps )
    {
      try
	{
	  //std::cout << "In subject: " << Idx << " val inv: " << Inv_C_eps << std::endl;
	  //std::cout << PIDN_ 
	  //	    << ", Group: " << group_ 
	  //	    << ". Number of new time points: " << time_points_ << std::endl;


	  //
	  // Load the posterior maps
	  int current_mat_coeff = 0;
	  for ( int d ; d < D_r ; d++ )
	    {
	      // Parameters
	      theta_y_(d,0) = Random_effect_ITK_model_.get_val( d, Idx );
	      // Covariance
	      for ( int c = d ; c < D_r ; c++)
		cov_theta_y_(d,c) = cov_theta_y_(c,d) = 
		  Random_effect_ITK_variance_.get_val( current_mat_coeff++, Idx );
	    }
      
	  //
	  // Process the prediction
	  std::map< int, Reader3D::Pointer >::const_iterator age_img = age_ITK_images_.begin();
	  for ( int tp = 0 ; tp < time_points_ ; tp++ )
	    {
	      //
	      // Design
	      Eigen::Matrix< double, D_r, 1 > x = X_1_rand_.row(tp).transpose();
	      double age_statistic = 0.;
	      if ( C2_ != 0.)
		// Normalization or standardization
		age_statistic = (static_cast<double>(age_img->first) - C1_) / C2_;
	      else
		// None or demean
		age_statistic = static_cast<double>(age_img->first) - C1_;
	      //
	      // check the order of ages
	      if ( age_statistic != x(1,0) )
		{
		  std::string mess = "Order of ages is not correct. Expected age: ";
		  mess += std::to_string( age_img->first ) + " and received age: ";
		  mess += std::to_string( x(1,0) );
		  //
		  throw NeuroBayes::NeuroBayesException( __FILE__, __LINE__,
							 mess.c_str(),
							 ITK_LOCATION );
		}
	      // response
	      double y = static_cast<double>( (age_img++)->second->GetOutput()->GetPixel( Idx ) );
	      
	      // variance
	      double variance  = 1. / Inv_C_eps + (x.transpose() * cov_theta_y_ * x)(0,0);
	      // Mean value
	      double mu = (x.transpose() * theta_y_)(0,0);
	      // argument of the Gaussian
	      double arg = - 0.5 * (y-mu) * (y-mu) / variance;
	      //
	      // record the value
	      Probability_prediction_map_.set_val( tp, Idx,
						   exp(arg) * inv_two_pi_squared / sqrt(variance) );
	      Error_prediction_map_.set_val( tp, Idx,
					     erf( inv_sqrt_2 * (y-mu) / sqrt(variance) ) );
 
	      //	      std::cout 
	      //				<< "theta_y_\n" << theta_y_
	      //		//		<< "\n C_eps_ = " <<  1. / Inv_C_eps
	      //		//		<< "\n variance_ = " <<  variance
	      //				<< "\n cov_ = \n" <<  cov_theta_y_
	      //		<< "\ny  = " <<  y
	      //		<< "\nmu  = " <<  mu
	      //		<< std::endl;
	    }
	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	  return exit( -1 );
	}
    }
  //
  //
  //
  template < int D_r, int D_f > void
    NeuroBayes::BmleSubject< D_r, D_f >::load_fitted_parameters( const std::string L2_param,
								 const int         Covariates) 
    {
      try
	{
	  w_score_ = true;
	  //
	  l2_posterior_parameters_map_ = NeuroBayes::NeuroBayesMakeITKImage( L2_param );
	  number_fitted_parameters_    = l2_posterior_parameters_map_.get_number_maps();
	  number_covariates_           = Covariates;
	  number_param_per_group_      = D_r * (Covariates+1);
	  number_groups_               = static_cast<int>( std::sqrt(number_fitted_parameters_ / number_param_per_group_) );
	  // Image model
	  std::string output_model = output_dir_ + "/" + "w-score_" 
	    + PIDN_ + "_" + std::to_string( group_ )
	    + ".nii.gz";
	  // Initiate the w-scores based on the background populations
	  groups_w_map_ = NeuroBayes::NeuroBayesMakeITKImage( number_groups_*time_points_,
							      output_model,
							      age_ITK_images_.begin()->second);

	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	  return exit( -1 );
	}
    }
  //
  //
  //
  template < int D_r, int D_f > void
    NeuroBayes::BmleSubject< D_r, D_f >::w_score( const MaskType::IndexType                 Idx, 
						  const double                              Inv_C_eps )
    {
      try
	{
	  //std::cout << "In subject: " << Idx << " val inv: " << Inv_C_eps << std::endl;
	  //std::cout << PIDN_ 
	  //	    << ", Group: " << group_ 
	  //	    << ". Number of new time points: " << time_points_ 
	  //	    << std::endl;

	  //
	  //std::cout 
	  //  << "number_fitted_parameters_: "                << number_fitted_parameters_
	  //  << "   number covariate(s): "     << number_covariates_
	  //  << "   number group(s): "         << number_groups_
	  //  << "   number param per group: " << number_param_per_group_
	  //  << std::endl;
      
	  //
	  // Vector of parameters
	  Eigen::MatrixXd beta = Eigen::MatrixXd::Zero( number_param_per_group_, number_groups_ );
	  // load the fitted parameters for each group
	  for ( int gr = 0 ; gr < number_groups_ ; gr++ )
	    for ( int d = 0 ; d < number_param_per_group_ ; d++ )
	      {
		const std::size_t img_num = gr * number_groups_ * number_param_per_group_ + d;
		beta(d,gr) = l2_posterior_parameters_map_.get_val( img_num, Idx );
	      }
	  //
	  //std::cout 
	  //  << "beta \n" <<  beta
	  //  << std::endl;

	  //
	  // Explanatory variable are baseline information and does not depend on the timepoint
	  // The EV should already be normalized/standardized or demean
	  Eigen::MatrixXd X = X_1_rand_ * X_2_;
	  // check The dimensions are fine
//	  if ( age_statistic != x(1,0) )
//	    {
//	      std::string mess = "Order of ages is not correct. Expected age: ";
//	      mess += std::to_string( age_img->first ) + " and received age: ";
//	      mess += std::to_string( x(1,0) );
//	      //
//	      throw NeuroBayes::NeuroBayesException( __FILE__, __LINE__,
//						     mess.c_str(),
//						     ITK_LOCATION );
//	    }


	  //
	  // fits
	  Eigen::MatrixXd Fitted_Y_tp = X * beta;
	  //std::cout 
	  //  << "y \n" << Fitted_Y_tp
	  //  << std::endl;
	  // Process the w-score
	  std::map< int, Reader3D::Pointer >::const_iterator age_img = age_ITK_images_.begin();
	  for ( int tp = 0 ; tp < time_points_ ; tp++ )
	    {
	      // response
	      double y = static_cast<double>( (age_img++)->second->GetOutput()->GetPixel( Idx ) );
	      // timepoint fit
	      for ( int gr = 0 ; gr < number_groups_ ; gr++ )
		{
		  double w_score = ( y - Fitted_Y_tp(tp,gr) ) * std::sqrt( Inv_C_eps );
		  groups_w_map_.set_val( gr * time_points_ + tp, 
					 Idx, w_score );
		  //std::cout 
		  //  << "gr: " << gr
		  //  << " tp: " << tp
		  //  << " y: " << y
		  //  << " Fitted_Y_tp: " << Fitted_Y_tp(tp,gr)
		  //  << " w_score: " << w_score
		  //  << std::endl;
		}
	    }
	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	  return exit( -1 );
	}
    }
}
#endif
