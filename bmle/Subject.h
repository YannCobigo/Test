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
//
#include "BmleException.h"
//
//
//
namespace MAC_bmle
{
  inline bool file_exists ( const std::string& name )
  {
    std::ifstream f( name.c_str() );
    return f.good();
  }

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
      PIDN_{0}, group_{0}, D_{0} {};
      //
      explicit BmleSubject( const int, const int );
    
      /** Destructor */
      virtual ~BmleSubject(){};

      //
      // Accessors
      inline const int get_PIDN() const { return PIDN_ ;}
      //
      inline const std::map< int, Reader3D::Pointer >&
	get_age_images() const { return age_ITK_images_ ;}
      //
      const Eigen::MatrixXd& get_random_matrix() const {return X_1_rand_;}
      const Eigen::MatrixXd& get_fixed_matrix() const {return X_1_fixed_;}
      const Eigen::MatrixXd& get_X2_matrix() const {return X_2_;}

      //
      //
      void set_fit( const MaskType::IndexType, const Eigen::MatrixXd , const Eigen::MatrixXd );

      //
      // Add time point
      void add_tp( const int, const std::list< double >&, const std::string& );
      // Convariates' model
      void build_design_matrices( const double );
      // Print
      void print() const;

    private:
      //
      // private member function
      //

      //
      // Add time point
      void create_theta_images();


      //
      // Subject parameters
      //
    
      // Identification number
      int PIDN_;
      // Group for multi-group comparison (controls, MCI, FTD, ...)
      // It can only take 1, 2, ... value
      int group_;
      // 
      // Age covariate map
      std::map< int, std::list< double > > age_covariates_;
      //
      // Age image maps
      // age-image name
      std::map< int, std::string > age_images_; 
      // age-ITK image
      std::map< int, Reader3D::Pointer > age_ITK_images_; 
      //
      // Number of time points
      int time_points_{0};

      //
      // Model parameters
      //

      // Model dimension
      int D_;
      //
      // Level 1
      // Random matrix
      Eigen::MatrixXd X_1_rand_;
      // Fixed matrix
      Eigen::MatrixXd X_1_fixed_;
      // Second level design matrix, Matrix of covariates
      Eigen::MatrixXd X_2_;
      //
      // Random effect results
      BmleMakeITKImage Random_effect_ITK_model_;
      // Random effect results
      BmleMakeITKImage Random_effect_ITK_variance_;
    };

  //
  //
  //
  template < int D_r, int D_f >
  MAC_bmle::BmleSubject< D_r, D_f >::BmleSubject( const int Pidn,
						  const int Group):
  PIDN_{Pidn}, group_{Group}, D_{2}
  {
    /* 
       g(t, \theta_{i}^{(1)}) = \sum_{d=1}^{D+1} \theta_{i,d}^{(1)} t^{d-1}
    */
  }
  //
  //
  //
  template < int D_r, int D_f > void
    MAC_bmle::BmleSubject< D_r, D_f >::build_design_matrices( const double Age_mean )
  {
    try
      {
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
    MAC_bmle::BmleSubject< D_r, D_f >::add_tp( const int                  Age,
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
		throw MAC_bmle::BmleException( __FILE__, __LINE__,
					       mess.c_str(),
					       ITK_LOCATION );
	      }
	    //
	    time_points_++;
	  }
	else
	  {
	    std::string mess = "Age " + std::to_string(Age) + " is already entered for the patient ";
	    mess += std::to_string(PIDN_) + ".";
	    //
	    throw MAC_bmle::BmleException( __FILE__, __LINE__,
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
    MAC_bmle::BmleSubject< D_r, D_f >::set_fit( const MaskType::IndexType Idx, 
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
    MAC_bmle::BmleSubject< D_r, D_f >::create_theta_images()
    {
      //std::cout << "We create output only one time" << std::endl;
      // Model output
      std::string output_model = "model_" 
	+ std::to_string( PIDN_ ) + "_" + std::to_string( group_ )
	+ "_" + std::to_string( time_points_ ) + "_" + std::to_string( D_ ) 
	+ ".nii.gz";
      Random_effect_ITK_model_ = BmleMakeITKImage( D_r,
						   output_model,
						   age_ITK_images_.begin()->second );
      // Variance output
      // We only record the diagonal sup elements
      //
      // | 1 2 3 |
      // | . 4 5 |
      // | . . 6 |
      std::string output_var = "var_" 
	+ std::to_string( PIDN_ ) + "_" + std::to_string( group_ )
	+ "_" + std::to_string( time_points_ ) + "_" + std::to_string( D_ ) 
	+ ".nii.gz";
      Random_effect_ITK_variance_ = BmleMakeITKImage( D_r * (D_r + 1) / 2 /*we make sure it is a int*/,
						      output_var,
						      age_ITK_images_.begin()->second );
    }
  //
  //
  //
  template < int D_r, int D_f > void
    MAC_bmle::BmleSubject< D_r, D_f >::print() const
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
}
#endif
