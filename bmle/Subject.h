#ifndef BMLESUBJECT_H
#define BMLESUBJECT_H
//
//
//
#include <iostream>
#include <fstream>
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
   * 
   */
  class BmleSubject
  {
    //
    // Some typedef
    using Image3DType = itk::Image< float, 3 >;
    using Reader3D    = itk::ImageFileReader< Image3DType >;

  public:
    /** Constructor. */
  BmleSubject():
    PIDN_{0}, group_{0}, D_{0} {};
    //
    explicit BmleSubject( const int, const int );
    
    /**  */
    virtual ~BmleSubject(){};

    //
    //
    inline const int get_PIDN() const { return PIDN_ ;};
    //
    //
    inline const std::map< int, Reader3D::Pointer >&
      get_age_images() const { return age_ITK_images_ ;};

    //
    // Add time point
    void add_tp( const int, const std::list< float >&, const std::string& );
    // Convariates' model
    void build_covariates_matrix();
    //
    // Print
    inline void print()
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

  private:
    //
    // Subject parameters
    //
    
    // Identification number
    int PIDN_;
    // Group for multi-group comparisin (controls, MCI, FTD, ...)
    int group_;
    // 
    // Age covariate map
    std::map< int, std::list< float > > age_covariates_;
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
    // Matrix of covariates
    Eigen::MatrixXf covariates_;
    //
    // Random effect
    // theta level 1
    Eigen::VectorXf theta_1_;
    // non-lin model in function of age
    Eigen::VectorXf model_age_;
  };
}
#endif
