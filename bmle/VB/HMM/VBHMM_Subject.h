#ifndef VBHMMSUBJECT_H
#define VBHMMSUBJECT_H
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
#include "Exception.h"
//
//
//
namespace VB
{
  namespace HMM
  {
    inline bool file_exists ( const std::string& name )
    {
      std::ifstream f( name.c_str() );
      return f.good();
    }

    /** \class Subject
     *
     * \brief 
     * Dim: dimension of the model.
     * Num_States: number of states accessible to the Markov Chain.
     * 
     */
    template< int Dim, int Num_States >
      class Subject
    {
      //
      // Some typedef
      using Image4DType = itk::Image< double, 4 >;
      using Reader4D    = itk::ImageFileReader< Image4DType >;
      using MaskType    = itk::Image< unsigned char, 3 >;
 
    public:
      /** Constructor. */
      Subject(){};
      //
      explicit Subject( const std::string, const std::string& );
    
      /** Destructor */
      virtual ~Subject(){};

      //
      // Accessors
      inline const std::string get_PIDN() const { return PIDN_ ;}
      //
      inline const std::map< int, Reader4D::Pointer >&
	get_age_images() const { return age_ITK_images_ ;}

      //
      // Write the fitted solution to the output image pointer
      void build_time_series( const MaskType::IndexType, 
			      std::vector< Eigen::Matrix< double, Dim, 1 > >&,
			      std::vector< Eigen::Matrix< double, 1, 1 > >& );
 
      //
      // Write the output matrix: fitted parameters and the covariance matrix
      void write_solution();

      //
      // Add time point
      void add_tp( const int, const std::list< double >&, const std::string& );
      // Print
      void print() const;


    private:
      //
      // private member function
      //

      //
      // output directory
      std::string output_dir_;


      //
      // Subject parameters
      //
    
      // Identification number
      std::string PIDN_;
      // 
      // Age covariate map
      std::map< int, std::list< double > > age_covariates_;
      //
      // Age image maps
      // age-image name
      std::map< int, std::string >       age_images_; 
      // age-ITK image
      std::map< int, Reader4D::Pointer > age_ITK_images_; 
      //
      // Number of time points
      int time_points_{0};

      //
      // Model parameters
      //

    };

    //
    //
    //
    template < int Dim, int Num_States >
      VB::HMM::Subject< Dim, Num_States >::Subject( const std::string Pidn,
						    const std::string& Output_dir ):
      PIDN_{Pidn}, output_dir_{Output_dir}
    {}
    //
    //
    //
    template < int Dim, int Num_States > void
      VB::HMM::Subject< Dim, Num_States >::add_tp( const int                  Age,
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
		    age_ITK_images_[ Age ] = Reader4D::New();
		    age_ITK_images_[ Age ]->SetFileName( image_ptr->GetFileName() );
		    age_ITK_images_[ Age ]->Update();
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
    template < int Dim, int Num_States > void
       VB::HMM::Subject< Dim, Num_States >::build_time_series( const MaskType::IndexType                       Idx, 
							       std::vector< Eigen::Matrix< double, Dim, 1 > >& Intensity,
							       std::vector< Eigen::Matrix< double, 1, 1 > >&   Age )
    {
    }
    //
    //
    //
    template < int Dim, int Num_States > void
      VB::HMM::Subject< Dim, Num_States >::write_solution()
      {
	//	Random_effect_ITK_model_.write();
	//	Random_effect_ITK_variance_.write();
      }
    //
    //
    //
    template < int Dim, int Num_States > void
      VB::HMM::Subject< Dim, Num_States >::print() const
      {
	//	std::cout << "PIDN: " << PIDN_ << std::endl;
	//	std::cout << "Group: " << group_ << std::endl;
	//	std::cout << "Number of time points: " << time_points_ << std::endl;
	//	//
	//	std::cout << "Age and covariates: " << std::endl;
	//	if ( !age_covariates_.empty() )
	//	  for ( auto age_cov : age_covariates_ )
	//	    {
	//	      std::cout << "At age " << age_cov.first << " covariates were:";
	//	      for( auto c : age_cov.second )
	//		std::cout << " " << c;
	//	      std::cout << std::endl;
	//	    }
	//	else
	//	  std::cout << "No age and covariates recorded." << std::endl;
	//	//
	//	std::cout << "Age and imagess: " << std::endl;
	//	if ( !age_images_.empty() )
	//	  for ( auto age_img : age_images_ )
	//	    std::cout << "At age " << age_img.first << " iamge was: "
	//		      << age_img.second << std::endl;
	//	else
	//	  std::cout << "No age and images recorded." << std::endl;
      }
  }
}
#endif
