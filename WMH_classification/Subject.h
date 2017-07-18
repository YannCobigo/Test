#ifndef SUBJECT_H
#define SUBJECT_H
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
#include "MACException.h"
#include "MACMakeITKImage.h"
//
//
//
namespace MAC
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
  template< int Dim >
    class Subject
    {
      //
      // Some typedef
      using Image3DType = itk::Image< double, 3 >;
      using Reader3D    = itk::ImageFileReader< Image3DType >;
      using MaskType    = itk::Image< unsigned char, 3 >;
 
    public:
      /** Constructor. */
    Subject():id_(-1){};
      /** Constructor. */
      Subject( const int );
    
      /** Destructor */
      virtual ~Subject(){};

      //
      // Accessors

      //
      // Write the output matrix: fitted parameters and the covariance matrix
      void write_solution();

      //
      // Get voxels
      //
      std::vector< double > get_modalities( const MaskType::IndexType ) const;
      //
      int get_label( const MaskType::IndexType ) const;
      //
      void set_fit( const MaskType::IndexType, 
		    const Eigen::MatrixXd, 
		    const Eigen::MatrixXd );
      // Add modality
      void add_modality( const int, const int );
      // Add label
      void add_label( const int );
      //
      void create_output_map();
      // get a sample image for image information
      Reader3D::Pointer get_sample() const
	{return modality_images_[0];};
      // Print
      void print() const;

    private:
      //
      // private members
      //

      //
      // Subject inner Id
      int id_;
      // Modality images
      std::vector< Reader3D::Pointer > modality_images_{ Dim }; 
      // Label
      Reader3D::Pointer                label_; 
      //
      // Output probability map and R2
      //MACMakeITKImage probability_map_;
    };
  
  //
  //
  //
  template < int D >
    MAC::Subject< D >::Subject( const int Id):id_{Id}
  {
  }
  //
  //
  //
  template < int D > void
    MAC::Subject< D >::add_label( const int Subject )
    {
      try
	{
	  if ( Subject == id_ )
	    {
	      std::string Image = MAC::Singleton::instance()->get_data()["inputs"]["labels"][Subject];
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
		  label_ = Reader3D::New();
		  label_->SetFileName( image_ptr->GetFileName() );
		  label_->Update();
		}
	      else
		{
		  std::string mess = "Image (" + std::to_string(Subject) + ") does not exists.";
		  throw MAC::MACException( __FILE__, __LINE__,
					    mess.c_str(),
					    ITK_LOCATION );
		}
	    }
	  else
	    {
	      std::string mess = "This modality does not match the subject. ";
	      //
	      throw MAC::MACException( __FILE__, __LINE__,
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
  template < int D > void
    MAC::Subject< D >::add_modality( const int Subject, const int Modality )
    {
      try
	{
	  if ( Subject == id_ )
	    {
	      std::string Image = MAC::Singleton::instance()->get_data()["inputs"]["images"][Modality][Subject];
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
		  modality_images_[ Modality ] = Reader3D::New();
		  modality_images_[ Modality ]->SetFileName( image_ptr->GetFileName() );
		  modality_images_[ Modality ]->Update();
		}
	      else
		{
		  std::string mess = "Image (" + std::to_string(Modality) + ",";
		  mess += std::to_string(Subject) + ") does not exists.";
		  throw MAC::MACException( __FILE__, __LINE__,
					    mess.c_str(),
					    ITK_LOCATION );
		}
	    }
	  else
	    {
	      std::string mess = "This modality does not match the subject. ";
	      //
	      throw MAC::MACException( __FILE__, __LINE__,
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
  template < int D > void
    MAC::Subject< D >::set_fit( const MaskType::IndexType Idx, 
				const Eigen::MatrixXd Model_fit, 
				const Eigen::MatrixXd Cov_fit )
    {
//      //
//      // ToDo: I would like to write the goodness of the score (r-square ...)
//      //
//      // copy Eigen Matrix information into a vector
//      // We only record the diagonal sup of the covariance.
//      std::vector< double > model( D_r ), cov( D_r * (D_r + 1) / 2 );
//      int current_mat_coeff = 0;
//      for ( int d ; d < D_r ; d++ )
//	{
//	  model[d] = Model_fit(d,0);
//	  Random_effect_ITK_model_.set_val( d, Idx, Model_fit(d,0) );
//	  for ( int c = d ; c < D_r ; c++)
//	    {
//	      cov[d]  = Cov_fit(d,c);
//	      Random_effect_ITK_variance_.set_val( current_mat_coeff++, Idx, Cov_fit(d,c) );
//	    }
//	}
    }
  //
  //
  //
  template < int D > std::vector< double >
    MAC::Subject< D >::get_modalities( const MaskType::IndexType Idx ) const
    {
      std::vector< double > mod(D);
      for ( int mod_image = 0 ; mod_image < D ; mod_image++ )
	mod[mod_image] = modality_images_[mod_image]->GetOutput()->GetPixel(Idx);
      //
      return mod;
    }
  //
  //
  //
  template < int D > int
    MAC::Subject< D >::get_label( const MaskType::IndexType Idx ) const 
    {
      return static_cast< int >( label_->GetOutput()->GetPixel(Idx) );
    }
  //
  //
  //
  template < int D > void
    MAC::Subject< D >::create_output_map()
    {
//      //std::cout << "We create output only one time" << std::endl;
//      // Model output
//      std::string output_model = "model_" 
//	+ std::to_string( PIDN_ ) + "_" + std::to_string( group_ )
//	+ "_" + std::to_string( time_points_ ) + "_" + std::to_string( D_ ) 
//	+ ".nii.gz";
//      Random_effect_ITK_model_ = MACMakeITKImage( D_r,
//						   output_model,
//						   age_ITK_images_.begin()->second );
//      // Variance output
//      // We only record the diagonal sup elements
//      //
//      // | 1 2 3 |
//      // | . 4 5 |
//      // | . . 6 |
//      std::string output_var = "var_" 
//	+ std::to_string( PIDN_ ) + "_" + std::to_string( group_ )
//	+ "_" + std::to_string( time_points_ ) + "_" + std::to_string( D_ ) 
//	+ ".nii.gz";
//      Random_effect_ITK_variance_ = MACMakeITKImage( D_r * (D_r + 1) / 2 /*we make sure it is a int*/,
//						      output_var,
//						      age_ITK_images_.begin()->second );
    }
  //
  //
  //
  template < int D > void
    MAC::Subject< D >::write_solution( )
    {
//      Random_effect_ITK_model_.write();
//      Random_effect_ITK_variance_.write();
    }
  //
  //
  //
  template < int D > void
    MAC::Subject< D >::print() const
    {
//      std::cout << "PIDN: " << PIDN_ << std::endl;
//      std::cout << "Group: " << group_ << std::endl;
//      std::cout << "Number of time points: " << time_points_ << std::endl;
//      //
//      std::cout << "Age and covariates: " << std::endl;
//      if ( !age_covariates_.empty() )
//	for ( auto age_cov : age_covariates_ )
//	  {
//	    std::cout << "At age " << age_cov.first << " covariates were:";
//	    for( auto c : age_cov.second )
//	      std::cout << " " << c;
//	    std::cout << std::endl;
//	  }
//      else
//	std::cout << "No age and covariates recorded." << std::endl;
//      //
//      std::cout << "Age and imagess: " << std::endl;
//      if ( !age_images_.empty() )
//	for ( auto age_img : age_images_ )
//	  std::cout << "At age " << age_img.first << " iamge was: "
//		    << age_img.second << std::endl;
//      else
//	std::cout << "No age and images recorded." << std::endl;
    }
}
#endif
