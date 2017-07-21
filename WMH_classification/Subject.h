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
      using FilterType  = itk::ChangeInformationImageFilter< Image3DType >;

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
      void set_fit( const MaskType::IndexType, const double );
      // Add modality
      void add_modality( const int, const int );
      // Add label
      void add_label( const int );
      //
      void create_output_map( const int, const std::string );
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
      // Output probability map
      Image3DType::Pointer             probability_map_;
      // Output probability map name
      std::string                      probability_map_name_;
      // Take the dimension of the first subject image:
      Reader3D::Pointer                template_img_;

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
				const double Fit_value )
    {
      probability_map_->SetPixel( Idx,Fit_value  );
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
    MAC::Subject< D >::create_output_map( const int Subject, const std::string Output_name )
    {
      try
	{
	  if ( Subject == id_ )
	    {
	      std::string Image = MAC::Singleton::instance()->get_data()["inputs"]["images"][0][Subject];
	      // load the ITK images
	      if ( file_exists(Image) )
		{
		  //
		  // Create a template image to get the output image dimension
		  template_img_ = Reader3D::New();
		  template_img_->SetFileName( Image );
		  template_img_->Update();
		  // 
		  Image3DType::RegionType region;
		  Image3DType::IndexType  start = { 0, 0, 0 };
		  //
		  Image3DType::Pointer  raw_subject_image_ptr = template_img_->GetOutput();
		  Image3DType::SizeType size = raw_subject_image_ptr->GetLargestPossibleRegion().GetSize();
		  //
		  region.SetSize( size );
		  region.SetIndex( start );
		  //
		  probability_map_ = Image3DType::New();
		  probability_map_->SetRegions( region );
		  probability_map_->Allocate();
		  probability_map_->FillBuffer( 0.0 );
		  // name of the output
		  probability_map_name_ = Output_name;
		}
	      else
		{
		  std::string mess = "Image (" + Image + ",";
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
    MAC::Subject< D >::write_solution()
    {

      //
      // ITK orientation, most likely does not match our orientation
      // We have to reset the orientation
      // Origin
      Image3DType::Pointer  raw_subject_image_ptr = template_img_->GetOutput();
      Image3DType::PointType origin = raw_subject_image_ptr->GetOrigin();
      // Spacing 
      Image3DType::SpacingType spacing = raw_subject_image_ptr->GetSpacing();
      // Direction
      Image3DType::DirectionType direction = raw_subject_image_ptr->GetDirection();
      //
      // image filter
      FilterType::Pointer images_filter;
      images_filter = FilterType::New();
      //
      images_filter->SetOutputSpacing( spacing );
      images_filter->ChangeSpacingOn();
      images_filter->SetOutputOrigin( origin );
      images_filter->ChangeOriginOn();
      images_filter->SetOutputDirection( direction );
      images_filter->ChangeDirectionOn();
      //
      images_filter->SetInput( probability_map_ );
  
      //
      // write 
      itk::NiftiImageIO::Pointer nifti_io = itk::NiftiImageIO::New();
      //
      itk::ImageFileWriter< Image3DType >::Pointer writer = itk::ImageFileWriter< Image3DType >::New();
      writer->SetFileName( probability_map_name_ );
      writer->SetInput( images_filter->GetOutput() );
      writer->SetImageIO( nifti_io );
      writer->Update();
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
