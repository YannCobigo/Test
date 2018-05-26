#include "Subject.h"
//
//
//
MAC_nip::NipSubject::NipSubject( const int Group,
				 const std::string Pidn,
				 const std::string Image,
				 const std::string Mask,
				 const std::list< double >& Covariates ):
  group_{Group}, PIDN_{Pidn}, image_{Image}, mask_{Mask}
{
  //
  // Build the image vector
  
  //
  // Image
  ImageReaderType::Pointer reader_image = ImageReaderType::New();
  reader_image->SetFileName( image_ );
  reader_image->Update();
  MaskReaderType::Pointer reader_mask = MaskReaderType::New();
  reader_mask->SetFileName( mask_ );
  reader_mask->Update();
  // Region to explore
  ImageType::RegionType region;
  ImageType::Pointer   image_in = reader_image->GetOutput();
  ImageType::SizeType  img_size = image_in->GetLargestPossibleRegion().GetSize();
  ImageType::IndexType start    = {0, 0, 0};
  region.SetSize( img_size );
  region.SetIndex( start );
  //
  itk::ImageRegionIterator< MaskType >  imageIterator_mask( reader_mask->GetOutput(), region );
  //
  std::list< double > in_mask_value;
  while( !imageIterator_mask.IsAtEnd() )
    {
      if (  static_cast<int>( imageIterator_mask.Value() ) != 0 )
	{
	  MaskType::IndexType idx = imageIterator_mask.GetIndex();
	  in_mask_value.push_back( reader_image->GetOutput()->GetPixel(idx) );
	}
      //
      ++imageIterator_mask;
    }
  //
  image_matrix_ = Eigen::MatrixXd::Zero( in_mask_value.size(), 1 );
  int pos_in_image = 0;
  for ( auto im : in_mask_value )
    image_matrix_( pos_in_image++, 0 ) = im;
  //
  // Explanatory variables
  ev_matrix_ = Eigen::MatrixXd::Zero( Covariates.size(), 1 );
  int pos_in_ev = 0;
  for ( auto cov : Covariates )
    ev_matrix_(pos_in_ev++,0) = cov;
}
