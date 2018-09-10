#include "Subjects_mapping.h"
#include "NipMakeITKImage.h"
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
using ImageType       = itk::Image< double, 3 >;
using ImageReaderType = itk::ImageFileReader< ImageType >;
using MaskType        = itk::Image< unsigned char, 3 >;
using MaskReaderType  = itk::ImageFileReader< MaskType >;
//
//
//
MAC_nip::NipSubject_Mapping::NipSubject_Mapping( const std::string& CSV_file,
						 const std::string& Mask,
						 const int Reduced_space ):
  csv_file_{CSV_file.c_str()}, mask_{Mask}
{
  try
    {
      std::string line;
      //skip the first line
      std::getline(csv_file_, line);
      //
      // then loop
      while( std::getline(csv_file_, line) )
	{
	  std::stringstream  lineStream( line );
	  std::string        cell;
	  std::cout << line << std::endl;
	  
	  //
	  // Get the PIDN
	  std::string PIDN;
	  std::getline(lineStream, PIDN, ',');
	  // Get the group
	  std::getline(lineStream, cell, ',');
	  const int group = std::stoi( cell );
	  if ( group == 0 )
	    throw NipException( __FILE__, __LINE__,
				"Select another group name than 0 (e.g. 1, 2, ...). 0 is reserved.",
				ITK_LOCATION );
	  // Get the image
	  std::string image;
	  std::getline(lineStream, image, ',');
	  // Covariates
	  std::list< double > covariates;
	  while( std::getline(lineStream, cell, ',') )
	    covariates.push_back( std::stof(cell) );
	  //
	  //
	  if ( PIDNs_.find(PIDN) == PIDNs_.end() )
	    {
	      PIDNs_.insert( PIDN );
	      groups_.insert( group );
	      group_pind_[ group ].push_back( NipSubject( group, PIDN, image, Mask, covariates ) );
	    }
	  else
	    {
	      std::string mess = "PIDN " + PIDN + " was already inserted in the dtaset.";
	      throw NipException( __FILE__, __LINE__,
				  mess.c_str(),
				  ITK_LOCATION );
	    }
	}

      //
      // Build the group matrices
      for ( auto g : groups_ )
	{
	  //
	  // 
	  int
	    n = group_pind_[g].size(),
	    p = group_pind_[g][0].get_image_matrix().rows(),
	    q = group_pind_[g][0].get_ev_matrix().rows();

	  //
	  // Build matrics
	  Eigen::MatrixXd Image_matrix = Eigen::MatrixXd::Zero(n,p);
	  Eigen::MatrixXd EV_matrix    = Eigen::MatrixXd::Zero(n,q);
	  for ( int s = 0 ; s < n ; s++ )
	    {
	      Image_matrix.row(s) = group_pind_[g][s].get_image_matrix().col(0);
	      // CCA
	      if ( q > 0 && Reduced_space == 0 )
		EV_matrix.row(s) = group_pind_[g][s].get_ev_matrix().col(0);
	    }

	  //
	  // Build spectrum
	  std::size_t K_spectrum = 0;
	  // CCA
	  if ( q > 0 && Reduced_space == 0 )
	    K_spectrum = (p > q ? q : p);
	  // SPC
	  else if ( Reduced_space > 0 )
	    K_spectrum = Reduced_space;
	  else
	    throw NipException( __FILE__, __LINE__,
				"ERROR: case unknown (chose CCA or SPC).",
				ITK_LOCATION );
	  //
	  Spectra matrix_spetrum( K_spectrum );
	  for ( int k = 0 ; k < K_spectrum ; k++ )
	    {
	      if ( q > 0 )
		{
		  // Coefficient
		  std::get< coeff_k >( matrix_spetrum[k] ) = 0.;
		  // vectors
		  std::get< Uk >( matrix_spetrum[k] ) = Eigen::MatrixXd::Random( p, 1 );
		  std::get< Vk >( matrix_spetrum[k] ) = Eigen::MatrixXd::Random( q, 1 );
		  // normalization
		  std::get< Uk >( matrix_spetrum[k] ) /= std::get< Uk >( matrix_spetrum[k] ).lpNorm< 2 >();
		  std::get< Vk >( matrix_spetrum[k] ) /= std::get< Vk >( matrix_spetrum[k] ).lpNorm< 2 >();
		}
	      else
		throw NipException( __FILE__, __LINE__,
				    "ERROR: need to build the case for PCA (SPC).",
				    ITK_LOCATION );
	    }

	  //
	  // Create the tuple
	  group_matrices_[g] = std::make_tuple( std::make_shared< const Eigen::MatrixXd >(Image_matrix),
						std::make_shared< const Eigen::MatrixXd >(EV_matrix),
						std::make_shared< Spectra >(matrix_spetrum) );
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
void
MAC_nip::NipSubject_Mapping::dump() 
{
  try
    {
      //
      // For each group
      for ( auto g : groups_ )
	{
	  //
	  // 
	  int
	    n = group_pind_[g].size(),
	    p = group_pind_[g][0].get_image_matrix().rows(),
	    q = group_pind_[g][0].get_ev_matrix().rows();
	  
	  //
	  // Build spectrum image
	  //

	  //
	  //
	  std::size_t K_spectrum = 0;
	  if ( q > 0 )
	    K_spectrum = (p > q ? q : p);
	  else
	    throw NipException( __FILE__, __LINE__,
				"ERROR: need to build the case for PCA (SPC).",
				ITK_LOCATION );
	  
	  //
	  // For each spectrum write the image
	  int K_spectrum_left = 0;
	  for ( int kk = 0 ; kk < K_spectrum ; kk++ )
	    if ( std::get<coeff_k>((*std::get<2>(group_matrices_[g]))[kk]) != 0. )
	      K_spectrum_left++;
	  // Buid the 4D output image
	  std::string group_spectrum_Uk_image_name = "spectrum_U_gr" + std::to_string(g) + ".nii.gz";
	  NipMakeITKImage group_spectrum_Uk( K_spectrum_left, group_spectrum_Uk_image_name,
					     group_pind_[g][0].get_image_reader() );
	  // loop over spectrum left
	  for ( int kk = 0 ; kk < K_spectrum ; kk++ )
	    if ( std::get<coeff_k>((*std::get<2>(group_matrices_[g]))[kk]) != 0. )
	      {
		// Region to explore
		ImageType::RegionType region;
		ImageType::Pointer    image_in = group_pind_[g][0].get_image_reader()->GetOutput();
		ImageType::SizeType   img_size = image_in->GetLargestPossibleRegion().GetSize();
		ImageType::IndexType  start    = {0, 0, 0};
		region.SetSize( img_size );
		region.SetIndex( start );
		//
		itk::ImageRegionIterator< MaskType > imageIterator_mask( group_pind_[g][0].get_mask_reader()->GetOutput(), region );
		//
		int pos = 0;
		while( !imageIterator_mask.IsAtEnd() )
		  {
		    if (  static_cast<int>( imageIterator_mask.Value() ) != 0 )
		      {
			MaskType::IndexType idx = imageIterator_mask.GetIndex();
			double val_U = std::get<Uk>((*std::get<2>(group_matrices_[g]))[kk])(pos++,0);
			group_spectrum_Uk.set_val( kk, idx, val_U );
		      }
		    //
		    ++imageIterator_mask;
		  }
	      }
	  //
	  group_spectrum_Uk.write();
	}
    }
  catch( itk::ExceptionObject & err )
    {
      std::cerr << err << std::endl;
      exit(-1);
    }
}
