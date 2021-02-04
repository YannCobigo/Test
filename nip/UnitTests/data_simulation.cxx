#include <iostream>
#include <cstdlib>
#include <string>
#include <cstdlib>
#include <fstream>
#include <sstream>  
#include <regex>  
#include <random>
//
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIterator.h>
#include <itkNiftiImageIO.h>
#include <itkOrientImageFilter.h>
#include <itkSpatialOrientation.h>
#include "itkImageDuplicator.h"
// Eigen
#include <Eigen/Core>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Sparse>
//
//
//
itk::ImageIOBase::Pointer
getImageIO( std::string input )
{
  itk::ImageIOBase::Pointer imageIO = itk::ImageIOFactory::CreateImageIO(input.c_str(), itk::ImageIOFactory::ReadMode);
  
  imageIO->SetFileName(input);
  imageIO->ReadImageInformation();
  //
  return imageIO;
}
//
itk::ImageIOBase::IOPixelType
pixel_type( itk::ImageIOBase::Pointer imageIO )
{
  return imageIO->GetPixelType();
}
//
//
//
int main(int argc, char const *argv[])
{
  //
  // 
  itk::ImageIOBase::Pointer model = NULL;
  itk::ImageIOBase::Pointer atlas = NULL;
  itk::ImageIOBase::Pointer mask  = NULL;
  std::string output_name         = "";

  //
  //
  if( argc == 5 )
    {
      output_name = argv[1];
      model       = getImageIO( argv[2] );
      atlas       = getImageIO( argv[3] );
      mask        = getImageIO( argv[4] );
    }
  else
    {
      std::cerr << "data_simulation requires: \"output_suffix\" \"file_model.nii.gz\" \"Atlas.nii.gz\" \"mask.nii.gz\" "
		<< std::endl;
      return EXIT_FAILURE;
    }

  
  //
  // reader
  typedef itk::Image< float, 3 > Image;
  typedef itk::ImageFileReader< Image >  Image_reader;
  typedef itk::ImageDuplicator< Image >  DuplicatorType;
  typedef itk::Image< char, 3 > Mask;
  typedef itk::ImageFileReader< Mask >  Mask_reader;
  typedef itk::ImageFileReader< Mask >  Atlas_reader;

  //
  // Image
  Image_reader::Pointer reader_image = Image_reader::New();
  reader_image->SetFileName( model->GetFileName() );
  reader_image->Update();
  Mask_reader::Pointer reader_mask = Mask_reader::New();
  reader_mask->SetFileName( mask->GetFileName() );
  reader_mask->Update();
  Atlas_reader::Pointer reader_atlas = Atlas_reader::New();
  reader_atlas->SetFileName( atlas->GetFileName() );
  reader_atlas->Update();

  //
  // Region to explore
  Mask::RegionType region;
  //
  Mask::Pointer   image_in = reader_mask->GetOutput();
  Mask::SizeType  img_size = image_in->GetLargestPossibleRegion().GetSize();
  Mask::IndexType start    = {0, 0, 0};
  //
  region.SetSize( img_size );
  region.SetIndex( start );

  //
  // Explanatory variables
  int
    n  = 10,   // number of subjects
    q  = 4;   // neuropsy metrics
  // CSV output
  std::ofstream X_out("../../data/data_test.csv");
  std::string
    X_output("PIDN,Gr,path,EV1,EV2,EV3,EV4\n"),
    sep(","), rtn("\n");
  // Random information
  std::default_random_engine generator;
  std::normal_distribution< float >
    rd_subject( 0.0, 0.5 ),
    rd_noise( 0.0, 0.05 );

  //
  // loop over subjects
  for ( int i = 0 ; i < n ; i++ )
    {
      //
      // Output
      X_output += "100" + std::to_string(i) + sep;
      if ( i < 50 )
	X_output += "1" + sep;
      else
	X_output += "2" + sep;
      // neurospy measures
      std::vector< float > X( q );

      //
      // Simulated data
      DuplicatorType::Pointer duplicator = DuplicatorType::New();
      duplicator->SetInputImage( reader_image->GetOutput() );
      duplicator->Update();
      //
      Image::Pointer image_out = duplicator->GetOutput();
      itk::ImageRegionIterator< Mask >  imageIterator_mask( reader_mask->GetOutput(), region );
      //
      while( !imageIterator_mask.IsAtEnd() )
	{
	  //
	  // Voxel index
	  auto vox_idx = imageIterator_mask.GetIndex();
	  int atlas_val = static_cast< int >( reader_atlas->GetOutput()->GetPixel( vox_idx ) );
	  if ( atlas_val > 0 )
	    {
	      //
	      // gray matter value for new subject
	      float gm = reader_image->GetOutput()->GetPixel( vox_idx );
	      if ( i < 50 )
		gm += rd_subject( generator );
	      else
		gm += rd_subject( generator ) - -0.5;
	      //
	      image_out->SetPixel( vox_idx, gm );
	      //
	      if ( atlas_val == 31 )
		{
		  X[0] += gm;
//		  X[1] += gm;
//		  X[2] += gm;
		}
//	      else if ( atlas_val == 19 )
//		{
//		  X[1] += gm;
//		  X[2] += -1.*gm;
//		}
	      if ( atlas_val == 1 )
		{
		  if ( i < 50 )
		    X[2] += 2. * gm;
		  else
		    X[2] += 0.5 * gm;
		}
	    }
	  else
	    image_out->SetPixel( imageIterator_mask.GetIndex(), 0.0 );
	  //
	  ++imageIterator_mask;
	}

      //
      // write output
      itk::NiftiImageIO::Pointer nifti_io = itk::NiftiImageIO::New();
      nifti_io->SetPixelType( pixel_type(model) );
      //
      itk::ImageFileWriter< Image >::Pointer writer = itk::ImageFileWriter< Image >::New();
      writer->SetFileName( output_name + "_" + std::to_string(i) + ".nii.gz" );
      writer->SetInput( image_out );
      writer->SetImageIO( nifti_io );
      writer->Update();

      //
      // output
      X_output += "/home/cobigo/devel/CPP/NIP//nip/UnitTests/build/" + output_name + "_" + std::to_string(i) + ".nii.gz" + sep;
      // explenatory variables
      X_output += std::to_string(X[0]+ rd_noise(generator)) + sep;
      X_output += std::to_string(X[1]+ rd_noise(generator)) + sep;
      X_output += std::to_string(X[2]+ rd_noise(generator)) + sep;
      X_output += std::to_string(X[3]+ rd_noise(generator)) + sep;
      X_output += rtn;
    }
  
  //
  // Write the output
  X_out << X_output;
  X_out.close();

  //
  //
  return EXIT_SUCCESS;
}
