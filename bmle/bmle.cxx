#include<iostream>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <vector>
//
// ITK
//
#include <itkImageFileReader.h>
#include <itkSpatialOrientationAdapter.h>
#include "itkChangeInformationImageFilter.h"
using MaskType       = itk::Image< unsigned char, 3 >;
using MaskReaderType = itk::ImageFileReader< MaskType >;
//
// 
//
#include "BmleException.h"
#include "Load_csv.h"
//
//
//
class InputParser{
public:
  explicit InputParser ( const int &argc, const char **argv )
  {
    for( int i = 1; i < argc; ++i )
      tokens.push_back( std::string(argv[i]) );
  }
  //
  const std::string getCmdOption( const std::string& option ) const
  {
    //
    //
    std::vector< std::string >::const_iterator itr = std::find( tokens.begin(),
								tokens.end(),
								option );
    if ( itr != tokens.end() && ++itr != tokens.end() )
      return *itr;

    //
    //
    return "";
  }
  //
  bool cmdOptionExists( const std::string& option ) const
  {
    return std::find( tokens.begin(), tokens.end(), option) != tokens.end();
  }
private:
  std::vector < std::string > tokens;
};
//
//
//
int
main( const int argc, const char **argv )
{
  try
    {
      //
      // Parse the arguments
      //
      if( argc > 1 )
	{
	  InputParser input( argc, argv );
	  if( input.cmdOptionExists("-h") )
	    throw MAC_bmle::BmleException( __FILE__, __LINE__,
					   "./bmle -c file.csv -m mask.nii.gz >",
					   ITK_LOCATION );

	  //
	  // takes the csv file ans the mask
	  const std::string& filename = input.getCmdOption("-c");
	  const std::string& mask     = input.getCmdOption("-m");
	  //
	  if ( !filename.empty() )
	    {
	      if ( mask.empty() )
		{
		  std::string mess = "No mask loaded. A mask must be loaded.\n";
		  mess += "./bmle -c file.csv -m mask.nii.gz";
		  throw MAC_bmle::BmleException( __FILE__, __LINE__,
						 mess.c_str(),
						 ITK_LOCATION );
		}

	      ////////////////////////////
	      ///////              ///////
	      ///////  PROCESSING  ///////
	      ///////              ///////
	      ////////////////////////////

	      //
	      // Load the CSV file
	      MAC_bmle::BmleLoadCSV< 3/*D_r*/, 2 /*D_f*/> subject_mapping( filename );
	      // create the 4D iamge with all the images
	      subject_mapping.build_groups_design_matrices();

	      //
	      // Expecttion Maximization
	      //

	      //
	      // Mask
	      MaskReaderType::Pointer reader_mask_{ MaskReaderType::New() };
	      reader_mask_->SetFileName( mask );
	      reader_mask_->Update();
	      // Visiting region (Mask)
	      MaskType::RegionType region;
	      //
	      MaskType::SizeType  img_size =
		reader_mask_->GetOutput()->GetLargestPossibleRegion().GetSize();
	      MaskType::IndexType start    = {0, 0, 0};
	      //
	      region.SetSize( img_size );
	      region.SetIndex( start );
	      //
	      itk::ImageRegionIterator< MaskType >
		imageIterator_mask( reader_mask_->GetOutput(), region );
	      
	      //
	      // loop over Mask area for every images
	      while( !imageIterator_mask.IsAtEnd() )
		{
		  if( static_cast<int>( imageIterator_mask.Value() ) != 0 )
		    {
		      //MaskType::IndexType idx = imageIterator_mask.GetIndex();
		      subject_mapping.Expectation_Maximization( imageIterator_mask.GetIndex() );
		    }
		  //
		  ++imageIterator_mask;
		}
	    }
	  else
	    throw MAC_bmle::BmleException( __FILE__, __LINE__,
					   "./bmle -c file.csv -m mask.nii.gz >",
					   ITK_LOCATION );
	}
      else
	throw MAC_bmle::BmleException( __FILE__, __LINE__,
				       "./bmle -c file.csv -m mask.nii.gz >",
				       ITK_LOCATION );
    }
  catch( itk::ExceptionObject & err )
    {
      std::cerr << err << std::endl;
      return EXIT_FAILURE;
    }

  //
  //
  //
  return EXIT_SUCCESS;
}
