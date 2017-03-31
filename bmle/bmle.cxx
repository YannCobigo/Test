#include<iostream>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <vector>
#include <chrono>
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
#include "Thread_dispatching.h"
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
	      // Number of THREADS in case of multi-threading
	      // this program hadles the multi-threading it self
	      // in no-debug mode
	      const int THREAD_NUM = 8;

	      //
	      // Load the CSV file
	      MAC_bmle::BmleLoadCSV< 2/*D_r*/, 0 /*D_f*/> subject_mapping( filename );
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
		imageIterator_mask( reader_mask_->GetOutput(), region ),
		imageIterator_progress( reader_mask_->GetOutput(), region );

	      //
	      // Task progress: elapse time
	      using  ms         = std::chrono::milliseconds;
	      using get_time    = std::chrono::steady_clock ;
	      auto start_timing = get_time::now();
	      
	      //
	      // loop over Mask area for every images
#ifndef DEBUG
	      std::cout << "Multi-threading" << std::endl;
	      // Start the pool of threads
	      {
		MAC_bmle::Thread_dispatching pool( THREAD_NUM );
#endif
	      while( !imageIterator_mask.IsAtEnd() )
		{
		  if( static_cast<int>( imageIterator_mask.Value() ) != 0 )
		    {
		      MaskType::IndexType idx = imageIterator_mask.GetIndex();
#ifdef DEBUG
		      if ( idx[0] > 25 && idx[0] < 35 && 
			   idx[1] > 65 && idx[1] < 75 &&
			   idx[2] > 55 && idx[2] < 65 )
			{
			  std::cout << imageIterator_mask.GetIndex() << std::endl;
			  subject_mapping.Expectation_Maximization( idx );
			}
#else
			// Please do not remove the bracket!!
		      if ( idx[0] > 25 && idx[0] < 35 && 
			   idx[1] > 65 && idx[1] < 75 &&
			   idx[2] > 55 && idx[2] < 65 )
//		      if ( idx[0] > 0 && idx[0] < 60 && 
//			   idx[1] > 0 && idx[1] < 140 &&
//			   idx[2] > 50 && idx[2] < 70 )
			{
			  pool.enqueue( std::ref(subject_mapping), idx );
			}
#endif
		    }
		  //
		  ++imageIterator_mask;
#ifndef DEBUG
		  // Keep the brack to end the pool of threads
		}
#endif
	      }

	      //
	      // Task progress
	      // End the elaps time
	      auto end_timing  = get_time::now();
	      auto diff        = end_timing - start_timing;
	      std::cout << "Process Elapsed time is :  " << std::chrono::duration_cast< ms >(diff).count()
			<< " ms "<< std::endl;

	      //
	      //
	      std::cout << "All the mask has been covered" << std::endl;
	      subject_mapping.write_subjects_solutions();
	      std::cout << "All output have been written." << std::endl;
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
