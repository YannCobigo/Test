#include<iostream>
#include <stdio.h>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <vector>
#include <chrono>
#include <sys/stat.h>
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
#include "Exception.h"
#include "Load_csv.h"
//
//
// Check the output directory exists
inline bool directory_exists( const std::string& Dir )
{
  struct stat sb;
  //
  if ( stat(Dir.c_str(), &sb ) == 0 && S_ISDIR( sb.st_mode ) )
    return true;
  else
    return false;
}
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
	    //
	    // It is the responsability of the user to create the 
	    // normalized/standardized hierarchical covariate
	    //
	    // -h                          : help
	    // -X  inv_cov_error.nii.gz    : (prediction) inverse of error cov on parameters
	    // -c   input.csv              : input file
	    // -m   mask.nii.gz            : mask
	    // -d                          : demeaning of age -> boolean
	    //
	    throw NeuroBayes::NeuroBayesException( __FILE__, __LINE__,
						   "./bmle -c file.csv -m mask.nii.gz -o output_dir < -d (demeaning age) >",
						   ITK_LOCATION );

	  //
	  // takes the csv file ans the mask
	  const std::string& filename       = input.getCmdOption("-c");
	  const std::string& mask           = input.getCmdOption("-m");
	  const std::string& output_dir     = input.getCmdOption("-o");
	  // Demean the age
	  const std::string& demeaning      = input.getCmdOption("-d");
	  // Prediction
	  const std::string& inv_cov_error  = input.getCmdOption("-X");
	  //
	  if ( !filename.empty() )
	    {
	      if ( mask.empty() && output_dir.empty() )
		{
		  std::string mess = "No mask loaded. A mask must be loaded.\n";
		  mess += "./bmle -c file.csv -m mask.nii.gz -o output_dir";
		  throw NeuroBayes::NeuroBayesException( __FILE__, __LINE__,
							 mess.c_str(),
							 ITK_LOCATION );
		}
	      // output directory exists?
	      if ( !directory_exists( output_dir ) )
		{
		  std::string mess = "The output directory is not correct.\n";
		  mess += "./bmle -c file.csv -m mask.nii.gz -o output_dir";
		  throw NeuroBayes::NeuroBayesException( __FILE__, __LINE__,
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
	      const int THREAD_NUM = 1;

	      //
	      // Load the CSV file
	      NeuroBayes::BmleLoadCSV< 2/*D_r*/, 0 /*D_f*/> subject_mapping( filename, output_dir,
									     input.cmdOptionExists("-d"), 
									     inv_cov_error );
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
		NeuroBayes::Thread_dispatching pool( THREAD_NUM );
#endif
	      while( !imageIterator_mask.IsAtEnd() )
		{
		  if( static_cast<int>( imageIterator_mask.Value() ) != 0 )
		    {
		      MaskType::IndexType idx = imageIterator_mask.GetIndex();
#ifdef DEBUG
		      if ( idx[0] > 33 && idx[0] < 35 && 
			   idx[1] > 30 && idx[1] < 32 &&
			   idx[2] > 61 && idx[2] < 63 )
			{
			  std::cout << imageIterator_mask.GetIndex() << std::endl;
			  subject_mapping.Expectation_Maximization( idx );
			}
#else
		      // Please do not remove the bracket!!
		      // vertex
		      if ( idx[0] > 92 - 5  && idx[0] < 92 + 5 && 
			   idx[1] > 94 - 5  && idx[1] < 94 + 5 &&
			   idx[2] > 63 - 5  && idx[2] < 63 + 5 )
//		      // ALL
//		      if ( idx[0] > 5 && idx[0] < 110 && 
//			   idx[1] > 5 && idx[1] < 140 &&
//			   idx[2] > 5 && idx[2] < 110 )
//		      // Octan 1
//		      if ( idx[0] > 5 && idx[0] < 60  & 
//			   idx[1] > 5 && idx[1] < 70  &&
//			   idx[2] > 2 && idx[2] < 60  )
//		      // Octan 2
//		      if ( idx[0] >= 60 && idx[0] < 110 && 
//			   idx[1] > 5 && idx[1] < 70  &&
//			   idx[2] > 2 && idx[2] < 60 )
//		      // Octan 3
//		      if ( idx[0] > 5 && idx[0] < 60  && 
//			   idx[1] >= 70 && idx[1] < 140 &&
//			   idx[2] > 2 && idx[2] < 60 )
//		      // Octan 4
//		      if ( idx[0] >= 60 && idx[0] < 110 && 
//			   idx[1] >= 70 && idx[1] < 140 &&
//			   idx[2] > 2 && idx[2] < 60 )
//		      // Octan 5
//		      if ( idx[0] > 5 && idx[0] < 60 && 
//			   idx[1] > 5 && idx[1] < 70 &&
//			   idx[2] >= 60 && idx[2] < 110 )
//		      // Octan 6
//		      if ( idx[0] >= 60 && idx[0] < 110 && 
//			   idx[1] > 5 && idx[1] < 70  &&
//			   idx[2] >= 60 && idx[2] < 110 )
//		      // Octan 7
//		      if ( idx[0] > 5 && idx[0] < 60  && 
//			   idx[1] >= 70 && idx[1] < 140 &&
//			   idx[2] >= 60 && idx[2] < 110 )
//		      // Octan 8
//		      if ( idx[0] >= 60 && idx[0] < 110 && 
//			   idx[1] >= 70 && idx[1] < 140 &&
//			   idx[2] >= 60 && idx[2] < 110 )
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
	    throw NeuroBayes::NeuroBayesException( __FILE__, __LINE__,
						   "./bmle -c file.csv -m mask.nii.gz >",
						   ITK_LOCATION );
	}
      else
	throw NeuroBayes::NeuroBayesException( __FILE__, __LINE__,
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
