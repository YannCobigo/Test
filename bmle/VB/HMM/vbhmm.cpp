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
#include <itkImageFileWriter.h>
#include <itkSpatialOrientationAdapter.h>
#include "itkChangeInformationImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
using MaskType       = itk::Image< unsigned char, 3 >;
using MaskReaderType = itk::ImageFileReader< MaskType >;
using Image3DType    = itk::Image< double, 3 >;
using Reader3D       = itk::ImageFileReader< Image3DType >;
using Writer3D       = itk::ImageFileWriter< Image3DType >;
using Image4DType    = itk::Image< double, 4 >;
using Reader4D       = itk::ImageFileReader< Image4DType >;
using Writer4D       = itk::ImageFileWriter< Image4DType >;
using FilterType     = itk::RescaleIntensityImageFilter< MaskType, Image3DType >;
//
// 
//
#include "Thread_dispatching.h"
#include "Exception.h"
#include "VBHMM_Subject_mapping.h"
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
	    // -c   input.csv              : input file
	    // -m   mask.nii.gz            : mask
	    // -o   output_dir             : output directory
	    //
	    throw NeuroBayes::NeuroBayesException( __FILE__, __LINE__,
						   "./vbhmm -c file.csv -m mask.nii.gz -o output_dir ",
						   ITK_LOCATION );

	  //
	  // takes the csv file ans the mask
	  const std::string& filename       = input.getCmdOption("-c");
	  const std::string& mask           = input.getCmdOption("-m");
	  const std::string& output_dir     = input.getCmdOption("-o");
	  // Demean the age
	  const std::string& transformation       = input.getCmdOption("-d");
	  NeuroStat::TimeTransformation time_tran = NeuroStat::TimeTransformation::NONE;
	  if ( !transformation.empty() )
	    {
	      if ( transformation.compare("demean") == 0 )
		time_tran = NeuroStat::TimeTransformation::DEMEAN;
	      else if ( transformation.compare("normalize") == 0 )
		time_tran = NeuroStat::TimeTransformation::NORMALIZE;
	      else if ( transformation.compare("standardize") == 0 )
		time_tran = NeuroStat::TimeTransformation::STANDARDIZE;
	      else if ( transformation.compare("load") == 0 )
		time_tran = NeuroStat::TimeTransformation::LOAD;
	      else
		{
		  std::string mess  = "The time transformation can be: ";
		   mess            += "none, demean, normalize, standardize.\n";
		   mess            += "Trans formation: " + transformation;
		   mess            += " is unknown.\n";
		   mess            += " please try ./bmle -h for all options.\n";
		   throw NeuroBayes::NeuroBayesException( __FILE__, __LINE__,
							  mess.c_str(),
							  ITK_LOCATION );
		}
	    }
	  // Slicing the space
	  //
	  if ( !filename.empty() )
	    {
	      if ( mask.empty() && output_dir.empty() )
		{
		  std::string mess = "No mask loaded. A mask must be loaded.\n";
		  mess += "./vbhmm -c file.csv -m mask.nii.gz -o output_dir";
		  throw NeuroBayes::NeuroBayesException( __FILE__, __LINE__,
							 mess.c_str(),
							 ITK_LOCATION );
		}
	      // output directory exists?
	      if ( !directory_exists( output_dir ) )
		{
		  std::string mess = "The output directory is not correct.\n";
		  mess += "./vbhmm -c file.csv -m mask.nii.gz -o output_dir";
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
	      // Saving the mask as double precision
	      FilterType::Pointer filter = FilterType::New();
	      filter->SetOutputMinimum( 0. );
	      filter->SetOutputMaximum( 1. );
	      filter->SetInput( reader_mask_->GetOutput() );
	      //
	      std::string output_mask_name = output_dir + "/mask_double_precision.nii.gz";
	      Writer3D::Pointer writer = Writer3D::New();
	      writer->SetInput( filter->GetOutput() );
	      writer->SetFileName( output_mask_name.c_str() );
	      writer->Update();
 

	      //
	      // Number of THREADS in case of multi-threading
	      // this program hadles the multi-threading it self
	      // in no-debug mode
	      const int THREAD_NUM = 4;
	      //
	      // Load the CSV file
	      // Dim is the number of modalities in the subject's timepoint
	      // number_of_states is the first guess on the number of states
	      VB::HMM::SubjectMapping< /*Dim*/ 3, /*number_of_states*/ 5 > subject_mapping( filename, output_dir, time_tran );


	      
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
		      if ( idx[0] > 76 - 1  && idx[0] < 76 + 1 && 
			   idx[1] > 78 - 1  && idx[1] < 78 + 1 &&
			   idx[2] > 35 - 1  && idx[2] < 35 + 1 )
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
						   "./vbhmm -c file.csv -m mask.nii.gz  -o output_dir ",
						   ITK_LOCATION );
	}
      else
	throw NeuroBayes::NeuroBayesException( __FILE__, __LINE__,
					       "./vbhmm -c file.csv -m mask.nii.gz  -o output_dir ",
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
