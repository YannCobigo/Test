#include<iostream>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <vector>
#include <chrono>
#include <memory>
//
// JSON interface
//
#include "json.hpp"
using json = nlohmann::json;
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
#include "MACException.h"
#include "MACLoadDataSet.h"
#include "Classification.h"
#include "Classification_linear_regression.h"
#include "Classification_logistic_regression.h"
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
	    throw MAC::MACException( __FILE__, __LINE__,
				     "./WMH_classification -c data_set.json",
				     ITK_LOCATION );

	  //
	  // takes the csv file ans the mask
	  const std::string& filename = input.getCmdOption("-c");
	  //const std::string& mask     = input.getCmdOption("-m");
	  //
	  if ( !filename.empty() )
	    {
	      //
	      // Load the data set
	      MAC::Singleton::instance( filename );
	      // print the data set
	      MAC::Singleton::instance()->print_data_set();
	      // load the mask
	      std::string mask = MAC::Singleton::instance()->get_data()["inputs"]["mask"].get< std::string >();
	      

	      if ( mask.empty() )
		{
		  std::string mess = "No mask loaded. A mask must be loaded.\n";
		  mess += "./WMH_classification -c data_set.json ";
		  throw MAC::MACException( __FILE__, __LINE__,
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
	      const int THREAD_NUM = 16;

	      //
	      // Create the feature mapping for each voxel
	      //MAC::Classification_linear_regression< /*Dim = */ 2 > features_mapping;
	      MAC::Classification_logistic_regression< /*Dim = */ 2 > features_mapping;
	      features_mapping.load_parameters_images();


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
	      using ms           = std::chrono::milliseconds;
	      using get_time     = std::chrono::steady_clock ;
	      auto  start_timing = get_time::now();
	      
	      //
	      // loop over Mask area for every images
#ifndef DEBUG
	      std::cout << "Multi-threading" << std::endl;
	      // Start the pool of threads
	      {
		MAC::Thread_dispatching pool( THREAD_NUM );
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
			    features_mapping.optimize( idx );
			  }
#else
			// Please do not remove the bracket!!
//			if ( idx[0] > 60 && idx[0] < 62 && 
//			     idx[1] > 90 && idx[1] < 92 &&
//			     idx[2] > 100 && idx[2] < 102 )
			if ( idx[0] > 0 && idx[0] < 180 && 
			     idx[1] > 0 && idx[1] < 210 &&
			     idx[2] > 0 && idx[2] < 182 )
			  {
			    pool.enqueue( std::ref(features_mapping), idx );
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
	      if ( MAC::Singleton::instance()->get_status() )
		features_mapping.write_parameters_images();
	      else
		features_mapping.write_subjects_map();
	      std::cout << "All output have been written." << std::endl;
	    }
	  else
	    throw MAC::MACException( __FILE__, __LINE__,
				     "./WMH_classification -c data_set.json",
				     ITK_LOCATION );
	}
      else
	throw MAC::MACException( __FILE__, __LINE__,
				 "./WMH_classification -c data_set.json",
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
