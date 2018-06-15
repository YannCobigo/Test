#include <iostream>
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
//#include "Thread_dispatching.h"
#include "NipException.h"
#include "Subjects_mapping.h"
#include "PMA.h"
#include "PMA_tools.h"
#include "PMD.h"
#include "SPC.h"
#include "PMA_cross_validation.h"
#include "PMD_cross_validation.h"
#include "SPC_cross_validation.h"
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
	    throw MAC_nip::NipException( __FILE__, __LINE__,
					   "./nip -c file.csv -m mask.nii.gz >",
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
		  mess += "./nip -c file.csv -m mask.nii.gz";
		  throw MAC_nip::NipException( __FILE__, __LINE__,
						 mess.c_str(),
						 ITK_LOCATION );
		}

	      ////////////////////////////
	      ///////              ///////
	      ///////  PROCESSING  ///////
	      ///////              ///////
	      ////////////////////////////

	      //
	      // Subject mapping
	      MAC_nip::NipSubject_Mapping mapping( filename, mask );
	      
	      //
	      // Task progress: elapse time
	      using  ms         = std::chrono::milliseconds;
	      using get_time    = std::chrono::steady_clock ;
	      auto start_timing = get_time::now();

	      //
	      // Optimize spectrum
	      MAC_nip::Nip_PMD_cross_validation< /* K-folds = */ 3, /* CPU */ 8 > pmd_cv( std::get< 0 /*image*/ >(mapping.get_PMA()[1]),
											  std::get< 1 /*EV*/ >(mapping.get_PMA()[1]));
	      pmd_cv.validation( std::get< 2 /*spectrum*/ >(mapping.get_PMA()[1]) );


	      
	      //
	      // Task progress
	      // End the elaps time
	      auto end_timing  = get_time::now();
	      auto diff        = end_timing - start_timing;
	      std::cout << "Process Elapsed time is :  " << std::chrono::duration_cast< ms >(diff).count()
			<< " ms "<< std::endl;

	    }
	}
      else
	throw MAC_nip::NipException( __FILE__, __LINE__,
				       "./nip -c file.csv -m mask.nii.gz >",
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
