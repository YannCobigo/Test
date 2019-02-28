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
#include "Thread_dispatching.h"
#include "Exception.h"
#include "Load_csv.h"
#include "Tools.h"
#include "IO/Command_line.h"
//
//
//
int
main( const int argc, const char **argv )
{
  try
    {
      //
      // mask coverage for specific sudies
      // SPM T1 mask: 121x145x121
      // 112 = 56 + 32 + 16 => max diviser 16 |
      // 136 = 128 + 8 => max diviser 8       | Global max diviser = 8
      std::vector< std::vector< int > > extrema(3);
      extrema[0] = {5,117}; //X
      extrema[1] = {5,141}; //Y
      extrema[2] = {5,117}; //Z

      //
      // Parse the arguments
      //
      if( argc > 1 )
	{


	  //
	  // Arguments
	  NeuroBayes::InputParser input( argc, argv );
	  // Print the commande line
	  std::cout << "############# Commande line: ################" << std::endl;
	  std::cout << argv[0];
	  std::cout << input.cmdOptionPrint() << std::endl;
	  std::cout << "#############################################" << std::endl;
	  //
	  if( input.cmdOptionExists("-h") )
	    {
	      //
	      // It is the responsability of the user to create the 
	      // normalized/standardized hierarchical covariate
	      //
	      // -h                          : help
	      // -X   inv_cov_error.nii.gz   : (prediction) inverse of error cov on parameters
	      // -c   input.csv              : input file
	      // -m   mask.nii.gz            : mask
	      // -d   statistics             : statistics = {demean,normalize,standardize,load} 
	      //                               The later loads a statistics file
	      // -v   edge_cut               : window = (edge / edge_cut)x(edge / edge_cut)x(edge / edge_cut)
	      //                               You will have edge_cut x edge_cut x edge_cut windows
	      //                               Option paired with -w
	      //                               edge_cut = {1,2,4,8}
	      // -w   window                 : process a specific window w (c.f. option -v)
	      //                               window is in {1, ... , edge_cut x edge_cut x edge_cut }
	      //
	      std::string help = "It is the responsability of the user to create the ";
	      help += "normalized/standardized hierarchical covariate.\n";
	      help += "-h                          : help\n";
	      help += "-X   inv_cov_error.nii.gz   : (prediction) inverse of error cov on parameters\n";
	      help += "-c   input.csv              : input file\n";
	      help += "-o   output_dir             : output directory\n";
	      help += "-m   mask.nii.gz            : mask\n";
	      help += "-d   statistics             : statistics = {demean,normalize,standardize,load}\n";
	      help += "                              The later loads a statistics file\n";
	      help += "-v   edge_cut               : window = (edge / edge_cut)x(edge / edge_cut)x(edge / edge_cut)\n";
	      help += "                              You will have edge_cut x edge_cut x edge_cut windows\n";
	      help += "                              Option paired with -w. edge_cut = {1,2,4,8}\n";
	      help += "-w   window                 : process a specific window w (c.f. option -v)\n";
	      help += "                              window is in { 1, ... , edge_cut x edge_cut x edge_cut }\n";
	      throw NeuroBayes::NeuroBayesException( __FILE__, __LINE__,
						     help.c_str(),
						     ITK_LOCATION );
	    }

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
	  const std::string& str_edge_cut = input.getCmdOption("-v");
	  const std::string& str_window   = input.getCmdOption("-w");
	  int edge_cut = 0;
	  int window   = 0;
	  std::vector< int > chuck(3,0);
	  // Windows will hold the three bounds of the window
	  std::vector< std::vector< std::vector< int > > > windows;
	  if ( !str_edge_cut.empty() )
	    {
	      // how many time we divide the edges
	      edge_cut = std::stoi( str_edge_cut );
	      window   = std::stoi( str_window ) - 1;
	      //
	      windows.resize( edge_cut*edge_cut*edge_cut );
	      chuck[0] = static_cast< int >( (extrema[0][1] - extrema[0][0]) / edge_cut );
	      chuck[1] = static_cast< int >( (extrema[1][1] - extrema[1][0]) / edge_cut );
	      chuck[2] = static_cast< int >( (extrema[2][1] - extrema[2][0]) / edge_cut );
	      //
	      // 
	      int window_count = 0;
	      std::cout << "################## Windows ##################" << std::endl;
	      for ( int k = 0 ; k < edge_cut ; k++ )
		for ( int j = 0 ; j < edge_cut ; j++ )
		  for ( int i = 0 ; i < edge_cut ; i++ )
		    {
		      // initialize the bounds
		      windows[ window_count ].resize(3);
		      for ( int u = 0 ; u < 3 ; u ++)
			windows[ window_count ][u].resize(2);
		      // X
		      windows[ window_count ][0][0] = extrema[0][0] + i*chuck[0];
		      windows[ window_count ][0][1] = extrema[0][0] + (i+1)*chuck[0];
		      // Y
		      windows[ window_count ][1][0] = extrema[1][0] + j*chuck[1];
		      windows[ window_count ][1][1] = extrema[1][0] + (j+1)*chuck[1];
		      // Z
		      windows[ window_count ][2][0] = extrema[2][0] + k*chuck[2];
		      windows[ window_count ][2][1] = extrema[2][0] + (k+1)*chuck[2];
		      //
		      //
		      std::cout << "windows["<<window_count+1<<"] = [(" 
				<< windows[ window_count ][0][0] << "," << windows[ window_count ][0][1] << "), ("
				<< windows[ window_count ][1][0] << "," << windows[ window_count ][1][1] << "), ("
				<< windows[ window_count ][2][0] << "," << windows[ window_count ][2][1] << ")] "
				<< std::endl;
		      //
		      window_count++;
		    }
	      std::cout << "#############################################" << std::endl;
	    }
	  else
	    {
	      windows.resize(1);
	      windows[0].resize(3);
	      //
	      std::cout << "################## Windows ##################" << std::endl;
	      std::cout << "windows[0] = [";
	      //
	      for ( int u = 0 ; u < 3 ; u ++)
		{
		  windows[0][u].resize(2);
		  //
		  windows[0][u][0] = extrema[u][0];
		  windows[0][u][1] = extrema[u][1];
		  //
		  std::cout << "("<< windows[0][u][0] << "," << windows[0][u][1] << "),";
		}
	      //
	      std::cout << "]" << std::endl;
	      std::cout << "#############################################" << std::endl;
	    }
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
	      if ( !NeuroBayes::directory_exists( output_dir ) )
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
	      const int THREAD_NUM = 24;

	      //
	      // Load the CSV file
	      NeuroBayes::BmleLoadCSV< 3/*D_r*/, 0 /*D_f*/> subject_mapping( filename, output_dir,
									     time_tran, 
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
//		      // vertex
//		      if ( idx[0] > 92 - 5  && idx[0] < 92 + 5 && 
//			   idx[1] > 94 - 5  && idx[1] < 94 + 5 &&
//			   idx[2] > 63 - 5  && idx[2] < 63 + 5 )
//		      // ALL
//		      if ( idx[0] > 5 && idx[0] < 110 && 
//			   idx[1] > 5 && idx[1] < 140 &&
//			   idx[2] > 5 && idx[2] < 110 )
//		      // Window w
		      if ( idx[0] >= windows[window][0][0] && idx[0] < windows[window][0][1] && 
			   idx[1] >= windows[window][1][0] && idx[1] < windows[window][1][1] &&
			   idx[2] >= windows[window][2][0] && idx[2] < windows[window][2][1] )
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
