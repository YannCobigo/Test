#include<iostream>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <vector>
//
// 
//
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
      {
	return *itr;
      }

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
  //
  // Parse the arguments
  //
  if( argc > 1 )
    {
      InputParser input( argc, argv );
      if( input.cmdOptionExists("-h") )
	{
	  std::cerr << "./bmle -c file.csv < -m mask.nii.gz >" << std::endl;
	  std::exit(0);
	}

      //
      // takes the csv file ans the mask
      const std::string& filename = input.getCmdOption("-c");
      const std::string& mask     = input.getCmdOption("-m");
      //
      if ( !filename.empty() )
	{
	  if ( mask.empty() )
	    std::cout << "No mask loaded. It would speed up the process to load a mask." << std::endl;
	  else
	    std::cout << "You are loading the mask: " << mask << std::endl;
	  //
	  // Load the CSV file
	  //
	  MAC_bmle::BmleLoadCSV file ( filename );
	}
      else
	{
	  std::cerr << "./bmle -c file.csv < -m mask.nii.gz >" << std::endl;
	  std::exit(0);
	}
    }
  else
    {
      std::cerr << "./bmle -c file.csv < -m mask.nii.gz >" << std::endl;
      std::exit(0);
    }

  //
  //
  //
  return EXIT_SUCCESS;
}
