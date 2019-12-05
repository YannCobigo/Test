#ifndef COMMAND_LINE_H
#define COMMAND_LINE_H
//
#include <stdio.h>
#include <sys/stat.h>
//
//
//
#include "Exception.h"
//
//
//
namespace NeuroBayes
{
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
  class InputParser
  {
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
    //
    std::string cmdOptionPrint() const
    {
      //
      //
      std::string options;
      std::vector< std::string >::const_iterator itr = tokens.begin();
      //
      for ( ; itr != tokens.end() ; ++itr )
	options += " " + *itr;
      std::cout << std::endl;
      //
      return options;
    }
  private:
    std::vector < std::string > tokens;
  };
}
#endif
