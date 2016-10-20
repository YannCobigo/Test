#include <algorithm>
#include <vector>
#include <list>
//
// ITK
//
//
//
//
#include "Load_csv.h" 
#include "BmleException.h"
//
//
//
MAC_bmle::BmleLoadCSV::BmleLoadCSV( const std::string& CSV_file ):
  csv_file_{ CSV_file.c_str() }
  {
    try
      {
	//
	//
	std::string line;
	//skip the first line
	std::getline(csv_file_, line);
	//
	// then loop
	while( std::getline(csv_file_, line) )
	  {
	    std::stringstream  lineStream( line );
	    std::string        cell;
	    std::cout << "ligne: " << line << std::endl;

	    //
	    // Get the PIDN
	    std::getline(lineStream, cell, ',');
	    const int PIDN = std::stoi( cell );
	    // Get the group
	    std::getline(lineStream, cell, ',');
	    const int group = std::stoi( cell );
	    // Get the age
	    std::getline(lineStream, cell, ',');
	    int age = std::stoi( cell );
	    // Get the image
	    std::string image;
	    std::getline(lineStream, image, ',');
	    // Covariates
	    std::list< float > covariates;
	    while( std::getline(lineStream, cell, ',') )
	      covariates.push_back( std::stof(cell) );

	    //
	    // check we have less than 10 groups
	    if( group > 10 )
	      throw BmleException( __FILE__, __LINE__,
				   "The CSV file should have less than 10 gourps.",
				   ITK_LOCATION );
	    // If the PIDN does not yet exist
	    if ( group_pind_[ group ].find( PIDN ) == group_pind_[ group ].end() )
	      {
		groups_.insert( group );
		group_pind_[ group ][PIDN] = BmleSubject( PIDN, group );
	      }
	    //
	    group_pind_[ group ][ PIDN ].add_tp( age, covariates, image );
	  }
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
	exit( -1 );
      }


    //
    //
    if (true)
      for ( auto g : groups_ )
	for ( auto s : group_pind_[g] )
	  {
	    s.second.print();
	    //s.second.build_covariates_matrix();
	  }
  }
//
//
//
void
MAC_bmle::BmleLoadCSV::image_cat()
{
  try
    {
    }
  catch( itk::ExceptionObject & err )
    {
      std::cerr << err << std::endl;
      exit( -1 );
    }
}
