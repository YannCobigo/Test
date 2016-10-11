#include <algorithm>
#include <vector>
#include <list>
#include <map>
//
//
//
#include "Load_csv.h" 
//
//
//
MAC_bmle::BmleLoadCSV::BmleLoadCSV( const std::string& CSV_file ):
  csv_file_{ CSV_file.c_str() }
  {
    //
    //
    std::string line;
    //skip the first line
    std::getline(csv_file_, line);
    //
    // then loop
    std::map< int, BmleSubject > subjects;
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

	// Scoped map we will free the space by move opertor
	if ( subjects.find( PIDN ) == subjects.end() )
	  subjects[ PIDN ] = BmleSubject( PIDN, group );
	//
	subjects[ PIDN ].add_tp( age, covariates, image );
      }

    //
    //
    if (true)
      for ( auto s : subjects )
	{
	  s.second.print();
	  s.second.build_covariates_matrix();
	}
  }
