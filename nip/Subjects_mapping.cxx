#include "Subjects_mapping.h"
//
//
//
MAC_nip::NipSubject_Mapping::NipSubject_Mapping( const std::string& CSV_file,
						 const std::string& Mask ):
  csv_file_{ CSV_file.c_str()}, mask_{Mask}
{
  try
    {
      std::string line;
      //skip the first line
      std::getline(csv_file_, line);
      //
      // then loop
      while( std::getline(csv_file_, line) )
	{
	  std::stringstream  lineStream( line );
	  std::string        cell;
	  std::cout << line << std::endl;
	  
	  //
	  // Get the PIDN
	  std::string PIDN;
	  std::getline(lineStream, PIDN, ',');
	  // Get the group
	  std::getline(lineStream, cell, ',');
	  const int group = std::stoi( cell );
	  if ( group == 0 )
	    throw NipException( __FILE__, __LINE__,
				"Select another group name than 0 (e.g. 1, 2, ...). 0 is reserved.",
				ITK_LOCATION );
	  // Get the image
	  std::string image;
	  std::getline(lineStream, image, ',');
	  // Covariates
	  std::list< double > covariates;
	  while( std::getline(lineStream, cell, ',') )
	    covariates.push_back( std::stof(cell) );
	  //
	  //
	  if ( PIDNs_.find(PIDN) == PIDNs_.end() )
	    {
	      PIDNs_.insert( PIDN );
	      groups_.insert( group );
	      group_pind_[ group ].push_back( NipSubject( group, PIDN, image, Mask, covariates ) );
	    }
	  else
	    {
	      std::string mess = "PIDN " + PIDN + " was already inserted in the dtaset.";
	      throw NipException( __FILE__, __LINE__,
				  mess.c_str(),
				  ITK_LOCATION );
	    }
	}
    }
  catch( itk::ExceptionObject & err )
    {
      std::cerr << err << std::endl;
      exit( -1 );
    }
}
