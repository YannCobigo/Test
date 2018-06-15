#include "Subjects_mapping.h"
//
//
//
MAC_nip::NipSubject_Mapping::NipSubject_Mapping( const std::string& CSV_file,
						 const std::string& Mask ):
  csv_file_{CSV_file.c_str()}, mask_{Mask}
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

      //
      // Build the group matrices
      for ( auto g : groups_ )
	{
	  //
	  // 
	  int
	    n = group_pind_[g].size(),
	    p = group_pind_[g][0].get_image_matrix().rows(),
	    q = group_pind_[g][0].get_ev_matrix().rows();

	  //
	  // Build matrics
	  Eigen::MatrixXd Image_matrix = Eigen::MatrixXd::Zero(n,p);
	  Eigen::MatrixXd EV_matrix    = Eigen::MatrixXd::Zero(n,q);
	  for ( int s = 0 ; s < n ; s++ )
	    {
	      Image_matrix.row(s) = group_pind_[g][s].get_image_matrix().col(0);
	      if ( q > 0 )
		EV_matrix.row(s) = group_pind_[g][s].get_ev_matrix().col(0);
	    }

	  //
	  // Build spectrum
	  std::size_t K_spectrum = 0;
	  if ( q > 0 )
	    K_spectrum = (p > q ? q : p);
	  else
	    throw NipException( __FILE__, __LINE__,
				"ERROR: need to build the case for PCA (SPC).",
				ITK_LOCATION );
	  //
	  Spectra matrix_spetrum( K_spectrum );
	  for ( int k = 0 ; k < K_spectrum ; k++ )
	    {
	      if ( q > 0 )
		{
		  // Coefficient
		  std::get< coeff_k >( matrix_spetrum[k] ) = 0.;
		  // vectors
		  std::get< Uk >( matrix_spetrum[k] ) = Eigen::MatrixXd::Random( p, 1 );
		  std::get< Vk >( matrix_spetrum[k] ) = Eigen::MatrixXd::Random( q, 1 );
		  // normalization
		  std::get< Uk >( matrix_spetrum[k] ) /= std::get< Uk >( matrix_spetrum[k] ).lpNorm< 2 >();
		  std::get< Vk >( matrix_spetrum[k] ) /= std::get< Vk >( matrix_spetrum[k] ).lpNorm< 2 >();
		}
	      else
		throw NipException( __FILE__, __LINE__,
				    "ERROR: need to build the case for PCA (SPC).",
				    ITK_LOCATION );
	    }

	  //
	  // Create the tuple
	  group_matrices_[g] = std::make_tuple( std::make_shared< const Eigen::MatrixXd >(Image_matrix),
						std::make_shared< const Eigen::MatrixXd >(EV_matrix),
						std::make_shared< Spectra >(matrix_spetrum) );
	}
    }
  catch( itk::ExceptionObject & err )
    {
      std::cerr << err << std::endl;
      exit( -1 );
    }
}
