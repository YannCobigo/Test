#ifndef VBHMMSUBJECTMAPPING_H
#define VBHMMSUBJECTMAPPING_H
//
//
//
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <algorithm>
#include <vector>
#include <list>
#include <map>
#include <set>
#include <math.h>
//#include <cmath.h>
// Eigen
#include <Eigen/Core>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/KroneckerProduct>
//
// ITK
//
#include <itkImageFileReader.h>
#include <itkSpatialOrientationAdapter.h>
#include "itkChangeInformationImageFilter.h"
using MaskType   = itk::Image< unsigned char, 3 >;
using MaskType4D = itk::Image< unsigned char, 4 >;
//
//
//
#include "Exception.h"
#include "MakeITKImage.h"
#include "Tools.h"
#include "VBHMM_Subject.h"
//
//
//
namespace VB
{
  namespace HMM
  {
    /** \class SubjectMapping
     *
     * \brief 
     * 
     */
    template< int Dim, int Num_States >
      class SubjectMapping
    {
    public:
      /** Constructor. */
      explicit SubjectMapping( const std::string&, const std::string& );
    
      /** Destructor */
      virtual ~SubjectMapping() {};


      //
      // Expectation maximization algorithm
      void Expectation_Maximization( MaskType::IndexType );
      // Write the output
      void write_subjects_solutions( );
      // multi-threading
      void operator ()( const MaskType::IndexType idx )
      {
	std::cout << "treatment for parameters: " 
		  << idx;
	Expectation_Maximization( idx );
      };


    private:
      //
      // Functions
      //


      //
      // Members
      //
    
    };
    //
    //
    //
    template< int Dim, int Num_States >
      SubjectMapping< Dim, Num_States >::SubjectMapping( const std::string& CSV_file,
							 const std::string& Output_dir )
      {
	try
	  {
//	    //
//	    //
//	    double mean_age = 0.;
//	    std::string line;
//	    //skip the first line
//	    std::getline(csv_file_, line);
//	    //
//	    // then loop
//	    while( std::getline(csv_file_, line) )
//	      {
//		std::stringstream  lineStream( line );
//		std::string        cell;
//		std::cout << "ligne: " << line << std::endl;
//
//		//
//		// Get the PIDN
//		std::getline(lineStream, cell, ',');
//		const std::string PIDN = cell;
//		// Get the group
//		std::getline(lineStream, cell, ',');
//		const int group = std::stoi( cell );
//		if ( group == 0 )
//		  //
//		  // The groups must be labeled 1, 2, 3, ...
//		  throw NeuroBayes::NeuroBayesException( __FILE__, __LINE__,
//							 "Select another group name than 0 (e.g. 1, 2, ...). 0 is reserved.",
//							 ITK_LOCATION );
//		// Get the age
//		std::getline(lineStream, cell, ',');
//		int age = std::stoi( cell );
//		mean_age += static_cast< double >( age );
//		// Get the image
//		std::string image;
//		std::getline(lineStream, image, ',');
//		// Covariates
//		std::list< double > covariates;
//		while( std::getline(lineStream, cell, ',') )
//		  covariates.push_back( std::stof(cell) );
//		num_covariates_ = covariates.size();
//
//		//
//		// check we have less than 10 groups
//		if( group > 10 )
//		  throw NeuroBayes::NeuroBayesException( __FILE__, __LINE__,
//							 "The CSV file should have less than 10 gourps.",
//							 ITK_LOCATION );
//		// If the PIDN does not yet exist
//		if ( group_pind_[ group ].find( PIDN ) == group_pind_[ group ].end() )
//		  {
//		    std::cout << PIDN << " " << group << std::endl;
//		    groups_.insert( group );
//		    group_pind_[ group ][PIDN] = BmleSubject< Dim, Num_States >( PIDN, group, Output_dir );
//		    group_num_subjects_[ group ]++;
//		    num_subjects_++;
//		  }
//		//
//		group_pind_[ group ][ PIDN ].add_tp( age, covariates, image );
//		num_3D_images_++;
//	      }
//	    //
//
//	    // 
//	    // Design Matrix for every subject
//	    //
//
//	    //
//	    // mean age
//	    mean_age /= static_cast< double >( num_3D_images_ );
//	    if ( Demeaning )
//	      std::cout << "mean age: " << mean_age << std::endl;
//	    else
//	      std::cout << "No demeaning " << std::endl;
//	    //
//	    Y_.resize( num_3D_images_ );
//	    int sub_image{0};
//	    for ( auto g : groups_ )
//	      for ( auto& s : group_pind_[g] )
//		{
//		  s.second.build_design_matrices( (Demeaning ? mean_age : 0) );
//		  // Create the vector of 3D measurements image
//		  for ( auto image : s.second.get_age_images() )
//		    Y_[ sub_image++ ] = image.second;
//		}
	  }
	catch( itk::ExceptionObject & err )
	  {
	    std::cerr << err << std::endl;
	    exit( -1 );
	  }
      }
    //
    //
    //
    template< int Dim, int Num_States > void
      SubjectMapping< Dim, Num_States >::Expectation_Maximization( MaskType::IndexType Idx )
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
    //
    //
    //
    template< int Dim, int Num_States > void
      SubjectMapping< Dim, Num_States >::write_subjects_solutions( )
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
  }
}
#endif
