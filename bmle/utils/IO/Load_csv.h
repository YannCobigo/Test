#ifndef LOAD_CSV_H
#define LOAD_CSV_H
//
#include <cstdlib>
#include <string>
#include <sstream>
#include <fstream>
// Eigen
#include <Eigen/Core>
#include <Eigen/Eigen>
#include <Eigen/Sparse>
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
  //
  /** \class Load_csv
   *
   * \brief Load CSV files
   * 
   *
   *
   */
  class Load_csv
  {
  public:
    // Constructor
  Load_csv( const std::string File_name,
	    const char        Delimiter = ',' ):
    file_name_{ File_name }, delimiter_{ Delimiter }
    {};
    // Destructor
    ~Load_csv(){};
    
    template < int /*Dim*/ Dim, int /*number_of_states*/ S >
      std::vector< std::vector< Eigen::Matrix< double, /*Dim*/ Dim, 1 > > >
      get_VB_HMM_date();

  private:
    //
    std::string  file_name_;
    char         delimiter_;
    
  };

  //
  //
  // The number of timepoints can be different from one
  // subject to the other.
  template < int /*Dim*/ Dim, int /*number_of_states*/ S >
    std::vector< std::vector< Eigen::Matrix< double, /*Dim*/ Dim, 1 > > >
    Load_csv::get_VB_HMM_date()
    {
      //
      //
      std::ifstream file( file_name_ );
      std::vector< std::vector< Eigen::Matrix< double, /*Dim*/ Dim, 1 > > > dataset;
      std::string line = "";
      // 
      int dim = 0;
      std::vector< Eigen::Matrix < double, Dim , 1 > > timepoints;
      //
      while ( file.good() )
	{
	  //
	  // grab the line
	  getline( file, line );
	  //
	  std::stringstream  lineStream( line );
	  std::string        cell;
	  //
	  if ( dim == 0 )
	    {
	      //
	      // new PIDN
	      // clear the timpoints
	      timepoints.clear();
	      //
	      while( std::getline( lineStream, cell, delimiter_ ) )
		{
		  Eigen::Matrix < double, Dim , 1 >
		    timepoint = Eigen::Matrix < double, Dim , 1 >::Zero();
		  timepoint(0,0) = std::stof(cell);
		  // load the matrix
		  timepoints.push_back( timepoint );
		}
	    }
	  else
	    {
	      //
	      // take the pointer of data
	      typename std::vector< Eigen::Matrix < double, Dim , 1 > >::iterator it;
	      it = timepoints.begin();
	      while( std::getline( lineStream, cell, delimiter_ ) )
		{
		  (*it)(dim,0) = std::stof(cell);
		  it++;
		}
	    }
	  //
	  // check the next dimension
	  dim++;
	  if ( dim == Dim )
	    {
	      // We change PIDN
	      if ( timepoints.size() > 1 )
		dataset.push_back( timepoints );
	      dim = 0;
	    }
	}
      //
      // Close the File
      file.close();

      //
      //
      return dataset;
    }
}
#endif
