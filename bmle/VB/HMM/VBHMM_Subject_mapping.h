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
#include "VBHMM.h"
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
      explicit SubjectMapping( const std::string&, const std::string&, const NeuroStat::TimeTransformation );
    
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
      // Arrange pidns into groups
      std::map< std::string /*pidn*/, VB::HMM::Subject< Dim, Num_States > > group_pind_;
      // CSV file
      std::ifstream csv_file_;
      // output directory
      std::string   output_dir_;
      // Statistic transformation of ages
      std::string   age_statistics_tranformation_;
      // number of PIDN
      long unsigned int num_subjects_{0};
      // number of 3D images = number of time points (TP)
      long unsigned int num_3D_images_{0};

      //
      // Records
      //
      // Lower bound
      NeuroBayes::NeuroBayesMakeITKImage lower_bound_;

    };
    //
    //
    //
    template< int Dim, int Num_States >
      SubjectMapping< Dim, Num_States >::SubjectMapping( const std::string& CSV_file,
							 const std::string& Output_dir,
							 // Age Dns: demean, normalize, standardize
							 const NeuroStat::TimeTransformation Dns ):
      csv_file_{ CSV_file.c_str() }, output_dir_{ Output_dir }
      {
	try
	  {
	    //
	    //
	    double 
	      mean_age = 0.;
	    int
	      max_age  = 0,
	      min_age  = 70000;
	    std::list< int > age_stat;
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
		const std::string PIDN = cell;
		// Get the group
		std::getline(lineStream, cell, ',');
		const int group = std::stoi( cell );
		if ( group == 0 )
		  //
		  // The groups must be labeled 1, 2, 3, ...
		  throw NeuroBayes::NeuroBayesException( __FILE__, __LINE__,
							 "Select another group name than 0 (e.g. 1, 2, ...). 0 is reserved.",
							 ITK_LOCATION );
		// Get the age
		std::getline(lineStream, cell, ',');
		int age = std::stoi( cell );
		age_stat.push_back(age);
		mean_age += static_cast< double >( age );
		if ( age < min_age )
		  min_age = age;
		if ( age > max_age )
		  max_age = age;
		// Get the image
		std::string image;
		std::getline(lineStream, image, ',');
		// Covariates
		std::list< double > covariates;
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
		// If the PIDN does not yet exist
		if ( group_pind_.find( PIDN ) == group_pind_.end() )
		  {
		    std::cout << PIDN << " " << group << std::endl;
		    group_pind_[PIDN] = VB::HMM::Subject< Dim, Num_States >( PIDN, Output_dir );
		    num_subjects_++;
		  }
		//
		group_pind_[ PIDN ].add_tp( age, covariates, image );
		num_3D_images_++;
	      } // while( std::getline(csv_file_, line) )
	    //

	    // 
	    // Design Matrix for every subject
	    //

	    //
	    // mean age
	    mean_age /= static_cast< double >( num_3D_images_ );
//	    //
//	    Y_.resize( num_3D_images_ );
//	    int sub_image{0};
	    age_statistics_tranformation_ = Output_dir + "/age_transformation.txt";
	    //
	    switch ( Dns )
	      {
	      case NeuroStat::TimeTransformation::NONE:
	      case NeuroStat::TimeTransformation::DEMEAN:
		{
		  std::cout 
		    << "Age will be demeaned with mean age: " 
		    << mean_age << " or 0 infunction of the option you have chosen.\n"
		    << std::endl;
		  // record the transformation
		  std::ofstream fout( age_statistics_tranformation_ );
		  fout << Dns << " " 
		       << ( Dns == NeuroStat::TimeTransformation::NONE ? 0 : mean_age )
		       << std::endl;
		  fout.close();
//		  //
//		  for ( auto g : groups_ )
//		    for ( auto& s : group_pind_[g] )
//		      {
//			//
//			s.second.build_design_matrices( (Dns == NeuroStat::TimeTransformation::NONE ? 
//							 0 : mean_age) );
//			// Create the vector of 3D measurements image
//			for ( auto image : s.second.get_age_images() )
//			  Y_[ sub_image++ ] = image.second;
//		      }
		  //
		  break;
		}
	      case NeuroStat::TimeTransformation::NORMALIZE:
		{
		  std::cout 
		    << "Age will be normalized between: (" 
		    << min_age << "," 
		    << max_age << ").\n" 
		    << std::endl;
		  //
		  double 
		    C1 = static_cast< double >(min_age),
		    C2 = static_cast< double >(max_age - min_age);
		  //
		  Eigen::Matrix< double, 2, 2 > M;
		  M << 1., -C1/C2, 0., 1./C2;
		  std::cout 
		    <<  "C1 = min_age = " << min_age
		    <<  ", C2 = (max_age - min_age) = " << max_age - min_age
		    <<  ". change of variable is u = (t-C1)/C2. \n"
		    <<  "y = a0 + a1xt + epsilon\n"
		    <<  "  = b0 + b2xu + epsilon\n"
		    <<  "a = M x b. Where M = \n"
		    <<  M
		    << std::endl;
		  // record the transformation
		  std::ofstream fout( age_statistics_tranformation_ );
		  fout << Dns << " " 
		       << C1 << " " << C2
		       << std::endl;
		  fout.close();
//		  //
//		  for ( auto g : groups_ )
//		    for ( auto& s : group_pind_[g] )
//		      {
//			s.second.build_design_matrices( C1, C2 );
//			// Create the vector of 3D measurements image
//			for ( auto image : s.second.get_age_images() )
//			  Y_[ sub_image++ ] = image.second;
//		      }
		  //
		  break;
		}
	      case NeuroStat::TimeTransformation::STANDARDIZE:
		{
		  //
		  // Standard deviation
		  double accum = 0.0;
		  std::for_each ( std::begin( age_stat ), std::end( age_stat ), 
				  [&](const double d) {accum += (d - mean_age) * (d - mean_age);} );
		  //
		  double stdev = sqrt( accum / static_cast< double >(age_stat.size()-1) );
		  //
		  //
		  std::cout 
		    << "Age will be standardized with: (mu = " 
		    << mean_age << ", std = " 
		    << stdev << ").\n" 
		    << std::endl;
		  //
		  double 
		    C1 = mean_age,
		    C2 = stdev;
		  //
		  Eigen::Matrix< double, 2, 2 > M;
		  M << 1., -C1/C2, 0., 1./C2;
		  std::cout 
		    <<  "C1 = mu = " << C1
		    <<  ", C2 = stdev = " << stdev
		    <<  ". change of variable is u = (t-C1)/C2. \n"
		    <<  "y = a0 + a1xt + epsilon\n"
		    <<  "  = b0 + b2xu + epsilon\n"
		    <<  "a = M x b. Where M = \n"
		    <<  M
		    << std::endl;
		  // record the transformation
		  std::ofstream fout( age_statistics_tranformation_ );
		  fout << Dns << " " 
		       << C1 << " " << C2
		       << std::endl;
		  fout.close();
//		  //
//		  for ( auto g : groups_ )
//		    for ( auto& s : group_pind_[g] )
//		      {
//			s.second.build_design_matrices( C1, C2 );
//			// Create the vector of 3D measurements image
//			for ( auto image : s.second.get_age_images() )
//			  Y_[ sub_image++ ] = image.second;
//		      }
		  //
		  break;
		}
	      case NeuroStat::TimeTransformation::LOAD:
		{
		  // record the transformation
		  std::ifstream fin( age_statistics_tranformation_ );
		  std::stringstream stat_string;
		  // Check the file exist
		  if ( fin )
		    stat_string << fin.rdbuf();
		  else
		    {
		      std::string mess = "The statistic transformation file does not exist.\n";
		      mess += "Please, check on: " + age_statistics_tranformation_;
		      throw NeuroBayes::NeuroBayesException( __FILE__, __LINE__,
							     mess.c_str(),
							     ITK_LOCATION );
		    }
		  //
		  std::string str_time_transfo;
		  int time_transfo = 0;
		  //
		  stat_string >> str_time_transfo;
		  time_transfo = std::stoi( str_time_transfo );
		  std::cout << "Transforamtion: " << time_transfo << std::endl;
		  //
		  //
		  switch ( time_transfo )
		    {
		    case NeuroStat::TimeTransformation::NONE:
		    case NeuroStat::TimeTransformation::DEMEAN:
		      {
			//
			std::string    str_C1;
			stat_string >> str_C1;
			//
			double             C1 = std::stod( str_C1 );
			std::cout << "C1 " << C1 << std::endl;
//			//
//			for ( auto g : groups_ )
//			  for ( auto& s : group_pind_[g] )
//			    {
//			      s.second.build_design_matrices( C1 );
//			      // Create the vector of 3D measurements image
//			      for ( auto image : s.second.get_age_images() )
//				Y_[ sub_image++ ] = image.second;
//			    }
			//
			break;
		      }
		    case NeuroStat::TimeTransformation::NORMALIZE:
		    case NeuroStat::TimeTransformation::STANDARDIZE:
		      {
			std::string 
			  str_C1,
			  str_C2;
			stat_string >> str_C1 >> str_C2;
			//
			double          
			  C1 = std::stod( str_C1 ),
			  C2 = std::stod( str_C2 );
			std::cout << "C1 " << C1 << std::endl;
			std::cout << "C2 " << C2 << std::endl;
			//
//			for ( auto g : groups_ )
//			  for ( auto& s : group_pind_[g] )
//			    {
//			      s.second.build_design_matrices( C1, C2 );
//			      // Create the vector of 3D measurements image
//			      for ( auto image : s.second.get_age_images() )
//				Y_[ sub_image++ ] = image.second;
//			    }
			//
			break;
		      }
		    default:
		      {
			std::string mess = "The statistic transformation requiered is unknown.\n";
			mess += "Please, check on: " + age_statistics_tranformation_;
			throw NeuroBayes::NeuroBayesException( __FILE__, __LINE__,
							       mess.c_str(),
							       ITK_LOCATION );
		      }
		    }

	      
		  //
		  //
		  fin.close();
		  //
		  break;
		}
	      }

	    //
	    // Create output images
	    std::string
	      lower_bound_name = output_dir_ + "/" + "lower_bounld.nii.gz";
	    //
	    //lower_bound_ = NeuroBayes::NeuroBayesMakeITKImage( 1, lower_bound_name, Y_[0] );
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
	    //
	    // Create the all subjects time series
	    std::vector< std::vector< Eigen::Matrix< double, Dim, 1 > > > intensity( num_subjects_ );
	    std::vector< std::vector< Eigen::Matrix< double, 1, 1 > > >   age( num_subjects_ );
	    // 
	    int subject_number = 0;
	    std::map< std::string /*pidn*/, int /*subject_number*/ > pind_subject_num;
	    for ( auto& s : group_pind_ )
	      {
		s.second.build_time_series( Idx, 
					    intensity[subject_number], 
					    age[subject_number] );
		//
		pind_subject_num[ s.first ] = subject_number++;
	      }
	    //
	    // Run the model
	    VB::HMM::Hidden_Markov_Model < Dim, Num_States > hidden_Markov_model( intensity, age );
	    //
	    //hidden_Markov_model.ExpectationMaximization();

	    for ( int s = 0 ; s < num_subjects_ ; s++ )
	      {
		std::cout << "Subject " << s << std::endl;
		int Ti = intensity[s].size();
		for ( int tp = 0 ; tp < Ti ; tp++ )
		  std::cout << intensity[s][tp] << std::endl; 
	      }
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
