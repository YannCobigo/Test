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
using MaskType    = itk::Image< unsigned char, 3 >;
using MaskType4D  = itk::Image< unsigned char, 4 >;
using Image3DType = itk::Image< double, 3 >;
using Reader3D    = itk::ImageFileReader< Image3DType >;
using Image4DType = itk::Image< double, 4 >;
using Reader4D    = itk::ImageFileReader< Image4DType >;
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
    /** \class Activity
     *
     * \brief Determine the type of action taken in the HMM calculation
     * 
     */
    enum Activity { UNKNOWN, FIT, PROJECTION };
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
      explicit SubjectMapping( const std::string&, const std::string&,
			       const NeuroStat::TimeTransformation );
      /** Constructor. */
      explicit SubjectMapping( const std::string&, const std::string&, const std::string&,
			       const std::string&, const std::string&, const std::string&,
			       const NeuroStat::TimeTransformation );
    
      /** Destructor */
      virtual ~SubjectMapping() {};


      //
      // Expectation maximization algorithm
      void Expectation_Maximization( MaskType::IndexType );
      //
      // Expectation maximization algorithm
      void projection_estimation( MaskType::IndexType );
      // Write the output
      void write_subjects_solutions( );
      // multi-threading
      void operator ()( const MaskType::IndexType idx )
      {
	std::cout << "treatment for parameters: " 
		  << idx;
	switch ( activity_ )
	  {
	  case FIT:
	    {
	      Expectation_Maximization( idx );
	      break;
	    }
	  case PROJECTION:
	    {
	      projection_estimation( idx );
	      break;
	    }
	  case UNKNOWN:
	  default:
	    {
	      std::string mess = "HMM activity is unknown.\n";
	      throw NeuroBayes::NeuroBayesException( __FILE__, __LINE__,
						     mess.c_str(),
						     ITK_LOCATION );
	    }
	  }
      };


    private:
      //
      // Functions
      //
      void build_outputs();

      //
      // Members
      //
      // Activity of the HMM
      Activity activity_{UNKNOWN};
      // Arrange pidns into groups
      std::map< std::string /*pidn*/, VB::HMM::Subject< Dim, Num_States > > group_pind_;
      // CSV file
      std::ifstream     csv_file_;
      // output directory
      std::string       output_dir_;
      // Statistic transformation of ages
      std::string       age_statistics_tranformation_;
      // number of PIDN
      long unsigned int num_subjects_{0};
      // number of 3D images = number of time points (TP)
      long unsigned int num_3D_images_{0};
      //
      // Projection variables
      //
      // PIDN last state
      Reader4D::Pointer PIDN_states_;
      // Transition matrix
      Reader4D::Pointer A_;
      // Transition matrix
      Reader4D::Pointer Gauss_;
      // Transition matrix
      Reader4D::Pointer Covariance_;
      // Number of time the projection is going to be done
      double            number_projection_;
      // number of iteration used to sample the projected image
      long int          number_sampling_iteration_{10000};
      
      //
      // Records
      //
      // Lower bound
      NeuroBayes::NeuroBayesMakeITKImage lower_bound_;
      // Transition matrix
      NeuroBayes::NeuroBayesMakeITKImage transition_matrix_;
      // First states
      NeuroBayes::NeuroBayesMakeITKImage first_states_;
      // Mu: cluster centroids
      NeuroBayes::NeuroBayesMakeITKImage mu_;
      // Precision of the matrix
      NeuroBayes::NeuroBayesMakeITKImage variance_;
      //
      // Projection variables
      //
      // Image projection
      NeuroBayes::NeuroBayesMakeITKImage projection_;
      // Image projection covariance
      NeuroBayes::NeuroBayesMakeITKImage projection_covariance_;
      // State projection
      NeuroBayes::NeuroBayesMakeITKImage projection_states_;
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
	  activity_ = FIT;
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
	      std::string image = "";
	      std::getline(lineStream, image, ',');
	      // Covariates11
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
	  build_outputs();
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
    template< int Dim, int Num_States >
      SubjectMapping< Dim, Num_States >::SubjectMapping( const std::string& PIDN_file,
							 const std::string& Transition_matrix,
							 const std::string& Gauss_clusters,
							 const std::string& Covariance_gaussians,
							 const std::string& Number_projections,
							 const std::string& Output_dir,
							 // Age Dns: demean, normalize, standardize
							 const NeuroStat::TimeTransformation Dns ):
      csv_file_{ "" }, output_dir_{ Output_dir }
    {
      try
	{
	  //
	  //
	  activity_ = PROJECTION;

	  //
	  // Load the last state of the PIDN
	  auto PIDN_ptr = itk::ImageIOFactory::CreateImageIO( PIDN_file.c_str(),
							      itk::ImageIOFactory::ReadMode );
	  PIDN_ptr->SetFileName( PIDN_file.c_str() );
	  PIDN_ptr->ReadImageInformation();
	  // Read the ITK image
	  PIDN_states_ = Reader4D::New();
	  PIDN_states_->SetFileName( PIDN_ptr->GetFileName() );
	  PIDN_states_->Update();

	  //
	  // Load the trasition matrix
	  auto A_ptr = itk::ImageIOFactory::CreateImageIO( Transition_matrix.c_str(),
							   itk::ImageIOFactory::ReadMode );
	  A_ptr->SetFileName( Transition_matrix.c_str() );
	  A_ptr->ReadImageInformation();
	  // Read the ITK image
	  A_ = Reader4D::New();
	  A_->SetFileName( A_ptr->GetFileName() );
	  A_->Update();

	  //
	  // Load the gaussian centroids
	  auto Gauss_ptr = itk::ImageIOFactory::CreateImageIO( Gauss_clusters.c_str(),
							       itk::ImageIOFactory::ReadMode );
	  Gauss_ptr->SetFileName( Gauss_clusters.c_str() );
	  Gauss_ptr->ReadImageInformation();
	  // Read the ITK image
	  Gauss_ = Reader4D::New();
	  Gauss_->SetFileName( Gauss_ptr->GetFileName() );
	  Gauss_->Update();

	  //
	  // Load the Gaussian covariance matrix
	  auto Covariance_ptr = itk::ImageIOFactory::CreateImageIO( Covariance_gaussians.c_str(),
								    itk::ImageIOFactory::ReadMode );
	  Covariance_ptr->SetFileName( Covariance_gaussians.c_str() );
	  Covariance_ptr->ReadImageInformation();
	  // Read the ITK image
	  Covariance_ = Reader4D::New();
	  Covariance_->SetFileName( Covariance_ptr->GetFileName() );
	  Covariance_->Update();

	  //
	  // Number of time the projection os going to be done
	  number_projection_ = std::stod( Number_projections );

	  //
	  // Create the projection file
	  // We load the mask we created in the main
	  std::string output_mask_name = output_dir_ + "/mask_double_precision.nii.gz";
	  auto image_ptr = itk::ImageIOFactory::CreateImageIO( output_mask_name.c_str(),
							       itk::ImageIOFactory::ReadMode );
	  image_ptr->SetFileName( output_mask_name.c_str() );
	  image_ptr->ReadImageInformation();
	  // Read the ITK image
	  Reader3D::Pointer tempo = Reader3D::New();
	  tempo->SetFileName( image_ptr->GetFileName() );
	  tempo->Update();
	  // outputs
	  std::string
	    projection_name            = output_dir_   + "/",
	    projection_covariance_name = output_dir_   + "/",
	    projection_states_name     = output_dir_   + "/";
	  projection_name             += "projection_" + Number_projections       + "_times_";
	  projection_name             += std::to_string(Dim)                      + "_dimensions_";
	  projection_name             += std::to_string(Num_States)               + "_states.nii.gz";
	  projection_covariance_name  += "projection_cov_" + Number_projections   + "_times_";
	  projection_covariance_name  += std::to_string(Dim)                      + "_dimensions_";
	  projection_covariance_name  += std::to_string(Num_States)               + "_states.nii.gz";
	  projection_states_name      += "state_projection_" + Number_projections + "_times_";
	  projection_states_name      += std::to_string(Dim)                      + "_dimensions_";
	  projection_states_name      += std::to_string(Num_States)               + "_states.nii.gz";
	  //
	  projection_            = NeuroBayes::NeuroBayesMakeITKImage( Dim,
								       projection_name, tempo );
	  projection_covariance_ = NeuroBayes::NeuroBayesMakeITKImage( Dim*Dim,
								       projection_covariance_name, tempo );
	  projection_states_     = NeuroBayes::NeuroBayesMakeITKImage( Num_States,
								       projection_states_name, tempo );
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
	    std::map< std::string /*pidn*/, int /*subject_number*/ > pidn_subject_num;
	    for ( auto& s : group_pind_ )
	      {
		s.second.build_time_series( Idx, 
					    intensity[subject_number], 
					    age[subject_number] );
		//
		pidn_subject_num[ s.first ] = subject_number++;
	      }
	    //
	    // Run the model
	    VB::HMM::Hidden_Markov_Model < Dim, Num_States > hidden_Markov_model( intensity, age );
	    //
	    hidden_Markov_model.ExpectationMaximization();

	    //
	    //
	    // record the outputs
	    //
	    //

	    //
	    // Globals
	    //
	    // lower bound
	    lower_bound_.set_val( 0, Idx,
				  hidden_Markov_model.get_lower_bound() );
	    //
	    // Transition matrix and first states
	    const Eigen::Matrix< double, Num_States, 1 >&
	      fs = hidden_Markov_model.get_first_states();
	    const Eigen::Matrix< double, Num_States, Num_States >&
	      tm = hidden_Markov_model.get_transition_matrix();
	    // Centroids and variances
	    const std::vector< Eigen::Matrix< double, Dim, 1 > >&
	      mu_vec = hidden_Markov_model.get_mu();
	    const std::vector< Eigen::Matrix< double, Dim, Dim > >&
	      var_mat = hidden_Markov_model.get_var();
	    //
	    int
	      mat_index = 0,
	      mu_index  = 0,
	      var_index = 0;
	    for ( int s = 0 ; s < Num_States ; s++ )
	      {
		// Centroids and variances
		for ( int d = 0 ; d < Dim ; d++ )
		  {
		    mu_index = d + Dim * s;
		    mu_.set_val( mu_index, Idx, mu_vec[s](d,0) );
		    //
		    for ( int dd = 0 ; dd < Dim ; dd++ )
		      {
			var_index = dd + Dim * d + Dim * Dim * s;
			variance_.set_val( var_index, Idx, var_mat[s](d,dd) );
		      }
		  }
		// Transition matrix
		for ( int ss = 0 ; ss < Num_States ; ss++ )
		  {
		    mat_index = ss + Num_States * s;
		    transition_matrix_.set_val( mat_index, Idx, tm(s,ss) );
		  }
		// First states
		first_states_.set_val( s, Idx, fs(s,0) );
	      }

	    //
	    //
	    // locals
	    //
	    // Subjects
	    for ( auto &s : group_pind_ )
	      {
		s.second.record_state( Idx,
				       (hidden_Markov_model.get_N())[ pidn_subject_num[s.first] ] );
	      }
	    

	    
	    //	    //
	    //	    // Other things to remove
	    //	    for ( int s = 0 ; s < num_subjects_ ; s++ )
	    //	      {
	    //		std::cout << "Subject " << s << std::endl;
	    //		int Ti = intensity[s].size();
	    //		for ( int tp = 0 ; tp < Ti ; tp++ )
	    //		  std::cout << intensity[s][tp] << std::endl; 
	    //	      }
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
      SubjectMapping< Dim, Num_States >::projection_estimation( MaskType::IndexType Idx )
      {
	try
	  {
	    //
	    // Results
	    std::vector< double >                  cumul_ppi( Num_States );
	    Eigen::Matrix< double, Dim, 1 >        projection = Eigen::Matrix< double, Dim, 1 >::Zero();
	    Eigen::Matrix< double, Dim, Dim >      proj_var   = Eigen::Matrix< double, Dim, Dim >::Zero();
	    Eigen::Matrix< double, Num_States, 1 > ppi        = Eigen::Matrix< double, Num_States, 1 >::Zero();

	    //
	    // Initiate the random generator
	    std::random_device rd;
	    std::mt19937                             generator( rd() );
	    //std::default_random_engine               generator;
	    std::uniform_real_distribution< double > uniform(0.0,1.0);

		  
	    //
	    // Get the last state and build the cumulted probabilities
	    Eigen::Matrix< double, Num_States, 1 > pii = Eigen::Matrix< double, Num_States, 1 >::Zero();
	    // Pull subject last state distribution
	    Image4DType::SizeType size = PIDN_states_->GetOutput()->GetLargestPossibleRegion().GetSize();
	    // extract the number of timepoints
	    int num_tp        = size[3] / Num_States;
	    int last_tp_state = Num_States * (num_tp - 1);
	    for ( int s = 0 ; s < Num_States ; s++ )
	      {
		Reader4D::IndexType idx4d = { Idx[0], Idx[1], Idx[2],
					      (last_tp_state + s) };
		pii(s,0) = PIDN_states_->GetOutput()->GetPixel( idx4d );
	      }
	    //std::cout << "pii \n" << pii << std::endl;


	    //
	    // Get the transition matrix
	    Eigen::Matrix< double, Num_States, Num_States > A = Eigen::Matrix< double, Num_States, Num_States >::Zero();
	    //
	    int mat_index = 0;
	    for ( int s = 0 ; s < Num_States ; s++ )
	      for ( int ss = 0 ; ss < Num_States ; ss++ )
		{
		  mat_index = ss + Num_States * s;
		  Reader4D::IndexType idx4d = {Idx[0], Idx[1], Idx[2], mat_index};
		  A(s,ss) = A_->GetOutput()->GetPixel( idx4d );
		}
	    std::cout << "A = \n" << A << std::endl;

	    //
	    // Get the gaussian centroid and the gussian covariance
	    std::vector< Eigen::Matrix< double, Dim, 1 > >   mu_vec(Num_States);
	    std::vector< Eigen::Matrix< double, Dim, Dim > > var_mat(Num_States);
	    //
	    int
	      mu_index  = 0,
	      var_index = 0;
	    for ( int s = 0 ; s < Num_States ; s++ )
	      {
		mu_vec[s]  = Eigen::Matrix< double, Dim, 1 >::Zero();
		var_mat[s] = Eigen::Matrix< double, Dim, Dim >::Zero();
		for ( int d = 0 ; d < Dim ; d++ )
		  {
		    mu_index                     = d + Dim * s;
		    Reader4D::IndexType idx4d_mu = {Idx[0], Idx[1], Idx[2], mu_index};
		    mu_vec[s](d,0) = Gauss_->GetOutput()->GetPixel( idx4d_mu );
		    //
		    for ( int dd = 0 ; dd < Dim ; dd++ )
		      {
			var_index = dd + Dim * d + Dim * Dim * s;
			Reader4D::IndexType idx4d_var = {Idx[0], Idx[1], Idx[2], var_index};
			var_mat[s](d,dd) = Covariance_->GetOutput()->GetPixel( idx4d_var );
		      }
		  }
		//std::cout << "mu_vec["<<s<<"] = \n" << mu_vec[s] << std::endl;
		//std::cout << "var_mat["<<s<<"] = \n" << var_mat[s] << std::endl;
	      }

		    
	    //
	    // Make the estimation
	    // Get the projected state
	    ppi = pii;
	    for ( int tp = 0 ; tp < number_projection_ ; tp++ )
	      ppi = ppi.transpose() * A;
	    ppi /= ppi.sum();
	    std::cout << "ppi = \n" << ppi<< std::endl;
	    // create an array of cunulative probabilities
	    for ( int s = 0 ; s < Num_States ; s++ )
	      {
		if ( s == 0 )
		  cumul_ppi[s] = ppi(s,0);
		else
		  cumul_ppi[s] = cumul_ppi[s-1] + ppi(s,0);
		//std::cout << "cumul_ppi["<<s<<"] = " << cumul_ppi[s] << std::endl;
	      }
	    //
	    // Extrapolation of the images
	    double u     = 0;
	    int    state = 0;
	    std::vector< Eigen::Matrix< double, Dim, 1 > > samples( number_sampling_iteration_ );
	    for ( int iteration = 0 ; iteration < number_sampling_iteration_ ; iteration++ )
	      {
		u = uniform( generator );
		state = 0;
		while ( u > cumul_ppi[state] && state < Num_States )
		  state++;
		//std::cout << "u = " << u << std::endl;
		//std::cout << "s = " << state << std::endl;
		samples[ iteration ] = NeuroBayes::gaussian_multivariate<Dim>( mu_vec[state], var_mat[state] );
	      }
	    //
	    // Moments
	    // mean
	    for ( auto vec : samples )
	      projection += vec;
	    projection /= static_cast<double>( number_sampling_iteration_ );
	    // variance
	    for ( auto vec : samples )
	      proj_var += (vec - projection) * (vec - projection).transpose();
	    proj_var /= static_cast<double>( number_sampling_iteration_ - 1 );
	    std::cout << "projection = \n" << projection << std::endl;
	    std::cout << "proj_var = \n"   << proj_var << std::endl;

		    
	    //
	    // Record the results
	    // record the projection
	    for ( int d = 0 ; d < Dim ; d++ )
	      {
		projection_.set_val( d, Idx, projection(d,0) );
		for ( int dd = 0 ; dd < Dim ; dd++ )
		  {
		    int cov_index = dd + Dim*d;
		    projection_covariance_.set_val( cov_index, Idx, proj_var(d,dd) );
		  }
	      }
	    // record the projected state
	    for ( int s = 0 ; s < Num_States ; s++ )
	      projection_states_.set_val( s, Idx, ppi(s,0) );
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
      SubjectMapping< Dim, Num_States >::build_outputs()
      {
	try
	  {
	    //
	    // We load the mask we created in the main
	    std::string output_mask_name = output_dir_ + "/mask_double_precision.nii.gz";
	    auto image_ptr = itk::ImageIOFactory::CreateImageIO( output_mask_name.c_str(),
								 itk::ImageIOFactory::ReadMode );
	    image_ptr->SetFileName( output_mask_name.c_str() );
	    image_ptr->ReadImageInformation();
	    // Read the ITK image
	    Reader3D::Pointer tempo = Reader3D::New();
	    tempo->SetFileName( image_ptr->GetFileName() );
	    tempo->Update();

	    //
	    // outputs
	    std::string
	      lower_bound_name       = output_dir_ + "/",
	      transition_matrix_name = output_dir_ + "/",
	      first_states_name      = output_dir_ + "/",
	      mu_name                = output_dir_ + "/",
	      variance_name          = output_dir_ + "/";
	    //
	    lower_bound_name        += "lower_bound_"       + std::to_string(Dim)  + "_dimensions_";
	    lower_bound_name        += std::to_string(Num_States)                  + "_states.nii.gz";
	    //			    
	    transition_matrix_name  += "transition_matrix_" + std::to_string(Dim)  + "_dimensions_";
	    transition_matrix_name  += std::to_string(Num_States)                  + "_states.nii.gz";
	    //			    
	    first_states_name       += "first_states_"      + std::to_string(Dim)  + "_dimensions_";
	    first_states_name       += std::to_string(Num_States)                  + "_states.nii.gz";
	    //			    
	    mu_name                 += "cluster_centers_"   + std::to_string(Dim)  + "_dimensions_";
	    mu_name                 += std::to_string(Num_States)                  + "_states.nii.gz";
	    //			    
	    variance_name           += "variance_"          + std::to_string(Dim)  + "_dimensions_";
	    variance_name           += std::to_string(Num_States)                  + "_states.nii.gz";
	    //
	    lower_bound_             = NeuroBayes::NeuroBayesMakeITKImage( 1,
									   lower_bound_name, tempo );
	    transition_matrix_       = NeuroBayes::NeuroBayesMakeITKImage( Num_States * Num_States,
									   transition_matrix_name, tempo );
	    first_states_            = NeuroBayes::NeuroBayesMakeITKImage( Num_States,
									   first_states_name, tempo );
	    mu_                      = NeuroBayes::NeuroBayesMakeITKImage( Num_States*Dim,
									   mu_name, tempo );
	    variance_                = NeuroBayes::NeuroBayesMakeITKImage( Num_States*Dim*Dim,
									   variance_name, tempo );

	    //
	    // Build the subject outputs
	    for ( auto &s : group_pind_ )
	      s.second.build_outputs( tempo );
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
	    switch ( activity_ )
	      {
	      case FIT:
		{
		  //
		  // Global metrics
		  lower_bound_.write();
		  //
		  transition_matrix_.write();
		  first_states_.write();
		  //
		  mu_.write();
		  variance_.write();
		  //
		  // write subjects outputs
		  for ( auto s : group_pind_ )
		    s.second.write_solution();
		  break;
		}
	      case PROJECTION:
		{
		  projection_.write();
		  projection_covariance_.write();
		  projection_states_.write();
		  break;
		}
	      case UNKNOWN:
	      default:
		{
		  std::string mess = "HMM activity is unknown.\n";
		  throw NeuroBayes::NeuroBayesException( __FILE__, __LINE__,
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
  }
}
#endif
