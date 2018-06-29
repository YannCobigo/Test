#ifndef NIP_SPC_CROSS_VALIDATION_H
#define NIP_SPC_CROSS_VALIDATION_H
#include <algorithm>    // std::next_permutation, std::sort
#include <mutex>
// Critical zone
std::mutex CRITICAL_ZONE_SPC;
//
//
//
#include "Thread_dispatching.h"
#include "NipException.h"
#include "PMA_cross_validation.h"
#include "SPC.h"
//
//
//
namespace MAC_nip
{
  std::size_t factorial_spc( std::size_t n )
    {
      return (n == 1 || n == 0) ? 1 : factorial_spc(n - 1) * n;
    }
  /** \class Nip_SPC_cross_validation
   *
   * \brief PMD: Penalized Matrices Decomposition
   * "A penalized matrix decomposition, with applications to sparse principal components and canonical correlation analysis" (PMID:19377034)
   * 
   */
  template< int K, int CPU >
    class Nip_SPC_cross_validation : public Nip_PMA_cross_validation
  {
  public:
    /*Constructor*/
    Nip_SPC_cross_validation( std::shared_ptr< const Eigen::MatrixXd >,
			      const int );
    /*Destructor*/
    virtual ~Nip_SPC_cross_validation(){};
      
    //
    //
    virtual void validation( std::shared_ptr< Spectra > matrices_spetrum_ );
      
    //
    //
    virtual void k_folds( const std::vector< double > );

    //
    //
    virtual void operator ()( const std::vector< double > Paramters )
    {
      std::cout
	<< "treatment for parameters c1: " << Paramters[0]
	<< ", and c2: "  << Paramters[1]
	<< std::endl;
      //
      k_folds( Paramters );
    };
    
    //
    //
  private:
    // Save Image matrix
    std::shared_ptr< const Eigen::MatrixXd > images_matrix_;
    //
    long int image_features_;
    //
    // image matrices for each fold
    std::vector< std::vector< Eigen::MatrixXd > > folds_images_matrices_{K};
    // Vector of values for c1 and c2
    std::vector< std::vector< double > > T2_{2};
    // grid size
    int grid_size_im_{5}; //1000
    //
    // Matrices and spectrum for the training and testing
    std::vector< Eigen::MatrixXd > fold_full_images_matrix_{K};
    //
    // permutation matrices to build the p-values
    std::size_t max_permutations_{2}; //200
    std::vector<  std::vector< Eigen::MatrixXd > > permutations_images_matrix_{K};
    //
    // Results
    double correlation_{0.};
    double correlation_sd_{100.};
    double p_value_{1.};
    double p_value_sd_{100.};
    double c1_{0.};
    double c2_{0.};
    // reduced space
    int reduced_space_;
  };
  
  //
  //
  template< int K, int CPU >
    Nip_SPC_cross_validation<K,CPU>::Nip_SPC_cross_validation( std::shared_ptr< const Eigen::MatrixXd > Images_matrix,
							       const int Reduced_space ):
    images_matrix_{Images_matrix}, image_features_{Images_matrix->cols()}, reduced_space_{Reduced_space}
  {
    try
      {
	//
	//  ToDo: Check same number of rows!!

	//
	// Initialization
	int
	  image_r = Images_matrix->rows(),
	  image_c = Images_matrix->cols();

	//
	// Build the grid
	double
	  c1_max = sqrt( static_cast< double > (image_c) ),
	  c2_max = sqrt( static_cast< double > (image_c) );
	double
	  c1_step = (c1_max - 1. ) / static_cast< double >( grid_size_im_ ),
	  c2_step = (c2_max - 1. ) / static_cast< double >( grid_size_im_ ),
	  c1_current = 1.,
	  c2_current = 1.;
	//
	T2_[0].resize(grid_size_im_);
	T2_[1].resize(grid_size_im_);
	//
	for ( int step = 0 ; step < grid_size_im_ ; step++ )
	  {
	    T2_[0][step] = T2_[1][step] = c1_current;
	    c1_current  += c1_step;
	  }

	//
	// Create the training and testing matrices
	int
	  fold_i      = image_r / K;
	int
	  last_fold_i = image_r - (K - 1) * fold_i,
	  im_pos      = 0;
	//
	for ( int k = 0 ; k < K  ; k++ )
	  {
	    //
	    if( k != K - 1 )
	      {
		folds_images_matrices_[k].resize( fold_i );
		fold_full_images_matrix_[k] = Eigen::MatrixXd::Zero(fold_i,image_c);
		//
		for ( int im = 0 ; im < fold_i ; im++ )
		  {
		    folds_images_matrices_[k][im]       = Images_matrix->row( im_pos );
		    fold_full_images_matrix_[k].row(im) = Images_matrix->row( im_pos );
		  }
	      }
	    else
	      {
		folds_images_matrices_[k].resize( last_fold_i );
		fold_full_images_matrix_[k] = Eigen::MatrixXd::Zero(last_fold_i,image_c);
		//
		for ( int im = 0 ; im < last_fold_i ; im++ )
		  {
		    folds_images_matrices_[k][im] = Images_matrix->row( im_pos );
		    fold_full_images_matrix_[k].row(im) = Images_matrix->row( im_pos );
		  }
	      }
	  }

	//
	// Permute the elements to build the p-value
	// The per permutations are made only on the images and applied on the original
	// explenatory variable matrix
	for ( int k = 0 ; k < K  ; k++ )
	  {
	    //MakeItBetter	    std::size_t permutation = factorial_spc( folds_images_matrices_[k].size() );
	    //MakeItBetter	    if ( permutation > max_permutations_ )
	    //MakeItBetter	      {
	    //MakeItBetter		std::cout << "The number of permutation requiered is " << permutation
	    //MakeItBetter			  << ". The maximum number of permutation is reached.\n"
	    //MakeItBetter			  << max_permutations_ << " permutation will be done."<< std::endl;
	    //MakeItBetter		permutation = max_permutations_;
	    //MakeItBetter	      }
	    //
	    permutations_images_matrix_[k].resize( max_permutations_ );
	    //
	    for ( int p = 0 ; p < max_permutations_ ; p++ )
	      {
		permutations_images_matrix_[k][p] = Eigen::MatrixXd::Zero( folds_images_matrices_[k].size(),
									   image_c );
		std::random_shuffle( folds_images_matrices_[k].begin(),
				     folds_images_matrices_[k].end() );
		//
		for ( int r = 0 ; r < folds_images_matrices_[k].size() ; r++ )
		  permutations_images_matrix_[k][p].row(r) = folds_images_matrices_[k][r];
	      }
	  }
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
	exit(-1);
      }
  }
  //
  //
  template< int K, int CPU > void
    Nip_SPC_cross_validation<K,CPU>::validation( std::shared_ptr< Spectra > Matrices_spetrum )
    {
      try
	{
	  //
	  // Multi-threading pool
	  // 
	  {
	    //
	    // Pool of threads
	    MAC_nip::Thread_dispatching pool( CPU );
	    //
	    // Find the best couple (c1,c2) using a k-fold cross-validation
	    std::vector< double > paramters( 2, 0. );
	    for ( auto c1 : T2_[0] )
	      for ( auto c2 : T2_[1] )
		{
		  paramters[0] = c1;
		  paramters[1] = c2;
		  //
		  pool.enqueue( std::ref( *this ), paramters );
		}
	  } // execute the jobs
	  //
	  // print the best values
	  // CSV;c1,c2,Corr,Corr_sd,p-val,p-val_ds
	  std::cout
	    << "CSV;" << c1_ << "," << c2_ << ","
	    << correlation_ << "," << correlation_sd_ << ","
	    << p_value_ << "," << p_value_sd_
	    << std::endl;
	  
	  
	  //
	  // When done, use the couple (c1,c2) to complete the
	  // optimization of the spectrum with the original matrices
	  //

	  //
	  // Create the spectrum
	  std::size_t K_cca = Matrices_spetrum->size();
	  MAC_nip::NipSPC pmd_spc;
	  pmd_spc.set_cs(c1_,c2_);
	  //
	  Eigen::MatrixXd
	    // training matrices
	    images_norm = MAC_nip::NipPMA_tools::normalize( *images_matrix_.get(),
							    MAC_nip::STANDARDIZE );
	  //
	  // SVD
	  if (false)
	    {
	      //
	      // X nxp
	      // X = UDV^T with U^TU = In and V^TV = Ip
	      Eigen::JacobiSVD< Eigen::MatrixXd >
		svd_V(images_norm.transpose() * images_norm, Eigen::ComputeThinV),
		svd_U(images_norm.transpose() * images_norm, Eigen::ComputeThinU);
	      //
	      for ( int k_factor = 0 ; k_factor < K_cca ; k_factor++ )
		{
		  // Coefficient
		  std::get< coeff_k >( (*Matrices_spetrum.get())[k_factor] ) = 0.;
		  // vectors
		  std::get< Uk >( (*Matrices_spetrum.get())[k_factor] ) = svd_U.matrixU().col( k_factor );
		  std::get< Vk >( (*Matrices_spetrum.get())[k_factor] ) = svd_V.matrixV().col( k_factor );
		}
	    }
	  //
	  // Full matrices optimization
	  pmd_spc.K_factors( images_norm.transpose() * images_norm,
			     *Matrices_spetrum.get(), L1, L1, true );
	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	  exit(-1);
	}
    }
  //
  //
  template< int K, int CPU > void
    Nip_SPC_cross_validation<K,CPU>::k_folds( const std::vector< double > Paramters )
    {
      try
	{
	  //
	  // principal componant
	  MAC_nip::NipSPC pmd_spc_k;
	  double
	    c1 = Paramters[0],
	    c2 = Paramters[1];
	  //
	  //
	  std::vector< double >
	    correlation(K,0.),
	    p_value(K,1.);

	  //
	  //
	  for ( int k = 0 ; k < K ; k++ )
	    {
	      //
	      // Concaten the training matrices together
	      int
		training_size = 0,
		testing_size  = fold_full_images_matrix_[k].rows();
	      for ( int kk = 0 ; kk < K ; kk++ )
		if ( kk != k )
		  training_size += fold_full_images_matrix_[kk].rows();
	      //
	      Eigen::MatrixXd
		images_training = Eigen::MatrixXd::Zero(training_size,image_features_);
	      //
	      int row_position = 0;
	      for ( int kk = 0 ; kk < K ; kk++ )
		if ( kk != k )
		  {
		    //
		    int number_rows = fold_full_images_matrix_[kk].rows();
		    images_training.block( row_position,
					   0,
					   number_rows,
					   image_features_) = fold_full_images_matrix_[kk].block(0,0,
												 number_rows,
												 image_features_);
		    //
		    row_position += number_rows;
		  }
		  
		  
	      //
	      // Train on the grid of (c1,c2) the (k-1)-samples
	      //
 
	      
	      //
	      // Create the matrices
	      // training matrices
	      Eigen::MatrixXd
		images_training_norm = MAC_nip::NipPMA_tools::normalize( images_training.transpose() * images_training,
									 MAC_nip::STANDARDIZE ),
		images_testing_norm = MAC_nip::NipPMA_tools::normalize( fold_full_images_matrix_[k].transpose()*fold_full_images_matrix_[k],
									MAC_nip::STANDARDIZE );

	      //
	      // Create the spectrum
	      std::size_t K_spc = reduced_space_;
	      Spectra matrix_spetrum_spc( K_spc );
	      // initialize the spectra
	      // SVD
	      if (false)
		{
		  //
		  // SVD on X nxp
		  // X = UDV^T with U^TU = In and V^TV = Ip
		  Eigen::JacobiSVD< Eigen::MatrixXd >
		    svd_V(images_training_norm, Eigen::ComputeThinV),
		    svd_U(images_training_norm, Eigen::ComputeThinU);
		  //
		  for ( int k_factor = 0 ; k_factor < K_spc ; k_factor++ )
		    {
		      // Coefficient
		      std::get< coeff_k >( matrix_spetrum_spc[k_factor] ) = 0.;
		      // vectors
		      std::get< Uk >( matrix_spetrum_spc[k_factor] ) = svd_U.matrixU().col( k_factor );
		      std::get< Vk >( matrix_spetrum_spc[k_factor] ) = svd_V.matrixV().col( k_factor );
		    }
		}
	      else
		{
		  for ( int k_factor = 0 ; k_factor < K_spc ; k_factor++ )
		    {
		      // Coefficient
		      std::get< coeff_k >( matrix_spetrum_spc[k_factor] ) = 0.;
		      // vectors
		      std::get< Uk >( matrix_spetrum_spc[k_factor] ) = Eigen::MatrixXd::Random( image_features_, 1 );
		      std::get< Vk >( matrix_spetrum_spc[k_factor] ) = Eigen::MatrixXd::Random( image_features_, 1 );
		      // normalization
		      std::get< Uk >( matrix_spetrum_spc[k_factor] ) /= std::get< Uk >( matrix_spetrum_spc[k_factor] ).lpNorm< 2 >();
		      std::get< Vk >( matrix_spetrum_spc[k_factor] ) /= std::get< Vk >( matrix_spetrum_spc[k_factor] ).lpNorm< 2 >();
		    }
		}
	      //
	      // Optimize the spectrum
	      pmd_spc_k.set_cs(c1,c2);
	      pmd_spc_k.K_factors( images_training_norm,
				   matrix_spetrum_spc, L1, L1, false );

	      //
	      // Compute correlation
	      // In sparse principal coefficient we use the correlation feature as the find out the highest
	      // rank coefficient for the first principal componant.
	      double
		rank_coeff =
		(std::get< Uk >( matrix_spetrum_spc[0] ).transpose() *
		 images_testing_norm *
		 std::get< Vk >( matrix_spetrum_spc[0] ))(0,0);
	      correlation[k] = rank_coeff;
	      p_value[k] = 1.;
	    } // for every fold

	  //
	  // Compute the mean values of results
	  // for a set of (c1, c2)
	  double
	    mean_corr  = 0.,
	    mean_p_val = 0.,
	    sd_corr  = 0.,
	    sd_p_val = 0.;
	  //
	  for ( int k = 0 ; k < K ; k++ )
	    {
	      mean_corr  += correlation[k];
	      mean_p_val += p_value[k];
	    }
	  //
	  mean_corr  /= static_cast< double >(K);
	  mean_p_val /= static_cast< double >(K);
	  //
	  for ( int k = 0 ; k < K ; k++ )
	    {
	      sd_corr  += ( correlation[k] - mean_corr) * ( correlation[k] - mean_corr);
	      sd_p_val += ( p_value[k] - mean_p_val ) * ( p_value[k] - mean_p_val );
	    }
	  //
	  sd_corr  /=  static_cast< double >(K-1);
	  sd_p_val /=  static_cast< double >(K-1); 
	  sd_corr  =  sqrt(sd_corr);
	  sd_p_val =  sqrt(sd_p_val);
	  // Critical zone
	  {
	    // lock the population
	    std::lock_guard< std::mutex > lock_critical_zone ( CRITICAL_ZONE_SPC );
	    std::cout
	      << "CSV;" << c1_ << "," << c2_ << ","
	      << correlation_ << "," << correlation_sd_ << ","
	      << p_value_ << "," << p_value_sd_
	      << std::endl;
	    //
	    if ( mean_corr > correlation_  )
	      {
		// save the best c1 and c2
		c1_ = c1;
		c2_ = c2;
		//
		correlation_    = mean_corr;
		correlation_sd_ = sd_corr;
		p_value_        = mean_p_val;
		p_value_sd_     = sd_p_val;

		std::cout
		  << "CSV;" << c1_ << "," << c2_ << ","
		  << correlation_ << "," << correlation_sd_ << ","
		  << p_value_ << "," << p_value_sd_
		  << std::endl;

	      }
	  }
	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	  exit(-1);
	}
    }
}
#endif
