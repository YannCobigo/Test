#ifndef NIP_PMD_CROSS_VALIDATION_H
#define NIP_PMD_CROSS_VALIDATION_H
#include <algorithm>    // std::next_permutation, std::sort
#include <mutex>
// Critical zone
std::mutex CRITICAL_ZONE;
//
//
//
#include "Thread_dispatching.h"
#include "NipException.h"
#include "PMA_cross_validation.h"
//
//
//
namespace MAC_nip
{
  std::size_t factorial( std::size_t n )
    {
      return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
    }
  /** \class Nip_PMD_cross_validation
   *
   * \brief PMD: Penalized Matrices Decomposition
   * "A penalized matrix decomposition, with applications to sparse principal components and canonical correlation analysis" (PMID:19377034)
   * 
   */
  template< int K, int CPU >
    class Nip_PMD_cross_validation : public Nip_PMA_cross_validation
  {
  public:
    /*Constructor*/
    Nip_PMD_cross_validation( std::shared_ptr< const Eigen::MatrixXd >,
			      std::shared_ptr< const Eigen::MatrixXd > );
    /*Destructor*/
    virtual ~Nip_PMD_cross_validation(){};
      
    //
    //
    virtual void validation( std::shared_ptr< Spectra > matrices_spetrum_ );
      
    //
    //
    void k_folds( const std::vector< double > );

    //
    void operator ()( const std::vector< double > Paramters )
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
    // Save Explenatory variables matrix
    std::shared_ptr< const Eigen::MatrixXd > ev_matrix_;
    //
    long int image_features_;
    long int ev_features_;
    //
    // image matrices for each fold
    std::vector< std::vector< Eigen::MatrixXd > > folds_images_matrices_{K};
    // Explanatory variable matrices for each fold
    std::vector< std::vector< Eigen::MatrixXd > >  folds_ev_matrices_{K};
    // Vector of values for c1 and c2
    std::vector< std::vector< double > > T2_{2};
    // grid size
    int grid_size_im_{50}; //1000
    int grid_size_ev_{10}; //100
    //
    // Matrices and spectrum for the training and testing
    // Matrices
    std::vector< Eigen::MatrixXd > fold_full_images_matrix_{K};
    std::vector< Eigen::MatrixXd > fold_full_ev_matrix_{K};
    //
    // permutation matrices to build the p-values
    std::size_t max_permutations_{1000}; //200
    std::vector<  std::vector< Eigen::MatrixXd > > permutations_images_matrix_{K};
    //
    // Results
    double correlation_{0.};
    double correlation_sd_{100.};
    double p_value_{1.};
    double p_value_sd_{100.};
    double c1_{0.};
    double c2_{0.};
  };
  
  //
  //
  template< int K, int CPU >
    Nip_PMD_cross_validation<K,CPU>::Nip_PMD_cross_validation( std::shared_ptr< const Eigen::MatrixXd > Images_matrix,
							       std::shared_ptr< const Eigen::MatrixXd > Ev_matrix ):
    images_matrix_{Images_matrix}, ev_matrix_{Ev_matrix}, image_features_{Images_matrix->cols()}, ev_features_{Ev_matrix->cols()}
  {
    try
      {
	//
	//  ToDo: Check same number of rows!!

	//
	// Initialization
	int
	  image_r = Images_matrix->rows(),
	  image_c = Images_matrix->cols(),
	  ev_r    = Ev_matrix->rows(),
	  ev_c    = Ev_matrix->cols();

	//
	// Build the grid
	double
	  c1_max = sqrt( static_cast< double > (image_c) ),
	  c2_max = sqrt( static_cast< double > (ev_c) );
	double
	  c1_step = (c1_max - 1. ) / static_cast< double >( grid_size_im_ ),
	  c2_step = (c2_max - 1. ) / static_cast< double >( grid_size_ev_ ),
	  c1_current = 1.,
	  c2_current = 1.;
	//
	T2_[0].resize(grid_size_im_);
	T2_[1].resize(grid_size_ev_);
	//
	for ( int step = 0 ; step < grid_size_im_ ; step++ )
	  {
	    T2_[0][step] = c1_current;
	    c1_current  += c1_step;
	  }
	for ( int step = 0 ; step < grid_size_ev_ ; step++ )
	  {
	    T2_[1][step] = c2_current;
	    c2_current  += c2_step;
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
		folds_ev_matrices_[k].resize( fold_i );
		fold_full_images_matrix_[k] = Eigen::MatrixXd::Zero(fold_i,image_c);
		fold_full_ev_matrix_[k]     = Eigen::MatrixXd::Zero(fold_i,ev_c);
		//
		for ( int im = 0 ; im < fold_i ; im++ )
		  {
		    folds_images_matrices_[k][im] = Images_matrix->row( im_pos );
		    folds_ev_matrices_[k][im]     = Ev_matrix->row( im_pos );
		    //
		    fold_full_images_matrix_[k].row(im) = Images_matrix->row( im_pos );
		    fold_full_ev_matrix_[k].row(im)     = Ev_matrix->row( im_pos++ );
		  }
	      }
	    else
	      {
		folds_images_matrices_[k].resize( last_fold_i );
		folds_ev_matrices_[k].resize( last_fold_i ) ;
		fold_full_images_matrix_[k] = Eigen::MatrixXd::Zero(last_fold_i,image_c);
		fold_full_ev_matrix_[k]     = Eigen::MatrixXd::Zero(last_fold_i,ev_c);
		//
		for ( int im = 0 ; im < last_fold_i ; im++ )
		  {
		    folds_images_matrices_[k][im] = Images_matrix->row( im_pos );
		    folds_ev_matrices_[k][im] = Ev_matrix->row( im_pos );
		    //
		    fold_full_images_matrix_[k].row(im) = Images_matrix->row( im_pos );
		    fold_full_ev_matrix_[k].row(im)     = Ev_matrix->row( im_pos++ );
		  }
	      }
	  }

	//
	// Permute the elements to build the p-value
	// The per permutations are made only on the images and applied on the original
	// explenatory variable matrix
	for ( int k = 0 ; k < K  ; k++ )
	  {
//MakeItBetter	    std::size_t permutation = factorial( folds_images_matrices_[k].size() );
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
    Nip_PMD_cross_validation<K,CPU>::validation( std::shared_ptr< Spectra > Matrices_spetrum )
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
	  MAC_nip::NipPMD pmd_cca;
	  pmd_cca.set_cs(c1_,c2_);
	  //
	  Eigen::MatrixXd
	    // training matrices
	    images_norm = MAC_nip::NipPMA_tools::normalize( *images_matrix_.get(),
							    MAC_nip::STANDARDIZE ),
	    ev_norm = MAC_nip::NipPMA_tools::normalize( *ev_matrix_.get(),
							MAC_nip::STANDARDIZE );
	  //
	  pmd_cca.K_factors( images_norm.transpose() * ev_norm,
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
    Nip_PMD_cross_validation<K,CPU>::k_folds( const std::vector< double > Paramters )
    {
      try
	{
	  MAC_nip::NipPMD pmd_cca;
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
	      // k is a testing set
	      //std::cout << "fold: " << k << ", and c1: " << c1 << ",c2: "  << c2 << std::endl;
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
		images_training = Eigen::MatrixXd::Zero(training_size,image_features_),
		ev_training     = Eigen::MatrixXd::Zero(training_size,ev_features_);
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
		    ev_training.block( row_position,
				       0,
				       number_rows,
				       ev_features_) = fold_full_ev_matrix_[kk].block(0,0,
										      number_rows,
										      ev_features_);
		    //
		    row_position += number_rows;
		  }
		  
		  
	      //
	      // Train on the grid of (c1,c2) the (k-1)-samples
	      //

	      //
	      // Create the spectrum
	      std::size_t K_cca = (image_features_ > ev_features_ ? ev_features_ : image_features_);
	      Spectra matrix_spetrum_cca( K_cca );
	      // initialize the spectra
	      // ToDo: the first vector should be the SVD highest eigen vector
	      for ( int k_factor = 0 ; k_factor < K_cca ; k_factor++ )
		{
		  // Coefficient
		  std::get< coeff_k >( matrix_spetrum_cca[k_factor] ) = 0.;
		  // vectors
		  std::get< Uk >( matrix_spetrum_cca[k_factor] ) = Eigen::MatrixXd::Random( image_features_, 1 );
		  std::get< Vk >( matrix_spetrum_cca[k_factor] ) = Eigen::MatrixXd::Random( ev_features_, 1 );
		  // normalization
		  std::get< Uk >( matrix_spetrum_cca[k_factor] ) /= std::get< Uk >( matrix_spetrum_cca[k_factor] ).lpNorm< 2 >();
		  std::get< Vk >( matrix_spetrum_cca[k_factor] ) /= std::get< Vk >( matrix_spetrum_cca[k_factor] ).lpNorm< 2 >();
		}
	      //
	      pmd_cca.set_cs(c1,c2);
	      //		  std::cout
	      //		    << "images_training\n" << images_training
	      //		    << "\n ev_training\n " << ev_training
	      //		    << std::endl;
	      Eigen::MatrixXd
		// training matrices
		images_training_norm = MAC_nip::NipPMA_tools::normalize( images_training,
									 MAC_nip::STANDARDIZE ),
		ev_training_norm = MAC_nip::NipPMA_tools::normalize( ev_training,
								     MAC_nip::STANDARDIZE ),
		// testing matrices
		images_testing_norm = MAC_nip::NipPMA_tools::normalize( fold_full_images_matrix_[k],
									MAC_nip::STANDARDIZE ),
		ev_testing_norm = MAC_nip::NipPMA_tools::normalize( fold_full_ev_matrix_[k],
								    MAC_nip::STANDARDIZE );
	      //
	      pmd_cca.K_factors( images_training_norm.transpose() * ev_training_norm,
				 matrix_spetrum_cca, L1, L1, false );
	      
	      // Compute correlation
	      for ( int k_factor = 0 ; k_factor < K_cca ; k_factor++ )
		{
		  double coefficient_to_rank = std::get< coeff_k >( matrix_spetrum_cca[k_factor] );
		  //
		  if ( coefficient_to_rank != 0. )
		    {
		      Eigen::MatrixXd
			za = images_training_norm * std::get< Uk >( matrix_spetrum_cca[k_factor] ),
			zb = ev_training_norm * std::get< Vk >( matrix_spetrum_cca[k_factor] ),
			za_test = images_testing_norm * std::get< Uk >( matrix_spetrum_cca[k_factor] ),
			zb_test = ev_testing_norm * std::get< Vk >( matrix_spetrum_cca[k_factor] );
		      double
			training_correlation = (za.transpose() * zb / (za.lpNorm< 2 >() * zb.lpNorm< 2 >()))(0,0),
			trained_correlation  = (za_test.transpose() * zb_test / (za_test.lpNorm< 2 >() * zb_test.lpNorm< 2 >()))(0,0);
		      
		      //std::cout
		      //  << "Correlation factor from training " << k_factor << ": "
		      //  << training_correlation
		      //  << " from testing " 
		      //  << trained_correlation
		      //  << std::endl;
		      //
		      // Compute the p-value
		      int p_increment = 0;
		      for ( int perm = 0 ; perm < max_permutations_ ; perm++ )
			{
			  Eigen::MatrixXd z_perm = MAC_nip::NipPMA_tools::normalize( permutations_images_matrix_[k][perm],
										     MAC_nip::STANDARDIZE ) * std::get< Uk >( matrix_spetrum_cca[k_factor] );
			  if ( trained_correlation < (z_perm.transpose() * zb_test / (z_perm.lpNorm< 2 >() * zb_test.lpNorm< 2 >()))(0,0) )
			    p_increment++;
			  //
			  //std::cout
			  //	<< "permutation " << perm << ": "
			  //	<< trained_correlation << " "
			  //	<< z_perm.transpose() * zb_test / (z_perm.lpNorm< 2 >() * zb_test.lpNorm< 2 >())
			  //	<< std::endl;
			}
		      //
		      double p = static_cast<double>(p_increment) / static_cast<double>(max_permutations_);
		      //std::cout << "p-value " << p << std::endl;
		      //
		      if ( k_factor == 0 )
			{
			  correlation[k] = trained_correlation;
			  p_value[k] = p;
			}
		    }
		  else
		    {
		      correlation[k] = 0.;
		      p_value[k] = 1.;
		    }
		} //for
	    }

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
	    std::lock_guard< std::mutex > lock_critical_zone ( CRITICAL_ZONE );
	    //
	    if ( mean_corr > correlation_ /*|| (mean_corr == correlation_ && correlation_sd_ > sd_corr)*/ )
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
