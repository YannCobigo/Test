#ifndef NIP_PMD_CROSS_VALIDATION_H
#define NIP_PMD_CROSS_VALIDATION_H
#include <algorithm>    // std::next_permutation, std::sort
//
//
//
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
  template< int K >
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
      virtual void validation();

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
      int grid_size_im_{1000};
      int grid_size_ev_{100};
      //
      // Matrices and spectrum for the training and testing
      // Matrices
      std::vector< Eigen::MatrixXd > fold_full_images_matrix_{K};
      std::vector< Eigen::MatrixXd > fold_full_ev_matrix_{K};
      //
      // permutation matrices to build the p-values
      std::size_t max_permutations_{1000};
      std::vector<  std::vector< Eigen::MatrixXd > > permutations_images_matrix_{K};
      
    };
  
  //
  //
  template< int K >
    Nip_PMD_cross_validation<K>::Nip_PMD_cross_validation( std::shared_ptr< const Eigen::MatrixXd > Images_matrix,
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
	    std::size_t permutation = factorial( folds_images_matrices_[k].size() );
	    if ( permutation > max_permutations_ )
	      {
		std::cout << "The number of permutation requiered is " << permutation
			  << ". The maximum number of permutation is reached.\n"
			  << max_permutations_ << " permutation will be done."<< std::endl;
		permutation = max_permutations_;
	      }
	    //
	    permutations_images_matrix_[k].resize( permutation );
	    //
	    for ( int p = 0 ; p < permutation ; p++ )
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
  template< int K > void
    Nip_PMD_cross_validation<K>::validation()
    {
      try
	{
	  //
	  // Find the best couple (c1,c2) using a k-fold cross-validation
	  MAC_nip::NipPMD pmd_cca;
	  for ( auto c1 : T2_[0] )
	    for ( auto c2 : T2_[1] )
	      for ( int k = 0 ; k < K ; k++ )
		{
		  //
		  // k is a testing set
		  std::cout << "fold: " << k << ", and c1: " << c1 << ",c2: "  << c2 << std::endl;
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
		    images_training_norm = MAC_nip::NipPMA_tools::normalize( images_training,
									     MAC_nip::STANDARDIZE ),
		    ev_training_norm = MAC_nip::NipPMA_tools::normalize( ev_training,
									     MAC_nip::STANDARDIZE );
		  pmd_cca.K_factors( images_training_norm.transpose() * ev_training_norm,
				     matrix_spetrum_cca, L1, L1 );
		  // Compute correlation
		  for ( int k_factor = 0 ; k_factor < K_cca ; k_factor++ )
		    {
		      Eigen::MatrixXd
			za = images_training_norm * std::get< Uk >( matrix_spetrum_cca[k_factor] ),
			zb = ev_training_norm * std::get< Vk >( matrix_spetrum_cca[k_factor] );
		      std::cout
			<< "Correlation factor " << k_factor << ": "
			<< za.transpose() * zb / (za.lpNorm< 2 >() * zb.lpNorm< 2 >())
			<< std::endl;
		    }

		  
		  //
		  // Compute the p-values
		  
		  //
		  // test on the 1-sample
		  //
		  
		  //
		  // Compute the p-value
		}

	  //
	  // When done, use the couple (c1,c2) to complete the determination
	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	  exit(-1);
	}
    }
}
#endif
