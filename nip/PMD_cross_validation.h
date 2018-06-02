#ifndef NIP_PMD_CROSS_VALIDATION_H
#define NIP_PMD_CROSS_VALIDATION_H
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
      Nip_PMD_cross_validation( const Eigen::MatrixXd&,
				const Eigen::MatrixXd& );
      /*Destructor*/
      virtual ~Nip_PMD_cross_validation(){};
      
      //
      //
      virtual void validation();

      //
      //
    private:
      // image matrices for each fold
      std::vector< Eigen::MatrixXd > folds_images_matrices_{K};
      // Explanatory variable matrices for each fold
      std::vector< Eigen::MatrixXd >  folds_ev_matrices_{K};
      // Vector of values for c1 and c2
      std::vector< std::vector< double > > T2_{2};
      // grid size
      int grid_size_im_{1000};
      int grid_size_ev_{100};
    };
  
  //
  //
  template< int K >
    Nip_PMD_cross_validation<K>::Nip_PMD_cross_validation( const Eigen::MatrixXd& Images_matrix,
							   const Eigen::MatrixXd& Ev_matrix )
    {
      try
	{
	  //
	  //  Check same number of rows!!

      
	  int
	    image_r = Images_matrix.rows(),
	    image_c = Images_matrix.cols(),
	    ev_r    = Ev_matrix.rows(),
	    ev_c    = Ev_matrix.cols();

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
	  //
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
		  folds_images_matrices_[k] = Eigen::MatrixXd::Zero( fold_i, image_c );
		  folds_ev_matrices_[k]     = Eigen::MatrixXd::Zero( fold_i, ev_c) ;
		  //
		  for ( int im = 0 ; im < fold_i ; im++ )
		    {
		      folds_images_matrices_[k].row( im ) = Images_matrix.row( im_pos );
		      folds_ev_matrices_[k].row( im ) = Ev_matrix.row( im_pos++ );
		    }
		}
	      else
		{
		  folds_images_matrices_[k] = Eigen::MatrixXd::Zero( last_fold_i, image_c );
		  folds_ev_matrices_[k]     = Eigen::MatrixXd::Zero( last_fold_i, ev_c) ;
		  //
		  for ( int im = 0 ; im < last_fold_i ; im++ )
		    {
		      folds_images_matrices_[k].row( im ) = Images_matrix.row( im_pos );
		      folds_ev_matrices_[k].row( im ) = Ev_matrix.row( im_pos++ );
		    }
		}

	      //
	      std::cout << " fold " << k << std::endl;
	      std::cout << folds_ev_matrices_[k] << std::endl;
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
	  for ( auto c1 : T2_[0] )
	    for ( auto c2 : T2_[1] )
	      {
		std::cout << "c1: " << c1 << ",c2: "  << c2 << std::endl;
		
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
