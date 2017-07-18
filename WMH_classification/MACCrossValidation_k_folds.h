#ifndef MACCROSSVALIDATION_K_FOLDS_H
#define MACCROSSVALIDATION_K_FOLDS_H
//
//
//
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>  
#include <map>
#include <list>
#include <memory>
// Egen
#include <Eigen/Core>
#include <Eigen/Eigen>
//
// ITK
//
#include <itkImageFileReader.h>
#include <itkSpatialOrientationAdapter.h>
#include "itkChangeInformationImageFilter.h"
using MaskType = itk::Image< unsigned char, 3 >;
//
//
//
#include "MACException.h"
#include "MACCrossValidation.h"
#include "Classification.h"
//
//
//
namespace MAC
{
  /** \class MACCrossValidation_k_folds
   *
   * \brief 
   * 
   */
  template< int Dim >
    class MACCrossValidation_k_folds : public MACCrossValidation< Dim >
    {
    public:
      /** Constructor. */
      explicit MACCrossValidation_k_folds( Classification< Dim >*,
					   const MaskType::IndexType,
					   const int, const int );
      
      /**  */
      virtual ~MACCrossValidation_k_folds(){};

      //
      //
      virtual void CV() const;

      
    private:
      // Number of fold of the cross-validation
      int k_;
      // Number of subjects
      int n_;
      // Size of the group for each fold 
      std::vector< int > groups_size_mapping_;
      // Design matrices
      std::vector< std::vector< Eigen::VectorXd > > k_groups_X_;
      // Responses
      std::vector< std::vector< double > >          k_groups_Y_;
      // Weights (beta)
      std::vector< std::vector< double > >          k_groups_W_;
    };

  //
  //
  //
  template< int Dim >
    MAC::MACCrossValidation_k_folds<Dim>::MACCrossValidation_k_folds( Classification< Dim >* Classify,
								      const MaskType::IndexType Idx,
								      const int K,
								      const int Num_subjects ): 
    MACCrossValidation<Dim>( Classify, Idx ),
    k_{ K }, n_{ Num_subjects }
  {
    try
      {
	//
	// Population
	std::cout << "n_: " << n_ << " k_: " << k_ << std::endl;

	//
	// Groups mapping
	// how many samples per fold in average
	int mean_size = n_ / k_;
	if ( mean_size < 1)
	  throw MAC::MACException( __FILE__, __LINE__,
				   "Too many folds are demanded for the population.",
				   ITK_LOCATION );
	
	//
	//std::cout << "mean_size: " << mean_size << std::endl;
	groups_size_mapping_ = std::vector< int >( k_, mean_size );
	// group redistribution of the remaining
	int remaining = n_ - ( k_ * mean_size );
	int increment_group = 0;
	while( remaining-- > 0 )
	  groups_size_mapping_[ increment_group++ ]++;

	//
	// Design matrices per fold
	k_groups_X_.resize( k_ );
	k_groups_Y_.resize( k_ );
	k_groups_W_.resize( k_ );
	// 
	int current_group         = 0;
	int current_samp_in_group = 0;
	// initialize the first group
	k_groups_X_[0].resize( groups_size_mapping_[ current_group ] );
	k_groups_Y_[0].resize( groups_size_mapping_[ current_group ] );
	k_groups_W_[0].resize( groups_size_mapping_[ current_group ] );
	//
	for ( int subject = 0 ; subject < n_ ; subject++ )
	  {
	    //
	    // Which group are we
	    if ( current_samp_in_group > groups_size_mapping_[ current_group ] - 1 )
	      {
		current_group++;
		current_samp_in_group = 0;
		//
		k_groups_X_[current_group].resize( groups_size_mapping_[ current_group ] );
		k_groups_Y_[current_group].resize( groups_size_mapping_[ current_group ] );
		k_groups_W_[current_group].resize( groups_size_mapping_[ current_group ] );
	      }

	    //
	    // Label
	    k_groups_Y_[ current_group ][current_samp_in_group] = 
	      static_cast< const double >( (MACCrossValidation<Dim>::classify_->get_subjects())[subject].get_label(Idx) );

	    //
	    // subject design matrix
	    Eigen::VectorXd X( Dim + 1 );
	    X(0) = 1.; // for beta_0
	    for ( int mod = 0 ; mod < Dim ; mod++ )
	      X( mod + 1 ) = ((MACCrossValidation<Dim>::classify_->get_subjects())[subject].get_modalities(Idx))[mod];
	    k_groups_X_[ current_group ][current_samp_in_group] = X;

	    //
	    // Increment
	    current_samp_in_group++;
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
  template< int Dim > void
    MAC::MACCrossValidation_k_folds<Dim>::CV( ) const
    {
      try
	{
	  //
	  // Accuracy and False discovery rate
	  std::vector< double > ACC, FDR;
	  //
	  // the current k will be the test fold
	  for ( int k = 0 ; k < k_ ; k++ )
	    {
	      // Number of training and testing
	      int 
		training = 0,
		testing  = groups_size_mapping_[k];
	      for ( int kk = 0 ; kk < k_ ; kk++ )
		if ( kk != k )
		  training += groups_size_mapping_[kk];

	      //
	      // Designs
	      Eigen::MatrixXd X_train( training, Dim + 1 );
	      Eigen::VectorXd Y_train( training );
	      Eigen::VectorXd W_train( training );
	      //
	      Eigen::MatrixXd X_test( testing, Dim + 1 );
	      Eigen::VectorXd Y_test( testing );
	      
	      //
	      // Filling the designs
	      int current_group_samp = 0;
	      for ( int kk = 0 ; kk < k_ ; kk++ )
		if ( kk != k )
		  {
		    for ( int samp = 0 ; samp < k_groups_Y_[ kk ].size() ; samp++ )
		      {
			// response
			Y_train(current_group_samp + samp)     = k_groups_Y_[ kk ][samp];
			// Desing
			X_train.row(current_group_samp + samp) = k_groups_X_[ kk ][samp];
		      }
		    // increment the group samp
		    current_group_samp += groups_size_mapping_[kk];
		  }		      
		else
		  for ( int samp = 0 ; samp < k_groups_Y_[ kk ].size() ; samp++ )
		    {
		      // response
		      Y_test(samp)     = k_groups_Y_[ kk ][samp];
		      // Desing
		      X_test.row(samp) = k_groups_X_[ kk ][samp];
		    }
	      
	      std::cout << "TRAIN k = " << k << "\n Y = \n" 
	       		<< Y_train
	       		<< "X = \n"
	       		<< X_train << std::endl;
	      std::cout << "TEST k = " << k << "\n Y = \n" 
	       		<< Y_test
	       		<< "X = \n"
			<< X_test << std::endl;
	      
	      //
	      //
	      // W_train = (X_train.transpose() * X_train).inverse() * X_train.transpose() * Y_train;
	      W_train = MACCrossValidation<Dim>::classify_->fit( X_train, Y_train );
	      // prediction
	      Eigen::VectorXd XW = X_test * W_train;
	      //
	      int 
		pop = Y_test.rows(),
		TP = 0, TN = 0,
		FP = 0, FN = 0;
	      double epsilon = 1.e-06;
	      for ( int l = 0 ; l < pop ; l++ )
		if ( Y_test(l) < 1 + epsilon &&  Y_test(l) > 1 - epsilon && XW(l) > 0.5 - epsilon )
		  TP++;
		else if ( Y_test(l) < 1 + epsilon &&  Y_test(l) > 1 - epsilon && XW(l) < 0.5 )
		  FN++;
		else if ( Y_test(l) < epsilon &&  Y_test(l) > - epsilon && XW(l) < 0.5 )
		  TN++;
		else if ( Y_test(l) < epsilon &&  Y_test(l) > - epsilon && XW(l) > 0.5 - epsilon )
		  FP++;
		else
		  throw MAC::MACException( __FILE__, __LINE__,
					   "Situation not covered",
					   ITK_LOCATION );
	      //
	      // Stat metrics
	      ACC.push_back( static_cast<double>(TP+TN) / static_cast<double>(pop) );
	      FDR.push_back( static_cast<double>(FP) / static_cast<double>(pop) );
	      
	      std::cout << "TEST W = " 
	       		<< W_train
	       		<< "Y_test = \n"
	       		<< Y_test 
			<< "W * X = \n"
			<< X_test * W_train
			<< std::endl;

	      
	      
	    } // end of k

	  //
	  // Stat metrics
	  // mean
	  double sum_acc  = std::accumulate(std::begin(ACC), std::end(ACC), 0.0);
	  double mean_acc =  sum_acc / ACC.size();
	  //
	  double sum_fdr  = std::accumulate(std::begin(FDR), std::end(FDR), 0.0);
	  double mean_fdr =  sum_fdr / FDR.size();
	  //
	  // variance
	  double accum_acc = 0.0;
	  std::for_each (std::begin(ACC), std::end(ACC), [&](const double d) {
	      accum_acc += (d - mean_acc) * (d - mean_acc);
	    });
	  double stdev_acc = sqrt( accum_acc / (ACC.size() - 1) );
	  //
	  double accum_fdr = 0.0;
	  std::for_each (std::begin(FDR), std::end(FDR), [&](const double d) {
	      accum_fdr += (d - mean_fdr) * (d - mean_fdr);
	    });
	  double stdev_fdr = sqrt( accum_fdr / (FDR.size() - 1) );
	  //
	  for ( int i = 0 ; i < ACC.size(); i++ )
	    {
	      std::cout << "ACC = " << ACC[i] 
			<< ", FDR = " << FDR[i]
			<< std::endl;
	    }
	  std::cout << "ACC = " << mean_acc << " +- " << stdev_acc
			<< ", FDR = " << mean_fdr << " +- " << stdev_fdr
			<< std::endl;

	  //
	  // Global coefficients
	  Eigen::MatrixXd X( n_, Dim + 1 );
	  Eigen::VectorXd Y( n_ );
	  Eigen::VectorXd W( n_ );
	  int current_group_samp = 0;
	  for ( int kk = 0 ; kk < k_ ; kk++ )
	    {
	      for ( int samp = 0 ; samp < k_groups_Y_[ kk ].size() ; samp++ )
		{
		  // response
		  Y(current_group_samp + samp)     = k_groups_Y_[ kk ][samp];
		  // Desing
		  X.row(current_group_samp + samp) = k_groups_X_[ kk ][samp];
		}
	      // increment the group samp
	      current_group_samp += groups_size_mapping_[kk];
	    }
	  // Calculate the coeff
	  //W = (X.transpose() * X).inverse() * X.transpose() * Y;
	  W = MACCrossValidation<Dim>::classify_->fit( X, Y );
	  // Record the weigths
	  for ( int w = 0 ; w < W.rows() ; w++ )
	    {
	      MACCrossValidation<Dim>::classify_->get_fit_weights().set_val( w, 
									     MACCrossValidation< Dim >::voxel_, 
									     W(w) );
	      std::cout << "W[" << w << "] = " <<  W(w) << std::endl;
	    }
	  // Record the statistics
	  MACCrossValidation<Dim>::classify_->get_ACC_FDR().set_val( 0, 
								     MACCrossValidation< Dim >::voxel_, 
								     mean_acc );
	  MACCrossValidation<Dim>::classify_->get_ACC_FDR().set_val( 1, 
								     MACCrossValidation< Dim >::voxel_, 
								     stdev_acc );
	  MACCrossValidation<Dim>::classify_->get_ACC_FDR().set_val( 2, 
								     MACCrossValidation< Dim >::voxel_, 
								     mean_fdr );
	  MACCrossValidation<Dim>::classify_->get_ACC_FDR().set_val( 3, 
								     MACCrossValidation< Dim >::voxel_, 
								     stdev_fdr );


	//std::cout << "Global W = " 
	//	    << W
	//	    << "Y = \n"
	//	    << Y 
	//	    << "W * X = \n"
	//	    << X * W
	//	    << std::endl;


	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	  exit( -1 );
	}
    }
}
#endif
