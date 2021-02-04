#ifndef CLASSIFICATION_LINEAR_REGRESSION_H
#define CLASSIFICATION_LINEAR_REGRESSION_H
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
//
// ITK
//
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIterator.h>
#include <itkNiftiImageIO.h>
#include <itkOrientImageFilter.h>
#include <itkSpatialOrientation.h>
using Image4DType = itk::Image< double, 4 >;
using Reader4D    = itk::ImageFileReader< Image4DType >;
using MaskType    = itk::Image< unsigned char, 3 >;
//
// JSON interface
//
#include "json.hpp"
using json = nlohmann::json;
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
using MaskType = itk::Image< unsigned char, 3 >;
//
//
//
#include "Classification.h"
#include "MACException.h"
#include "MACMakeITKImage.h"
#include "MACCrossValidation_k_folds.h"
//#include "Subject.h"
//
//
//
namespace MAC
{
  /** \class Classification_linear_regression
   *
   * \brief 
   * 
   */
  template< int Dim >
    class Classification_linear_regression : public Classification< Dim >
  {
  public:
    /** Constructor. */
    explicit Classification_linear_regression();
    
    /** Destructor */
    virtual ~Classification_linear_regression(){};


    //
    // train the calssification engin
    virtual void train(){};
    // use the calssification engin
    virtual void use(){};
    // Fit the model
    virtual Eigen::VectorXd fit( const Eigen::MatrixXd& X, const Eigen::VectorXd& Y ) const
    { return (X.transpose() * X).inverse() * X.transpose() * Y;};
    // Prediction from the model
    virtual Eigen::VectorXd prediction( const Eigen::MatrixXd& X, const Eigen::VectorXd& W ) const
    {return X * W;};
    // write the subject maps
    virtual void write_subjects_map();
    // Optimization
    // Optimization algorithm implemented in this class
    virtual void optimize( const MaskType::IndexType );
    // multi-threading
    void operator () ( const MaskType::IndexType Idx )
    {
      std::cout << "treatment for parameters: " 
		<< Idx << std::endl;
      optimize( Idx );
    };

  private:
    //
    // For each of the Dim modalities we load the measures of 3D images
  };
  //
  //
  //
  template< int Dim >
    Classification_linear_regression< Dim >::Classification_linear_regression():
  Classification< Dim >()
    {
    };
  //
  //
  //
  template< int Dim > void
    Classification_linear_regression< Dim >::optimize( const MaskType::IndexType Idx)
    {
      //std::cout << "IDX: " << Idx << std::endl;
      //std::cout << "image: " << MAC::Singleton::instance()->get_data()["inputs"]["images"][0][0]
      //<< std::endl;

      //
      // If training
      if ( MAC::Singleton::instance()->get_status() )
	{
	  //
	  // Cross validation
	  MACCrossValidation_k_folds<Dim> statistics( this, 
						      Idx,
						      /*k = */ MAC::Singleton::instance()->get_data()["strategy"]["CV_k_fold"], 
						      /*n = */ Classification<Dim>::get_subject_number() );
	  statistics.CV();
	}
      // Using
      else
	{
	  //
	  //
	  int n = Classification<Dim>::get_subject_number();
	  //
	  Eigen::MatrixXd X( n, Dim + 1 );
	  Eigen::VectorXd W( Dim + 1 );
	  W(0) = Classification<Dim>::weights_fitted_->GetOutput()->GetPixel( {Idx[0],Idx[1],Idx[2],0} );
	  for ( int subject = 0 ; subject < n ; subject++ )
	    {
	      X(subject, 0) = 1.; // for beta_0
	      for ( int mod = 0 ; mod < Dim ; mod++ )
		{
		  X( subject, mod + 1 ) = (( Classification<Dim>::get_subjects() )[subject].get_modalities(Idx))[mod];
		  W( mod + 1 ) = Classification<Dim>::weights_fitted_->GetOutput()->GetPixel( {Idx[0],Idx[1],Idx[2],mod+1} );
		}
	    }
	  //
	  // Process the results
	  Eigen::VectorXd XW = prediction( X, W );
	  //std::cout << "Idx: " << Idx
	  //	    << "\n X: " << X
	  //	    << "\n W: " << W
	  //	    << "\n XW: " << XW
	  //	    << std::endl;
	  //
	  for ( int subject = 0 ; subject < n ; subject++ )
	    ( Classification<Dim>::subjects_ )[subject].set_fit( Idx, XW(subject) );
	}
    };
  //
  //
  //
  template< int Dim > void
    Classification_linear_regression< Dim >::write_subjects_map()
    {
      int n = Classification<Dim>::get_subject_number();
      //
      for ( int subject = 0 ; subject < n ; subject++ )
	Classification<Dim>::subjects_[subject].write_solution();
    };
}
#endif
