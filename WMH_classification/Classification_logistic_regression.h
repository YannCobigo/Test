#ifndef CLASSIFICATION_LOGISTIC_REGRESSION_H
#define CLASSIFICATION_LOGISTIC_REGRESSION_H
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
//#include "Subject.h"
//
//
//
namespace MAC
{
  /** \class Classification_logistic_regression
   *
   * \brief 
   * 
   */
  template< int Dim >
    class Classification_logistic_regression : public Classification< Dim >
  {
  public:
    /** Constructor. */
    explicit Classification_logistic_regression();
    
    /** Destructor */
    virtual ~Classification_logistic_regression(){};


    //
    // train the calssification engin
    virtual void train(){};
    // use the calssification engin
    virtual void use(){};
    // Fit the model
    virtual Eigen::VectorXd fit( const Eigen::MatrixXd& X, const Eigen::VectorXd& Y ) const {};
    // write the subject maps
    virtual void write_subjects_map(){};
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
    Classification_logistic_regression< Dim >::Classification_logistic_regression():
  Classification< Dim >()
    {
    };
  //
  //
  //
  template< int Dim > void
    Classification_logistic_regression< Dim >::optimize( const MaskType::IndexType Idx)
    {
      std::cout << "IDX: " << Idx << std::endl;
      std::cout << "image: " << MAC::Singleton::instance()->get_data()["inputs"]["images"][0][0]
		<< std::endl;

      //
      // Design matrix
      Eigen::MatrixXd X( Classification<Dim>::get_subject_number(), Dim + 1 );
      Eigen::VectorXd Y( Classification<Dim>::get_subject_number() );
      Eigen::VectorXd W( Classification<Dim>::get_subject_number() );
      for ( int subject = 0 ; subject < Classification<Dim>::get_subject_number() ; subject++ )
	{
	  //
	  // Label
	  Y(subject) = static_cast< double >( Classification<Dim>::subjects_[subject].get_label(Idx) );
	  //
	  // subject design matrix
	  X( subject, 0 ) = 1.;
	  for ( int mod = 0 ; mod < Dim ; mod++ )
	    X( subject, mod + 1 ) = (Classification<Dim>::subjects_[subject].get_modalities(Idx))[mod];
	}
      std::cout << Y << std::endl;
      std::cout << X << std::endl;

      //
      // Linear regression
      // \hat{W} = (XX^{T})^(-1)X^{T}Y
      W = X.transpose() * Y;
      std::cout << W << std::endl;


      //
      // Record the weigths
      for ( int w = 0 ; w < W.rows() ; w++ )
	Classification<Dim>::fit_weights_.set_val( w, Idx, W(w) );
    };
}
#endif
