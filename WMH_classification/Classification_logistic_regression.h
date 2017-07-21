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
#include "MACCrossValidation_k_folds.h"
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
    virtual Eigen::VectorXd fit( const Eigen::MatrixXd& X, const Eigen::VectorXd& Y ) const;
    // Prediction from the model
    virtual Eigen::VectorXd prediction( const Eigen::MatrixXd& X, const Eigen::VectorXd& W ) const;
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
    // Sigmoid
    double sigmoid( const Eigen::VectorXd& B,  const Eigen::VectorXd& X ) const
    { return 1. / ( 1. + exp( X.transpose() * B ) );};
    //
    // Error function
    // E(W) = - \sum_{i=1}^{n} t_{i} \ln y_{i} + (1+t_{i}) \ln (1+y_{i})
    // Using the derivative of \sigma(a):
    // \frac{d \sigma}{d a} = \sigma(a) ( 1 - \sigma(a) )
    // \nabla E(W) = \sum_{i=1}^{n} (y_{i} - t_{i}) X_{i}
    Eigen::VectorXd nabla_E( const Eigen::VectorXd&,  
			     const Eigen::MatrixXd& ) const;
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
    Classification_logistic_regression< Dim >::optimize( const MaskType::IndexType Idx )
    {
      std::cout << "IDX: " << Idx << std::endl;
      //std::cout << "image: " << MAC::Singleton::instance()->get_data()["inputs"]["images"][0][0]
      // << std::endl;

      //
      // Cross validation
      MACCrossValidation_k_folds<Dim> statistics( this, 
						  Idx,
						  /*k = */ MAC::Singleton::instance()->get_data()["strategy"]["CV_k_fold"], 
						  /*n = */ Classification<Dim>::get_subject_number() );
      statistics.CV();
    };
  //
  //
  // The logistic regression is based on the sigmoid: S(x) = \frac{1.}{1. + \exp{a + b.x}}
  // W = (a, b_{1}, b_{2}, ..., b_{Dim})
  template< int Dim > Eigen::VectorXd
    Classification_logistic_regression< Dim >::fit( const Eigen::MatrixXd& X, 
						    const Eigen::VectorXd& Y ) const
    {
      double 
	epsilon = 1.e-03, // convergence of the model
	// learning rate
	rho     = MAC::Singleton::instance()->get_data()["strategy"]["learning_rate"]; 
      // sigmoid coefficients
      Eigen::VectorXd W = Eigen::VectorXd::Ones( Dim + 1 );
      // result
      Eigen::VectorXd 
	hat_Y = Eigen::VectorXd::Ones( Y.rows() ),
	res   = Eigen::VectorXd::Ones( Y.rows() );
      
      //
      int count = 0;
      while ( res.transpose() * res > epsilon )
	{
	  W  += rho * nabla_E( res, X );
	  // reset the data
	  for ( int r = 0 ; r < Y.rows() ; r++ )
	    hat_Y(r) = sigmoid( W, X.row(r) );
	  res = ( hat_Y - Y );
	//std::cout << "At the iteration : " << count++ 
	//	    << " the Residual: " << res.transpose() * res
	//	    << " W = \n" << W
	//	    << "\n the Y-hat: \n" << hat_Y
	//	    << "\n the tag: \n" << Y
	//	    << std::endl;
	}

      //std::cout << "Final W is " << W << std::endl;
      
      //
      //
      return W;
    };
  //
  //
  // The logistic regression is based on the sigmoid: S(x) = \frac{1.}{1. + \exp{a + b.x}}
  // W = (a, b_{1}, b_{2}, ..., b_{Dim})
  template< int Dim > Eigen::VectorXd
    Classification_logistic_regression< Dim >::prediction( const Eigen::MatrixXd& X, 
							   const Eigen::VectorXd& W ) const
    {
      Eigen::VectorXd res = Eigen::VectorXd::Zero( X.rows() );
      for ( int r = 0 ; r < X.rows() ; r++ )
	res(r) = sigmoid( W, X.row(r) );

      //
      //
      return res;
    }
  //
  //
  // The logistic regression is based on the sigmoid: S(x) = \frac{1.}{1. + \exp{a + b.x}}
  // W = (a, b_{1}, b_{2}, ..., b_{Dim})
  template< int Dim > Eigen::VectorXd
    Classification_logistic_regression< Dim >::nabla_E( const Eigen::VectorXd& Res,  
							const Eigen::MatrixXd& X ) const
    {
      Eigen::VectorXd nabla = Eigen::VectorXd::Zero( X.cols() );
      //
      for ( int r = 0 ; r < Res.rows() ; r++ )
	nabla += Res(r) * X.row(r).transpose();
      //
      //
      return nabla;
    };
}
#endif
