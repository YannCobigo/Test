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
    // write the optimaized parameter of the classifiaction engine
    virtual void write_parameters_images(){};
    // load the optimaized parameter of the classifiaction engine
    virtual void load_parameters_images(){};
    // write the subject maps
    virtual void write_subjects_map(){};
    // Optimization
    // Optimization algorithm implemented in this class
    virtual void optimize( const MaskType::IndexType );
    // multi-threading
    void operator () ( const MaskType::IndexType Idx )
    {
      std::cout << "treatment for parameters: " 
		<< Idx;
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
      std::cout << Classification<Dim>::subjects_[0].get_label(Idx) << std::endl;
      std::cout << (Classification<Dim>::subjects_[0].get_modalities(Idx))[0] << std::endl;
      std::cout << (Classification<Dim>::subjects_[0].get_modalities(Idx))[0] << std::endl;
    };
}
#endif
