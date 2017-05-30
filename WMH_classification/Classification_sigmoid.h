#ifndef CLASSIFICATION_SIGMOID_H
#define CLASSIFICATION_SIGMOID_H
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
using MaskType = itk::Image< unsigned char, 3 >;
//
//
//
#include "Classification.h"
#include "MACException.h"
#include "MACMakeITKImage.h"
#include "Subject.h"
//
//
//
namespace MAC
{
  /** \class Classification_sgmoid
   *
   * \brief 
   * 
   */
  template< int Dim >
    class Classification_sgmoid : public Classification< Dim >
  {
  public:
    /** Constructor. */
    explicit Classification_sgmoid(){};
    
    /** Destructor */
    virtual ~Classification_sgmoid(){};


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
    virtual void optimize( const MaskType::IndexType ){};
    // multi-threading
    void operator () ( const MaskType::IndexType idx )
    {
      std::cout << "treatment for parameters: " 
		<< idx;
      optimize( idx );
    };



  };
}
#endif
