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
    virtual void optimize( const MaskType::IndexType ){};
    // multi-threading
    void operator () ( const MaskType::IndexType idx )
    {
      std::cout << "treatment for parameters: " 
		<< idx;
      optimize( idx );
    };

    
  private:
    //
    // Number of modalities
    int modalities_number_{ Dim };
    // Number of subjects
    int subject_number_;
    //
    // For each of the Dim modalities we load the measures of 3D images
    using Image3DType  = itk::Image< double, 3 >;
    using Reader3DType = itk::ImageFileReader< Image3DType >;
    std::vector< std::vector< Reader3DType::Pointer > > modalities_{Dim}; 
    
  };
  //
  //
  //
  template< int Dim >
    Classification_linear_regression< Dim >::Classification_linear_regression():
  Classification< Dim >()
    {
      //
      // We check the number of modalities is the same as the number of dimensions
      if ( MAC::Singleton::instance()->get_data()["inputs"]["images"].size() != Dim )
	{
	  std::string mess = "This code has been compiled for " + std::to_string(Dim);
	  mess += " modalities.\n";
	  mess += "The data set is asking for ";
	  mess += std::to_string( MAC::Singleton::instance()->get_data()["inputs"]["images"].size() );
	  mess += " modalities.\n ";
	  throw MAC::MACException( __FILE__, __LINE__,
				   mess.c_str(),
				   ITK_LOCATION );
	}

      //
      //
      subject_number_ = MAC::Singleton::instance()->get_data()["inputs"]["images"][0].size();
      //std::cout << "Number of sujbjects: " << subject_number_ << std::endl;
    };
}
#endif
