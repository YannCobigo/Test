#ifndef BMLEMAKEITKIMAGE_H
#define BMLEMAKEITKIMAGE_H
//
//
//
#include <iostream>
#include <fstream>
#include <math.h>  
#include <map>
#include <list>
// Egen
#include <Eigen/Core>
#include <Eigen/Eigen>
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
#include "itkChangeInformationImageFilter.h"
//
//
//
#include "BmleException.h"
//
//
//
namespace MAC_bmle
{
  /** \class BmleMakeITKImage
   *
   * \brief 
   * D_r: number of random degres of the model.
   * D_f: number of fixed degres of the model.
   * 
   */
  class BmleMakeITKImage
    {
      //
      // Some typedef
      using Image3DType = itk::Image< double, 3 >;
      using Reader3D    = itk::ImageFileReader< Image3DType >;
      using Image4DType = itk::Image< double, 4 >;
      using Reader4D    = itk::ImageFileReader< Image4DType >;
      using Writer4D    = itk::ImageFileWriter< Image4DType >;

    public:
      /** Constructor. */
      BmleMakeITKImage():D_{0},image_name_{""}{};
      //
      explicit BmleMakeITKImage( long unsigned int , std::string& );
    
      /**  */
      virtual ~BmleMakeITKImage(){};

      //
      // Record results
      void record(){};
      // Write image
      void write();

    private:
      //
      // Dimension of the case: random or fixed
      long unsigned int D_;
      // Image name
      std::string image_name_;
      // Measures grouped in vector of 3D image
      std::vector< Reader3D::Pointer > images_;
    };
}
#endif
