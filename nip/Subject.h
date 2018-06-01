#ifndef NIPSUBJECT_H
#define NIPSUBJECT_H
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
using ImageType       = itk::Image< double, 3 >;
using ImageReaderType = itk::ImageFileReader< ImageType >;
using MaskType        = itk::Image< unsigned char, 3 >;
using MaskReaderType  = itk::ImageFileReader< MaskType >;
//
//
//
#include "NipException.h"
//
//
//
namespace MAC_nip
{
  inline bool file_exists ( const std::string& name )
  {
    std::ifstream f( name.c_str() );
    return f.good();
  }

  /** \class NipSubject
   *
   * \brief 
   * 
   */
  class NipSubject
  {
    //
    // Some typedef
    using Image3DType = itk::Image< double, 3 >;
    using Reader3D    = itk::ImageFileReader< Image3DType >;
    using MaskType    = itk::Image< unsigned char, 3 >;
 
  public:
    /** Constructor. */
  NipSubject():
    PIDN_{0}, group_{0} {};
    //
    explicit NipSubject( const int, const std::string,
			 const std::string, const std::string,
			 const std::list< double >& );
    
    /** Destructor */
    virtual ~NipSubject(){};

    //
    // Accessors
    inline const std::string get_PIDN()      const { return PIDN_ ;}
    const Eigen::MatrixXd get_image_matrix() const { return image_matrix_ ;}
    const Eigen::MatrixXd get_ev_matrix()    const { return ev_matrix_ ;}


    // Print
    void print() const;

  private:
    //
    // Subject parameters
    //
    
    // Group for multi-group comparison (controls, MCI, FTD, ...)
    // It can only take 1, 2, ... value
    int group_;
    // Identification number
    std::string PIDN_;
    // Image path
    std::string image_;
    // Mask path
    std::string mask_;
    
    //
    // Matrix holding the image features
    Eigen::MatrixXd image_matrix_;
    // matrix holding the explanatory variables
    Eigen::MatrixXd ev_matrix_;
  };
}
#endif
