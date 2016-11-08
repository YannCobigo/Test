#ifndef BMLEGROUP_H
#define BMLEGROUP_H
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
//
//
//
#include "BmleException.h"
//
//
//
namespace MAC_bmle
{
  /** \class BmleGroup
   *
   * \brief 
   * D_r: number of random degres of the model.
   * D_f: number of fixed degres of the model.
   * 
   */
  template< int D_r, int D_f >
    class BmleGroup
    {
      //
      // Some typedef
      using Image3DType = itk::Image< float, 3 >;
      using Reader3D    = itk::ImageFileReader< Image3DType >;

    public:
      /** Constructor. */
    BmleGroup():
      group_{0}{};
      //
      explicit BmleGroup( const int, const int );
    
      /**  */
      virtual ~BmleGroup(){};


    private:
      //
      // private member function
      //
    };

  //
  //
  //
  template < int D_r, int D_f >
  MAC_bmle::BmleGroup< D_r, D_f >::BmleGroup( const int Group):
  group_{Group}
  {}
}
#endif
