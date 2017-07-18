#ifndef MACCROSSVALIDATION_H
#define MACCROSSVALIDATION_H
//
//
//
#include <iostream>
#include <fstream>
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
#include "Classification.h"
//
//
//
namespace MAC
{
  /** \class MACCrossValidation
   *
   * \brief 
   * 
   */
  template< int Dim >
    class MACCrossValidation
    {
    public:
      /** Constructor. */
    MACCrossValidation( const MAC::Classification< Dim >* Classify,
			const MaskType::IndexType         Idx ):
      classify_{Classify}
      {
	voxel_ = Idx;
      };
      
      /**  */
      virtual ~MACCrossValidation( )
	{
	  // no delete, it is a strategy: delete classify_;
	  classify_ = nullptr;
	};

      //
      //
      virtual void CV() const = 0;
      
    protected:
      const Classification< Dim >* classify_;
      MaskType::IndexType          voxel_;
    };
}
#endif
