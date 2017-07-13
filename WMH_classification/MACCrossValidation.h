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
    MACCrossValidation( const MAC::Classification< Dim >* Classify ):
      classify_{Classify}{};
      
      /**  */
      virtual ~MACCrossValidation( )
	{
	  // no delete, it is a strategy: delete classify_;
	  classify_ = nullptr;
	};
      
    private:
      const Classification< Dim >* classify_;
    };
}
#endif
