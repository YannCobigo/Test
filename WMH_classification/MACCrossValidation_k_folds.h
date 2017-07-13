#ifndef MACCROSSVALIDATION_K_FOLDS_H
#define MACCROSSVALIDATION_K_FOLDS_H
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
#include "MACCrossValidation.h"
#include "Classification.h"
//
//
//
namespace MAC
{
  /** \class MACCrossValidation_k_folds
   *
   * \brief 
   * 
   */
  template< int Dim >
    class MACCrossValidation_k_folds : public MACCrossValidation< Dim >
    {
    public:
      /** Constructor. */
      explicit MACCrossValidation_k_folds( const Classification< Dim >* ,
					   const int, const int );
      
      /**  */
      virtual ~MACCrossValidation_k_folds(){};
      
    private:
      // number of fold of the cross-validation
      int k_;
      // Number of subjects
      int n_;
    };

  //
  //
  //
  template< int Dim >
    MAC::MACCrossValidation_k_folds<Dim>::MACCrossValidation_k_folds( const Classification< Dim >* Classify,
								      const int K,
								      const int Num_subjects ): 
    MACCrossValidation<Dim>( Classify ),
    k_{ K }, n_{ Num_subjects }
  {
    
  }
}
#endif
