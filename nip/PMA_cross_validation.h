#ifndef NIP_PMA_CROSS_VALIDATION_H
#define NIP_PMA_CROSS_VALIDATION_H
//
//
//
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <math.h>  
#include <map>
#include <list>
#include <memory>
//
#include "PMA.h"
#include "PMA_tools.h"
//
//
//
#include "NipException.h"

//
//
//
namespace MAC_nip
{
  /** \class Nip_PMA_cross_validation
   *
   * \brief PMA: Penalized Matrices Analysise 
   * This is a class of cross validation.
   * "A penalized matrix decomposition, with applications to sparse principal components and canonical correlation analysis" (PMID:19377034)
   * 
   */
  class Nip_PMA_cross_validation
  {
  public:
    //
    //
    virtual void validation( std::shared_ptr< Spectra > ) = 0;
    //
    //
    virtual void k_folds( const std::vector< double > ) = 0;
    //
    //
    virtual void operator ()( const std::vector< double > Paramters ) = 0;
  };
}
#endif
