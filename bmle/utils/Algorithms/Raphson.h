#ifndef RAPHSON_H
#define RAPHSON_H
//
#include "Algorithm.h"

//
//
//
namespace NeuroBayes
{
  /** \class Raphson
   *
   * \brief Newton-Raphson algorithm
   * 
   */
  class Raphson : public Algorithm
  {
  public:
    /** Constructor. */
    Raphson();
    
    //
    //
    virtual void update();
  };
}
#endif
