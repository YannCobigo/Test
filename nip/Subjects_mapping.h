#ifndef NIPSUBJECT_MAPPING_H
#define NIPSUBJECT_MAPPING_H
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
//
//
#include "NipException.h"
#include "Subject.h"
//
//
//
namespace MAC_nip
{
  /** \class NipSubject_Mapping
   *
   * \brief 
   * 
   */
  class NipSubject_Mapping
  {
  public:
    /** Constructor. */
    explicit NipSubject_Mapping( const std::string&,
				 const std::string& );
    /** Destructor */
    virtual ~NipSubject_Mapping() {};


  private:
    //
    // Members
    //
    
    //
    // CSV file
    std::ifstream csv_file_;
    std::string   mask_;
    //
    // Arrange pidns into groups
    std::set< std::string > PIDNs_;
    std::set< int > groups_;
    std::map< int /*group*/, std::vector< NipSubject > > group_pind_;
    
  };
}
#endif
