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
#include "PMA.h"
//
using PMA_type = std::tuple< std::shared_ptr< const Eigen::MatrixXd >,
                             std::shared_ptr< const Eigen::MatrixXd >,
                             std::shared_ptr< Spectra > >;
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
				 const std::string&,
				 const int );
    /** Destructor */
    virtual ~NipSubject_Mapping() {};

    /** Display the spectrum **/
    void dump() ;

    /** Access the PMA type **/
    std::map< int /*group*/, PMA_type >& get_PMA(){ return group_matrices_;};

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
    // first matrix is the group image matrix and the second is the
    // group explenatory variables
    std::map< int /*group*/, PMA_type > group_matrices_;
  };
}
#endif
