#ifndef BMLELOADCSV_H
#define BMLELOADCSV_H
//
//
//
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <map>
#include <set>
#include <vector>
//
//
//
#include "Subject.h"
//
//
//
namespace MAC_bmle
{
  /** \class BmleLoadCSV
   *
   * \brief 
   * 
   */
  class BmleLoadCSV
  {
  public:
    /** Constructor. */
    explicit BmleLoadCSV( const std::string& );
    
    /** Destructor */
    virtual ~BmleLoadCSV() {};


    //
    // Functions
    //

    //
    void image_cat();
    //

    
  private:
    // CSV file
    std::ifstream csv_file_;
    // Arrange pidns inti groups
    std::set< int > groups_;
    std::vector< std::map< int /*pidn*/, BmleSubject > > group_pind_{10};
   
  };
}
#endif
