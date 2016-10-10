#ifndef BMLELOADCSV_H
#define BMLELOADCSV_H
//
//
//
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
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
    
    /**  */
    virtual ~BmleLoadCSV() {};

  private:
    // CSV file
    std::ifstream csv_file_;
   
  };
}
#endif
