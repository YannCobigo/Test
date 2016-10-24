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

    // This function will load all the patients images into a 4D image.
    void image_concat();
    //

    
  private:
    //
    // CSV file
    std::ifstream csv_file_;
    //
    // Arrange pidns inti groups
    std::set< int > groups_;
    std::vector< std::map< int /*pidn*/, BmleSubject > > group_pind_{10};
    //
    // Measures in  4D image
    using Image4DType = itk::Image< float, 4 >;
    Image4DType::Pointer Y_;
    // number of 3D images
    long unsigned int num_3D_images_{0};
  };
}
#endif
