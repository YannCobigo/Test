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
    // This function will load all the patients images into a 4D image.
    void build_groups_design_matrices();

    
  private:
    //
    // Functions
    //

    // This function will load all the patients images into a 4D image.
    void image_concat();

    //
    // Members
    //
    
    //
    // CSV file
    std::ifstream csv_file_;
    //
    // Arrange pidns inti groups
    std::set< int > groups_;
    std::vector< std::map< int /*pidn*/, BmleSubject< 3, 3 > > > group_pind_{10};
    // Number of subjects per group
    std::vector< int > group_num_subjects_{0,0,0,0,0,0,0,0,0,0};
    //
    // Measures in  4D image
    using Image4DType = itk::Image< float, 4 >;
    Image4DType::Pointer Y_;
    // number of PIDN
    long unsigned int num_subjects_{0};
    // number of 3D images = number of time points (TP)
    long unsigned int num_3D_images_{0};
    //
    int num_covariates_{0};
  };
}
#endif
