#ifndef CLASSIFICATION_H
#define CLASSIFICATION_H
//
// ITK
//
#include <itkImageFileReader.h>
using MaskType = itk::Image< unsigned char, 3 >;
//
//
//
#include "MACException.h"
//
//
//
namespace MAC
{
  /** \class Classification
   *
   * \brief 
   *
   * The template argument Dim represents modalities dimensions of the linear/non-linear models used 
   * for the classification. The feature space is composed of Dim dimensions of modality: Dim = 1, 
   * we are using only one modality, Dim = 2: two modalities.
   * The rest of the feature space dimensions are not yet accessible from the input data set.
   * 
   */
  template< int Dim >
    class Classification
  {
  public:
    /** Constructor. */
    explicit Classification(){};
    
    /** Destructor */
    virtual ~Classification(){};


    //
    // train the calssification engin
    virtual void train() = 0;
    // use the calssification engin
    virtual void use()   = 0;
    // write the optimaized parameter of the classifiaction engine
    virtual void write_parameters_images()  = 0;
    // load the optimaized parameter of the classifiaction engine
    virtual void load_parameters_images()   = 0;
    // write the subject maps
    virtual void write_subjects_map()       = 0;
    // Optimization
    virtual void optimize( const MaskType::IndexType ) = 0;
  };
}
#endif
