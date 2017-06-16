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
#include "Subject.h"
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
    explicit Classification();
    
    /** Destructor */
    virtual ~Classification(){};


    //
    // write the optimaized parameter of the classifiaction engine
     void write_parameters_images();
    // load the optimaized parameter of the classifiaction engine
     void load_parameters_images(){};
    // train the calssification engin
    virtual void train() = 0;
    // use the calssification engin
    virtual void use()   = 0;
    // write the subject maps
    virtual void write_subjects_map()       = 0;
    // Optimization
    virtual void optimize( const MaskType::IndexType ) = 0;

    //
    //
    int get_subject_number(){return subject_number_;};


  private:
    //
    // Number of modalities
    int modalities_number_{ Dim };
    // Number of subjects
    int subject_number_;

  protected:
    //
    // For each of the Dim modalities we load the measures of 3D images
    std::vector< Subject< Dim > > subjects_;

    //
    // Output
    MACMakeITKImage fit_weights_;
  };
  //
  //
  //
  template< int Dim >
    Classification< Dim >::Classification()
    {
      //
      // We check the number of modalities is the same as the number of dimensions
      if ( MAC::Singleton::instance()->get_data()["inputs"]["images"].size() != Dim )
	{
	  std::string mess = "This code has been compiled for " + std::to_string(Dim);
	  mess += " modalities.\n";
	  mess += "The data set is asking for ";
	  mess += std::to_string( MAC::Singleton::instance()->get_data()["inputs"]["images"].size() );
	  mess += " modalities.\n ";
	  throw MAC::MACException( __FILE__, __LINE__,
				   mess.c_str(),
				   ITK_LOCATION );
	}
      //
      subject_number_ = MAC::Singleton::instance()->get_data()["inputs"]["images"][0].size();
      //std::cout << "Number of sujbjects: " << subject_number_ << std::endl;

      
      
      //
      // Load the subjects data and mask
      subjects_.resize( subject_number_ );
      for ( int sub = 0 ; sub < subject_number_ ; sub++ )
	{
	  // modalities
	  subjects_[sub] = Subject< Dim >( sub );
	  for ( int mod = 0 ; mod < modalities_number_ ; mod++ )
	    subjects_[sub].add_modality( sub, mod );
	  // if trainning: mask
	  if ( MAC::Singleton::instance()->get_status() )
	    subjects_[sub].add_label( sub );
	}

      
      
      //
      // Output weights
      std::string output_model = MAC::Singleton::instance()->get_data()["strategy"]["weights"];
      fit_weights_ = MACMakeITKImage( Dim + 1,
				      output_model,
				      subjects_[0].get_sample() );
      
    };
  //
  //
  //
  template< int Dim > void
    Classification< Dim >::write_parameters_images()
    {
      fit_weights_.write();
    }
}
#endif
