#ifndef CLASSIFICATION_H
#define CLASSIFICATION_H
//
// ITK
//
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIterator.h>
#include <itkNiftiImageIO.h>
#include <itkOrientImageFilter.h>
#include <itkSpatialOrientation.h>
using Image4DType = itk::Image< double, 4 >;
using Reader4D    = itk::ImageFileReader< Image4DType >;
using MaskType    = itk::Image< unsigned char, 3 >;
// Eigen
#include <Eigen/Core>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/KroneckerProduct>
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
    // Fit the model
    virtual Eigen::VectorXd fit ( const Eigen::MatrixXd&, const Eigen::VectorXd& ) const = 0;
    // Prediction from the model
    virtual Eigen::VectorXd prediction( const Eigen::MatrixXd& X, const Eigen::VectorXd& W ) const = 0;
    // write the subject maps
    virtual void write_subjects_map() = 0;
    // Optimization
    virtual void optimize( const MaskType::IndexType ) = 0;

    //
    //
    Eigen::MatrixXd normalization( const Eigen::MatrixXd );

    
    //
    //
    int get_subject_number(){return subject_number_;};
    //
    const std::vector< Subject< Dim > >& get_subjects() const {return subjects_;};
    MACMakeITKImage& get_fit_weights(){return fit_weights_;};
    MACMakeITKImage& get_ACC_FDR(){return ACC_FDR_;};


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
    // Input/Outputs
    //
    
    //
    // Training 
    MACMakeITKImage fit_weights_;
    // Accuracy and False descovery rate
    MACMakeITKImage ACC_FDR_;
    
    //
    // Using 
    Reader4D::Pointer weights_fitted_;
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
	  // if training
	  if ( MAC::Singleton::instance()->get_status() )
	    subjects_[sub].add_label( sub );
	  // if using
	  else
	    subjects_[sub].create_output_map( sub,
					      MAC::Singleton::instance()->get_data()["output"]["map"][sub] );
	}

      
      
      //
      // If training
      std::string output_model = MAC::Singleton::instance()->get_data()["strategy"]["weights"];
      if ( MAC::Singleton::instance()->get_status() )
	{
	  //
	  // Output weights
	  fit_weights_ = MACMakeITKImage( Dim + 1,
					  output_model,
					  subjects_[0].get_sample() );
	  // Output weights
	  ACC_FDR_ = MACMakeITKImage( 4,
				      "Acc_sd_FDR_sd.nii.gz",
				      subjects_[0].get_sample() );
	}
      // If using
      else
	{
	  //
	  // Load the regression coefficients
	  weights_fitted_ = Reader4D::New();
	  weights_fitted_->SetFileName( output_model );
	  weights_fitted_->Update();
	  // Check the dimensions comply
	  Image4DType::SizeType size = 
	    Classification<Dim>::weights_fitted_->GetOutput()->GetLargestPossibleRegion().GetSize();
	  if ( size[3] != Dim + 1 )
	    throw MAC::MACException( __FILE__, __LINE__,
				     "The dimension of the design and the number of weight don't comply.",
				     ITK_LOCATION );
	}
    };
  //
  //
  //
  template< int Dim > void
    Classification< Dim >::write_parameters_images()
    {
      fit_weights_.write();
      ACC_FDR_.write();
    }
  //
  //
  //
  template< int Dim > Eigen::MatrixXd
    Classification< Dim >::normalization( const Eigen::MatrixXd Mat)
    {
      int ls = Mat.rows();
      int cs = Mat.cols();
      //
      Eigen::MatrixXd tempo = Mat;
      //
      for ( int col = 0 ; col < cs ; col++ )
	{
	  double
	    mean      = 0.,
	    sigma_sqr = 0.;
	  //
	  for ( int l = 0 ; l < ls ; l++ )
	    mean += Mat.col( col )(l);
	  mean /= static_cast< double >( ls );
	  //
	  for ( int l = 0 ; l < ls ; l++ )
	    sigma_sqr += (mean - Mat.col( col )(l))*(mean - Mat.col( col )(l));
	  sigma_sqr /= static_cast< double >( ls - 1 );
	  //
	  for ( int l = 0 ; l < ls ; l++ )
	    tempo.col( col )(l) = ( Mat.col( col )(l) - mean ) / sqrt( sigma_sqr );

	  //
	  //
	  tempo.col( 0 ) = Mat.col( 0 );
	}
      return tempo;
    }
}
#endif
