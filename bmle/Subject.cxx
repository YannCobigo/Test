#include "Subject.h"


MAC_bmle::BmleSubject::BmleSubject( const int Pidn,
				    const int Group):
  PIDN_{Pidn}, group_{Group}, D_{2}
{
  /* 
     g(t, \theta_{i}^{(1)}) = \sum_{d=1}^{D+1} \theta_{i,d}^{(1)} t^{d-1}
  */
}
//
//
//
void
MAC_bmle::BmleSubject::build_covariates_matrix()
{
  try
    {
      //
      // Initialize the covariate matrix
      // and the random parameters
      std::map< int, std::list< float > >::const_iterator age_cov_it = age_covariates_.begin();
      //
      covariates_.resize( age_covariates_.size() * (D_ + 1), (*age_cov_it).second.size() * (D_ + 1));
      covariates_ = Eigen::MatrixXf::Zero(age_covariates_.size() * (D_ + 1), ((*age_cov_it).second.size() + 1 )* (D_ + 1));
      //
//      theta_1_.resize( D_ + 1 );
//      theta_1_ = Eigen::VectorXf::Random();
//      model_age_.resize( D_ + 1 );
      //
      int line = 0;
      int col  = 0;
      for ( ; age_cov_it != age_covariates_.end() ; age_cov_it++ )
	{
	  covariates_.block( line * (D_ + 1), 0, D_ + 1, D_ + 1 ) = Eigen::MatrixXf::Identity( D_ + 1, D_ + 1 );
	  col = 0;
	  // covariates
	  for ( auto cov : (*age_cov_it).second )
	    {
	      covariates_.block( line * (D_ + 1), ++col * (D_ + 1), D_ + 1, D_ + 1 ) = cov * Eigen::MatrixXf::Identity( D_ + 1, D_ + 1 );
	    }
	  // next age
	  line++;
	}
      
      std::cout << covariates_ << std::endl;
    }
  catch( itk::ExceptionObject & err )
    {
      std::cerr << err << std::endl;
      return exit( -1 );
    }
}
//
//
//
void
MAC_bmle::BmleSubject::add_tp( const int                 Age,
			       const std::list< float >& Covariates,
			       const std::string&        Image )
{
  try
    {
      if ( age_covariates_.find( Age ) == age_covariates_.end() )
	{
	  age_covariates_[ Age ] = Covariates;
	  age_images_[ Age ]     = Image;
	  //
	  // load the ITK images
	  if ( file_exists(Image) )
	    {
	      //
	      // load the image ITK pointer
	      auto image_ptr = itk::ImageIOFactory::CreateImageIO( Image.c_str(),
								   itk::ImageIOFactory::ReadMode );
	      image_ptr->SetFileName( Image );
	      image_ptr->ReadImageInformation();
	      // Read the ITK image
	      age_ITK_images_[ Age ] = Reader3D::New();
	      age_ITK_images_[ Age ]->SetFileName( image_ptr->GetFileName() );
	      age_ITK_images_[ Age ]->Update();
	    }
	  else
	    {
	      std::string mess = "Image " + Image + " does not exists.";
	      throw MAC_bmle::BmleException( __FILE__, __LINE__,
					     mess.c_str(),
					     ITK_LOCATION );
	    }
	  //
	  time_points_++;
	}
      else
	{
	  std::string mess = "Age " + std::to_string(Age) + " is already entered for the patient ";
	  mess += std::to_string(PIDN_) + ".";
	  //
	  throw MAC_bmle::BmleException( __FILE__, __LINE__,
					 mess.c_str(),
					 ITK_LOCATION );
	}
    }
  catch( itk::ExceptionObject & err )
    {
      std::cerr << err << std::endl;
      return exit( -1 );
    }
}
