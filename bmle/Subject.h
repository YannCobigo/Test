#ifndef BMLESUBJECT_H
#define BMLESUBJECT_H
//
//
//
#include <iostream>
#include <map>
#include <list>
// Egen
#include <Eigen/Core>
#include <Eigen/Eigen>
//
//
//
namespace MAC_bmle
{
  /** \class BmleSubject
   *
   * \brief 
   * 
   */
  class BmleSubject
  {
  public:
    /** Constructor. */
  BmleSubject():
    PIDN_{0}, group_{0}, D_{0} {};
    //
    explicit BmleSubject( const int, const int );
    
    /**  */
    virtual ~BmleSubject(){};

    //
    //
    inline const int get_PIDN() const { return PIDN_ ;};

    //
    // Add time point
    inline void add_tp( const int Age, const std::list< float >& Covariates,
			const std::string& Image )
    {
      if ( age_covariates_.find( Age ) == age_covariates_.end() )
	{
	  age_covariates_[ Age ] = Covariates;
	  age_images_[ Age ]     = Image;
	  time_points_++;
	}
      else
	std::cerr << "Age " << Age << " is already entered for the patient " << PIDN_
		 << "." << std::endl;
    }
    // Convariates' model
    void build_covariates_matrix();
    //
    // Print
    inline void print()
    {
      std::cout << "PIDN: " << PIDN_ << std::endl;
      std::cout << "Group: " << group_ << std::endl;
      std::cout << "Number of time points: " << time_points_ << std::endl;
      //
      std::cout << "Age and covariates: " << std::endl;
      if ( !age_covariates_.empty() )
	for ( auto age_cov : age_covariates_ )
	  {
	    std::cout << "At age " << age_cov.first << " covariates were:";
	    for( auto c : age_cov.second )
	      std::cout << " " << c;
	    std::cout << std::endl;
	  }
      else
	std::cout << "No age and covariates recorded." << std::endl;
      //
      std::cout << "Age and imagess: " << std::endl;
      if ( !age_images_.empty() )
	for ( auto age_img : age_images_ )
	  std::cout << "At age " << age_img.first << " iamge was: "
		    << age_img.second << std::endl;
      else
	std::cout << "No age and images recorded." << std::endl;
    }

  private:
    //
    // Subject parameters
    //
    
    // Identification number
    int PIDN_;
    // Group for multi-group comparisin (controls, MCI, FTD, ...)
    int group_;
    // image ITK
    // Age covariate map
    std::map< int, std::list< float > > age_covariates_;
    // Age covariate map
    std::map< int, std::string > age_images_;
    // Number of time points
    int time_points_{0};

    //
    // Model parameters
    //

    // Model dimension
    int D_;
    // Matrix of covariates
    Eigen::MatrixXf covariates_;
    //
    // Random effect
    // theta level 1
    Eigen::VectorXf theta_1_;
    // non-lin model in function of age
    Eigen::VectorXf model_age_;
  };
}
#endif
