


#include "Subject.h"


MAC_bmle::BmleSubject::BmleSubject( const int Pidn, const int Group):
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
  //
  // Initialize the covariate matrix
  // and the random parameters
  std::map< int, std::list< float > >::const_iterator age_cov_it = age_covariates_.begin();
  //
  covariates_.resize( age_covariates_.size() * (D_ + 1), (*age_cov_it).second.size() * (D_ + 1));
  covariates_ = Eigen::MatrixXf::Zero(age_covariates_.size() * (D_ + 1), ((*age_cov_it).second.size() + 1 )* (D_ + 1));
  //
  theta_1_.resize( D_ + 1 );
  theta_1_ = Eigen::VectorXf::Random();
  model_age_.resize( D_ + 1 );
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
