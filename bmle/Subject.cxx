#include "Subject.h"


MAC_bmle::BmleSubject::BmleSubject( const int Pidn, const int Group):
  PIDN_{Pidn}, group_{Group}, D_{2}
{}
//
//
//
void
MAC_bmle::BmleSubject::build_covariates_matrix()
{
  //
  std::map< int, std::list< float > >::const_iterator age_cov_it = age_covariates_.begin();
  covariates_.resize( age_covariates_.size() * (D_ + 1), (*age_cov_it).second.size() * (D_ + 1));
  //

  
  std::cout << covariates_ << std::endl;
}
