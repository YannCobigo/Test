#include "Raphson.h"

//
//
//
void 
NeuroBayes::Raphson::set_matrices( const Eigen::MatrixXd& Kappa, 
				   const Eigen::MatrixXd& Nabla, 
				   const Eigen::MatrixXd& Hessian )
{ 
  kappa_ = Kappa; 
  nabla_ = Nabla; 
  H_     = Hessian; 
};
//
//
//
void
NeuroBayes::Raphson::update()
{
  learning_rate_   /= 2.;
  Eigen::MatrixXd d = H_.inverse() * nabla_;
  kappa_           -= learning_rate_ * d;
}
