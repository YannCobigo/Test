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
  learning_rate_   *= 0.5;
//  if ( learning_rate_ < 1.e-05 )
//    learning_rate_ = 1.;
  //
  Eigen::MatrixXd d = H_.inverse() * nabla_;
  kappa_           += learning_rate_ * d;
}
