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
  kappa_ = kappa_ - H_.inverse() * nabla_;
}
