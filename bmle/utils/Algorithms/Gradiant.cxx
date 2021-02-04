#include "Gradiant.h"
#include <iostream>

//
//
//
void 
NeuroBayes::Gradiant::set_matrices( const Eigen::MatrixXd& Kappa, 
				    const Eigen::MatrixXd& Nabla, 
				    const Eigen::MatrixXd& Hessian )
{ 
  kappa_ = Kappa; 
  nabla_ = Nabla; 
};
//
//
//
void
NeuroBayes::Gradiant::update()
{
  learning_rate_   *= 0.5;
  Eigen::MatrixXd d = learning_rate_ * nabla_;
  kappa_           -= d;
}
