#ifndef MAXIMUM_LIKELIHOOD_H
#define MAXIMUM_LIKELIHOOD_H
//
#include "Optimizer.h"

//
//
//
namespace NeuroBayes
{
  /** \class Maximum_likelihood
   *
   * \brief Newton-Maximum_likelihood algorithm
   * 
   */
  template< class Algo, int DimY >
  class Maximum_likelihood : public Optimizer
  {
  public:
    /** Constructor. */
    Maximum_likelihood();
    
    //
    //
    virtual void update(){};

  private:
    // Algorithm
    Algo algo_;
    // covariances
    Eigen::MatrixXd sigma_;
    // covariances
    Eigen::MatrixXd Sigma_;
  };

  //
  //
  template< class Algo, int DimY >
  Maximum_likelihood<Algo,DimY>::Maximum_likelihood()
    {
      // Random symmetric positive definit 
      Eigen::MatrixXd X = Eigen::MatrixXd::Random( DimY, DimY );
      sigma_            = X * X.transpose();
    };
}
#endif
