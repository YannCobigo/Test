#ifndef RAPHSON_H
#define RAPHSON_H
//
#include "Algorithm.h"

//
//
//
namespace NeuroBayes
{
  /** \class Raphson
   *
   * \brief Newton-Raphson algorithm
   * 
   */
  class Raphson : public Algorithm
  {
  public:
    /** Constructor. */
    Raphson(){};
    
    //
    //
    virtual void update();

    //
    // Setters
    void set_matrices( const Eigen::MatrixXd&, const Eigen::MatrixXd&, const Eigen::MatrixXd& );
    // Getters
    const Eigen::MatrixXd& get_parameters() const {return kappa_;}
    //
    //
  private:
    //
    double learning_rate_{1.e-02};

    // Parameters
    Eigen::MatrixXd kappa_;
    // Gradiant
    Eigen::MatrixXd nabla_;
    // Hessian
    Eigen::MatrixXd H_;
  };
}
#endif
