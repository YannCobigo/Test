#ifndef GRADIANT_H
#define GRADIANT_H
//
#include "Algorithm.h"

//
//
//
namespace NeuroBayes
{
  /** \class Gradiant
   *
   * \brief Newton-Gradiant algorithm
   * 
   */
  class Gradiant : public Algorithm
  {
  public:
    /** Constructor. */
    Gradiant()
      {learning_rate_ = learning_rate_orig_;};
    
    //
    //
    virtual void         update();
    virtual const double get_learning_rate() const {return learning_rate_;};

    //
    // Setters
    void set_matrices( const Eigen::MatrixXd&, const Eigen::MatrixXd&, const Eigen::MatrixXd& );
    // Getters
    const Eigen::MatrixXd& get_parameters() const {return kappa_;}
    //
    void reset(){learning_rate_ = learning_rate_orig_;};
    //
    //
  private:
    //
    double learning_rate_{0.};
    double learning_rate_orig_{1.e-02};

    // Parameters
    Eigen::MatrixXd kappa_;
    // Gradiant
    Eigen::MatrixXd nabla_;
  };
}
#endif
