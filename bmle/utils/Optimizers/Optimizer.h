#ifndef OPTIMIZER_H
#define OPTIMIZER_H
//
#include <limits>       // std::numeric_limits
//
//
// Eigen
#include <Eigen/Core>
#include <Eigen/Eigen>
#include <Eigen/Sparse>

//
//
//
namespace NeuroBayes
{
  class Optimizer
  {
  public:
    virtual void update() = 0;
  };
}
#endif
