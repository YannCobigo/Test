#ifndef ALGORITHM_H
#define ALGORITHM_H
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
  class Algorithm
  {
  public:
    virtual void update() = 0;
  };
}
#endif
