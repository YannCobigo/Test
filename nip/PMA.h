#ifndef NIPPMA_H
#define NIPPMA_H
//
//
//
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <math.h>  
#include <map>
#include <list>
// Egen
#include <Eigen/Core>
#include <Eigen/Eigen>
using Spectra = std::vector< std::tuple< double, Eigen::MatrixXd, Eigen::MatrixXd> >;
//
//
//
#include "NipException.h"
// Penalization
enum Penality { L1, L2, FUSION };
enum Spectrum {coeff_k = 0, Uk = 1, Vk = 2};

//
//
//
namespace MAC_nip
{
  /** \class NipPMA
   *
   * \brief PMA: Penalized Matrices Analysise
   * "A penalized matrix decomposition, with applications to sparse principal components and canonical correlation analysis" (PMID:19377034)
   * 
   */
  class NipPMA
  {
  public:
    //
    //
    virtual double single_factor( const Eigen::MatrixXd&, const Eigen::MatrixXd&,
				  Eigen::MatrixXd, const Penality, 
				  Eigen::MatrixXd, const Penality,
				  const int ) = 0;
    //
    virtual void K_factors( const Eigen::MatrixXd&,
				Spectra&, Penality, Penality ) = 0;
  };
}
#endif
