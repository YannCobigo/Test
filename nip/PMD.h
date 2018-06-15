#ifndef NIPPMD_H
#define NIPPMD_H
//
//
//
#include "NipException.h"
#include "PMA.h"
#include "PMA_tools.h"
//
//
//
namespace MAC_nip
{
  /** \class NipPMD
   *
   * \brief PMD: Penalized Matrices Decomposition
   * "A penalized matrix decomposition, with applications to sparse principal components and canonical correlation analysis" (PMID:19377034)
   * 
   */
  class NipPMD : public NipPMA
  {
  public:
    /*Constructor*/
    NipPMD(){};
    /*Destructor*/
    virtual ~NipPMD(){};

    //
    //
    virtual double single_factor( const Eigen::MatrixXd&, const Eigen::MatrixXd&,
				  Eigen::MatrixXd&, const Penality, 
				  Eigen::MatrixXd&, const Penality,
				  const int );
    //
    virtual void K_factors( const Eigen::MatrixXd&,
			    Spectra&,
			    Penality, Penality );
    //
    virtual void K_factors_trained( const Eigen::MatrixXd&,
				    const Spectra&,
				    Penality, Penality );
    //
    virtual void set_cs( double C1, double C2 )
    {c1_ = C1; c2_ = C2;};

    //
    double c1_{0.};
    double c2_{0.};
  };
}
#endif
