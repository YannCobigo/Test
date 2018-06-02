#ifndef NIP_SPC_CROSS_VALIDATION_H
#define NIP_SPC_CROSS_VALIDATION_H
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
  /** \class Nip_SPC_cross_validation
   *
   * \brief SPC: Sparce Principal Component
   * "A penalized matrix decomposition, with applications to sparse principal components and canonical correlation analysis" (PMID:19377034)
   * 
   */
  template< typename Matrix, int K >
    class Nip_SPC_cross_validation : public NipPMA
    {
    public:
      /*Constructor*/
      Nip_SPC_cross_validation();
      /*Destructor*/
      virtual ~Nip_SPC_cross_validation(){};
      
      //
      //
      virtual void validation();
    };

  //
  //
  template< typename Matrix, int K >
    Nip_SPC_cross_validation<Matrix,K>::Nip_SPC_cross_validation()
    {}
  //
  //
  template< typename Matrix, int K > void
    Nip_SPC_cross_validation<Matrix,K>::validation()
    {}
}
#endif
