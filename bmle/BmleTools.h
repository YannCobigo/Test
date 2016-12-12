#ifndef BMLETOOLS_H
#define BMLETOOLS_H
//
//
//
namespace MAC_bmle
{
  int power( double val, unsigned n )
  {
    return ( n == 0 ? 1 : val * power( val, n-1 ) );
  }
  
  template< int n >
    struct TMP_power
    {
      enum{value *= TMP_power<n-1>::value};
    };
  template<  >
    struct TMP_power<0>
    {
      enum{value = 1};
    };

  
}
#endif
