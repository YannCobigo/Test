#include "PMD.h"
//
// Algorithm 1: computation of a single factor PMD model
// If X = Y^T Z, Y and Z are standardized matrices (column-wise)
// It is suggested to start the algorithm with V as the first SVD vector.
//
// Algo1 include algorithm 3 and 4
//
// Pu and Pv are the penalities associated, repectively, to U and V.
// 
// The algorithm output the the rank coefficient d_{k}
double
MAC_nip::NipPMD::single_factor( const Eigen::MatrixXd& X, const Eigen::MatrixXd& UPrev,
				Eigen::MatrixXd& U, const Penality Pu, 
				Eigen::MatrixXd& V, const Penality Pv,
				const int Niter = 1000 )
{
  try
    {
      //
      // Tests
      if ( U.rows() != X.rows() || V.rows() != X.cols() )
	throw MAC_nip::NipException( __FILE__, __LINE__,
				     "Check the dimentions of the matrices. U must have the same number of rows as X and V the same number of columns.",
				     ITK_LOCATION );

      //
      // initialization
      int
	xl = X.rows(), xc = X.cols();
      //
      if ( c1_ == 0 || c2_ == 0 )
	{
	  c1_ = sqrt( static_cast< double >(xl) );
	  c2_ = sqrt( static_cast< double >(xc) );
	}
      double
	Ul1Norm = U.lpNorm< 1 >(),
	Vl1Norm = V.lpNorm< 1 >(),
	delta_1 = 0., delta_2 = 0.;
      Eigen::MatrixXd
	v_old = Eigen::MatrixXd::Random( xc, 1 ),
	Xv, XTu;
      //std::cout << "v_old \n" << v_old << std::endl;
  
      //
      switch ( Pv )
	{
	case L1:
	  {
	    // Algorithm 3: computtion of a single factor PMD(L1,L1) model
	    // ToDo: make sure v is L2-norm 1.
	    int count = 0;
	    while ( ( V - v_old ).lpNorm< 1 >() > 1.e-06 && ++count < Niter )
	      {
		//
		v_old = V;
		//
		// Update v
		Xv      = X * V;
		Ul1Norm = U.lpNorm< 1 >();
		//
		if ( Ul1Norm > c1_ )
		  delta_1 = NipPMA_tools::dichotomy_search( Xv, Ul1Norm, c1_ );
		else
		  delta_1 = 0.;
		//std::cout << "delta_1 " << delta_1 << std::endl;
		//
		for ( int l = 0 ; l < xl ; l++ )
		  U(l,0) = NipPMA_tools::soft_threshold( Xv(l,0), delta_1 );
		U /= U.lpNorm< 2 >();
		//
		// Update u
		XTu     = X.transpose() * U;
		Vl1Norm = V.lpNorm< 1 >();
		//
		if ( Vl1Norm > c2_ )
		  delta_2 = NipPMA_tools::dichotomy_search( XTu, Vl1Norm, c2_ );
		else
		  delta_2 = 0.;
		//std::cout << "delta_2 " << delta_2 << std::endl;
		//
		for ( int c = 0 ; c < xc ; c++ )
		  V(c,0) = NipPMA_tools::soft_threshold( XTu(c,0), delta_2 );
		V /= V.lpNorm< 2 >();
		//  std::cout << "v_old \n" << v_old << std::endl;
		// std::cout << "V \n" << V <<  " " << V.lpNorm< 2 >() << std::endl;
		//  std::cout << "V-vold " << ( V - v_old ).lpNorm< 1 >()  << std::endl;
	      }
	    //
	    break;
	  }
	case FUSION:
	  // Algorithm 4: computtion of a single factor PMD(L1,L1) model
	  // ToDo: make sure v is L2-norm 1.
	default:
	  {
	    throw MAC_nip::NipException( __FILE__, __LINE__,
					 "Type of penality not yet implemented.",
					 ITK_LOCATION );
	  }
	}

      //
      // output
      // coefficient associated with the rank decomposition level of X (input)
      // compute the rank coefficient
      return (U.transpose() * X * V)(0,0);
    }
  catch( itk::ExceptionObject & err )
    {
      std::cerr << err << std::endl;
      return EXIT_FAILURE;
    }
}

//
// Algorithm 2: computation of K factor of PMD
// the input matrix X is at max rank K.
void
MAC_nip::NipPMD::K_factors( const Eigen::MatrixXd& X,
			    Spectra& Matrix_spetrum,
			    Penality Pu, Penality Pv,
			    bool Verbose = false )
{
  try
    {
      //
      //
      std::size_t K = Matrix_spetrum.size();
 
      //
      // algorithm
      Eigen::MatrixXd XX = X;
      for ( int k = 0 ; k < K ; k++ )
	{
	  std::get< coeff_k >( Matrix_spetrum[k] ) =
	    single_factor( XX, Eigen::MatrixXd::Zero(0,0),
			   std::get< Uk >(Matrix_spetrum[k]), Pu,
			   std::get< Vk >(Matrix_spetrum[k]), Pv );

	  //
	  // Update the rest of the matrix
	  XX -= std::get< coeff_k >( Matrix_spetrum[k] )
	    * std::get< Uk >(Matrix_spetrum[k])
	    * std::get< Vk >(Matrix_spetrum[k]).transpose();
	  //std:: cout << XX << std::endl;
	}
      //
      Eigen::MatrixXd Sol = Eigen::MatrixXd::Zero(X.rows(),X.cols());
      if ( Verbose )
	for ( int k = 0 ; k < K ; k++ )
	  {
	    std::cout << "dk[" << k << "] = " << std::get< coeff_k >( Matrix_spetrum[k] ) << std::endl;
	    std::cout << "uk[" << k << "] = \n" << std::get< Uk >(Matrix_spetrum[k]) << std::endl;
	    std::cout << "vk[" << k << "] = \n" << std::get< Vk >(Matrix_spetrum[k]) << std::endl;
	    Sol += std::get< coeff_k >( Matrix_spetrum[k] )
	      * std::get< Uk >(Matrix_spetrum[k])
	      * std::get< Vk >(Matrix_spetrum[k]).transpose();
	    std::cout << "Sol: \n" << Sol << std::endl;
	  }
    }
  catch( itk::ExceptionObject & err )
    {
      std::cerr << err << std::endl;
      exit( -1 );
    }
}
