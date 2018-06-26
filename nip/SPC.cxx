#include "SPC.h"
//
// Algorithm 1bis: computation of a single factor PMD model adapted to PCA
// If X = Y^T Z, Y and Z are standardized matrices (column-wise)
// It is suggested to start the algorithm with V as the first SVD vector.
//
// Algo1 include algorithm 3 and 4
//
// Uprev is the set of orthogonal U_{k-1}
// Pu and Pv are the penalities associated, repectively, to U and V.
// 
// The algorithm output the the rank coefficient d_{k}
double
MAC_nip::NipSPC::single_factor( const Eigen::MatrixXd& X, const Eigen::MatrixXd& Uprev,
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
	Vl1Norm = V.lpNorm< 1 >(),
	delta_1 = 0., delta_2 = 0.;
      Eigen::MatrixXd
	v_old = Eigen::MatrixXd::Random( xc, 1 ),
	Xv    = X * V, XTu = X.transpose() * U;
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
		Xv = X * V;
		if ( Uprev.cols() > 0 )
		  {
		    //		U  = Xv;
		    //		U /= X.lpNorm< 2 >();
		    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(Uprev.rows(), Uprev.rows());
		    U  = ( I - Uprev * Uprev.transpose() ) * Xv;
		    if ( U.lpNorm< 2 >() != 0. )
		      U /= U.lpNorm< 2 >();
		  }
		else
		  {
		    U  = Xv;
		    if ( Xv.lpNorm< 2 >() != 0. )
		      U /= Xv.lpNorm< 2 >();
		  }
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
		if ( V.lpNorm< 2 >() != 0 )
		  V /= V.lpNorm< 2 >();
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
      // compute the rank coefficient
      // coefficient associated with the rank decomposition level of X (input)
      return (U.transpose() * X * V)(0,0);
    }
  catch( itk::ExceptionObject & err )
    {
      std::cerr << err << std::endl;
      return EXIT_FAILURE;
    }
}

//
// Algorithm 2: computation of K factor of SPC
// the input matrix X is at max rank K.
void
MAC_nip::NipSPC::K_factors( const Eigen::MatrixXd& X,
			    Spectra& Matrix_spetrum,
			    Penality Pu, Penality Pv,
			    bool Verbose = false )
{std::cout << "YOYOYOYOYOYOOYOYOYOYOYOYOYOYOYYYYYYYYYYYYYYYYYYY\n" ;
  try
    {
      
      //
      //
      std::size_t K = Matrix_spetrum.size();
std::cout << "je passe SPC " << std::endl;
      //
      // algorithm
      Eigen::MatrixXd XX = X;
      bool coefficient_too_small = false;
      for ( int k = 0 ; k < K ; k++ )
	{
	  if( !coefficient_too_small )
	    {
	      //
	      // Build the orthogonal spectrum Uprev
	      Eigen::MatrixXd Uprev;
	      if ( k > 0 )
		{
		  Uprev = Eigen::MatrixXd::Zero( std::get< Uk >( Matrix_spetrum[0] ).rows(), k );
		  for ( int i = 0 ; i < k ; i++ )
		    Uprev.col(i) = std::get< Uk >( Matrix_spetrum[i] );
		}
	      
	      //
	      // Sparse Principal Componant
	      double coefficient_to_rank =
		single_factor( XX, Uprev,
			       std::get< Uk >(Matrix_spetrum[k]), Pu,
			       std::get< Vk >(Matrix_spetrum[k]), Pv );
	      
	      //
	      //
	      if ( isnan(coefficient_to_rank) )
		{
		  coefficient_too_small = true;
		  std::get< coeff_k >( Matrix_spetrum[k] ) = 0.;
		}
	      else
		{
		  // Update the rest of the matrix
		  std::get< coeff_k >( Matrix_spetrum[k] ) = coefficient_to_rank;
		  XX -= std::get< coeff_k >( Matrix_spetrum[k] )
		    * std::get< Uk >(Matrix_spetrum[k])
		    * std::get< Vk >(Matrix_spetrum[k]).transpose();
		}
	    }
	  else
	    std::get< coeff_k >( Matrix_spetrum[k] ) = 0.;
	}
      //
      Eigen::MatrixXd Sol = Eigen::MatrixXd::Zero(X.rows(),X.cols());
      Eigen::MatrixXd D   = Eigen::MatrixXd::Zero(X.rows(),X.cols());
      Eigen::MatrixXd I   = Eigen::MatrixXd::Identity(X.rows(),X.cols());
      Eigen::MatrixXd P   = Eigen::MatrixXd::Zero(std::get< Uk >( Matrix_spetrum[0] ).rows(), K );
      Eigen::VectorXd diagonal = Eigen::VectorXd::Zero(X.rows());
      for ( int k = 0 ; k < K ; k++ )
	{
	  std::cout << "dk[" << k << "] = " << std::get< coeff_k >( Matrix_spetrum[k] ) << std::endl;
	  std::cout << "uk[" << k << "] = \n" << std::get< Uk >(Matrix_spetrum[k]) << std::endl;
	  std::cout << "vk[" << k << "] = \n" << std::get< Vk >(Matrix_spetrum[k]) << std::endl;
	  //
	  P.col(k)    = std::get< Uk >(Matrix_spetrum[k]);
	  diagonal(k) = std::get< coeff_k >( Matrix_spetrum[k] );
	  if ( k > 0 )
	    {
	      std::cout << "uk[" << 0 << "] . uk[" << k << "] = "
			<< std::get< Uk >(Matrix_spetrum[0]).transpose() * std::get< Uk >(Matrix_spetrum[k])
			<< std::endl;
	      std::cout << "vk[" << 0 << "] . vk[" << k << "] = "
			<< std::get< Vk >(Matrix_spetrum[0]).transpose() * std::get< Vk >(Matrix_spetrum[k])
			<< std::endl;
	    }
	}
      D = diagonal.asDiagonal();
      std::cout << "D:\n" << D << std::endl;
      std::cout << "P:\n" << P << std::endl;
      std::cout << "Sol:\n: " << P * D * P.inverse() << std::endl;
      //
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
