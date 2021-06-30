#ifndef CLASSIFIER_DEPP_ML_PERCEPTRON_H
#define CLASSIFIER_DEPP_ML_PERCEPTRON_H
#include <random>
#include <math.h> 
#include <tuple> 
#include <limits>
//
//
//
//
// DLib
//
//#include <dlib/svm.h>
//
//
//
/*! \namespace Classification
 *
 * Name space for classification package.
 *
 */
namespace MAC_classification
{
  //
  //
  //
  /*! \class Classifier_deep_ml_perceptron
   *
   *
   */
  template< typename Classifier_type >
    class Classifier_deep_ml_perceptron
    {
    public:
      /*!
       * \brief Default Constructor
       *
       * Constructor of the class Classifier_deep_ml_perceptron
       *
       */
      Classifier_deep_ml_perceptron(){};
      /*!
       * \brief Default Constructor
       *
       * Constructor of the class Classifier_deep_ml_perceptron
       *
       */
      Classifier_deep_ml_perceptron(const char* );
      /*!
       * \brief Default ConstructorClassifier_deep_ml_perceptron
       *
       * Constructor of the class 
       *
       */
      Classifier_deep_ml_perceptron( const int, const sample_type&  );
      /*!
       * \brief feed-forward
       *
       * predict the results with a trained set of weights.
       *
       */
      const std::tuple< std::vector< double >, std::vector< double > > feed_forward();
      /*!
       * \brief feed-backward
       *
       * Backward propagation of the error cost function.
       *
       */
      const std::list< double > feed_backward( const sample_type&, const int );
      /*!
       * \brief group to vector
       *
       * Change the input class value into a vector.
       *
       */
      inline const int  weights_size(){return weights_.size();};
      /*!
       * \brief 
       *
       * Cost function
       *
       */
      inline const double  cost_function() const {return E_i_;};
      /*!
       * \brief group to vector
       *
       * Change the input class value into a vector.
       *
       */
      std::vector< double > group_to_vector( const int );
      /*!
       * \brief update_weights
       *
       * .
       *
       */
      void update_weights( const std::list<double>&  );
      /*!
       * \brief delta E
       *
       * .
       *
       */
      double delta_E() const;
      /*!
       * \brief Write weights vectors
       *
       * This function dump the neural network weights
       *
       */
      void write_weights( const char* ) const;
      /*!
       * \brief Read weights vectors
       *
       * This function read the neural network weights
       *
       */
      void read_weights( const char* );
      /*!
       * \brief Load image
       *
       * This function read the neural network weights
       *
       */
      void load_image( const sample_type& );

    private:
      // Number of hidden layers
      int number_layers_;
      // ToDo: remove Number of hidden units
      int neuron_per_layer_;
      // Number of output class
      int Number_of_classes_;
      // Entries
      std::vector< double > X_;
      // Entries
      std::vector< std::vector< double > > Z_;
      // Weights of the neural network
      std::list< std::tuple<
	double, /* weight */
	int,    /* upper layer */
	int,    /* lower layer */
	int     /* level */
	> > weights_;
      // neurons per layers
      std::vector< long int > neurons_;
      // Sizes of the layers
      // {x_i} i = 0 .. N
      int N_;
      // {z_i} i = 0 .. D
      int D_;
      // {y_i} i = 1 .. K
      int K_;
      // learning rate
      double eta_;
      // Regulation constante
      double lambda_;
      // Cost function
      double E_i_;
    };
  //
  //
  //
  template< typename Classifier_type >
    Classifier_deep_ml_perceptron< Classifier_type >::Classifier_deep_ml_perceptron( const int Number_of_output_class,
										     const sample_type& Sample_normalised ): 
  K_( Number_of_output_class ), number_layers_( 4 ), neuron_per_layer_( 100 ), eta_( 1.e-08 ), lambda_( 0. ), E_i_( std::numeric_limits< double >::max() )
    {
      try
	{
	  //
	  // Create the first set of random weights
	  std::default_random_engine generator;
	  std::uniform_real_distribution<double> distribution(-1.,1.);

	  // ToDo: check size of layers_ == neurons_;
	  neurons_ = { 6, 2, 3, 4, Sample_normalised.nr() }; // later we add the bias
	  Z_.resize( neurons_.size() );
	  //
	  int dimension = 0;
	  for ( int layer = 0 ; layer < neurons_.size() - 1 ; layer++ )
	    {
	      dimension  += ( neurons_[layer] + 1) * ( neurons_[layer+1] + 1 );
	      Z_[layer].resize( neurons_[layer] + (layer == 0 ? 0 : 1) );
	      //
	      //
	      for ( int i = 0 ;  i < neurons_[layer] + (layer == 0 ? 0 : 1 ); i++ )
		for ( int j = 0 ; j < neurons_[layer+1] + 1 ; j++ )
		  weights_.push_back( std::tuple<double, int, int,int>( distribution(generator),
									i, j, layer) );
	    }
	  dimension--; // The layer k does not have bias term
	  //
	  // Copy of the inputs in the last layer
	  Z_[ neurons_.size() - 1 ][0] = 1.; // bias term
	  for ( long r = 1 ; r < Sample_normalised.nr() + 1 ; ++r )
	    Z_[ neurons_.size() - 1 ][r] = Sample_normalised( r-1, 0 );
	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	  abort();
	}
    }
  //
  //
  //
  template< typename Classifier_type >
    Classifier_deep_ml_perceptron< Classifier_type >::Classifier_deep_ml_perceptron( const char* Decision_function ): 
  K_( 0 ), number_layers_( 1 ), neuron_per_layer_( 100 ), eta_( 0. ), lambda_( 0.0 ), E_i_( std::numeric_limits< double >::max() )
    {
      try
	{
	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	  abort();
	}
    }
  //
  //
  //
  template< typename Classifier_type > const std::tuple< std::vector< double >, std::vector< double > >
    Classifier_deep_ml_perceptron< Classifier_type >::feed_forward()
    {
      try
	{
	  //
	  int current_i = neurons_[ neurons_.size() - 1 ] -1 +1;
	  double a      = 0.;
	  // First weights are last layer
	  for ( std::list< std::tuple<double, int, int, int > >::reverse_iterator weight = weights_.rbegin() ;
		weight != weights_.rend() ; ++weight )
	    {
	      //
	      //
	      double W = std::get< 0 /*weight*/ >( *weight );
	      int
		I     = std::get< 1 /*i*/ >(*weight),
		J     = std::get< 2 /*j*/ >(*weight),
		layer = std::get< 3 /*layer*/ >(*weight);
	      //
	      if ( current_i == I )
		{
		  a += W * Z_[layer][J];
		}
	      else
		{
		  Z_[layer-1][current_i] = ( layer-1 == 0 ? tanh(a) : exp(a) );
		  a         = 0.;
		  current_i = I;
		  weight--;
		}
	    }

	  //
	  // ToDo: normalize level 0 (k): Z_[0][:]
	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	  abort();
	}
    }
  //
  //
  //
  template< typename Classifier_type > const std::list< double >
    Classifier_deep_ml_perceptron< Classifier_type >::feed_backward( const sample_type& Sample_normalised,
								     const int Image_class )
    {
      try
	{
	  //
	  // Gradient of the loss function
	  std::list< double > grad_E_n;
	  // we skip the level k
	  std::vector< std::vector< double > > sum_functions( neurons_.size() - 1 ); 
	  for ( int layer = 1 ; layer < neurons_.size() ; layer++)
	    sum_functions[ layer - 1 ].resize( neurons_[layer] + 1 );
	  //
	  for( auto weight : weights_ )
	    {
	      //
	      //
	      double W = std::get< 0 /*weight*/ >(weight);
	      int
		I     = std::get< 1 /*i*/ >(weight),
		J     = std::get< 2 /*i*/ >(weight),
		layer = std::get< 3 /*layer*/ >(weight);
	      //
	      switch ( layer )
		{
		case 0:
		  {
		    grad_E_n.push_back( 1.1 );
		    // Sum functions of the next layer
		    sum_functions[layer][J] += 2.2;
		    break;
		  }
		case 1:
		  {
		    // gradient of the layer
		    double grad_e = 0.;
		    for ( auto neurons_pair : sum_functions[layer-1] )
		      grad_e += neurons_pair * 3.3;
		    grad_E_n.push_back( grad_e );
		    // Sum functions of the next layer
		    sum_functions[layer][J] += sum_functions[layer-1][I] * 4.4;
		    break;
		  }
		case 2:
		  {
		    // gradient of the layer
		    double grad_e = 0.;
		    for ( auto neurons_pair : sum_functions[layer-1] )
		      grad_e += neurons_pair * 3.3;
		    grad_E_n.push_back( grad_e );
		    // Sum functions of the next layer
		    sum_functions[layer][J] += sum_functions[layer-1][I] * 4.4;
		    break;
		  }
		case 3:
		  {
		    // gradient of the layer
		    double grad_e = 0.;
		    for ( auto neurons_pair : sum_functions[layer-1] )
		      grad_e += neurons_pair * 3.3;
		    grad_E_n.push_back( grad_e );
		    // Sum functions of the next layer
		    sum_functions[layer][J] += sum_functions[layer-1][I] * 4.4;
		    break;
		  }
		default:
		  {
		    // ToDo MacErrorHandler
		    std::cerr << "Number of layers must be less than X." << std::endl;
		  }
		}
	    }
	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	  abort();
	}
    }
  //
  //
  //
  template< typename Classifier_type > std::vector< double >
    Classifier_deep_ml_perceptron< Classifier_type >::group_to_vector( const int Image_class )
    {
      std::vector< double > t_k( K_, 0 );
      t_k[Image_class] = 1;
      //
      return t_k;
    }
  //
  //
  //
  template< typename Classifier_type > void
    Classifier_deep_ml_perceptron< Classifier_type >::update_weights( const std::list<double>& Gradient )
    {
      try
	{
	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	  abort();
	}
    }
  //
  //
  //
  template< typename Classifier_type > double
    Classifier_deep_ml_perceptron< Classifier_type >::delta_E() const
    {
      try
	{
	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	  abort();
	}
    }
  //
  //
  //
  template< typename Classifier_type > void
    Classifier_deep_ml_perceptron< Classifier_type >::write_weights( const char* Weights_file ) const
    {
      try
	{
	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	  abort();
	}
    }
  //
  //
  //
  template< typename Classifier_type > void
    Classifier_deep_ml_perceptron< Classifier_type >::read_weights( const char* Weights_file )
    {
      try
	{
	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	  abort();
	}
    }
  //
  //
  //
  template< typename Classifier_type > void
    Classifier_deep_ml_perceptron< Classifier_type >::load_image( const sample_type& Sample_normalised )
    {
      try
	{ 
	}
      catch( itk::ExceptionObject & err )
	{
	  std::cerr << err << std::endl;
	  abort();
	}
    }
}
#endif
