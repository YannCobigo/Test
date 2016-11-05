#include <algorithm>
#include <vector>
#include <list>
// Egen
#include <Eigen/Core>
#include <Eigen/Eigen>
#include <Eigen/Sparse>
//
// ITK
//
#include <itkImageFileReader.h>
#include <itkSpatialOrientationAdapter.h>
#include "itkChangeInformationImageFilter.h"
//
//
//
#include "Load_csv.h" 
#include "BmleException.h"
//
//
//
MAC_bmle::BmleLoadCSV::BmleLoadCSV( const std::string& CSV_file ):
  csv_file_{ CSV_file.c_str() }
  {
    try
      {
	//
	//
	float mean_age = 0.;
	std::string line;
	//skip the first line
	std::getline(csv_file_, line);
	//
	// then loop
	while( std::getline(csv_file_, line) )
	  {
	    std::stringstream  lineStream( line );
	    std::string        cell;
	    std::cout << "ligne: " << line << std::endl;

	    //
	    // Get the PIDN
	    std::getline(lineStream, cell, ',');
	    const int PIDN = std::stoi( cell );
	    // Get the group
	    std::getline(lineStream, cell, ',');
	    const int group = std::stoi( cell );
	    // Get the age
	    std::getline(lineStream, cell, ',');
	    int age = std::stoi( cell );
	    mean_age += static_cast< float >( age );
	    // Get the image
	    std::string image;
	    std::getline(lineStream, image, ',');
	    // Covariates
	    std::list< float > covariates;
	    while( std::getline(lineStream, cell, ',') )
	      covariates.push_back( std::stof(cell) );
	    num_covariates_ = covariates.size();

	    //
	    // check we have less than 10 groups
	    if( group > 10 )
	      throw BmleException( __FILE__, __LINE__,
				   "The CSV file should have less than 10 gourps.",
				   ITK_LOCATION );
	    // If the PIDN does not yet exist
	    if ( group_pind_[ group ].find( PIDN ) == group_pind_[ group ].end() )
	      {
		groups_.insert( group );
		group_pind_[ group ][PIDN] = BmleSubject< 3 /*D_r*/, 3 /*D_f*/ >( PIDN, group );
		group_num_subjects_[ group ]++;
		num_subjects_++;
	      }
	    //
	    group_pind_[ group ][ PIDN ].add_tp( age, covariates, image );
	    num_3D_images_++;
	  }

	// 
	// Design Matrix for every subject
	//

	//
	// mean age
	mean_age /= static_cast< float >( num_3D_images_ );
	std::cout << "mean age: " << mean_age << std::endl;
	//
	for ( auto g : groups_ )
	  for ( auto& s : group_pind_[g] )
	    s.second.build_design_matrices( mean_age );
	//
	// Create the 4D measurements image
	image_concat();
      }
    catch( itk::ExceptionObject & err )
      {
	std::cerr << err << std::endl;
	exit( -1 );
      }


    //
    //
    if (true)
      for ( auto g : groups_ )
	for ( auto s : group_pind_[g] )
	  {
	    s.second.print();
	    //s.second.build_covariates_matrix();
	  }
  }
//
//
//
void
MAC_bmle::BmleLoadCSV::image_concat()
{
  try
    {
      //
      // ITK image types
      using Image3DType = itk::Image< float, 3 >;
      using Reader3D    = itk::ImageFileReader< Image3DType >;
      // 
      using Iterator3D = itk::ImageRegionConstIterator< Image3DType >;
      using Iterator4D = itk::ImageRegionIterator< Image4DType >;

      //
      // Create the 4D image of measures
      //

      //
      // Set the measurment 4D image
      Y_ = Image4DType::New();
      //
      Image4DType::RegionType region;
      Image4DType::IndexType  start = { 0, 0, 0, 0 };
      //
      // Take the dimension of the first subject image:
      Reader3D::Pointer subject_image_reader_ptr =
	group_pind_[ (*groups_.begin()) ].begin()->second.get_age_images().begin()->second;
      //
      Image3DType::Pointer  raw_subject_image_ptr = subject_image_reader_ptr->GetOutput();
      Image3DType::SizeType size = raw_subject_image_ptr->GetLargestPossibleRegion().GetSize();
      Image4DType::SizeType size_4D{ size[0], size[1], size[2], num_3D_images_ };
      //
      region.SetSize( size_4D );
      region.SetIndex( start );
      //
      Y_->SetRegions( region );
      Y_->Allocate();
      //
      // ITK orientation, most likely does not match our orientation
      // We have to reset the orientation
      using FilterType = itk::ChangeInformationImageFilter< Image4DType >;
      FilterType::Pointer filter = FilterType::New();
      // Origin
      Image3DType::PointType orig_3d = raw_subject_image_ptr->GetOrigin();
      Image4DType::PointType origin;
      origin[0] = orig_3d[0]; origin[1] = orig_3d[1]; origin[2] = orig_3d[2]; origin[3] = 0.;
      // Spacing 
      Image3DType::SpacingType spacing_3d = raw_subject_image_ptr->GetSpacing();
      Image4DType::SpacingType spacing;
      spacing[0] = spacing_3d[0]; spacing[1] = spacing_3d[1]; spacing[2] = spacing_3d[2]; spacing[3] = 1.;
      // Direction
      Image3DType::DirectionType direction_3d = raw_subject_image_ptr->GetDirection();
      Image4DType::DirectionType direction;
      direction[0][0] = direction_3d[0][0]; direction[0][1] = direction_3d[0][1]; direction[0][2] = direction_3d[0][2]; 
      direction[1][0] = direction_3d[1][0]; direction[1][1] = direction_3d[1][1]; direction[1][2] = direction_3d[1][2]; 
      direction[2][0] = direction_3d[2][0]; direction[2][1] = direction_3d[2][1]; direction[2][2] = direction_3d[2][2];
      direction[3][3] = 1.; // 
      //
      filter->SetOutputSpacing( spacing );
      filter->ChangeSpacingOn();
      filter->SetOutputOrigin( origin );
      filter->ChangeOriginOn();
      filter->SetOutputDirection( direction );
      filter->ChangeDirectionOn();
      //
      //
      Iterator4D it4( Y_, Y_->GetBufferedRegion() );
      it4.GoToBegin();
      //
      for ( auto group : groups_ )
	for ( auto subject : group_pind_[group] )
	  for ( auto image : subject.second.get_age_images() )
	    {
	      std::cout << image.second->GetFileName() << std::endl;
	      Image3DType::RegionType region = image.second->GetOutput()->GetBufferedRegion();
	      Iterator3D it3( image.second->GetOutput(), region );
	      it3.GoToBegin();
	      while( !it3.IsAtEnd() )
		{
		  it4.Set( it3.Get() );
		  ++it3; ++it4;
		}
	    }

//      //
//      // Writer
//      filter->SetInput( Y_ );
//      itk::NiftiImageIO::Pointer nifti_io = itk::NiftiImageIO::New();
//      //
//      itk::ImageFileWriter< Image4DType >::Pointer writer = itk::ImageFileWriter< Image4DType >::New();
//      writer->SetFileName( "measures_4D.nii.gz" );
//      writer->SetInput( filter->GetOutput() );
//      writer->SetImageIO( nifti_io );
//      writer->Update();

    }
  catch( itk::ExceptionObject & err )
    {
      std::cerr << err << std::endl;
      exit( -1 );
    }
}
//
//
//
void
MAC_bmle::BmleLoadCSV::build_groups_design_matrices()
{
  try
    {
      //Eigen::SparseMatrix< float > X1( num_3D_images_, num_subjects_ * 3 /*D_r*/ + 3 /*D_f*/);
      int
	X1_lines = num_3D_images_,
	X1_cols  = num_subjects_ * 3 /*D_r*/ + groups_.size() * 3 /*D_f*/,
	X2_lines = groups_.size() * ( num_subjects_ * 3 /*D_r*/ + 3 /*D_f*/ ),
	X2_cols  = groups_.size() * ( 3 /*D_r*/ * (num_covariates_ + 1) + 3 /*D_f*/ );
      Eigen::MatrixXf
	X1( X1_lines, X1_cols ),
	X2( X1_lines, X1_cols );
      X1 = Eigen::MatrixXf::Zero( X1_lines, X1_cols );
      X2 = Eigen::MatrixXf::Zero( X2_lines, X2_cols );
      //
      int line_x1 = 0, col_x1 = 0;
      int line_x2 = 0, col_x2 = 0;
      int current_gr = ( *groups_.begin() ), increme_dist_x1 = 0, increme_dist_x2 = 0;
      for ( auto g : groups_ )
	for ( auto subject : group_pind_[g] )
	  {
	    //
	    // we change group
	    if ( current_gr != g )
	      {
		increme_dist_x1 += group_num_subjects_[g] * 3 /*D_r*/ + 3 /*D_f*/;
		increme_dist_x2 +=  3 /*D_r*/ * (num_covariates_ + 1) + 3 /*D_f*/;
		col_x1          += 3 /*D_f*/;
		current_gr       = g;
	      }
	    //
	    // X1 design
	    int
	      sub_line_x1 = subject.second.get_random_matrix().rows(),
	      sub_col_x1  = subject.second.get_random_matrix().cols();
	    X1.block( line_x1, col_x1, sub_line_x1, sub_col_x1 ) = subject.second.get_random_matrix();
	    X1.block( line_x1, increme_dist_x1 + group_num_subjects_[g]  * 3 /*D_r*/,
		      sub_line_x1, sub_col_x1 ) = subject.second.get_fixed_matrix();
	    //
	    line_x1 += sub_line_x1;
	    col_x1  += sub_col_x1;
	    //
	    // X2 design
	    int
	      sub_line_x2 = subject.second.get_X2_matrix().rows(),
	      sub_col_x2  = subject.second.get_X2_matrix().cols();
	    X2.block( line_x2, increme_dist_x2, sub_line_x2, sub_col_x2 ) = subject.second.get_X2_matrix();
	    X2.block( line_x2 + 3 /*D_f*/, increme_dist_x2 + 3 /*D_f*/,
	    	      3 /*D_f*/, 3 /*D_f*/ ) = Eigen::MatrixXf::Identity( 3 /*D_f*/, 3 /*D_f*/ );
	    ////
	    line_x2 += sub_line_x2 + 3 /*D_f*/;
	    ////col_x2  += sub_col_x2;
	  }
      std::cout << X1 << std::endl;
      std::cout << X2 << std::endl;
    }
  catch( itk::ExceptionObject & err )
    {
      std::cerr << err << std::endl;
      exit( -1 );
    }
}
