//#include "QuickView.h"
#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <fstream>
// Eigen
#include <Eigen/Core>
#include <Eigen/Eigen>
//
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIterator.h>
#include <itkNiftiImageIO.h>
#include <itkOrientImageFilter.h>
#include <itkSpatialOrientation.h>

//#include "EM.h"
#include "VBGM.h"
#include "MakeITKImage.h"

itk::ImageIOBase::Pointer getImageIO(std::string input){
  itk::ImageIOBase::Pointer imageIO = itk::ImageIOFactory::CreateImageIO(input.c_str(), itk::ImageIOFactory::ReadMode);

  imageIO->SetFileName(input);
  imageIO->ReadImageInformation();

  return imageIO;
}

itk::ImageIOBase::IOComponentType component_type(itk::ImageIOBase::Pointer imageIO){
  return imageIO->GetComponentType();
}

itk::ImageIOBase::IOPixelType pixel_type(itk::ImageIOBase::Pointer imageIO){
  return imageIO->GetPixelType();
}

size_t num_dimensions(itk::ImageIOBase::Pointer imageIO){
  return imageIO->GetNumberOfDimensions();
}

int main(int argc, char const *argv[]){
//  if( argc < 3 )
//    {
//      std::cerr << "Long_extraction requires: \"\" " << std::endl;
//      return EXIT_FAILURE;
//    }

  //
  // input images
  //
  // 10072
 int N = 6+7;
 itk::ImageIOBase::Pointer CM[N];
 CM[0] = getImageIO( "/mnt/tempo/ana_res-2018-07-09_SPM/T1/Change_maps/wch_c1_map_10072_2009-11-06_to_2011-10-06.nii" );
 CM[1] = getImageIO( "/mnt/tempo/ana_res-2018-07-09_SPM/T1/Change_maps/wch_c1_map_10072_2011-10-06_to_2012-10-12.nii" );
 CM[2] = getImageIO( "/mnt/tempo/ana_res-2018-07-09_SPM/T1/Change_maps/wch_c1_map_10072_2012-10-12_to_2013-10-16.nii" );
 CM[3] = getImageIO( "/mnt/tempo/ana_res-2018-07-09_SPM/T1/Change_maps/wch_c1_map_10072_2013-10-16_to_2014-09-30.nii" );
 CM[4] = getImageIO( "/mnt/tempo/ana_res-2018-07-09_SPM/T1/Change_maps/wch_c1_map_10072_2014-09-30_to_2015-09-17.nii" );
 CM[5] = getImageIO( "/mnt/tempo/ana_res-2018-07-09_SPM/T1/Change_maps/wch_c1_map_10072_2015-09-17_to_2016-10-20.nii" );
 CM[6] = getImageIO( "/mnt/tempo/ana_res-2018-07-09_SPM/T1/wc1avg_T1_10072_2009-11-06_xj_2009-11-06_TPM.nii.gz" );
 CM[7] = getImageIO( "/mnt/tempo/ana_res-2018-07-09_SPM/T1/wc1avg_T1_10072_2009-11-06_xj_2011-10-06_TPM.nii.gz" );
 CM[8] = getImageIO( "/mnt/tempo/ana_res-2018-07-09_SPM/T1/wc1avg_T1_10072_2009-11-06_xj_2012-10-12_TPM.nii.gz" );
 CM[9] = getImageIO( "/mnt/tempo/ana_res-2018-07-09_SPM/T1/wc1avg_T1_10072_2009-11-06_xj_2013-10-16_TPM.nii.gz" );
 CM[10] = getImageIO( "/mnt/tempo/ana_res-2018-07-09_SPM/T1/wc1avg_T1_10072_2009-11-06_xj_2014-09-30_TPM.nii.gz" );
 CM[11] = getImageIO( "/mnt/tempo/ana_res-2018-07-09_SPM/T1/wc1avg_T1_10072_2009-11-06_xj_2015-09-17_TPM.nii.gz" );
 CM[12] = getImageIO( "/mnt/tempo/ana_res-2018-07-09_SPM/T1/wc1avg_T1_10072_2009-11-06_xj_2016-10-20_TPM.nii.gz" );
// 9983  //
// 9983  // 9983
// 9983  int N = 4+5;
// 9983  itk::ImageIOBase::Pointer CM[N];
// 9983  CM[0] = getImageIO( "/mnt/production/study/LEFFTDs_long/anares/ana_res-2018-07-09_SPM/T1/Change_maps/wch_c1_map_9983_2009-10-20_to_2013-04-15.nii" );
// 9983  CM[1] = getImageIO( "/mnt/production/study/LEFFTDs_long/anares/ana_res-2018-07-09_SPM/T1/Change_maps/wch_c1_map_9983_2013-04-15_to_2015-04-28.nii" );
// 9983  CM[2] = getImageIO( "/mnt/production/study/LEFFTDs_long/anares/ana_res-2018-07-09_SPM/T1/Change_maps/wch_c1_map_9983_2015-04-28_to_2016-09-27.nii" );
// 9983  CM[3] = getImageIO( "/mnt/production/study/LEFFTDs_long/anares/ana_res-2018-07-09_SPM/T1/Change_maps/wch_c1_map_9983_2016-09-27_to_2017-06-23.nii" );
// 9983  CM[4] = getImageIO( "/mnt/production/study/LEFFTDs_long/anares/ana_res-2018-07-09_SPM/T1/wc1avg_T1_9983_2009-10-20_xj_2009-10-20_TPM.nii.gz" );
// 9983  CM[5] = getImageIO( "/mnt/production/study/LEFFTDs_long/anares/ana_res-2018-07-09_SPM/T1/wc1avg_T1_9983_2009-10-20_xj_2013-04-15_TPM.nii.gz" );
// 9983  CM[6] = getImageIO( "/mnt/production/study/LEFFTDs_long/anares/ana_res-2018-07-09_SPM/T1/wc1avg_T1_9983_2009-10-20_xj_2015-04-28_TPM.nii.gz" );
// 9983  CM[7] = getImageIO( "/mnt/production/study/LEFFTDs_long/anares/ana_res-2018-07-09_SPM/T1/wc1avg_T1_9983_2009-10-20_xj_2016-09-27_TPM.nii.gz" );
// 9983  CM[8] = getImageIO( "/mnt/production/study/LEFFTDs_long/anares/ana_res-2018-07-09_SPM/T1/wc1avg_T1_9983_2009-10-20_xj_2017-06-23_TPM.nii.gz" );
//  //
//  // 5888
//  int N = 6+7;
//  itk::ImageIOBase::Pointer CM[N];
//  CM[0] = getImageIO( "/mnt/production/study/LEFFTDs_long/anares/ana_res-2018-07-09_SPM/T1/Change_maps/wch_c1_map_5888_2009-05-27_to_2010-10-27.nii" );
//  CM[1] = getImageIO( "/mnt/production/study/LEFFTDs_long/anares/ana_res-2018-07-09_SPM/T1/Change_maps/wch_c1_map_5888_2010-10-27_to_2011-11-01.nii" );
//  CM[2] = getImageIO( "/mnt/production/study/LEFFTDs_long/anares/ana_res-2018-07-09_SPM/T1/Change_maps/wch_c1_map_5888_2011-11-01_to_2013-05-28.nii" );
//  CM[3] = getImageIO( "/mnt/production/study/LEFFTDs_long/anares/ana_res-2018-07-09_SPM/T1/Change_maps/wch_c1_map_5888_2013-05-28_to_2016-01-14.nii" );
//  CM[4] = getImageIO( "/mnt/production/study/LEFFTDs_long/anares/ana_res-2018-07-09_SPM/T1/Change_maps/wch_c1_map_5888_2016-01-14_to_2017-01-12.nii" );
//  CM[5] = getImageIO( "/mnt/production/study/LEFFTDs_long/anares/ana_res-2018-07-09_SPM/T1/Change_maps/wch_c1_map_5888_2017-01-12_to_2018-01-29.nii" );
//  CM[6] = getImageIO( "/mnt/production/study/LEFFTDs_long/anares/ana_res-2018-07-09_SPM/T1/wc1avg_T1_5888_2009-05-27_xj_2009-05-27_TPM.nii.gz" );
//  CM[7] = getImageIO( "/mnt/production/study/LEFFTDs_long/anares/ana_res-2018-07-09_SPM/T1/wc1avg_T1_5888_2009-05-27_xj_2010-10-27_TPM.nii.gz" );
//  CM[8] = getImageIO( "/mnt/production/study/LEFFTDs_long/anares/ana_res-2018-07-09_SPM/T1/wc1avg_T1_5888_2009-05-27_xj_2011-11-01_TPM.nii.gz" );
//  CM[9] = getImageIO( "/mnt/production/study/LEFFTDs_long/anares/ana_res-2018-07-09_SPM/T1/wc1avg_T1_5888_2009-05-27_xj_2013-05-28_TPM.nii.gz" );
//  CM[10] = getImageIO( "/mnt/production/study/LEFFTDs_long/anares/ana_res-2018-07-09_SPM/T1/wc1avg_T1_5888_2009-05-27_xj_2016-01-14_TPM.nii.gz" );
//  CM[11] = getImageIO( "/mnt/production/study/LEFFTDs_long/anares/ana_res-2018-07-09_SPM/T1/wc1avg_T1_5888_2009-05-27_xj_2017-01-12_TPM.nii.gz" );
//  CM[12] = getImageIO( "/mnt/production/study/LEFFTDs_long/anares/ana_res-2018-07-09_SPM/T1/wc1avg_T1_5888_2009-05-27_xj_2018-01-29_TPM.nii.gz" );

  //
  itk::ImageIOBase::Pointer mask    = getImageIO( "/mnt/macdata/groups/imaging_core/matt/projects/mac-loni/Change_rates_Shooting/Change_rate_all_intervals/mask_80_split0000.nii.gz" );
  itk::ImageIOBase::Pointer Desikan = getImageIO( "/mnt/production/study/LEFFTDs_long/wDesikan.nii" );

  //
  //
  // reader
  typedef itk::Image< double, 3 >         Proba;
  typedef itk::Image< double, 3 >         Mask;
  typedef itk::ImageFileReader< Proba >  Reader;
  typedef itk::ImageFileReader< Mask >   Mask_reader;
  //
  // Change maps
  Reader::Pointer reader_CM[N];
  for ( int i = 0 ; i < N ; i++ )
    {
      reader_CM[i] = Reader::New();
      reader_CM[i]->SetFileName( CM[i]->GetFileName() );
      reader_CM[i]->Update();
    }
  //
  Mask_reader::Pointer reader_mask = Mask_reader::New();
  reader_mask->SetFileName( mask->GetFileName() );
  reader_mask->Update();
  //
  Mask_reader::Pointer reader_Desikan = Mask_reader::New();
  reader_Desikan->SetFileName( Desikan->GetFileName() );
  reader_Desikan->Update();

  //
  // Walk through the image
  Mask::RegionType region;
  //
  Mask::Pointer   image_atlas = reader_mask->GetOutput();
  Mask::SizeType  img_size    = reader_mask->GetOutput()->GetLargestPossibleRegion().GetSize();
  Mask::IndexType start       = {0, 0, 0};
  //
  region.SetSize( img_size );
  region.SetIndex( start );
  //
  //
  std::list< Eigen::Matrix< double, 3, 1 > > X_pos;
  std::list< Eigen::Matrix< double, 1, 1 > > X_intensity;
  //
  //
  itk::ImageRegionIterator<Mask> imageIterator_mask( reader_mask->GetOutput(), region );
  //
  std::string csv_string = "PIDN,X,Y,Z,ROI,dtp1,dtp2,dtp3,dtp4,dtp5,dtp6,tp1,tp2,tp3,tp4,tp5,tp6,tp7,\n";
  while( !(++imageIterator_mask).IsAtEnd() )
    {
      int mask_val = static_cast<int>( imageIterator_mask.Value() );
      if( mask_val != 0 )
	{
	  //
	  Mask::IndexType idx = imageIterator_mask.GetIndex();
	  //
	  int atlas_val = static_cast<int>( reader_Desikan->GetOutput()->GetPixel(idx) );
	  
	  std::string vox_csv_string = "ch_pidn,";
	  //
	  vox_csv_string += std::to_string(static_cast<int>( idx[0] )) + ",";
	  vox_csv_string += std::to_string(static_cast<int>( idx[1] )) + ",";
	  vox_csv_string += std::to_string(static_cast<int>( idx[2] )) + ",";
	  vox_csv_string += std::to_string(atlas_val) + ",";
	  //
	  for ( int i = 0 ; i < N ; i++ )
	    vox_csv_string += std::to_string(static_cast<float>( reader_CM[i]->GetOutput()->GetPixel(idx) )) + ",";
	  vox_csv_string += "\n";
	  //
	  csv_string += vox_csv_string;

	  //
	  //
	  Eigen::Matrix< double, 3, 1 > data;
	  data(0,0) = static_cast<double>( idx[0] );
	  data(1,0) = static_cast<double>( idx[1] );
	  data(2,0) = static_cast<double>( idx[2] );
	  //data(3,0) = static_cast<double>( reader_CM[1]->GetOutput()->GetPixel(idx) );
	  //
	  X_pos.push_back( data );
	  //
	  Eigen::Matrix< double, 1, 1 > data_int;
	  data_int(0,0) = static_cast<double>( reader_CM[1]->GetOutput()->GetPixel(idx) );
	  //
	  X_intensity.push_back( data_int );
	}
    }

//tempo  //
//tempo  // Write the csv file
//tempo  std::ofstream csv_out ( "PIDN.csv");
//tempo  if ( csv_out.is_open() )
//tempo    {
//tempo      csv_out << csv_string;
//tempo      csv_out.close();
//tempo    }
//tempo  else std::cout << "Unable to open file" << std::endl;
  

///  //
///  //
///  const int K = 2;
///  const int K_clus = 20;
///  EM< /*Dim*/ 1, /*K_gaussians*/ K > GaussianMixture_intensity;
///  GaussianMixture_intensity.ExpectationMaximization( X_intensity );
///  //
///  MAC::MakeITKImage output;
///  output = MAC::MakeITKImage( K_clus, "Clusters_probabilities.nii.gz", reader_CM[0] );
///
/////  VBGaussianMixture < /*Dim*/ 1, /*K_gaussians*/ K > VBGaussianMixture_intensity( X_intensity );
/////  VBGaussianMixture_intensity.ExpectationMaximization();
///  //
///
///
///  //
///  //
///  std::list< Eigen::Matrix< double, 3, 1 > >::iterator it_x = X_pos.begin();
///  std::list< Eigen::Matrix< double, 3, 1 > > filtered_pos;
///  int size_x = X_pos.size();
///  // Filter the position on the Intensity EM
///  for ( int x = 0 ; x < size_x ; x++ )
///    {
/////      for ( int k = 0 ; k < K ; k++ )
/////	{
///      double posterior = GaussianMixture_intensity.get_posterior_probabilities()[1/*k*/][x];
///      if ( posterior > 0.5 )
///	{
///	  filtered_pos.push_back( *(it_x) );
///	  std::cout 
///	    << (*it_x)(0,0) << ","
///	    << (*it_x)(1,0) << ","
///	    << (*it_x)(2,0) << ","
///	    << std::endl;
///	}
/////	}
///      ++it_x;
///    }
///  // Filter the position on the Intensity EM
///  VB::GM::VBGaussianMixture < /*Dim*/ 3, /*K_gaussians*/ 10 > VBGaussianMixture( filtered_pos );
///  VBGaussianMixture.ExpectationMaximization();
///  //
///  std::list< Eigen::Matrix< double, 3, 1 > >::iterator filterd_it_x = filtered_pos.begin();
///  int filterd_size_x = filtered_pos.size();
///  std::cout << "filterd_size_x " << filterd_size_x << std::endl;
///  for ( int x = 0 ; x < filterd_size_x ; x++ )
///    {
///      for ( int k = 0 ; k < K_clus ; k++ )
///	{
///	  int idx[3];
///	  idx[0] = (*filterd_it_x)(0,0);
///	  idx[1] = (*filterd_it_x)(1,0);
///	  idx[2] = (*filterd_it_x)(2,0);
///	  output.set_val( k, 
///			  {static_cast< long int >( (*filterd_it_x)(0,0) ),
///			      static_cast< long int >( (*filterd_it_x)(1,0) ),
///			      static_cast< long int >( (*filterd_it_x)(2,0) )}, 
///			  VBGaussianMixture.get_posterior_probabilities()[k][x] );
///	}
///      ++filterd_it_x;
///    }
///  //
///  output.write();


  //
  //
  return EXIT_SUCCESS;
}

