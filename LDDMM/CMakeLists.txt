cmake_minimum_required(VERSION 2.8)
 
project(LDDMM)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
if (ITKVtkGlue_LOADED)
  find_package(VTK REQUIRED)
  include(${VTK_USE_FILE})
endif()


set(Header_Displacement Modules/Filtering/DisplacementField/include/ )
set(Header_Registration Modules/Registration/RegistrationMethodsv4/include/ )

include_directories( ${Header_Displacement} ${Header_Registration} ) 


add_executable(LDDMM MACOSX_BUNDLE LDDMM.cxx)
 
if( "${ITK_VERSION_MAJOR}" LESS 4 )
  target_link_libraries(LDDMM ITKReview ${ITK_LIBRARIES})
else( "${ITK_VERSION_MAJOR}" LESS 4 )
  target_link_libraries(LDDMM ${ITK_LIBRARIES})
endif( "${ITK_VERSION_MAJOR}" LESS 4 )
