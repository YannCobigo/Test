cmake_minimum_required(VERSION 2.8)

#project(mk_test)
 
if(CMAKE_COMPILER_IS_GNUCXX)
  add_definitions(-std=c++11)
endif()

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
if (ITKVtkGlue_LOADED)
  find_package(VTK REQUIRED)
  include(${VTK_USE_FILE})
else()
  find_package(ItkVtkGlue REQUIRED)
  include(${ItkVtkGlue_USE_FILE})
  set(Glue ItkVtkGlue)
endif()

#
# Eigen
set( EIGEN /mnt/neuroimaging/Eigen-3/install/include/eigen3/ )
#
include_directories( ${EIGEN})
#
file( GLOB HEADERS_ ../*.h )
list( APPEND HEADERS ${HEADERS_} )
#
file( GLOB SOURCES_ ../PMD.cxx ../SPC.cxx )
list( APPEND SOURCES ${SOURCES_} )

MESSAGE( ${SOURCES} )

#simple_matrix.cxx
add_executable( simple_matrix MACOSX_BUNDLE simple_matrix.cxx ${HEADERS} ${SOURCES} ) 
# cmake 3.1 target_compile_features(mk_test PRIVATE cxx_range_for)
target_link_libraries(simple_matrix 
  ${Glue} ${VTK_LIBRARIES} ${ITK_LIBRARIES} ${HEADER_FILES})
#
#simple_matrix_cv.cxx
add_executable( simple_matrix_cv MACOSX_BUNDLE simple_matrix_cv.cxx ${HEADERS} ${SOURCES} ) 
# cmake 3.1 target_compile_features(mk_test PRIVATE cxx_range_for)
target_link_libraries(simple_matrix_cv 
  ${Glue} ${VTK_LIBRARIES} ${ITK_LIBRARIES} ${HEADER_FILES})
