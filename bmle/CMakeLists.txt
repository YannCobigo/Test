cmake_minimum_required(VERSION 2.8)

project(bmle)

#
# Compilation

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_definitions(/DYOURDEFINITION)
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
  add_definitions(-std=c++11)
  set(CMAKE_CXX_FLAGS "-O3")
  set(CMAKE_CXX_FLAGS_DEBUG "-DDEBUG")
#  if(CMAKE_BUILD_TYPE STREQUAL "DEBUG")
#    set(CMAKE_CXX_FLAGS "-O3")
#  else()
#  endif()
endif()

#
# Eigen
#set( EIGEN /mnt/neuroimaging/Eigen-3/install/include/eigen3/ /mnt/neuroimaging/Eigen-3/install/include/eigen3/unsupported/ )
set( EIGEN /home/cobigo/devel/CPP/Eigen/install/include/eigen3/  /home/cobigo/devel/CPP/Eigen/install/include/eigen3/unsupported/)
#
include_directories( ${EIGEN}/ ${ITK_INC})
list( APPEND SOURCES ${DLIB_ALL_SOURCES} )



#
# ITK
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

#
# project
include_directories( ${bmle_SOURCE_DIR}  )
#
set( bmle_H ${bmle_SOURCE_DIR}/bmle.h )
install( FILES ${bmle_H} DESTINATION ${CMAKE_INSTALL_PREFIX}/include )

#
file( GLOB HEADERS_ bmle_*.h )
install( FILES ${HEADERS_}  DESTINATION ${CMAKE_INSTALL_PREFIX}/include/ )
#
file( GLOB SOURCES_ *.cxx )
list( APPEND SOURCES ${SOURCES_} )
#message(STATUS ${SOURCES})


add_executable( bmle MACOSX_BUNDLE bmle.cxx ${SOURCES} )
# cmake 3.1 target_compile_features(bmle PRIVATE cxx_range_for)
target_link_libraries( bmle -lX11 ${ITK_LIBRARIES} ${HEADER_FILES} )
