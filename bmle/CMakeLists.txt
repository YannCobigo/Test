cmake_minimum_required(VERSION 2.8)

project(NeuroBayes)

#
# Compilation

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_definitions(/DYOURDEFINITION)
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
  add_definitions(-std=c++11)
  set(CMAKE_CXX_FLAGS "-O3")
#  set(CMAKE_CXX_FLAGS_DEBUG "-DDEBUG")
#  set(CMAKE_CXX_FLAGS "-g")
#  if(CMAKE_BUILD_TYPE STREQUAL "DEBUG")
#    set(CMAKE_CXX_FLAGS "-O3")
#  else()
#  endif()
endif()

#
# Eigen
set( EIGEN /mnt/neuroimaging/Eigen-3/install/include/eigen3/ /mnt/neuroimaging/Eigen-3/install/include/eigen3/unsupported/ )
#set( EIGEN /home/cobigo/devel/Singularity/neuroimaging_CentOS7/Eigen-3/install/include/eigen3/  /home/cobigo/devel/Singularity/neuroimaging_CentOS7/Eigen-3/install/include/eigen3/unsupported/)
#
include_directories( ${EIGEN}/ ${ITK_INC})
list( APPEND SOURCES ${DLIB_ALL_SOURCES} )



#
# ITK
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

#
# Utils
file( GLOB HEADERS_       utils/*.h )
file( GLOB HEADERS_       utils/IO/*.h )
file( GLOB HEADERS_       utils/Algorithms/*.h )
file( GLOB HEADERS_       utils/Optimizers/*.h )
file( GLOB UTILS_SOURCES_ utils/*.cxx utils/Algorithms/*.cxx  )
#
list( APPEND HEADERS ${HEADERS_}  )
list( APPEND SOURCES ${UTILS_SOURCES_} )
#
include_directories(
  ${NeuroBayes_SOURCE_DIR}/utils
   )
# Libraries
add_library( _NeuroBayes_utils SHARED ${UTILS_SOURCES_} )


#
# project BMLE
include_directories( ${NeuroBayes_SOURCE_DIR}/BMLE )
#
file( GLOB BMLE_HEADERS_ BMLE/*.h )
file( GLOB BMLE_SOURCES_ BMLE/*.cxx )
#
list( APPEND HEADERS ${BMLE_HEADERS_} )
list( APPEND SOURCES ${BMLE_SOURCES_} )
#message(STATUS ${SOURCES})
#
add_executable( bmle MACOSX_BUNDLE BMLE/bmle.cpp  ${HEADERS_} ${BMLE_SOURCES_}  )
# cmake 3.1 target_compile_features(bmle PRIVATE cxx_range_for)
target_link_libraries( bmle -lX11 ${ITK_LIBRARIES} _NeuroBayes_utils )

#
# project mle
include_directories( ${NeuroBayes_SOURCE_DIR}/mle )
#
file( GLOB BMLE_HEADERS_ mle/*.h )
file( GLOB BMLE_SOURCES_ mle/*.cxx )
#
list( APPEND HEADERS ${MLE_HEADERS_} )
list( APPEND SOURCES ${MLE_SOURCES_} )
#message(STATUS ${SOURCES})
#
add_executable( mle MACOSX_BUNDLE mle/mle.cpp  ${HEADERS_} ${BMLE_SOURCES_}  )
# cmake 3.1 target_compile_features(bmle PRIVATE cxx_range_for)
target_link_libraries( mle -lX11 ${ITK_LIBRARIES} _NeuroBayes_utils )


#
# project Variational Bayes HMM
include_directories( ${NeuroBayes_SOURCE_DIR}/VB/HMM  ${NeuroBayes_SOURCE_DIR}/noVB/HMM )
#
file( GLOB VBHMM_HEADERS_ VB/HMM/*.h )
file( GLOB VBHMM_SOURCES_ VB/HMM/*.cxx )
file( GLOB HMM_HEADERS_ noVB/HMM/*.h )
file( GLOB HMM_SOURCES_ noVB/HMM/*.cxx )
#
list( APPEND HEADERS ${VBHMM_HEADERS_} ${HMM_HEADERS_} )
list( APPEND SOURCES ${VBHMM_SOURCES_} ${HMM_SOURCES_} )
#message(STATUS ${SOURCES})
#
add_executable( vbhmm     MACOSX_BUNDLE VB/HMM/vbhmm.cpp            ${HEADERS_} ${VBHMM_SOURCES_}  )
add_executable( vbhmmproj MACOSX_BUNDLE VB/HMM/vbhmm_projection.cpp ${HEADERS_} ${VBHMM_SOURCES_}  )
add_executable( vbhmmtest MACOSX_BUNDLE VB/HMM/vbhmm_test.cpp       ${HEADERS_} ${VBHMM_SOURCES_}  )
add_executable( hmm     MACOSX_BUNDLE noVB/HMM/hmm.cpp              ${HEADERS_} ${HMM_SOURCES_}  )
add_executable( hmmtest MACOSX_BUNDLE noVB/HMM/hmm_test.cpp         ${HEADERS_} ${HMM_SOURCES_}  )
# cmake 3.1 target_compile_features(bmle PRIVATE cxx_range_for)
target_link_libraries( vbhmm     -lX11  -lgsl -lm ${ITK_LIBRARIES} _NeuroBayes_utils )
target_link_libraries( vbhmmproj -lX11  -lgsl -lm ${ITK_LIBRARIES} _NeuroBayes_utils )
target_link_libraries( vbhmmtest -lX11  -lgsl -lm ${ITK_LIBRARIES} _NeuroBayes_utils )
target_link_libraries( hmm     -lX11  -lgsl -lm ${ITK_LIBRARIES} _NeuroBayes_utils )
target_link_libraries( hmmtest -lX11  -lgsl -lm ${ITK_LIBRARIES} _NeuroBayes_utils )


#
# project Variational Bayes LGSSM
include_directories( ${NeuroBayes_SOURCE_DIR}/noVB/LGSSM )
#
#file( GLOB VBLGSSM_HEADERS_ VB/LGSSM/*.h )
#file( GLOB VBLGSSM_SOURCES_ VB/LGSSM/*.cxx )
file( GLOB LGSSM_HEADERS_ noVB/LGSSM/*.h )
file( GLOB LGSSM_SOURCES_ noVB/LGSSM/*.cxx )
#
list( APPEND HEADERS ${LGSSM_HEADERS_} )
list( APPEND SOURCES ${LGSSM_SOURCES_} )
#message(STATUS ${SOURCES})
#
#add_executable( vbhmm     MACOSX_BUNDLE VB/LGSSM/vbhmm.cpp            ${HEADERS_} ${VBLGSSM_SOURCES_}  )
#add_executable( vbhmmproj MACOSX_BUNDLE VB/LGSSM/vbhmm_projection.cpp ${HEADERS_} ${VBLGSSM_SOURCES_}  )
#add_executable( vbhmmtest MACOSX_BUNDLE VB/LGSSM/vbhmm_test.cpp       ${HEADERS_} ${VBLGSSM_SOURCES_}  )
add_executable( lgssm     MACOSX_BUNDLE noVB/LGSSM/lgssm.cpp              ${HEADERS_} ${LGSSM_SOURCES_}  )
add_executable( lgssmtest MACOSX_BUNDLE noVB/LGSSM/lgssm_test.cpp         ${HEADERS_} ${LGSSM_SOURCES_}  )
# cmake 3.1 target_compile_features(bmle PRIVATE cxx_range_for)
#target_link_libraries( vbhmm     -lX11  -lgsl -lm ${ITK_LIBRARIES} _NeuroBayes_utils )
#target_link_libraries( vbhmmproj -lX11  -lgsl -lm ${ITK_LIBRARIES} _NeuroBayes_utils )
#target_link_libraries( vbhmmtest -lX11  -lgsl -lm ${ITK_LIBRARIES} _NeuroBayes_utils )
target_link_libraries( lgssm     -lX11  -lgsl -lm ${ITK_LIBRARIES} _NeuroBayes_utils )
target_link_libraries( lgssmtest -lX11  -lgsl -lm ${ITK_LIBRARIES} _NeuroBayes_utils )


#
# project Variational Bayes GM
include_directories( ${NeuroBayes_SOURCE_DIR}/VB/GM )
#
file( GLOB VBGM_HEADERS_ VB/GM/*.h )
file( GLOB VBGM_SOURCES_ VB/GM/*.cxx )
file( GLOB GM_HEADERS_ noVB/GM/*.h )
file( GLOB GM_SOURCES_ noVB/GM/*.cxx )
#
list( APPEND HEADERS ${VBGM_HEADERS_}  ${GM_HEADERS_} )
list( APPEND SOURCES ${VBGM_SOURCES_}  ${GM_SOURCES_} )
#message(STATUS ${SOURCES})
#
add_executable( vbgm     MACOSX_BUNDLE   VB/GM/vbgm.cpp       ${HEADERS_} ${VBGM_SOURCES_}  )
#add_executable( vbgmtest MACOSX_BUNDLE   VB/GM/vbgm_test.cpp  ${HEADERS_} ${VBGM_SOURCES_}  )
add_executable( gm       MACOSX_BUNDLE noVB/GM/gm.cpp         ${HEADERS_} ${GM_SOURCES_}  )
#add_executable( gmtest   MACOSX_BUNDLE noVB/GM/gm_test.cpp    ${HEADERS_} ${GM_SOURCES_}  )
# cmake 3.1 target_compile_features(bmle PRIVATE cxx_range_for)
target_link_libraries( vbgm     -lX11  -lgsl -lm ${ITK_LIBRARIES} _NeuroBayes_utils )
#target_link_libraries( vbgmtest -lX11  -lgsl -lm ${ITK_LIBRARIES} _NeuroBayes_utils )
target_link_libraries( gm       -lX11  -lgsl -lm ${ITK_LIBRARIES} _NeuroBayes_utils )
#target_link_libraries( gmtest   -lX11  -lgsl -lm ${ITK_LIBRARIES} _NeuroBayes_utils )


#
# Install
install( FILES ${HEADERS_} ${BMLE_HEADERS_} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/ )
install( FILES ${UTILS_SOURCES_} ${BMLE_SOURCES_} DESTINATION ${CMAKE_INSTALL_PREFIX}/src/ )
