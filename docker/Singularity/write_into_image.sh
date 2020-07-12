#!/usr/bin/sh
#
# CUDA info
# ATTENTION CHANGE cuda-${VERSION}
SRC=/usr/local/src/
###################
CUDA_VERSION=10.2 #
###################
CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-${CUDA_VERSION}
cp -a /usr/local/cuda/ ${CUDA_TOOLKIT_ROOT_DIR}/
rm -rf /usr/local/cuda
ln -s ${CUDA_TOOLKIT_ROOT_DIR} /usr/local/cuda
#
PATH=${CUDA_TOOLKIT_ROOT_DIR}/bin:/usr/local/bin:$PATH 
LD_LIBRARY_PATH=${CUDA_TOOLKIT_ROOT_DIR}/lib64:${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs:/usr/local/lib64:/usr/local/lib:$LD_LIBRARY_PATH

#########
# DPCPP #
#########
# https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md
# https://github.com/intel/llvm/
# Tag version
###################
DPCPP_TAG=2020-06 #
###################
export DPCPP_HOME=${SRC}/DPCPP
export DPCPP_BUILD=${DPCPP_HOME}/build
mkdir $DPCPP_HOME
mkdir $DPCPP_BUILD
cd $DPCPP_HOME
# prepare the space for the drivers
mkdir /etc/OpenCL
mkdir /etc/OpenCL/vendors
#
# Installation of the low level machine
# Instead of installing the low level CPU runtime, it is possible to build and install the Khronos ICD loader, which contains all the symbols required.
git clone https://github.com/KhronosGroup/OpenCL-Headers /usr/local/include/OpenCL-Headers
ln -s /usr/local/include/OpenCL-Headers ${DPCPP_BUILD}/OpenCL-Headers
git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader ${DPCPP_BUILD}/OpenCL-ICD-Loader
#OpenCL defines an Installable Client Driver (ICD) mechanism to allow developers to build applications against an Installable Client Driver loader (ICD loader) rather than linking their applications against a specific OpenCL implementation.
cd ${DPCPP_BUILD}/OpenCL-ICD-Loader
mkdir build
cd build
cmake -D OPENCL_ICD_LOADER_HEADERS_DIR:PATH=${DPCPP_BUILD}/OpenCL-Headers/ ..
make
# The OpenCL ICD Loader Tests use a "stub" ICD, which must be set up manually. The method to install the "stub" ICD is operating system dependent.
echo ${DPCPP_BUILD}/OpenCL-ICD-Loader/build/test/driver_stub/libOpenCLDriverStub.so > /etc/OpenCL/vendors/test.icd
#
# DPC++
cd $DPCPP_HOME
git clone https://github.com/intel/llvm -b sycl
cd ${DPCPP_HOME}/llvm
git checkout $DPCPP_TAG
#
#    --system-ocl -> Don't Download OpenCL deps via cmake but use the system ones
#    --no-werror -> Don't treat warnings as errors when compiling llvm
#    --cuda -> use the cuda backend (see Nvidia CUDA)
#    --shared-libs -> Build shared libraries
#    -t -> Build type (debug or release)
#    -o -> Path to build directory
#    --cmake-gen -> Set build system type (e.g. --cmake-gen "Unix Makefiles")
python $DPCPP_HOME/llvm/buildbot/configure.py --system-ocl --no-werror --cuda --shared-libs -t release --cmake-gen "Unix Makefiles" --cmake-opt="-DCMAKE_LIBRARY_PATH=${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs" -o $DPCPP_BUILD
#python $DPCPP_HOME/llvm/buildbot/compile.py -o $DPCPP_BUILD
cd $DPCPP_BUILD && make && make install 

#########
# CMake #
#########
CM_MAJ=3
CM_MIN=18
CM_PATH=0
# 
cd $SRC
curl -fsSL https://github.com/Kitware/CMake/releases/download/v${CM_MAJ}.${CM_MIN}.${CM_PATH}-rc3/cmake-${CM_MAJ}.${CM_MIN}.${CM_PATH}-rc3.tar.gz -O && \
tar zxvf cmake-${CM_MAJ}.${CM_MIN}.${CM_PATH}-rc3.tar.gz && cd cmake-${CM_MAJ}.${CM_MIN}.${CM_PATH}-rc3 && mkdir build && cd build 
#
cmake \
-D CMAKE_BUILD_TYPE=Release \
-D BUILD_SHARED_LIBS=OFF \
-D CMAKE_INSTALL_PREFIX=/usr/local/ .. && make -j 8 && make install
ln -s /usr/local/share/cmake-${CM_MAJ}.${CM_MIN} /usr/local/share/cmake

#######
# VTK #
#######
VTK_MAJ=9
VTK_MIN=0
VTK_PATCH=1
#
cd $SRC
https://www.vtk.org/files/release/9.0/VTK-9.0.1.tar.gz
curl -fsSL https://www.vtk.org/files/release/${VTK_MAJ}.${VTK_MIN}/VTK-${VTK_MAJ}.${VTK_MIN}.${VTK_PATCH}.tar.gz -O && \
tar zxvf VTK-${VTK_MAJ}.${VTK_MIN}.${VTK_PATCH}.tar.gz && cd VTK-${VTK_MAJ}.${VTK_MIN}.${VTK_PATCH} && mkdir build && cd build
cmake \
-D CMAKE_BUILD_TYPE=Release \
-D BUILD_SHARED_LIBS=ON \
-D CMAKE_INSTALL_PREFIX=/usr/local/ .. && make -j 8 && make install

#######
# ITK #
#######
ITK_MAJ=5
ITK_MIN=1
ITK_PATCH=0
#
cd $SRC
curl -fsSL https://github.com/InsightSoftwareConsortium/ITK/releases/download/v${ITK_MAJ}.${ITK_MIN}.${ITK_PATCH}/InsightToolkit-${ITK_MAJ}.${ITK_MIN}.${ITK_PATCH}.tar.gz -O && \
tar zxvf InsightToolkit-${ITK_MAJ}.${ITK_MIN}.${ITK_PATCH}.tar.gz && cd InsightToolkit-${ITK_MAJ}.${ITK_MIN}.${ITK_PATCH} && mkdir build && cd build
cmake \
       -D CMAKE_BUILD_TYPE:STRING=RELEASE \
       -D BUILD_SHARED_LIBS=ON \
       -D CMAKE_VERBOSE_MAKEFILE:BOOL=FALSE \
       -D ITK_LEGACY_REMOVE=OFF \
       -D Module_ITKVtkGlue:BOOL=ON \
       -D VTK_DIR:PATH=/usr/local/lib64/ \
       -D CMAKE_INSTALL_PREFIX:PATH=/usr/local/ .. && make -j 8 && make install

##########
# Eigen3 #
##########
EG_MAJ=3
EG_MIN=3
EG_PATCH=7
#
cd $SRC
git clone https://gitlab.com/libeigen/eigen && cd eigen && git checkout ${EG_MAJ}.${EG_MIN}.${EG_PATCH}
mkdir build && cd build
cmake \
       -D CMAKE_BUILD_TYPE:STRING=RELEASE \
       -D BUILD_SHARED_LIBS=ON \
       -D CMAKE_VERBOSE_MAKEFILE:BOOL=FALSE \
       -D CMAKE_INSTALL_PREFIX:PATH=/usr/local/ ..  && make install

########
# CGAL #
########
CG_MAJ=5
CG_MIN=0
CG_PATCH=2
#
cd $SRC
git clone https://github.com/CGAL/cgal && cd cgal && git checkout releases/CGAL-${CG_MAJ}.${CG_MIN}.${CG_PATCH}

########
# JSON #
########
JS_MAJ=3
JS_MIN=7
JS_PATCH=3
#
cd $SRC
git clone https://github.com/nlohmann/json && cd json && git checkout v${JS_MAJ}.${JS_MIN}.${JS_PATCH}
mkdir build && cd build
cmake \
       -D CMAKE_BUILD_TYPE:STRING=RELEASE \
       -D BUILD_SHARED_LIBS=ON \
       -D CMAKE_VERBOSE_MAKEFILE:BOOL=FALSE \
       -D CMAKE_INSTALL_PREFIX:PATH=/usr/local/ ..  && make -j 8 && make install


############
## MRtrix3 #
############
# https://mrtrix.readthedocs.io/en/latest/installation/build_from_source.html
#MR_MAJ=3
#MR_MIN=0
#MR_PATCH=1
##
#cd $SRC
#git clone https://github.com/MRtrix3/mrtrix3.git && cd mrtrix3 && git checkout ${MR_MAJ}.${MR_MIN}.${MR_PATCH}
#export EIGEN_CFLAGS="-isystem /usr/local/include/eigen3"
#./configure --prefix=/usr/local/ && make -j 8 && make install
#cmake \
#       -D CMAKE_BUILD_TYPE:STRING=RELEASE \
#       -D BUILD_SHARED_LIBS=ON \
#       -D CMAKE_VERBOSE_MAKEFILE:BOOL=FALSE \
#       -D CMAKE_INSTALL_PREFIX:PATH=/usr/local/ ..  && make install



########
## FSL #
########
## https://mrtrix.readthedocs.io/en/latest/installation/build_from_source.html
#FSL_MAJ=6
#FSL_MIN=0
#FSL_PATCH=3
##
## Missing fslview & fsleyes
#cd $SRC
#curl -fsSL https://fsl.fmrib.ox.ac.uk/fsldownloads/fsl-${FSL_MAJ}.${FSL_MIN}.${FSL_PATCH}-sources.tar.gz -O && \
#    tar zxvf fsl-${FSL_MAJ}.${FSL_MIN}.${FSL_PATCH}-sources.tar.gz && cd fsl && export FSLDIR=$PWD && \
#    source ${FSLDIR}/etc/fslconf/fsl.sh && ./build 


