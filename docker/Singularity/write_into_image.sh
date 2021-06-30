#!/usr/bin/sh
#
# CUDA info
# ATTENTION CHANGE cuda-${VERSION}
SRC=/usr/local/src/
###################
CUDA_VERSION=10.1 #
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
export DPCPP_NVIDIA=${DPCPP_HOME}/built_NVIDIA
#export DPCPP_CPU=${DPCPP_HOME}/built_CPU
mkdir $DPCPP_HOME
mkdir $DPCPP_NVIDIA
#mkdir $DPCPP_CPU
cd $DPCPP_HOME
# prepare the space for the drivers
mkdir /etc/OpenCL
mkdir /etc/OpenCL/vendors
#
# Installation of the low level machine
# Instead of installing the low level CPU runtime, it is possible to build and install the Khronos ICD loader, which contains all the symbols required.
git clone https://github.com/KhronosGroup/OpenCL-Headers /usr/local/include/OpenCL-Headers
ln -s /usr/local/include/OpenCL-Headers ${DPCPP_NVIDIA}/OpenCL-Headers
git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader ${DPCPP_NVIDIA}/OpenCL-ICD-Loader
# OpenCL defines an Installable Client Driver (ICD) mechanism to allow developers to build applications against an Installable Client Driver loader (ICD loader) rather than linking their applications against a specific OpenCL implementation.
cd ${DPCPP_NVIDIA}/OpenCL-ICD-Loader
mkdir build
cd build
cmake -D OPENCL_ICD_LOADER_HEADERS_DIR:PATH=${DPCPP_NVIDIA}/OpenCL-Headers/ ..
make
echo ${DPCPP_NVIDIA}/OpenCL-ICD-Loader/build/test/driver_stub/libOpenCLDriverStub.so > /etc/OpenCL/vendors/test.icd
# The OpenCL ICD Loader Tests use a "stub" ICD, which must be set up manually. The method to install the "stub" ICD is operating system dependent.
#
### DPC++
##cd $DPCPP_HOME
##git clone https://github.com/intel/llvm -b sycl
##cd ${DPCPP_HOME}/llvm
##git checkout $DPCPP_TAG
###
###    --system-ocl -> Don't Download OpenCL deps via cmake but use the system ones
###    --no-werror -> Don't treat warnings as errors when compiling llvm
###    --cuda -> use the cuda backend (see Nvidia CUDA)
###    --shared-libs -> Build shared libraries
###    -t -> Build type (debug or release)
###    -o -> Path to build directory
###    --cmake-gen -> Set build system type (e.g. --cmake-gen "Unix Makefiles")
###python $DPCPP_HOME/llvm/buildbot/configure.py --system-ocl --no-werror --cuda --shared-libs -t release --cmake-gen "Unix Makefiles" --cmake-opt="-DCMAKE_LIBRARY_PATH=${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs" -o $DPCPP_NVIDIA
###python $DPCPP_HOME/llvm/buildbot/compile.py -o $DPCPP_NVIDIA
##
##cd $DPCPP_NVIDIA
###
##cmake -G "Unix Makefiles" \
##      -DCMAKE_BUILD_TYPE=Release \
##      -DLLVM_ENABLE_ASSERTIONS=ON  \
##      -DLLVM_EXTERNAL_XPTI_SOURCE_DIR=${DPCPP_HOME}/llvm/xpti \
##      -DLLVM_EXTERNAL_LIBDEVICE_SOURCE_DIR=/usr/local/src/DPCPP/llvm/libdevice  \
##      -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
##      -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
##      -DLLVM_EXTERNAL_PROJECTS="llvm-spirv;sycl;opencl-aot;xpti;libdevice" \
##      -DLLVM_ENABLE_PROJECTS="clang;llvm-spirv;sycl" \
##      -DLLVM_EXTERNAL_SYCL_SOURCE_DIR=${DPCPP_HOME}/llvm/sycl \
##      -DLLVM_EXTERNAL_LLVM_SPIRV_SOURCE_DIR=${DPCPP_HOME}/llvm/llvm-spirv \
##      -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR} \
##      -DLLVM_ENABLE_PROJECTS="clang;llvm-spirv;sycl;opencl-aot;xpti;libdevice;libclc" \
##      -DSYCL_BUILD_PI_CUDA=ON \
##      -DCUDA_CUDA_LIBRARY:FILEPATH=/usr/local/cuda/lib64/stubs/libcuda.so \
##      -DCMAKE_LIBRARY_PATH:FILEPATH=${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs \
##      -DLLVM_BUILD_TOOLS=ON \
##      -DCMAKE_INSTALL_PREFIX=${DPCPP_NVIDIA}/install \
##      -DSYCL_INCLUDE_TESTS=ON \
##      -DBUILD_SHARED_LIBS=ON \
##      -DOpenCL_INCLUDE_DIR=/usr/local/include/OpenCL-Headers \
##      -DOpenCL_LIBRARY=${DPCPP_NVIDIA}/OpenCL-ICD-Loader/build/libOpenCL.so \
##      -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
##      -DLIBCLC_TARGETS_TO_BUILD="nvptx64--;nvptx64--nvidiacl" \
##      -DSYCL_ENABLE_WERROR=OFF \
##      -DSYCL_ENABLE_XPTI_TRACING=ON \
##      ${DPCPP_HOME}/llvm/llvm
###
##cmake --build ${DPCPP_NVIDIA} -- deploy-sycl-toolchain deploy-opencl-aot -j 16
##make install

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

#########
# BOUML #
#########
BO_MAJ=7
BO_MIN=10
BO_PATCH=1
#
cd $SRC
# https://www.bouml.fr/download.html
wget http://bouml.free.fr/files/bouml-${BO_MAJ}.${BO_MIN}-CentOS7.x86_64.rpm
dnf install -y bouml-${BO_MAJ}.${BO_MIN}-CentOS7.x86_64.rpm

#######
# VTK #
#######
VTK_MAJ=9
VTK_MIN=0
VTK_PATCH=1
#
cd $SRC
#https://www.vtk.org/files/release/9.0/VTK-9.0.1.tar.gz
curl -fsSL https://www.vtk.org/files/release/${VTK_MAJ}.${VTK_MIN}/VTK-${VTK_MAJ}.${VTK_MIN}.${VTK_PATCH}.tar.gz -O && \
tar zxvf VTK-${VTK_MAJ}.${VTK_MIN}.${VTK_PATCH}.tar.gz && cd VTK-${VTK_MAJ}.${VTK_MIN}.${VTK_PATCH} && mkdir build && cd build
/usr/local/bin/cmake \
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
/usr/local/bin/cmake \
       -D CMAKE_BUILD_TYPE:STRING=RELEASE \
       -D BUILD_SHARED_LIBS=ON \
       -D CMAKE_VERBOSE_MAKEFILE:BOOL=FALSE \
       -D ITK_LEGACY_REMOVE=ON \
       -D Module_ITKVtkGlue:BOOL=OFF \
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
/usr/local/bin/cmake \
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

##########
# libigl #
##########
lgl_MAJ=2
lgl_MIN=2
lgl_PATCH=0
#
cd $SRC
git clone https://github.com/libigl/libigl.git && cd libigl && git checkout v${lgl_MAJ}.${lgl_MIN}.${lgl_PATCH} && mkdir build && cd build
/usr/local/bin/cmake \
       -D CMAKE_INSTALL_PREFIX:PATH=/usr/local/ ..  && make && make install

############
# GeomView #
############
GV_MAJ=1
GV_MIN=9
GV_PATCH=5
# wget https://sourceforge.net/projects/geomview/files/geomview/1.9.5/geomview-1.9.5.tar.gz/download -o geomview-1.9.5.tar.gz
#https://sourceforge.net/projects/geomview/files/geomview/1.9.5/geomview-1.9.5.tar.gz/download
cd $SRC
wget https://sourceforge.net/projects/geomview/files/geomview/${GV_MAJ}.${GV_MIN}.${GV_PATCH}/geomview-${GV_MAJ}.${GV_MIN}.${GV_PATCH}.tar.gz/download && mv download geomview-${GV_MAJ}.${GV_MIN}.${GV_PATCH}.tar.gz && tar zxvf geomview-${GV_MAJ}.${GV_MIN}.${GV_PATCH}.tar.gz && cd geomview-${GV_MAJ}.${GV_MIN}.${GV_PATCH} && ./configure --prefix=/usr/local/ && make -j8 && make install 

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
/usr/local/bin/cmake \
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
#/usr/local/bin/cmake \
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


