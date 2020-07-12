Container information - Software versions
=========================================

The container is built with [Singularity](https://sylabs.io/docs/) (2.6.1-HEAD.9103f0155).

* `CUDA` - 10.2
* `DPC++` - 
* `CMake` - 3.18.0
* `VTK` - 9.0.1
* `ITK` - 5.1.0
* `Eigen3` - 3.3.7 (includes in `/usr/local/include/eigen3`)
* `CGAL` - 5.0.2 (includes in `/usr/local/src/cgal`)

CUDA_VERSION=10.2
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-${CUDA_VERSION}
export DPCPP=/usr/local/src/DPCPP/build/install

PATH=${CUDA_TOOLKIT_ROOT_DIR}/bin:${DPCPP}/bin:/usr/local/bin:$PATH 
LD_LIBRARY_PATH=${CUDA_TOOLKIT_ROOT_DIR}/lib64:${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs:${DPCPP}/lib/:/usr/local/lib64:/usr/local/lib:$LD_LIBRARY_PATH


Create a new Singularity image
==============================

The container must be built first in a writable manner using:

``` {.bash}
singularity image.create --size `echo "1024 64" | awk -F" " '{print $1*$2}'` /path/to/GPGPU-imaging_core.img
singularity build --writable /path/to/GPGPU-imaging_core.img GPGPU-imaging_core.def
```

when the image is built, the rest of the intsallation has to be done by launching the container

``` {.bash}
singularity shell --writable  --nv /path/to/GPGPU-imaging_core.img
```

then write in the image using the script

``` {.bash}
Singularity GPGPU-imaging_core.def:~/devel/Container/Docker/docker/Singularity> ./write_into_image.sh
```
