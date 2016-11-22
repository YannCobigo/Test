#
# CentOS
# cat /etc/centos-release
# sudo docker build -f Dockerfile --tag="imaging_core:03" .
# ex. FROM centos:6 version 6 of CentOS
FROM centos:7

MAINTAINER Yann Cobigo yann.cobigo@ucsf.edu

# File system
RUN mkdir /mnt/neuroimaging/ /mnt/macdata /mnt/images /mnt/tempo

# 
# Modified PYTHON libraries
# We have to copy /mnt/scratch/neuroimaging_CentOS7/Python/QAP
# where we have the Dockerfile to be able to process this data
# cp -r /mnt/scratch/neuroimaging_CentOS7/Python/QAP .
# cp -r /mnt/scratch/neuroimaging_CentOS7/Python/dipy .
COPY QAP /mnt/tempo/QAP
COPY dipy /mnt/tempo/dipy

#
# Update the image
RUN yum update -y && yum install  -y epel-release.noarch && yum groupinstall -y "Development tools" && yum makecache fast
#
RUN yum install  -y openssh.x86_64 openssh-clients.x86_64 openssh-server.x86_64 fuse-sshfs.x86_64 \
    emacs tmux htop git wget tree.x86_64 cmake gcc.x86_64 gcc-c++ \
    fftw-libs-single-3.3.3-8.el7.x86_64 python-devel.x86_64 \
    xz zlib-devel bzip2-devel bzip2-libs-1.0.6-13.el7.x86_64 \
    openssl-devel ncurses-devel sqlite-devel libreoffice.x86_64 gwenview.x86_64 \
    libpng12.x86_64 mlocate-0.26-5.el7.x86_64 \
    libyaml-0.1.4-11.el7_0.x86_64 yaml-cpp.x86_64 yaml-cpp-devel.x86_64 gsl.x86_64 \
    gsl-devel.x86_64 gimp.x86_64 evince.x86_64 bc.x86_64 \
    gedit.x86_64 gedit-plugins.x86_64 gedit-plugins-data.x86_64 \
    gmp-devel.x86_64 gmp-static.x86_64 mpfr.x86_64 mpfr-devel.x86_64  glibc.i686 \
    sudo.x86_64 net-tools initscripts nautilus.x86_64
# GL
RUN yum install -y mesa-libGL-devel.x86_64 \
    mesa-libGLU.x86_64 mesa-libGLU-devel.x86_64 mesa-libGLU.i686 mesa-libGLU-devel.i686 \
     mesa-libGLw.x86_64 mesa-libGLw-devel.x86_64 \
     mesa-libOSMesa.x86_64 mesa-libOSMesa-devel.x86_64 \
     mesa-libgbm-devel.x86_64

#
# X11
RUN yum install -y xorg-x11-apps.x86_64 \
    xorg-x11-xauth xorg-x11-drivers.x86_64 \
      xorg-x11-utils.x86_64 xorg-x11-xinit-session.x86_64

# Qt
RUN yum install -y qt-config.x86_64 qt-creator.x86_64 qt-devel.x86_64 \
     qt-examples.x86_64 qt-mysql.x86_64 qt-postgresql.x86_64 qt5-designer.x86_64 \
       qt5-qt3d.x86_64 qt5-qt3d-devel.x86_64 qt5-qtbase.x86_64 qt5-qtbase-devel.x86_64 \
	 qt5-qtbase-examples.x86_64 qt5-qtconfiguration.x86_64 \
	  qt5-qtconfiguration-devel.x86_64 qt5ct.x86_64 qt5-qtx11extras.x86_64 qt5-qtx11extras-devel.x86_64 \
	   qt5-qtsvg.x86_64 qt5-qtsvg-devel.x86_64 qt5-qtsvg-examples.x86_64 \
	   qt5-qtbase-gui.x86_64 qt5-qtsvg.x86_64 qt5-qtsvg-devel.x86_64 qt5-qtsvg-examples.x86_64\
	    graphviz.x86_64 graphviz-devel.x86_64 graphviz-doc.x86_64 graphviz-gd.x86_64 graphviz-graphs.x86_64 \
	    graphviz-guile.x86_64 graphviz-python.x86_64

#
# Sublime text 3
# PyCharm
RUN  wget --directory-prefix=/mnt/tempo/  http://c758482.r82.cf2.rackcdn.com/sublime_text_3_build_3065_x64.tar.bz2 \
     && tar -vxjf /mnt/tempo/sublime_text_3_build_3065_x64.tar.bz2 -C /opt/ \
     && ln -s /opt/sublime_text_3/sublime_text /usr/bin/sublime3 

# 
# Python 
RUN yum install -y  python-pip && pip install --upgrade pip setuptools
RUN yum install -y  numpy scipy python-matplotlib ipython python-pandas sympy python-nose  \
    libxml2-devel.x86_64 libxml2-python.x86_64 libxslt.x86_64 libxslt-devel.x86_64 libxslt-python.x86_64 
# pip
# Pip is part of Extra Packages for Enterprise Linux (EPEL),
RUN pip install --upgrade pip nibabel networkx numpy rdflib sphinx graphviz scipy sklearn progressbar \
    && pip install lockfile future simplejson prov statsmodels psutil cython untangle jsonpath-rw \
    bctpy h5py networkx 
RUN pip install -e git://github.com/nipy/nipype.git#egg=Package && pip install --upgrade nipype 
RUN pip install --upgrade pygraphviz Pillow python-igraph 
# Python louvain
# https://pypi.python.org/pypi/python-louvain/0.3
RUN wget --directory-prefix=/mnt/tempo/ https://pypi.python.org/packages/f1/5f/1cd2040382f56b21ef731f09fd7a818ad2bbcd5a8201fd2ebd4ec15297bb/python-louvain-0.3.tar.gz  \
    && tar -zxvf /mnt/tempo/python-louvain-0.3.tar.gz \
    && python /mnt/tempo/python-louvain-0.3/setup.py install 

#
# Modified PYTHON libraries
# Quality Assessment Protocol software
RUN cd /mnt/tempo/QAP && chmod u+x setup.py && python setup.py build && python setup.py install \
    && cd /mnt/tempo/dipy && chmod u+x setup.py && python setup.py build && python setup.py install 

#
# user features
# To avoid the interactive questions by adduser, you can call it with these parameters: RUN adduser --disabled-password --gecos '' newuser
# we give the user its Cloud-Instance specific user ID
# run id in a terminal
# uid=1002(ycobigo) gid=1002(ycobigo) groups=1002(ycobigo),4(adm),27(sudo),110(admin) \
RUN groupadd ic        -g 1100 && \
    groupadd ycobigo   -g 1002 && \
    groupadd gmarx     -g 1003 && \
    groupadd guest     -g 1004 && \
    groupadd vsamboju  -g 1005 && \
    groupadd kchiang   -g 1006 && \
    groupadd mmandelli -g 1007 && \
    groupadd pmumford  -g 1008 && \
    useradd -u 1001 -G 1100 -g 1002 -p "ycobigo_tempo"  -ms /bin/bash ycobigo && \
    useradd -u 1002 -G 1100 -g 1003 -p "gmarx_tempo"    -ms /bin/bash gmarx && \
    useradd -u 1003 -G 1100 -g 1004 -p "guest_tempo"    -ms /bin/bash guest && \
    useradd -u 1004 -G 1100 -g 1005 -p "vsamboju_tempo" -ms /bin/bash vsamboju && \
    useradd -u 1005 -G 1100 -g 1006 -p "kchiang_tempo"  -ms /bin/bash kchiang && \
    useradd -u 1006 -G 1100 -g 1007 -p "mmandelli_tempo" -ms /bin/bash mmandelli && \
    useradd -u 1007 -G 1100 -g 1008 -p "pmumford_tempo"  -ms /bin/bash pmumford && \
    echo "ycobigo ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/ycobigo && \
    echo "gmarx ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/gmarx && \
    chmod 0440 /etc/sudoers.d/ycobigo && \
    chmod 0440 /etc/sudoers.d/gmarx 

USER guest

