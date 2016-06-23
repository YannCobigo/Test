#
# CentOS
# cat /etc/centos-release
#
# ex. FROM centos:6 version 6 of CentOS
FROM centos:7

MAINTAINER Yann Cobigo yann.cobigo@ucsf.edu

# File system
RUN mkdir /mnt/neuroimaging/ /mnt/macdata /mnt/images 


# Update the image
RUN yum update -y
RUN yum install  -y epel-release.noarch

RUN yum groupinstall -y "Development tools"

RUN yum install  -y openssh.x86_64 \
    openssh-clients.x86_64 \
    openssh-server.x86_64 \
    emacs \
    cmake \
    gcc.x86_64 \
    gcc-c++ \
    python-devel.x86_64 \
    tmux \
    htop \
    git \
    wget \
    xz \
    zlib-devel \
    bzip2-devel \
    openssl-devel \
    ncurses-devel \
    sqlite-devel \
    libpng12.x86_64 \
    mlocate-0.26-5.el7.x86_64 \
    tree.x86_64 \
    sudo.x86_64

RUN yum install -y mesa-libGL-devel.x86_64 \
    mesa-libGLU.x86_64 \
     mesa-libGLU-devel.x86_64 \
     mesa-libGLw.x86_64 \
     mesa-libGLw-devel.x86_64 \
     mesa-libOSMesa.x86_64 \
     mesa-libOSMesa-devel.x86_64 \
     mesa-libgbm-devel.x86_64

# X11
RUN yum install -y xorg-x11-apps.x86_64 \
     xorg-x11-drivers.x86_64 \
      xorg-x11-utils.x86_64 \
       xorg-x11-xinit-session.x86_64

# Qt
RUN yum install -y qt-config.x86_64 \
    qt-creator.x86_64 qt-devel.x86_64 \
     qt-examples.x86_64 qt-mysql.x86_64 \
      qt-postgresql.x86_64 qt5-designer.x86_64 \
       qt5-qt3d.x86_64 qt5-qt3d-devel.x86_64 \
        qt5-qtbase.x86_64 qt5-qtbase-devel.x86_64 \
	 qt5-qtbase-examples.x86_64 qt5-qtconfiguration.x86_64 \
	  qt5-qtconfiguration-devel.x86_64 qt5ct.x86_64 \
	  qt5-qtx11extras.x86_64 qt5-qtx11extras-devel.x86_64 \
	   qt5-qtsvg.x86_64 qt5-qtsvg-devel.x86_64 qt5-qtsvg-examples.x86_64 \
	   qt5-qtbase-gui.x86_64 qt5-qtsvg.x86_64 qt5-qtsvg-devel.x86_64 qt5-qtsvg-examples.x86_64

###
### Python
##RUN cd && wget --no-check-certificate https://www.python.org/ftp/python/2.7.6/Python-2.7.6.tar.xz && tar -xf Python-2.7.6.tar.xz 
##RUN cd Python-2.7.6 && ./configure --prefix=/usr/local && make && make altinstall 
##
### It is important to use altinstall instead of install, otherwise you will end up with two different versions of Python in the filesystem both named python.
##
##RUN ln -s /usr/local/bin/python2.7 /usr/local/bin/python
##RUN cd && wget https://bootstrap.pypa.io/ez_setup.py && /usr/local/bin/python2.7 ez_setup.py && /usr/local/bin/easy_install-2.7 pip

RUN yum install -y  python-pip && pip install --upgrade pip 
RUN yum install -y  numpy scipy python-matplotlib ipython python-pandas sympy python-nose  \
    libxml2-devel.x86_64 libxml2-python.x86_64 libxslt.x86_64 libxslt-devel.x86_64 libxslt-python.x86_64 
    
    


# 
# Python 
# pip
# Pip is part of Extra Packages for Enterprise Linux (EPEL),
# matplotlib scipy
RUN pip install --upgrade pip nibabel networkx numpy rdflib sphinx graphviz  scipy && pip install future simplejson prov

#
# user features
#To avoid the interactive questions by adduser, you can call it with these parameters: RUN adduser --disabled-password --gecos '' newuser
RUN useradd -ms /bin/bash ycobigo
USER ycobigo
WORKDIR /home/ycobigo
