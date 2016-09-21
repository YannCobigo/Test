#
# CentOS
# cat /etc/centos-release
# sudo docker build -f Dockerfile --tag="imaging_core:01" .
# ex. FROM centos:6 version 6 of CentOS
FROM centos:7

MAINTAINER Yann Cobigo yann.cobigo@ucsf.edu

# File system
RUN mkdir /mnt/neuroimaging/ /mnt/macdata /mnt/images 

#
# Update the image
RUN yum update -y && yum install  -y epel-release.noarch && yum groupinstall -y "Development tools" && yum makecache fast
#
RUN yum install  -y openssh.x86_64 openssh-clients.x86_64 openssh-server.x86_64 \
    emacs tmux htop git wget tree.x86_64 cmake gcc.x86_64 gcc-c++ \
    fftw-libs-single-3.3.3-8.el7.x86_64 python-devel.x86_64 \
    xz zlib-devel bzip2-devel bzip2-libs-1.0.6-13.el7.x86_64 \
    openssl-devel ncurses-devel sqlite-devel \
    libpng12.x86_64 mlocate-0.26-5.el7.x86_64 \
    libyaml-0.1.4-11.el7_0.x86_64 yaml-cpp.x86_64 yaml-cpp-devel.x86_64 gsl.x86_64 
    gsl-devel.x86_64 gimp.x86_64 evince.x86_64 bc.x86_64 \
    sudo.x86_64 net-tools initscripts
#
RUN yum install -y mesa-libGL-devel.x86_64 \
    mesa-libGLU.x86_64 mesa-libGLU-devel.x86_64 \
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
	   qt5-qtbase-gui.x86_64 qt5-qtsvg.x86_64 qt5-qtsvg-devel.x86_64 qt5-qtsvg-examples.x86_64

#
RUN yum install -y graphviz.x86_64 graphviz-devel.x86_64 \
    graphviz-doc.x86_64 graphviz-gd.x86_64 graphviz-graphs.x86_64 \
    graphviz-guile.x86_64 graphviz-python.x86_64

# 
# Python 
RUN yum install -y  python-pip && pip install --upgrade pip 
RUN yum install -y  numpy scipy python-matplotlib ipython python-pandas sympy python-nose  \
    libxml2-devel.x86_64 libxml2-python.x86_64 libxslt.x86_64 libxslt-devel.x86_64 libxslt-python.x86_64 
# pip
# Pip is part of Extra Packages for Enterprise Linux (EPEL),
RUN pip install --upgrade pip nibabel networkx numpy rdflib sphinx graphviz  scipy && pip install lockfile future simplejson prov statsmodels psutil
RUN pip install -e git://github.com/nipy/nipype.git#egg=Package && pip install --upgrade nipype 
RUN pip install dipy && pip install --upgrade dipy 
RUN pip install --upgrade qap pygraphviz Pillow

#
# Quality Assessment Protocol software
RUN cd /mnt/scratch/neuroimaging_CentOS7/Python/QAP && chmod u+x setup.py && sudo python setup.py build && sudo python setup.py install


#
# user features
# To avoid the interactive questions by adduser, you can call it with these parameters: RUN adduser --disabled-password --gecos '' newuser
# we give the user its Cloud-Instance specific user ID
# run id in a terminal
# uid=1002(ycobigo) gid=1002(ycobigo) groups=1002(ycobigo),4(adm),27(sudo),110(admin) \
RUN groupadd ic       -g 1100 && \
    groupadd ycobigo  -g 1002 && \
    groupadd gmarx    -g 1003 && \
    groupadd guest    -g 1004 && \
    groupadd vsamboju -g 1005 && \
    useradd -u 1001 -G 1100 -g 1002 -p "ycobigo_tempo"  -ms /bin/bash ycobigo && \
    useradd -u 1002 -G 1100 -g 1003 -p "gmarx_tempo"    -ms /bin/bash gmarx && \
    useradd -u 1003 -G 1100 -g 1004 -p "guest_tempo"    -ms /bin/bash guest && \
    useradd -u 1004 -G 1100 -g 1005 -p "vsamboju_tempo" -ms /bin/bash vsamboju && \
    useradd -u 1005 -G 1100 -g 1006 -p "kchiang_tempo"  -ms /bin/bash vsamboju && \
    echo "ycobigo ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/ycobigo && \
    chmod 0440 /etc/sudoers.d/ycobigo 

USER guest

