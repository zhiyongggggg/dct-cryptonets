#!/usr/bin/env bash

# Install libjpeg-turbo
sudo apt install -y nasm
sudo apt install -y cmake
sudo apt install -y libsm6 libxext6 libxrender-dev

mkdir ~/Repos/compressed-cryptonets/lib/
wget https://downloads.sourceforge.net/libjpeg-turbo/libjpeg-turbo-3.0.1.tar.gz
tar xvf libjpeg-turbo-3.0.1.tar.gz --directory ~/Repos/compressed-cryptonets/lib/
cd ~/Repos/compressed-cryptonets/lib/libjpeg-turbo-3.0.1

mkdir build
cd    build

cmake -DCMAKE_INSTALL_PREFIX=/usr \
      -DCMAKE_BUILD_TYPE=RELEASE  \
      -DENABLE_STATIC=FALSE       \
      -DCMAKE_INSTALL_DOCDIR=/usr/share/doc/libjpeg-turbo-3.0.1 \
      -DCMAKE_INSTALL_DEFAULT_LIBDIR=lib  \
      ..
make
sudo make install