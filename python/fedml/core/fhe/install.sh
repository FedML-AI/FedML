#!/bin/sh
apt-get install clang
pip3 install pybind11

cd tools
git clone https://github.com/weidai11/cryptopp.git
cd cryptopp && make && make test && make install

cd ..
git clone -b release-v1.11.2 https://gitlab.com/palisade/palisade-development.git
cd palisade-development && mkdir build && cd build
cmake .. && make && make install

cd ../../fhe-/fed/palisade_pybind/SHELFI_FHE/src
pip3 install ../
