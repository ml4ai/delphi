#!/bin/bash

# Install Boost from source

wget https://dl.bintray.com/boostorg/release/1.71.0/source/boost_1_71_0.tar.gz
tar xfz boost_1_71_0.tar.gz
pushd boost_1_71_0
  ./bootstrap.sh
  if [[ $? -ne 0 ]]; then exit 1; fi;
  sudo ./b2 install
  if [[ $? -ne 0 ]]; then exit 1; fi;
popd
rm -rf boost_1_71_0*
