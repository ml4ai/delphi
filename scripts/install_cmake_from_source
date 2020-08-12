#!/bin/bash

# Install CMake from source

install_cmake_from_source() {
  local version=3.16
  local build=2
  local filename="cmake-$version.$build-Linux-x86_64.sh"
  wget https://cmake.org/files/v$version/$filename
  if [[ $? -ne 0 ]]; then exit 1; fi;
  sudo sh $filename --prefix=/usr/local --skip-license
  if [[ $? -ne 0 ]]; then exit 1; fi;
  rm $filename
  if [[ $? -ne 0 ]]; then exit 1; fi;
}

install_cmake_from_source
