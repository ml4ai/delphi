#!/bin/bash

# build_and_run.sh

rm -rf build
rm -rf delphi/cpp/*

mkdir -p build
cd build
cmake ..
cmake --build . -- -j12 DelphiPython
cp *.so ../delphi/cpp
cd ..

time python test.py
