cmake_minimum_required(VERSION 2.8.12)
project(delphi)

set(PYBIND11_CPP_STANDARD -std=c++1z)
add_subdirectory(pybind11)
add_subdirectory(nlohmann)
pybind11_add_module(extension wrapper.cpp)
