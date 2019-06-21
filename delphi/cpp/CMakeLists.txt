cmake_minimum_required(VERSION 2.8.12)
project(delphi)

set(PYBIND11_CPP_STANDARD -std=c++17)
set(PYBIND11_PYTHON_VERSION 3.6)
find_package(sqlite REQUIRED)
add_subdirectory(pybind11)
include_directories(${CMAKE_SOURCE_DIR})
pybind11_add_module(extension wrapper.cpp)
