cmake_minimum_required(VERSION 3.11)
project(delphi)

set(Boost_NO_BOOST_CMAKE ON)
# Add and link pybind11 modules
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake:${CMAKE_MODULE_PATH}")
find_package(SQLite3)
find_package(Boost REQUIRED)
include_directories(lib)
include_directories(external)
include_directories(${Boost_INCLUDE_DIR})
include_directories(external/nlohmann-json)
find_package(Eigen3 REQUIRED)
find_package(PythonInterp)
set(PYBIND11_PYTHON_VERSION "${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
set(PYBIND11_CPP_STANDARD -std=c++17)
add_subdirectory(external/pybind11)
pybind11_add_module(kde lib/kde_wrapper.cpp)
pybind11_add_module(AnalysisGraph lib/AnalysisGraph_wrapper.cpp)
target_link_libraries(AnalysisGraph PRIVATE ${Boost_LIBRARIES} Eigen3::Eigen)
target_link_libraries(kde PRIVATE ${Boost_LIBRARIES})



# Tests
add_library(doctest INTERFACE)
add_executable(cpptests tests/cpptests.cpp)
target_include_directories(doctest INTERFACE external/doctest)
target_link_libraries(cpptests PRIVATE doctest)
