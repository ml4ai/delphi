cmake_minimum_required(VERSION 3.11)
project(delphi)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(Boost_NO_BOOST_CMAKE ON)

# Add and link pybind11 modules
find_package(Boost REQUIRED)
include_directories(lib)
include_directories(external)
include_directories(${Boost_INCLUDE_DIR})

include_directories(external/nlohmann-json)
include_directories(external/cppitertools)

find_package(FMT)

if (FMT_FOUND)
  message("-- Found fmt.")
else()
  message("-- fmt package not found on system, using the version bundled with Delphi.")
  option(FMT_DOC OFF)
  option(FMT_INSTALL OFF)
  option(FMT_TEST OFF)
  add_subdirectory(external/fmt)
endif()

find_package(PythonInterp REQUIRED)

set(PYBIND11_PYTHON_VERSION "${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
set(PYBIND11_CPP_STANDARD -std=c++17)
add_subdirectory(external/pybind11)

pybind11_add_module(kde lib/kde_wrapper.cpp)
pybind11_add_module(AnalysisGraph lib/AnalysisGraph_wrapper.cpp)

target_link_libraries(AnalysisGraph PRIVATE ${Boost_LIBRARIES} fmt::fmt)
target_link_libraries(kde PRIVATE ${Boost_LIBRARIES} fmt::fmt)

find_package(Eigen3)
if (EIGEN3_FOUND)
  message("-- Found Eigen3: ${EIGEN3_INCLUDE_DIR}")
  target_link_libraries(AnalysisGraph PRIVATE Eigen3::Eigen)
else()
  message("-- Eigen3 not found on system, using version bundled with Delphi")
  include_directories(external/eigen)
endif()


find_package(SQLite3 REQUIRED)
include_directories(${SQLITE3_INCLUDE_DIRS})
target_link_libraries (AnalysisGraph PRIVATE "sqlite3")


# Tests
add_library(doctest INTERFACE)
add_executable(cpptests tests/cpptests.cpp)
target_include_directories(doctest INTERFACE external/doctest)
target_link_libraries(cpptests PRIVATE doctest)
