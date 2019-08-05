cmake_minimum_required(VERSION 3.11)
project(delphi)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(Boost_NO_BOOST_CMAKE ON)
# Add and link pybind11 modules
find_package(Boost REQUIRED)
include_directories(lib)
include_directories(external)
include_directories(${Boost_INCLUDE_DIR})
include_directories(external/nlohmann-json)
find_package(PythonInterp REQUIRED)
set(PYBIND11_PYTHON_VERSION "${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
set(PYBIND11_CPP_STANDARD -std=c++17)
add_subdirectory(external/pybind11)
pybind11_add_module(kde lib/kde_wrapper.cpp)
pybind11_add_module(AnalysisGraph lib/AnalysisGraph_wrapper.cpp)
target_link_libraries(AnalysisGraph PRIVATE ${Boost_LIBRARIES})
target_link_libraries(kde PRIVATE ${Boost_LIBRARIES})

if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  include_directories("/usr/include/eigen3")
endif()

find_package(SQLite3 REQUIRED)

if (NOT ${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  target_link_libraries (AnalysisGraph PRIVATE Eigen3::Eigen)
endif()

include_directories(${SQLITE3_INCLUDE_DIRS})
target_link_libraries (AnalysisGraph PRIVATE ${SQLITE3_LIBRARIES})


# Tests
add_library(doctest INTERFACE)
add_executable(cpptests tests/cpptests.cpp)
target_include_directories(doctest INTERFACE external/doctest)
target_link_libraries(cpptests PRIVATE doctest)
