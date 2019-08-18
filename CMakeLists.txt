cmake_minimum_required(VERSION 3.11)
project(delphi)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(Boost_NO_BOOST_CMAKE ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden")


find_package(Graphviz)

include_directories(${GRAPHVIZ_INCLUDE_DIRS})
# Add and link pybind11 modules
find_package(Boost REQUIRED)
include_directories(lib)
include_directories(external)
include_directories(${Boost_INCLUDE_DIR})

include_directories(external/nlohmann-json)
include_directories(external/cppitertools)

add_subdirectory(external/fmt)
include_directories(external/fmt/include)

find_package(PythonInterp REQUIRED)

set(PYBIND11_PYTHON_VERSION "${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
set(PYBIND11_CPP_STANDARD -std=c++17)
add_subdirectory(external/pybind11)

file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/lib/*.cpp)

message("-- Using Eigen3 version 3.3 bundled with Delphi")
include_directories(external/eigen)

find_package(SQLite3 REQUIRED)
include_directories(${SQLITE3_INCLUDE_DIRS})

add_library(Delphi ${SRC_FILES})
target_link_libraries(Delphi
                      PRIVATE ${Boost_LIBRARIES}
                              fmt::fmt
                              "sqlite3"
                              ${GRAPHVIZ_LIBS}
                              )


# Tests
add_library(doctest INTERFACE)
add_executable(cpptests tests/cpptests.cpp)
target_include_directories(doctest INTERFACE external/doctest)
target_link_libraries(cpptests PRIVATE doctest Delphi)

pybind11_add_module(DelphiPython MODULE lib/wrappers/AnalysisGraph_wrapper.cpp)
target_link_libraries(DelphiPython PRIVATE Delphi)

# Executables
add_executable(create_model apps/create_model.cpp)
target_link_libraries(create_model PRIVATE Delphi)
