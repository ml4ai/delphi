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
find_package(Boost COMPONENTS program_options REQUIRED)
include_directories(lib)
include_directories(external)
include_directories(${Boost_INCLUDE_DIR})

include_directories(external/nlohmann-json)
include_directories(external/cppitertools)

option(USE_BUNDLED_FMT "Use fmtlib bundled with Delphi" ON)
option(USE_BUNDLED_SPDLOG "Use spdlog bundled with Delphi" ON)

if(USE_BUNDLED_FMT)
  include_directories(external/fmt/include)
  add_subdirectory(external/fmt)
else()
  find_package(fmt REQUIRED)
endif()


if(USE_BUNDLED_SPDLOG)
  add_subdirectory(external/spdlog)
  include_directories(external/spdlog/include)
else()
  find_package(spdlog 1.4.1 REQUIRED)
endif()

find_package(Python3 REQUIRED)

set(PYBIND11_PYTHON_VERSION "${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
set(PYBIND11_CPP_STANDARD -std=c++17)
add_subdirectory(external/pybind11)

file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/lib/*.cpp)

message("-- Using Eigen3 version 3.3 bundled with Delphi")
include_directories(external/eigen)

find_package(SQLite3 REQUIRED)

add_library(Delphi
  lib/AnalysisGraph.hpp
  lib/construct_beta_pdfs.cpp
  lib/data.cpp
  lib/data.hpp
  lib/DiGraph.hpp
  lib/graphviz_interface.cpp
  lib/graphviz_interface.hpp
  lib/graphviz.cpp
  lib/kde.cpp
  lib/kde.hpp
  lib/random_variables.cpp
  lib/random_variables.hpp
  lib/rng.cpp
  lib/rng.hpp
  lib/train_model.cpp
  lib/tran_mat_cell.hpp
  lib/utils.cpp
  lib/utils.hpp
  lib/Node.hpp
  lib/Edge.hpp
  lib/Indicator.hpp
  lib/Indicator.cpp
  lib/subgraphs.cpp
  lib/constructors.cpp
  lib/sampling.cpp
  lib/inference.cpp
  lib/synthetic_data.cpp
  lib/graph_building.cpp
  lib/graph_modification.cpp
  lib/graph_utils.cpp
  lib/printing.cpp
  lib/indicator_manipulation.cpp
)

target_include_directories(Delphi PRIVATE external/range-v3/include)
target_link_libraries(Delphi
                      PRIVATE ${Boost_LIBRARIES}
                              fmt::fmt
                              spdlog::spdlog
                              SQLite::SQLite3
                              ${GRAPHVIZ_LIBS})

# Tests
add_library(doctest INTERFACE)
add_executable(cpptests tests/cpptests.cpp)
target_include_directories(doctest INTERFACE external/doctest)
target_link_libraries(cpptests PRIVATE doctest Delphi)

pybind11_add_module(DelphiPython MODULE lib/DelphiPython.cpp)
target_link_libraries(DelphiPython PRIVATE Delphi)

# Executables
add_executable(create_model apps/create_model.cpp)
target_link_libraries(create_model PRIVATE Delphi)
