cmake_minimum_required(VERSION 3.11)
project(delphi)

# Uncomment this line to generate debug code
#set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(Boost_NO_BOOST_CMAKE ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden -pthread -g")

find_package(Graphviz)

# Add MacOs Boost directories
SET(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} "/opt/local/libexec/boost/1.76/include/boost")
SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} "/opt/local/libexec/boost/1.76/lib")

#find_package(OpenMP)

# Disable multi-threading in Eigen as it significantly slows down matrix exponential calculation
add_definitions(-DEIGEN_DONT_PARALLELIZE)

# Add and link pybind11 modules
find_package(Boost COMPONENTS graph program_options REQUIRED)

find_package(range-v3 REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(fmt REQUIRED)
find_package(nlohmann_json REQUIRED)

# To micro benchmark the MCMC loop compile as:
# cmake -DTIME=ON ..
option(TIME "Micro benchmark the MCMC loop" OFF)
if (TIME)
    add_definitions(-DTIME)
    message(STATUS "Timing - on")
endif()

# To turn on and off multithreading compile as:
# cmake -DMULTI_THREADING=OFF ..
option(MULTI_THREADING "Turning on multithreading" ON)
if (MULTI_THREADING)
    add_definitions(-DMULTI_THREADING)
    message(STATUS "Multithreading - on")
endif()

option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)

# Optional building of Python bindings using Pybind11
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 CONFIG REQUIRED)
    message(STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIRS}")
    find_package(Python3 REQUIRED)
    set(PYBIND11_PYTHON_VERSION "${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
    set(PYBIND11_CPP_STANDARD -std=c++17)
endif()

find_package(SQLite3 REQUIRED)

add_library(Delphi
  lib/parameter_initialization.cpp
  lib/to_json.cpp
  lib/data.cpp
  lib/graphviz_interface.cpp
  lib/graphviz.cpp
  lib/KDE.cpp
  lib/Random_Variables.cpp
  lib/rng.cpp
  lib/train_model.cpp
  lib/Tran_Mat_Cell.cpp
  lib/utils.cpp
  lib/Indicator.cpp
  lib/subgraphs.cpp
  lib/constructors.cpp
  lib/causemos_integration.cpp
  lib/sampling.cpp
  lib/prediction.cpp
  lib/synthetic_data.cpp
  lib/graph_building.cpp
  lib/graph_modification.cpp
  lib/graph_utils.cpp
  lib/printing.cpp
  lib/indicator_manipulation.cpp
  lib/sandbox.cpp
  lib/format_output.cpp
  lib/database.cpp
  lib/DatabaseHelper.cpp
  lib/head_nodes.cpp
  lib/TrainingStatus.cpp
  lib/profiler.cpp
  lib/CSVWriter.hpp)

target_link_libraries(Delphi
                      PRIVATE ${Boost_LIBRARIES}
                              fmt::fmt
                              SQLite::SQLite3
                              nlohmann_json::nlohmann_json
                              Eigen3::Eigen
                              ${GRAPHVIZ_LIBS})
target_link_libraries(Delphi INTERFACE range-v3)
target_include_directories(Delphi PRIVATE ${GRAPHVIZ_INCLUDE_DIRS} lib external)
#if(OpenMP_CXX_FOUND)
#    target_link_libraries(Delphi PUBLIC OpenMP::OpenMP_CXX)
#endif()

if(BUILD_PYTHON_BINDINGS)

    pybind11_add_module(DelphiPython MODULE NO_EXTRAS lib/DelphiPython.cpp)
    target_link_libraries(DelphiPython PRIVATE Delphi Eigen3::Eigen)
    target_include_directories(DelphiPython PRIVATE ${GRAPHVIZ_INCLUDE_DIRS})
    add_custom_command(TARGET DelphiPython POST_BUILD
        COMMAND echo ""
        COMMAND echo "Copying..."
        COMMAND ls DelphiPython*
        COMMAND echo "to ${PROJECT_SOURCE_DIR}/delphi/cpp/"
        COMMAND echo ""
        COMMAND rm -rf ${PROJECT_SOURCE_DIR}/delphi/cpp/DelphiPython*
        COMMAND cp DelphiPython* ${PROJECT_SOURCE_DIR}/delphi/cpp/
    )

endif()

# Find the served library
find_package(PkgConfig REQUIRED)
pkg_check_modules(SERVED REQUIRED IMPORTED_TARGET served)

# Executables
add_executable(create_model apps/create_model.cpp)
target_link_libraries(create_model PRIVATE Delphi Eigen3::Eigen)
target_include_directories(create_model PRIVATE ${Boost_INCLUDE_DIR} lib external ${GRAPHVIZ_INCLUDE_DIRS})

## Sandbox tester
add_executable(sandbox_tester apps/sandbox_tester.cpp)
target_link_libraries(sandbox_tester PRIVATE Delphi Eigen3::Eigen)
target_include_directories(sandbox_tester PRIVATE lib ${GRAPHVIZ_INCLUDE_DIRS})

## Timer program
add_executable(timer
        apps/timer.cpp
        lib/Timer.hpp
        lib/CSVWriter.hpp)
target_link_libraries(timer PRIVATE Delphi Eigen3::Eigen ${Boost_LIBRARIES})
target_include_directories(timer PRIVATE ${Boost_INCLUDE_DIR} lib external ${GRAPHVIZ_INCLUDE_DIRS})

## Matirx Exponential Timer program
add_executable(mat_exp_timer
        apps/mat_exp_timer.cpp
        lib/Timer.hpp
        lib/CSVWriter.hpp)
target_link_libraries(mat_exp_timer PRIVATE Delphi Eigen3::Eigen ${Boost_LIBRARIES})
target_include_directories(mat_exp_timer PRIVATE ${Boost_INCLUDE_DIR} lib external ${GRAPHVIZ_INCLUDE_DIRS})

## Program to check whether Delphi is compiled with OpenMP
add_executable(check_OpenMP apps/check_OpenMP.cpp)
target_link_libraries(check_OpenMP PRIVATE Delphi Eigen3::Eigen ${Boost_LIBRARIES})
target_include_directories(check_OpenMP PRIVATE ${Boost_INCLUDE_DIR} lib external ${GRAPHVIZ_INCLUDE_DIRS})

# Build the REST API program
add_executable(delphi_rest_api apps/delphi_rest_api.cpp)
target_link_libraries(delphi_rest_api PRIVATE Delphi PkgConfig::SERVED Eigen3::Eigen pthread)
target_include_directories(delphi_rest_api PRIVATE ${Boost_INCLUDE_DIR} lib external ${GRAPHVIZ_INCLUDE_DIRS})
