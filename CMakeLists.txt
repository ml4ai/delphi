cmake_minimum_required(VERSION 3.11)
project(delphi)

# Uncomment this line to generate debug code
set(CMAKE_BUILD_TYPE Debug)

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
find_package(Boost COMPONENTS graph program_options REQUIRED)
include_directories(lib)
include_directories(external)
include_directories(${Boost_INCLUDE_DIR})

find_package(range-v3 REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(fmt REQUIRED)
find_package(nlohmann_json REQUIRED)

option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)

# Optional building of Python bindings using Pybind11
if(BUILD_PYTHON_BINDINGS)

    find_package(pybind11 CONFIG REQUIRED)
    message(STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIRS}")
    find_package(Python3 REQUIRED)
    set(PYBIND11_PYTHON_VERSION "${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
    set(PYBIND11_CPP_STANDARD -std=c++17)
endif()

file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/lib/*.cpp)
find_package(SQLite3 REQUIRED)

add_library(Delphi
    lib/AnalysisGraph.hpp
    lib/causemos_integration.cpp
    lib/constructors.cpp
    lib/CSVWriter.hpp
    lib/database.cpp
    lib/DatabaseHelper.cpp
    lib/DatabaseHelper.hpp
    lib/data.cpp
    lib/data.hpp
    #lib/DelphiPython.cpp
    lib/DiGraph.hpp
    lib/Edge.hpp
    lib/exceptions.hpp
    lib/format_output.cpp
    lib/graph_building.cpp
    lib/graph_modification.cpp
    lib/graph_utils.cpp
    lib/graphviz.cpp
    lib/graphviz_interface.cpp
    lib/graphviz_interface.hpp
    lib/head_nodes.cpp
    lib/Indicator.cpp
    lib/Indicator.hpp
    lib/indicator_manipulation.cpp
    lib/KDE.cpp
    lib/KDE.hpp
    lib/Node.hpp
    lib/parameter_initialization.cpp
    lib/prediction.cpp
    lib/printing.cpp
    lib/PybindTester.hpp
    lib/Random_Variables.cpp
    lib/Random_Variables.hpp
    lib/rng.cpp
    lib/rng.hpp
    lib/sampling.cpp
    lib/sandbox.cpp
    lib/subgraphs.cpp
    lib/synthetic_data.cpp
    lib/Timer.hpp
    lib/to_json.cpp
    lib/train_model.cpp
    lib/Tran_Mat_Cell.cpp
    lib/Tran_Mat_Cell.hpp
    lib/utils.cpp
    lib/utils.hpp
)

target_link_libraries(Delphi
                      PRIVATE ${Boost_LIBRARIES}
                              fmt::fmt
                              SQLite::SQLite3
                              nlohmann_json::nlohmann_json
                              Eigen3::Eigen
                              ${GRAPHVIZ_LIBS})
target_link_libraries(Delphi INTERFACE range-v3)

if(BUILD_PYTHON_BINDINGS)

    pybind11_add_module(DelphiPython MODULE NO_EXTRAS lib/DelphiPython.cpp)
    target_link_libraries(DelphiPython PRIVATE Delphi Eigen3::Eigen)
    add_custom_command(TARGET DelphiPython POST_BUILD
        COMMAND echo ""
        COMMAND echo "Copying..."
        COMMAND ls DelphiPython*
        COMMAND echo "to ${PROJECT_SOURCE_DIR}/delphi/cpp/"
        COMMAND echo ""
        COMMAND rm -rf ${PROJECT_SOURCE_DIR}/delphi/cpp/DelphiPython*
        COMMAND cp DelphiPython* ${PROJECT_SOURCE_DIR}/delphi/cpp/
    )
#add_executable(pybind_tester_driver apps/pybind_tester_driver.cpp)

endif()

# Find the served library
find_package(PkgConfig REQUIRED)
pkg_check_modules(SERVED REQUIRED IMPORTED_TARGET served)

# Executables
add_executable(create_model apps/create_model.cpp)
target_link_libraries(create_model PRIVATE Delphi Eigen3::Eigen)
add_executable(sandbox_tester apps/sandbox_tester.cpp)
target_link_libraries(sandbox_tester PRIVATE Delphi Eigen3::Eigen)
add_executable(timer
        scripts/timer.cpp
        lib/Timer.hpp
        lib/CSVWriter.hpp)
target_link_libraries(timer PRIVATE Delphi Eigen3::Eigen ${Boost_LIBRARIES})

# Build the REST API program
add_executable(delphi_rest_api apps/delphi_rest_api.cpp)
target_link_libraries(delphi_rest_api PRIVATE Delphi PkgConfig::SERVED)
target_link_libraries(delphi_rest_api PRIVATE Eigen3::Eigen pthread)
