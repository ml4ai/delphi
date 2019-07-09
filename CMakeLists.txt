cmake_minimum_required(VERSION 3.11)
project(delphi)

# Conan setup
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

# Add and link pybind11 modules
include_directories(lib)
include_directories(external)
set(PYBIND11_CPP_STANDARD -std=c++17)
set(PYBIND11_PYTHON_VERSION 3.7)
add_subdirectory(external/pybind11)
pybind11_add_module(kde lib/kde_wrapper.cpp)
pybind11_add_module(AnalysisGraph lib/AnalysisGraph_wrapper.cpp)
target_link_libraries(AnalysisGraph PUBLIC ${CONAN_LIBS})
target_link_libraries(kde PUBLIC ${CONAN_LIBS})

# Tests
add_executable(cpptests tests/cpptests.cpp)
target_link_libraries(cpptests PUBLIC ${CONAN_LIBS})
