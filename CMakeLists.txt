cmake_minimum_required(VERSION 2.8.12)
project(delphi)


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/delphi/cpp)
set(PYBIND11_CPP_STANDARD -std=c++17)
set(PYBIND11_PYTHON_VERSION 3.6)
add_subdirectory(external/pybind11)
include_directories(lib)
include_directories(external)
pybind11_add_module(kde lib/kde_wrapper.cpp)
pybind11_add_module(AnalysisGraph lib/AnalysisGraph_wrapper.cpp)

# Finding and linking SQLite3
include(FindPackageHandleStandardArgs)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}"/cmake")
find_package(SQLite3)
if (SQLITE3_FOUND)
  include_directories(${SQLITE3_INCLUDE_DIRS})
  target_link_libraries (AnalysisGraph PUBLIC ${SQLITE3_LIBRARIES})
endif (SQLITE3_FOUND)
