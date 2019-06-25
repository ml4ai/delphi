#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "AnalysisGraph.cpp"

namespace py = pybind11;

PYBIND11_MODULE(AnalysisGraph, m) {
  py::class_<AnalysisGraph>(m, "AnalysisGraph")
      .def("from_json_file", &AnalysisGraph::from_json_file)
      .def("print_nodes", &AnalysisGraph::print_nodes)
      .def("to_dot", &AnalysisGraph::to_dot)
      .def("construct_beta_pdfs", &AnalysisGraph::construct_beta_pdfs);
}
