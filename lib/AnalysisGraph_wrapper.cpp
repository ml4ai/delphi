#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "AnalysisGraph.cpp"

namespace py = pybind11;

PYBIND11_MODULE(AnalysisGraph, m) {
  py::class_<AnalysisGraph>(m, "AnalysisGraph")
      .def("from_json_file", &AnalysisGraph::from_json_file)
      .def("print_nodes", &AnalysisGraph::print_nodes)
      .def("print_edges", &AnalysisGraph::print_edges)
      .def("to_dot", &AnalysisGraph::to_dot)
      .def("construct_beta_pdfs", &AnalysisGraph::construct_beta_pdfs)
      .def("sample_from_prior", &AnalysisGraph::sample_from_prior)
      .def("add_node", &AnalysisGraph::add_node)
      .def("add_edge", &AnalysisGraph::add_edge)
      .def("find_all_paths", &AnalysisGraph::find_all_paths)
      .def("print_all_paths", &AnalysisGraph::print_all_paths)
      //.def("simple_paths", &AnalysisGraph::simple_paths)
      .def("print_cells_affected_by_beta", &AnalysisGraph::print_cells_affected_by_beta)
      .def("initialize", &AnalysisGraph::initialize, py::return_value_policy::reference_internal)
      .def("sample_from_prior", &AnalysisGraph::sample_from_prior, py::return_value_policy::reference_internal)
      .def("sample_from_likelihood", &AnalysisGraph::sample_from_likelihood, py::return_value_policy::reference_internal)
    ;
}
