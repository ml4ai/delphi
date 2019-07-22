#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "AnalysisGraph.cpp"

namespace py = pybind11;

PYBIND11_MODULE(AnalysisGraph, m) {
  py::class_<AnalysisGraph>(m, "AnalysisGraph")
      .def("from_json_file", &AnalysisGraph::from_json_file)
      .def("from_statements", &AnalysisGraph::from_statements)
      .def("print_nodes", &AnalysisGraph::print_nodes)
      .def("print_edges", &AnalysisGraph::print_edges)
      .def("print_name_to_vertex", &AnalysisGraph::print_name_to_vertex)
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
      .def("sample_from_posterior", &AnalysisGraph::sample_from_posterior, py::return_value_policy::reference_internal)
      .def("sample_from_proposal_debug", &AnalysisGraph::sample_from_proposal_debug, py::return_value_policy::reference_internal)
      .def("set_initial_state", &AnalysisGraph::set_initial_state)
      .def("get_beta", &AnalysisGraph::get_beta)
      .def("take_step", &AnalysisGraph::take_step)
      .def("print_name_to_vertex", &AnalysisGraph::print_name_to_vertex)
      .def("map_concepts_to_indicators", &AnalysisGraph::map_concepts_to_indicators)
      .def("print_indicators", &AnalysisGraph::print_indicators)
    ;
}
