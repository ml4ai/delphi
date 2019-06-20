#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "AnalysisGraph.cpp"

namespace py = pybind11;

PYBIND11_MODULE(extension, m) {
  py::class_<AnalysisGraph>(m, "AnalysisGraph")
    .def("from_json_file", &AnalysisGraph::from_json_file)
    .def("print_nodes", &AnalysisGraph::print_nodes)
    .def("to_dot", &AnalysisGraph::to_dot)
    .def("construct_beta_pdfs", &AnalysisGraph::construct_beta_pdfs)
  ;
  py::class_<KDE>(m, "KDE")
    .def(py::init<vector<double> >())
    .def("resample", &KDE::resample)
    .def("evaluate", &KDE::evaluate)
  ;
}
