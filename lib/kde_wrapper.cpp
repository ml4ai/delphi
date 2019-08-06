#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kde.hpp"

using std::vector;

namespace py = pybind11;

PYBIND11_MODULE(kde, m) {
  py::class_<KDE>(m, "KDE")
      .def(py::init<vector<double>>())
      .def("resample", &KDE::resample)
      .def("pdf", py::overload_cast<double>(&KDE::pdf),
           "Evaluate pdf for a single value")
      .def("pdf", py::overload_cast<vector<double>>(&KDE::pdf),
           "Evaluate pdf for a list of values")
      .def("logpdf", &KDE::logpdf)
      .def_readwrite("dataset", &KDE::dataset)
      .def(py::pickle([](KDE &kde) { return py::make_tuple(kde.dataset); },
                      [](py::tuple t) {
                        if (t.size() != 1)
                          throw std::runtime_error("Invalid state!");

                        KDE kde(t[0].cast<vector<double>>());
                        return kde;
                      }));
}
