#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "AnalysisGraph.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace std;

PYBIND11_MODULE(DelphiPython, m) {
  py::enum_<InitialBeta>(m, "InitialBeta")
      .value("ZERO", InitialBeta::ZERO)
      .value("ONE", InitialBeta::ONE)
      .value("HALF", InitialBeta::HALF)
      .value("MEAN", InitialBeta::MEAN)
      .value("RANDOM", InitialBeta::RANDOM);

  py::class_<AnalysisGraph>(m, "AnalysisGraph")
      .def_readwrite("id", &AnalysisGraph::id)
      .def("to_json_string", &AnalysisGraph::to_json_string, "indent"_a = 0)
      .def("from_json_string", &AnalysisGraph::from_json_string)
      .def("generate_create_model_response", &AnalysisGraph::generate_create_model_response)
      .def("get_edge_weights_for_causemos_viz", &AnalysisGraph::get_edge_weights_for_causemos_viz)
      .def_readwrite("data_heuristic", &AnalysisGraph::data_heuristic)
      .def_readwrite("res", &AnalysisGraph::res)
      .def_property("s0",
                    &AnalysisGraph::get_initial_latent_state,
                    &AnalysisGraph::set_initial_latent_state)
      .def_static("from_indra_statements_json_file",
                  &AnalysisGraph::from_indra_statements_json_file,
                  "filename"_a,
                  "belief_score_cutoff"_a = 0.9,
                  "grounding_score_cutoff"_a = 0.0,
                  "ontology"_a = "WM")
      .def_static("from_causemos_json_string",
                  &AnalysisGraph::from_causemos_json_string)
      .def_static("from_causemos_json_file",
                  &AnalysisGraph::from_causemos_json_file)
      .def_static("from_causal_fragments",
                  &AnalysisGraph::from_causal_fragments,
                  "causal_fragments"_a)
      .def("__len__", &AnalysisGraph::num_vertices)
      .def("__getitem__", [](AnalysisGraph& G, string name) { return G[name]; })
      .def("__getitem__",
           [](AnalysisGraph& G, int node_index) { return G[node_index]; })
      .def("get_subgraph_for_concept",
           &AnalysisGraph::get_subgraph_for_concept,
           "concept"_a,
           "inward"_a = false,
           "depth"_a = -1)
      .def("get_subgraph_for_concept_pair",
           &AnalysisGraph::get_subgraph_for_concept_pair,
           "source_concept"_a,
           "target_concept"_a,
           "cutoff"_a = -1)
      .def("prune", &AnalysisGraph::prune, "cutoff"_a = 2)
      .def("merge_nodes",
           &AnalysisGraph::merge_nodes,
           "concept_1"_a,
           "concept_2"_a,
           "same_polarity"_a = true)
      .def(
          "__iter__",
          [](AnalysisGraph& g) {
            return py::make_iterator(g.begin(), g.end());
          },
          py::keep_alive<0, 1>())
      .def("num_vertices", &AnalysisGraph::num_vertices)
      .def("num_edges", &AnalysisGraph::num_edges)
      .def("edge", py::overload_cast<string, string>(&AnalysisGraph::edge))
      .def("print_nodes", &AnalysisGraph::print_nodes)
      .def("print_edges", &AnalysisGraph::print_edges)
      .def("print_name_to_vertex", &AnalysisGraph::print_name_to_vertex)
      .def("to_dot", &AnalysisGraph::to_dot)
      .def("to_png",
           &AnalysisGraph::to_png,
           "filename"_a = "CAG.png",
           "simplified_labels"_a = true,
           "label_depth"_a = 1,
           "node_to_highlight"_a = "",
           "rankdir"_a = "TB")
      .def("add_node", &AnalysisGraph::add_node, "concept"_a)
      .def("remove_node",
           py::overload_cast<string>(&AnalysisGraph::remove_node),
           "concept"_a)
      .def("remove_nodes", &AnalysisGraph::remove_nodes, "concepts"_a)
      .def("add_edge",
           py::overload_cast<CausalFragment>(&AnalysisGraph::add_edge),
           "causal_fragment"_a)
      .def("change_polarity_of_edge",
           &AnalysisGraph::change_polarity_of_edge,
           "source_concept"_a,
           "source_polarity"_a,
           "target_concept"_a,
           "target_polarity"_a)
      .def("remove_edge", &AnalysisGraph::remove_edge, "source"_a, "target"_a)
      .def("remove_edges", &AnalysisGraph::remove_edges, "edges"_a)
      .def("find_all_paths", &AnalysisGraph::find_all_paths)
      .def("print_all_paths", &AnalysisGraph::print_all_paths)
      .def("print_cells_affected_by_beta",
           &AnalysisGraph::print_cells_affected_by_beta,
           "source"_a,
           "target"_a)
      .def("print_name_to_vertex", &AnalysisGraph::print_name_to_vertex)
      .def("map_concepts_to_indicators",
           &AnalysisGraph::map_concepts_to_indicators,
           "n"_a = 1,
           "country"_a = "South Sudan")
      .def("print_indicators", &AnalysisGraph::print_indicators)
      .def("set_indicator",
           &AnalysisGraph::set_indicator,
           "concept"_a,
           "indicator"_a,
           "source"_a)
      .def("delete_indicator",
           &AnalysisGraph::delete_indicator,
           "concept"_a,
           "indicator"_a)
      .def("delete_all_indicators",
           &AnalysisGraph::delete_all_indicators,
           "concept"_a)
      .def("test_inference_with_synthetic_data",
           &AnalysisGraph::test_inference_with_synthetic_data,
           "start_year"_a = 2015,
           "start_month"_a = 1,
           "end_year"_a = 2015,
           "end_month"_a = 12,
           "res"_a = 100,
           "burn"_a = 900,
           "country"_a = "South Sudan",
           "state"_a = "",
           "county"_a = "",
           py::arg("units") = map<std::string, std::string>{},
           "initial_beta"_a = InitialBeta::HALF)
      .def("train_model",
           &AnalysisGraph::train_model,
           "start_year"_a = 2012,
           "start_month"_a = 1,
           "end_year"_a = 2017,
           "end_month"_a = 12,
           "res"_a = 200,
           "burn"_a = 10000,
           "country"_a = "South Sudan",
           "state"_a = "",
           "county"_a = "",
           py::arg("units") = map<std::string, std::string>{},
           "initial_beta"_a = InitialBeta::ZERO,
           "use_heuristic"_a = false,
           "use_continuous"_a = true)
      .def("generate_prediction",
           &AnalysisGraph::generate_prediction,
           "start_year"_a,
           "start_month"_a,
           "end_year"_a,
           "end_month"_a)
      .def("generate_causemos_projection",
           &AnalysisGraph::generate_causemos_projection,
           "json_projection"_a)
      .def("prediction_to_array",
           &AnalysisGraph::prediction_to_array,
           "indicator"_a)
      .def("set_derivative", &AnalysisGraph::set_derivative)
      .def("set_default_initial_state",
           &AnalysisGraph::set_default_initial_state)
      .def("set_random_seed", &AnalysisGraph::set_random_seed);

  py::class_<RV>(m, "RV")
      .def(py::init<std::string>())
      .def("sample", &RV::sample);

  py::class_<Indicator, RV>(m, "Indicator")
      .def("__repr__", &Indicator::get_name)
      .def_readwrite("name", &Indicator::name)
      .def("set_source", &Indicator::set_source)
      .def("set_unit", &Indicator::set_unit)
      .def("set_mean", &Indicator::set_mean)
      .def("set_value", &Indicator::set_value)
      .def("set_stdev", &Indicator::set_stdev)
      .def("set_time", &Indicator::set_time)
      .def("set_aggaxes", &Indicator::set_aggaxes)
      .def("set_aggregation_method", &Indicator::set_aggregation_method)
      .def("set_timeseries", &Indicator::set_timeseries)
      .def("set_samples", &Indicator::set_samples)
      .def("get_source", &Indicator::get_source)
      .def("get_unit", &Indicator::get_unit)
      .def("get_mean", &Indicator::get_mean)
      .def("get_value", &Indicator::get_value)
      .def("get_stdev", &Indicator::get_stdev)
      .def("get_time", &Indicator::get_time)
      .def("get_aggaxes", &Indicator::get_aggaxes)
      .def("get_aggregation_method", &Indicator::get_aggregation_method)
      .def("get_timeseries", &Indicator::get_timeseries)
      .def("get_samples", &Indicator::get_samples);

  py::class_<Node>(m, "Node")
      .def_readwrite("name", &Node::name)
      .def("__repr__", &Node::to_string)
      .def("get_indicator", &Node::get_indicator)
      .def("replace_indicator", &Node::replace_indicator)
      .def_readwrite("indicators", &Node::indicators);

  py::class_<Edge>(m, "Edge")
      .def_readwrite("evidence", &Edge::evidence)
      .def_readwrite("kde", &Edge::kde);

  py::class_<Statement>(m, "Statement")
      .def_readwrite("subject", &Statement::subject)
      .def_readwrite("object", &Statement::subject);

  py::class_<KDE>(m, "KDE")
      .def(py::init<vector<double>>())
      .def("pdf",
           py::overload_cast<double>(&KDE::pdf),
           "Evaluate pdf for a single value")
      .def("pdf",
           py::overload_cast<vector<double>>(&KDE::pdf),
           "Evaluate pdf for a list of values")
      .def("logpdf", &KDE::logpdf)
      .def_readwrite("dataset", &KDE::dataset)
      .def(py::pickle([](KDE& kde) { return py::make_tuple(kde.dataset); },
                      [](py::tuple t) {
                        if (t.size() != 1)
                          throw std::runtime_error("Invalid state!");

                        KDE kde(t[0].cast<vector<double>>());
                        return kde;
                      }));
}
