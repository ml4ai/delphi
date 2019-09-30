// Usage: ./create_model <path_to_JSON_file>

#include "AnalysisGraph.hpp"
#include "spdlog/spdlog.h"
#include <boost/program_options.hpp>

int main(int argc, char* argv[]) {
  using namespace std;
  using namespace boost::program_options;
  using namespace spdlog;

  options_description desc("Allowed options");
  positional_options_description pd;

  // Path to JSON-serialized INDRA statements
  string stmts;
  string cag_png_filename;

  desc.add_options()("help,h", "Executable for creating Delphi models")(
      "stmts,i",
      value<string>(&stmts),
      "Path to input JSON-serialized INDRA statements")(
      "belief_score_cutoff,b",
      value<double>()->default_value(0.9),
      "INDRA belief score cutoff for statements to be included in the model.")(
      "grounding_score_cutoff,g",
      value<double>()->default_value(0.7),
      "Grounding score cutoff for statements to be included in the model")(
      "draw_graph,d",
      bool_switch()->default_value(false),
      "Draw causal analysis graph using Graphviz")(
      "quantify,q",
      bool_switch()->default_value(false),
      "Quantify graph with probability distributions")(
      "train_model,t", bool_switch()->default_value(false), "Train model")(
      "map_concepts",
      bool_switch()->default_value(false),
      "Map concepts to indicators")(
      "simplified_labels",
      bool_switch()->default_value(true),
      "Use simplified node labels without showing the whole ontology path.")(
      "cag_filename,o",
      value<string>(&cag_png_filename)->default_value("CAG.png"),
      "Filename for the output visualized CAG")(
      "label_depth",
      value<int>()->default_value(1),
      "Ontology depth for simplified labels in CAG visualization");

  // Setting positional arguments
  pd.add("stmts", 1);
  pd.add("belief_score_cutoff", 2);
  pd.add("grounding_score_cutoff", 3);

  variables_map vm;
  store(parse_command_line(argc, argv, desc), vm);
  notify(vm);

  if (vm.count("help") || argc == 1) {
    cout << desc << endl;
    return 1;
  }

  RNG* R = RNG::rng();
  R->set_seed(87);

  set_level(spdlog::level::debug);
  auto G =
      AnalysisGraph::from_json_file(stmts,
                                    vm["belief_score_cutoff"].as<double>(),
                                    vm["grounding_score_cutoff"].as<double>());

  debug("Number of vertices: {}", G.num_vertices());
  debug("Number of edges: {}", G.num_edges());

  if (vm["map_concepts"].as<bool>()) {
    G.map_concepts_to_indicators();
  }
  if (vm["quantify"].as<bool>()) {
    G.construct_beta_pdfs();
  }
  if (vm["draw_graph"].as<bool>() == true) {
    G.to_png(vm["cag_filename"].as<string>(),
             vm["simplified_labels"].as<bool>(),
             vm["label_depth"].as<int>());
  }
  if (vm["train_model"].as<bool>()) {
    G.train_model(2015, 1, 2015, 12, 100, 900);
  }
  return EXIT_SUCCESS;
}
