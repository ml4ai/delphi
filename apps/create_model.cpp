// Usage: ./create_model <path_to_JSON_file>

#include "AnalysisGraph.hpp"
#include "rng.hpp"
#include "spdlog/spdlog.h"
#include <boost/program_options.hpp>

using namespace std;
using namespace boost::program_options;
using namespace spdlog;

/* Function used to check that 'opt1' and 'opt2' are not specified
   at the same time. */
void conflicting_options(const variables_map& vm,
                         const char* opt1,
                         const char* opt2) {
  if (vm.count(opt1) && !vm[opt1].defaulted() && vm.count(opt2) &&
      !vm[opt2].defaulted())
    throw logic_error(string("Conflicting options '") + opt1 + "' and '" +
                      opt2 + "'.");
}
int main(int argc, char* argv[]) {

  options_description general_options("Allowed options");
  options_description graph_drawing_options("Options for graph drawing");
  positional_options_description pd;

  // Path to JSON-serialized INDRA statements
  string indra_json_file;
  string uncharted_json_file;
  string cag_png_filename;
  string country;
  string rankdir;

  graph_drawing_options.add_options()(
      "draw_graph,d",
      bool_switch()->default_value(false),
      "Draw causal analysis graph using Graphviz")(
      "rankdir",
      value<string>(&rankdir)->default_value("TB"),
      "Direction of graph layout ['LR' | 'TB'] ");

  general_options.add_options()("help,h",
                                "Executable for creating Delphi models")(
      "stmts,i",
      value<string>(&indra_json_file),
      "Path to input JSON-serialized INDRA statements")(
      "belief_score_cutoff,b",
      value<double>()->default_value(0.9),
      "INDRA belief score cutoff for statements to be included in the model.")(
      "grounding_score_cutoff,g",
      value<double>()->default_value(0.7),
      "Grounding score cutoff for statements to be included in the model")(
      "quantify,q",
      bool_switch()->default_value(false),
      "Quantify graph with probability distributions")(
      "train_model,t", bool_switch()->default_value(false), "Train model")(
      "n_indicators",
      value<int>()->default_value(1),
      "Number of indicators to map to each concept")(
      "map_concepts",
      bool_switch()->default_value(false),
      "Map concepts to indicators")(
      "simplified_labels",
      bool_switch()->default_value(false),
      "Use simplified node labels without showing the whole ontology path.")(
      "country",
      value<string>(&country)->default_value("South Sudan"),
      "The country for which the model is being built")(
      "cag_filename,o",
      value<string>(&cag_png_filename)->default_value("CAG.png"),
      "Filename for the output visualized CAG")(
      "label_depth",
      value<int>()->default_value(1),
      "Ontology depth for simplified labels in CAG visualization")(
      "causemos_json",
      value<string>(&uncharted_json_file),
      "Path to CauseMos JSON file");

  // Setting positional arguments
  pd.add("stmts", 1);
  pd.add("belief_score_cutoff", 2);
  pd.add("grounding_score_cutoff", 3);

  variables_map vm;
  options_description all_options("Allowed options");
  all_options.add(general_options).add(graph_drawing_options);
  store(parse_command_line(argc, argv, all_options), vm);
  notify(vm);

  if (vm.count("help") || argc == 1) {
    cout << all_options << endl;
    return 0;
  }

  conflicting_options(vm, "stmts", "causemos_json");

  RNG* R = RNG::rng();
  R->set_seed(87);

  set_level(spdlog::level::debug);
  AnalysisGraph G;

  if (vm.count("stmts")) {
    G = AnalysisGraph::from_json_file(
        indra_json_file,
        vm["belief_score_cutoff"].as<double>(),
        vm["grounding_score_cutoff"].as<double>());
  }
  if (vm.count("causemos_json")) {
    G = AnalysisGraph::from_uncharted_json_file(uncharted_json_file);
  }

  debug("Number of vertices: {}", G.num_vertices());
  debug("Number of edges: {}", G.num_edges());

  if (vm["map_concepts"].as<bool>()) {
    G.map_concepts_to_indicators(vm["n_indicators"].as<int>(),
                                 vm["country"].as<string>());
  }
  if (vm["quantify"].as<bool>()) {
    G.construct_beta_pdfs(RNG::rng()->get_RNG());
  }

  if (vm["draw_graph"].as<bool>() == true) {
    G.to_png(vm["cag_filename"].as<string>(),
             vm["simplified_labels"].as<bool>(),
             vm["label_depth"].as<int>(),
             "",
             vm["rankdir"].as<string>());
  }
  if (vm["train_model"].as<bool>()) {
    G.train_model(2015, 1, 2015, 12, 100, 900, vm["country"].as<string>());
  }
  return EXIT_SUCCESS;
}
