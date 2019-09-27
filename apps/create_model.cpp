// Usage: ./create_model <path_to_JSON_file>

#include "AnalysisGraph.hpp"
#include "spdlog/spdlog.h"
#include <boost/program_options.hpp>

int main(int argc, char* argv[]) {
  using namespace std;
  using namespace boost::program_options;

  options_description desc("Allowed options");
  positional_options_description pd;

  // Path to JSON-serialized INDRA statements
  string stmts;

  desc.add_options()
    ("help,h", "Executable for creating Delphi models")
    ("stmts", value<string>(&stmts), "Path to JSON-serialized INDRA statements")
    ("belief_score_cutoff", value<double>()->default_value(0.9),
     "INDRA belief score cutoff for statements to be included in the model.")
    ("grounding_score_cutoff", value<double>()->default_value(0.7),
     "Grounding score cutoff for statements to be included in the model")
  ;

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

  RNG *R = RNG::rng();
  R->set_seed(87);
  spdlog::set_level(spdlog::level::debug);
  auto G = AnalysisGraph::from_json_file(
      stmts,
      vm["belief_score_cutoff"].as<double>(), 
      vm["grounding_score_cutoff"].as<double>());

  G.map_concepts_to_indicators();
  G.construct_beta_pdfs();
  G.to_png();
  G.train_model(2015, 1, 2015, 12, 100, 900);
  return EXIT_SUCCESS;
}
