// Usage: ./create_model <path_to_JSON_file>

#include "AnalysisGraph.hpp"
#include "spdlog/spdlog.h"

using namespace std;

int main(int argc, char* argv[]) {
  RNG *R = RNG::rng();
  R->set_seed(87);
  spdlog::set_level(spdlog::level::debug);
  auto G = AnalysisGraph::from_json_file(argv[1], 0.9, 0.0);
  G.map_concepts_to_indicators();
  G.construct_beta_pdfs();
  G.to_png();
  G.train_model(2015, 1, 2015, 12, 100, 900);
  return EXIT_SUCCESS;
}
