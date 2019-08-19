// Usage: ./create_model <path_to_JSON_file>

#include "AnalysisGraph.hpp"

using namespace std;

int main(int argc, char* argv[]) {
  RNG *R = RNG::rng();
  R->set_seed(87);
  //auto G = AnalysisGraph::from_json_file(argv[1], 0.9, 0.0);
  vector<CausalFragment> causal_fragments = {
    {{"large", -1, "UN/events/human/conflict"},
     {"small", 1, "UN/events/human/physical_insecurity"}},
    {{"large", 1, "UN/events/human/conflict"},
     {"small", -1, "UN/events/human/human_migration"}},
    {{"large", 1, "UN/events/human/physical_insecurity"},
     {"small", 1, "UN/events/human/human_migration"}},
    {{"large", -1, "UN/entities/human/population"},
     {"small", 1, "UN/events/human/human_migration"}},
  };
  AnalysisGraph G = AnalysisGraph::from_causal_fragments(causal_fragments);
  G = G.get_subgraph_for_concept("UN/events/human/human_migration", 2, true);
  G.print_nodes();
  G.print_all_paths();
  G.map_concepts_to_indicators();
  G.replace_indicator("UN/events/human/human_migration",
                      "Net migration",
                      "New asylum seeking applicants",
                      "UNHCR");
  G.construct_beta_pdfs();
  G.to_png();
  G.train_model(2015, 1, 2015, 12, 100, 900);
  return EXIT_SUCCESS;
}
