#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "AnalysisGraph.hpp"
#include "doctest.h"
#include <fmt/core.h>

using namespace std;
vector<CausalFragment> causal_fragments = {
    {{"large", -1, "UN/entities/human/financial/economic/inflation"},
     {"small", 1, "UN/events/human/human_migration"}},
    {{"large", 1, "UN/events/human/human_migration"},
     {"small", -1, "UN/entities/human/food/food_security"}},
    {{"large", 1, "UN/events/human/human_migration"},
     {"small", 1, "UN/entities/human/food/food_insecurity"}},
};

TEST_CASE("Testing model training") {
  RNG *R = RNG::rng();
  R->set_seed(87);

  AnalysisGraph G = AnalysisGraph::from_causal_fragments(causal_fragments);

  G.map_concepts_to_indicators();

  G.replace_indicator("UN/events/human/human_migration",
                      "Net migration",
                      "New asylum seeking applicants",
                      "UNHCR");
  G.to_png();
  cout << G.to_dot() << endl;

  G.construct_beta_pdfs();
  G.train_model(2015, 1, 2015, 12, 100, 900);

  pair<vector<string>,
       vector<vector<unordered_map<string, unordered_map<string, double>>>>>
      preds = G.generate_prediction(2015, 1, 2015, 12);
  fmt::print("Prediction to array\n");

  try {
    vector<vector<double>> result =
        G.prediction_to_array("New asylum seeking applicants");
    fmt::print("Size of result is {} x {}\nFirst value of result is {}\n",
               result.size(),
               result[0].size(),
               result[0][0]);
  }
  catch (IndicatorNotFoundException &infe) {
    fmt::print(infe.what());
  }
}

TEST_CASE("Testing merge_nodes") {
  AnalysisGraph G = AnalysisGraph::from_causal_fragments(causal_fragments);
  REQUIRE(G.num_nodes() == 4);
  G.merge_nodes("UN/entities/human/food/food_security",
                "UN/entities/human/food/food_insecurity",
                false);
  REQUIRE(G.num_nodes() == 3);
}
