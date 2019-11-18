#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "AnalysisGraph.hpp"
#include "doctest.h"
#include <fmt/core.h>

using namespace std;
string inflation = "wm/concept/causal_factor/economic_and_commerce/economic "
                   "activity/market/inflation";
string migration =
    "wm/concept/causal_factor/social_and_political/migration/human_migration";
string food_security = "wm/concept/causal_factor/condition/food_security";

vector<CausalFragment> causal_fragments = {
    {{"large", -1, inflation}, {"small", 1, migration}},
    {{"large", 1, migration}, {"small", -1, food_security}},
    {{"large", 1, migration}, {"small", -1, food_security}},
};

TEST_CASE("Testing model training") {
  RNG* R = RNG::rng();
  R->set_seed(87);

  AnalysisGraph G = AnalysisGraph::from_causal_fragments(causal_fragments);

  G.map_concepts_to_indicators();
  G.to_png();

  G[migration].replace_indicator(
      "Net migration", "New asylum seeking applicants", "UNHCR");

  G.train_model(2015, 1, 2015, 12, 100, 900);

  Prediction preds = G.generate_prediction(2015, 1, 2015, 12);
  fmt::print("Prediction to array\n");

  try {
    vector<vector<double>> result =
        G.prediction_to_array("New asylum seeking applicants");
    fmt::print("Size of result is {} x {}\nFirst value of result is {}\n",
               result.size(),
               result[0].size(),
               result[0][0]);
  }
  catch (IndicatorNotFoundException& infe) {
    fmt::print(infe.what());
  }
}
