#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "AnalysisGraph.cpp"
#include "rng.hpp"

TEST_CASE("Testing model training") {
  RNG *R = RNG::rng();
  R->set_seed(87);

  vector<pair<tuple<string, int, string>, tuple<string, int, string>>>
      statements = {
          {{"large", -1, "UN/entities/human/financial/economic/inflation"},
           {"small", 1, "UN/events/human/human_migration"}}};
  AnalysisGraph G = AnalysisGraph::from_statements(statements);

  G.map_concepts_to_indicators();

  G.replace_indicator("UN/events/human/human_migration", "Net migration",
                      "New asylum seeking applicants", "UNHCR");

  G.train_model(2015, 1, 2015, 12, 100, 900);

  pair<vector<string>,
       vector<vector<unordered_map<string, unordered_map<string, double>>>>>
      preds = G.generate_prediction(2015, 1, 2015, 12);
  fmt::print("Prediction to array\n");

  try {
    vector<vector<double>> result =
        G.prediction_to_array("New asylum seeking applicants");
    fmt::print("Size of result is {} x {}\nFirst value of result is {}\n",
               result.size(), result[0].size(), result[0][0]);
  } catch (IndicatorNotFoundException &infe) {
    fmt::print(infe.what());
  }
} 
