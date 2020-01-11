#include "AnalysisGraph.hpp"
#include "utils.hpp"
#include <fstream>
#include <range/v3/all.hpp>

using namespace std;
using namespace delphi::utils;
// CauseMos integration
AnalysisGraph AnalysisGraph::from_causemos_json_dict(nlohmann::json json_data) {
  AnalysisGraph G;

  auto statements = json_data["statements"];

  for (auto stmt : statements) {
    auto evidence = stmt["evidence"];

    if (evidence.is_null()) {
      continue;
    }

    auto subj = stmt["subj"];
    auto obj = stmt["obj"];

    if (subj.is_null() or obj.is_null()) {
      continue;
    }

    auto subj_db_ref = subj["db_refs"];
    auto obj_db_ref = obj["db_refs"];

    if (subj_db_ref.is_null() or obj_db_ref.is_null()) {
      continue;
    }

    auto subj_concept_json = subj_db_ref["concept"];
    auto obj_concept_json = obj_db_ref["concept"];

    if (subj_concept_json.is_null() or obj_concept_json.is_null()) {
      continue;
    }

    // TODO: Not sure why python version is doing a split on / and
    // then again a join on /!!
    string subj_name = subj_concept_json.get<string>();
    string obj_name = obj_concept_json.get<string>();

    auto subj_delta = stmt["subj_delta"];
    auto obj_delta = stmt["obj_delta"];

    auto subj_polarity_json = subj_delta["polarity"];
    auto obj_polarity_json = obj_delta["polarity"];

    int subj_polarity = 1;
    int obj_polarity = 1;
    if (!subj_polarity_json.is_null()) {
      subj_polarity = subj_polarity_json.get<int>();
    }

    if (!obj_polarity_json.is_null()) {
      obj_polarity = obj_polarity_json.get<int>();
    }

    auto subj_adjectives = subj_delta["adjectives"];
    auto obj_adjectives = obj_delta["adjectives"];
    auto subj_adjective =
        (!subj_adjectives.is_null() and subj_adjectives.size() > 0)
            ? subj_adjectives[0]
            : "None";
    auto obj_adjective =
        (obj_adjectives.size() > 0) ? obj_adjectives[0] : "None";

    string subj_adj_str = subj_adjective.get<string>();
    string obj_adj_str = obj_adjective.get<string>();

    auto causal_fragment =
        CausalFragment({subj_adj_str, subj_polarity, subj_name},
                       {obj_adj_str, obj_polarity, obj_name});
    G.add_edge(causal_fragment);
  }

  for (auto& [concept, mapping] : json_data["conceptIndicators"].items()) {
    string indicator_name = mapping["name"].get<string>();
    string indicator_source = mapping["name"].get<string>();
    G[concept].add_indicator(indicator_name, indicator_source);
    // Calculate aggregate from the values given
    vector<double> values = {};
    for (auto& data_point : mapping["values"]) {
      values.push_back(data_point["value"].get<double>());
    }
    // Aggregation function
    string func = mapping["func"].get<string>();
    double aggregated_value = 0.0;
    if (func == "max") {
      aggregated_value = ranges::max(values);
    }
    else if (func == "min") {
      aggregated_value = ranges::min(values);
    }
    else if (func == "mean") {
      aggregated_value = mean(values);
    }
    else {
      throw runtime_error("\"func\" must be one of [max|min|mean]");
    }
    G[concept].get_indicator(indicator_name).set_mean(aggregated_value);
  }
  G.initialize_random_number_generator();
  G.construct_beta_pdfs();
  return G;
}

AnalysisGraph AnalysisGraph::from_causemos_json_string(string json_string) {
  auto json_data = nlohmann::json::parse(json_string);
  return AnalysisGraph::from_causemos_json_dict(json_data);
}

AnalysisGraph AnalysisGraph::from_causemos_json_file(string filename) {
  auto json_data = load_json(filename);
  return AnalysisGraph::from_causemos_json_dict(json_data);
}
