#include "AnalysisGraph.hpp"
#include "tqdm.hpp"
#include "spdlog/spdlog.h"
#include <fstream>

using namespace std;
using tq::tqdm;
using spdlog::debug;
using spdlog::error;
using spdlog::warn;

nlohmann::json load_json(string filename) {
  ifstream i(filename);
  nlohmann::json j = nlohmann::json::parse(i);
  return j;
}

AnalysisGraph AnalysisGraph::from_json_file(string filename,
                                            double belief_score_cutoff,
                                            double grounding_score_cutoff,
                                            string ontology) {
  debug("Loading INDRA statements JSON file.");
  auto json_data = load_json(filename);

  AnalysisGraph G;

  unordered_map<string, int> name_to_vertex = {};

  debug("Processing INDRA statements...");
  for (auto stmt : tqdm(json_data)) {
    if (stmt["type"] == "Influence") {
      auto subj_ground = stmt["subj"]["concept"]["db_refs"][ontology][0][1];
      auto obj_ground = stmt["obj"]["concept"]["db_refs"][ontology][0][1];
      bool grounding_check = (subj_ground >= grounding_score_cutoff) and
                             (obj_ground >= grounding_score_cutoff);
      if (grounding_check) {
        auto subj = stmt["subj"]["concept"]["db_refs"]["WM"][0][0];
        auto obj = stmt["obj"]["concept"]["db_refs"]["WM"][0][0];
        if (!subj.is_null() and !obj.is_null()) {
          if (stmt["belief"] < belief_score_cutoff) {
            continue;
          }

          string subj_str = subj.get<string>();
          string obj_str = obj.get<string>();

          if (subj_str.compare(obj_str) != 0) { // Guard against self loops
            // Add the nodes to the graph if they are not in it already
            for (string name : {subj_str, obj_str}) {
              G.add_node(name);
            }

            // Add the edge to the graph if it is not in it already
            for (auto evidence : stmt["evidence"]) {
              auto annotations = evidence["annotations"];
              auto subj_adjectives = annotations["subj_adjectives"];
              auto obj_adjectives = annotations["obj_adjectives"];
              auto subj_adjective =
                  (!subj_adjectives.is_null() and subj_adjectives.size() > 0)
                      ? subj_adjectives[0]
                      : "None";
              auto obj_adjective =
                  (obj_adjectives.size() > 0) ? obj_adjectives[0] : "None";
              auto subj_polarity = annotations["subj_polarity"];
              auto obj_polarity = annotations["obj_polarity"];

              if (subj_polarity.is_null()) {
                subj_polarity = 1;
              }
              if (obj_polarity.is_null()) {
                obj_polarity = 1;
              }
              string subj_adj_str = subj_adjective.get<string>();
              string obj_adj_str = subj_adjective.get<string>();
              auto causal_fragment =
                  CausalFragment({subj_adj_str, subj_polarity, subj_str},
                                 {obj_adj_str, obj_polarity, obj_str});

              G.add_edge(causal_fragment);
            }
          }
        }
      }
    }
  }
  return G;
}

AnalysisGraph
AnalysisGraph::from_uncharted_json_dict(nlohmann::json json_data) {
  AnalysisGraph G;

  unordered_map<string, int> name_to_vertex = {};

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
    string text = stmt["evidence"][0]["text"].get<string>();
    G.add_edge(causal_fragment);
  }
  return G;
}

AnalysisGraph AnalysisGraph::from_uncharted_json_string(string json_string) {
  auto json_data = nlohmann::json::parse(json_string);
  return AnalysisGraph::from_uncharted_json_dict(json_data);
}

AnalysisGraph AnalysisGraph::from_uncharted_json_file(string filename) {
  auto json_data = load_json(filename);
  return AnalysisGraph::from_uncharted_json_dict(json_data);
}

