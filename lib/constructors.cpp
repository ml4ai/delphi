#include "AnalysisGraph.hpp"
#include "spdlog/spdlog.h"
#include "tqdm.hpp"
#include "utils.hpp"
#include <fstream>
#include <range/v3/all.hpp>

using namespace std;
using namespace delphi::utils;
using spdlog::debug, spdlog::error, spdlog::warn, tq::tqdm;

AnalysisGraph
AnalysisGraph::from_indra_statements_json_dict(nlohmann::json json_data,
                                               double belief_score_cutoff,
                                               double grounding_score_cutoff,
                                               string ontology) {
  debug("Loading INDRA statements JSON file.");
  AnalysisGraph G;

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
AnalysisGraph::from_indra_statements_json_string(string json_string,
                                                 double belief_score_cutoff,
                                                 double grounding_score_cutoff,
                                                 string ontology) {
  auto json_data = nlohmann::json::parse(json_string);
  return AnalysisGraph::from_indra_statements_json_dict(
      json_data, belief_score_cutoff, grounding_score_cutoff, ontology);
}

AnalysisGraph
AnalysisGraph::from_indra_statements_json_file(string filename,
                                               double belief_score_cutoff,
                                               double grounding_score_cutoff,
                                               string ontology) {
  debug("Loading INDRA statements JSON file.");
  auto json_data = load_json(filename);

  return AnalysisGraph::from_indra_statements_json_dict(
      json_data, belief_score_cutoff, grounding_score_cutoff, ontology);
}

AnalysisGraph
AnalysisGraph::from_causal_fragments(vector<CausalFragment> causal_fragments) {
  AnalysisGraph G;

  for (CausalFragment cf : causal_fragments) {
    Event subject = Event(cf.first);
    Event object = Event(cf.second);

    string subj_name = subject.concept_name;
    string obj_name = object.concept_name;

    if (subj_name.compare(obj_name) != 0) { // Guard against self loops
      // Add the nodes to the graph if they are not in it already
      for (string name : {subj_name, obj_name}) {
        G.add_node(name);
      }
      G.add_edge(cf);
    }
  }
  return G;
}

AnalysisGraph AnalysisGraph::from_json_string(string json_string) {
  auto data = nlohmann::json::parse(json_string);
  AnalysisGraph G;
  G.id = data["id"];
  for (auto e : data["edges"]) {
    string source = e["source"].get<string>();
    string target = e["target"].get<string>();
    G.add_node(source);
    G.add_node(target);
    G.add_edge(source, target);
    G.edge(source, target).kde.dataset = e["kernels"].get<vector<double>>();
  }
  for (auto& [concept, indicator] : data["indicatorData"].items()) {
    string indicator_name = indicator["name"].get<string>();
    G[concept].add_indicator(indicator_name, indicator["source"].get<string>());
    G[concept]
        .get_indicator(indicator_name)
        .set_mean(indicator["mean"].get<double>());
  }
  return G;
}
