#include "AnalysisGraph.hpp"
#include "utils.hpp"
#include <fstream>
#include <range/v3/all.hpp>

using namespace std;
using namespace delphi::utils;
using fmt::print;

AnalysisGraph
AnalysisGraph::from_indra_statements_json_dict(nlohmann::json json_data,
                                               double belief_score_cutoff,
                                               double grounding_score_cutoff,
                                               string ontology) {
  AnalysisGraph G;

  for (auto stmt : json_data) {
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

/**
 * copy constructor
 * TODO: Most probably the copy is sharing the same
 * random number generation class RNG
 * TODO: If at any point we make a copy of AnalysisGraph
 * and find that the copy does not behave as intended,
 * we might have not copied something or we might have
 * copied something incorrectly. This is one place to
 * look for bugs.
 */
AnalysisGraph::AnalysisGraph(const AnalysisGraph& rhs) {
  // Copying private members
  this->indicators_in_CAG = rhs.indicators_in_CAG;
  // NOTE: Copying this gives a segmentation fault
  //       Investigate
  //this->name_to_vertex = rhs.name_to_vertex;
  this->causemos_call = rhs.causemos_call;
  this->trained = rhs.trained;
  this->n_timesteps = rhs.n_timesteps;
  this->pred_timesteps = rhs.pred_timesteps;
  this->training_range = rhs.training_range;
  this->train_start_epoch = rhs.train_start_epoch;
  this->train_end_epoch = rhs.train_end_epoch;
  this->pred_start_epoch = rhs.pred_start_epoch;
  this->pred_end_epoch = rhs.pred_end_epoch;
  this->pred_start_timestep = rhs.pred_start_timestep;
  this->observation_timesteps = rhs.observation_timesteps;
  this->modeling_period = rhs.modeling_period;
  this->pred_range = rhs.pred_range;
  this->t = rhs.t;
  this->delta_t = rhs.delta_t;
  this->s0 = rhs.s0;
  this->A_original = rhs.A_original;
  this->continuous = rhs.continuous;

  // NOTE: This assumes that node indices and indicator indices for each node
  // does not change when copied. This data structure is indexed using those
  // indices. If they gets changed while copying, assigned indicator data would
  // be mixed up and hence training gets mixed up.
  this->observed_state_sequence = rhs.observed_state_sequence;

  this->predicted_latent_state_sequences = rhs.predicted_latent_state_sequences;
  this->predicted_observed_state_sequences = rhs.predicted_observed_state_sequences;
  this->test_observed_state_sequence = rhs.test_observed_state_sequence;
  this->one_off_constraints = rhs.one_off_constraints;
  this->perpetual_constraints = rhs.perpetual_constraints;
  this->is_one_off_constraints = rhs.is_one_off_constraints;
  this->transition_matrix_collection = rhs.transition_matrix_collection;
  this->initial_latent_state_collection = rhs.initial_latent_state_collection;
  this->synthetic_latent_state_sequence= rhs.synthetic_latent_state_sequence;
  this->synthetic_data_experiment = rhs.synthetic_data_experiment;

  // Copying public members
  this->id = rhs.id;
  this->data_heuristic = rhs.data_heuristic;
  this->res = rhs.res;

  for_each(rhs.node_indices(), [&](int v) {
    Node node_rhs = rhs.graph[v];

    // Add nodes in the same order as rhs so that indices does not chance
    this->add_node(node_rhs.name);

    // Copy all the data for node v in rhs graph to this graph.
    // This data includes all the indicators.
    (*this)[v] = rhs.graph[v];

    /*
    for(const Indicator& ind : node_rhs.indicators) {
      this->set_indicator(node_rhs.name, ind.name, ind.source);
    }
    */
  });
  /*
  for (auto [vert_name, vert_id] : rhs.name_to_vertex) {
    this->add_node(vert_name);
  }
  */

  // Add all the edges
  for_each(rhs.edges(), [&](auto e_rhs) {
    auto [e_lhs, exists] = this->add_edge(rhs.graph[boost::source(e_rhs, rhs.graph)].name,
		                          rhs.graph[boost::target(e_rhs, rhs.graph)].name);

    // Copy all the edge data structures
    this->graph[e_lhs] = rhs.graph[e_rhs];
    /*
    this->graph[e_lhs].evidence = rhs.graph[e_rhs].evidence;
    this->graph[e_lhs].kde = rhs.graph[e_rhs].kde;
    this->graph[e_lhs].name = rhs.graph[e_rhs].name;
    this->graph[e_lhs].theta = rhs.graph[e_rhs].theta;
    */

  });
}
