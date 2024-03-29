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

AnalysisGraph
AnalysisGraph::from_causal_fragments_with_data(pair<vector<CausalFragment>,
                                  ConceptIndicatorAlignedData> cag_ind_data,
                                               int kde_kernels) {
  AnalysisGraph G = from_causal_fragments(cag_ind_data.first);

  G.n_kde_kernels = kde_kernels;

  G.observed_state_sequence.clear();
  G.n_timesteps = 0;

  // NOTE: Only one indicator per concept
  for (const auto & [ concept, ind_data ] : cag_ind_data.second) {
    G.set_indicator(concept, ind_data.first, "");

    if (G.n_timesteps < ind_data.second.size()) {
      G.n_timesteps = ind_data.second.size();
    }
  }

  // Access (concept is a vertex in the CAG)
  // [ timestep ][ concept ][ indicator ][ observation ]
  G.observed_state_sequence = ObservedStateSequence(G.n_timesteps);

  int num_verts = G.num_vertices();

  // Fill in observed state sequence
  // NOTE: This code is very similar to the implementations in
  // set_observed_state_sequence_from_data and get_observed_state_from_data
  for (int ts = 0; ts < G.n_timesteps; ts++) {
    G.observed_state_sequence[ts] = vector<vector<vector<double>>>(num_verts);

    for (const auto & [ concept, ind_data ] : cag_ind_data.second) {
      int v = G.name_to_vertex.at(concept);

      Node& n = G[v];
      G.observed_state_sequence[ts][v] = vector<vector<double>>(n.indicators.size());

      // Only one indicator per concept => i = 0
      for (int i = 0; i < n.indicators.size(); i++) {
        G.observed_state_sequence[ts][v][i] = vector<double>();

        if (ts < ind_data.second.size()) {
          G.observed_state_sequence[ts][v][i].push_back(ind_data.second[ts]);
        }
      }
    }
  }

  G.num_modeling_timesteps_per_one_observation_timestep = 1;
  G.train_start_epoch = 0;
  G.modeling_timestep_gaps.clear();
  G.modeling_timestep_gaps = vector<double>(G.n_timesteps, 1.0);
  G.modeling_timestep_gaps[0] = 0;

  return G;
}

AnalysisGraph AnalysisGraph::from_json_string(string json_string) {
  auto data = nlohmann::json::parse(json_string);
  AnalysisGraph G;
  G.id = data["id"];
  G.experiment_id = data["experiment_id"];
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
  #ifdef TIME
    this->durations = rhs.durations;
    this->mcmc_part_duration = rhs.mcmc_part_duration;
    this->writer = rhs.writer;
    this->timing_file_prefix = rhs.timing_file_prefix;
    this->timing_run_number = rhs.timing_run_number;
  #endif
  this->causemos_call = rhs.causemos_call;

  this->uni_dist = rhs.uni_dist;
  this->norm_dist = rhs.norm_dist;
  this->uni_disc_dist = rhs.uni_disc_dist;
  this->uni_disc_dist_edge = rhs.uni_disc_dist_edge;

  this->res = rhs.res;
  this->n_kde_kernels = rhs.n_kde_kernels;

  this->indicators_in_CAG = rhs.indicators_in_CAG;
  this->A_beta_factors = rhs.A_beta_factors;
  this->beta_dependent_cells = rhs.beta_dependent_cells;
  this->beta2cell = rhs.beta2cell;

  this->generated_latent_sequence = rhs.generated_latent_sequence;
  this->generated_concept = rhs.generated_concept;

  this->trained = rhs.trained;
  this->stopped = rhs.stopped;

  this->n_timesteps = rhs.n_timesteps;
  this->pred_timesteps = rhs.pred_timesteps;
  this->training_range = rhs.training_range;
  this->pred_range = rhs.pred_range;
  this->train_start_epoch = rhs.train_start_epoch;
  this->train_end_epoch = rhs.train_end_epoch;
  this->pred_start_timestep = rhs.pred_start_timestep;
  this->observation_timesteps_sorted = rhs.observation_timesteps_sorted;
  this->modeling_timestep_gaps = rhs.modeling_timestep_gaps;
  this->observation_timestep_unique_gaps = rhs.observation_timestep_unique_gaps;
  this->model_data_agg_level = rhs.model_data_agg_level;

  this->e_A_ts = rhs.e_A_ts;
  this->e_A_fourier_ts = rhs.e_A_fourier_ts;
  this->num_modeling_timesteps_per_one_observation_timestep = rhs.num_modeling_timesteps_per_one_observation_timestep;

  this->external_concepts = rhs.external_concepts;
  this->concept_sample_pool = rhs.concept_sample_pool;
  this->edge_sample_pool = rhs.edge_sample_pool;

  this->t = rhs.t;
  this->delta_t = rhs.delta_t;

  this->log_likelihood = rhs.log_likelihood;
  this->previous_log_likelihood = rhs.previous_log_likelihood;
  this->log_likelihood_MAP = rhs.log_likelihood_MAP;
  this->MAP_sample_number = rhs.MAP_sample_number;
  this->log_likelihoods = rhs.log_likelihoods;

  this->coin_flip = rhs.coin_flip;
  this->coin_flip_thresh = rhs.coin_flip_thresh;

  this->previous_theta = rhs.previous_theta;
  this->changed_derivative = rhs.changed_derivative;
  this->previous_derivative = rhs.previous_derivative;

  this->s0 = rhs.s0;
  this->s0_prev = rhs.s0_prev;
  this->derivative_prior_variance = rhs.derivative_prior_variance;

  this->A_original = rhs.A_original;

  this->head_node_model = rhs.head_node_model;
  this->A_fourier_base = rhs.A_fourier_base;
  this->s0_fourier = rhs.s0_fourier;

  this->continuous = rhs.continuous;

  this->current_latent_state = rhs.current_latent_state;

  // NOTE: This assumes that node indices and indicator indices for each node
  // does not change when copied. This data structure is indexed using those
  // indices. If they gets changed while copying, assigned indicator data would
  // be mixed up and hence training gets mixed up.
  this->observed_state_sequence = rhs.observed_state_sequence;

  this->predicted_latent_state_sequences = rhs.predicted_latent_state_sequences;
  this->predicted_observed_state_sequences = rhs.predicted_observed_state_sequences;
  this->test_observed_state_sequence = rhs.test_observed_state_sequence;

  this->one_off_constraints = rhs.one_off_constraints;
  this->head_node_one_off_constraints = rhs.head_node_one_off_constraints;
  this->perpetual_constraints = rhs.perpetual_constraints;
  this->is_one_off_constraints = rhs.is_one_off_constraints;
  this->clamp_at_derivative = rhs.clamp_at_derivative;
  this->rest_derivative_clamp_ts = rhs.rest_derivative_clamp_ts;

  this->transition_matrix_collection = rhs.transition_matrix_collection;
  this->initial_latent_state_collection = rhs.initial_latent_state_collection;
  this->synthetic_latent_state_sequence = rhs.synthetic_latent_state_sequence;
  this->synthetic_data_experiment = rhs.synthetic_data_experiment;

  // Copying public members
  this->id = rhs.id;
  this->experiment_id = rhs.experiment_id;
  this->data_heuristic = rhs.data_heuristic;

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


/** Copy assignment operator (copy-and-swap idiom)*/
AnalysisGraph& AnalysisGraph::operator=(AnalysisGraph rhs) {
    #ifdef TIME
        swap(durations, rhs.durations);
        swap(mcmc_part_duration, rhs.mcmc_part_duration);
        swap(writer, rhs.writer);
        swap(timing_file_prefix, rhs.timing_file_prefix);
        swap(timing_run_number, rhs.timing_run_number);
    #endif

    swap(causemos_call, rhs.causemos_call);

    swap(graph, rhs.graph);

    swap(uni_dist, rhs.uni_dist);
    swap(norm_dist, rhs.norm_dist);
    swap(uni_disc_dist, rhs.uni_disc_dist);
    swap(uni_disc_dist_edge, rhs.uni_disc_dist_edge);

    swap(res, rhs.res);
    swap(n_kde_kernels, rhs.n_kde_kernels);

    swap(name_to_vertex, rhs.name_to_vertex);
    swap(indicators_in_CAG, rhs.indicators_in_CAG);
    swap(A_beta_factors, rhs.A_beta_factors);
    swap(beta_dependent_cells, rhs.beta_dependent_cells);
    swap(beta2cell, rhs.beta2cell);

    swap(body_nodes, rhs.body_nodes);
    swap(head_nodes, rhs.head_nodes);
    swap(generated_latent_sequence, rhs.generated_latent_sequence);
    swap(generated_concept, rhs.generated_concept);

    swap(trained, rhs.trained);
    swap(stopped, rhs.stopped);

    swap(n_timesteps, rhs.n_timesteps);
    swap(pred_timesteps, rhs.pred_timesteps);
    swap(training_range, rhs.training_range);
    swap(pred_range, rhs.pred_range);
    swap(train_start_epoch, rhs.train_start_epoch);
    swap(train_end_epoch, rhs.train_end_epoch);
    swap(pred_start_timestep, rhs.pred_start_timestep);
    swap(observation_timesteps_sorted, rhs.observation_timesteps_sorted);
    swap(modeling_timestep_gaps, rhs.modeling_timestep_gaps);
    swap(observation_timestep_unique_gaps, rhs.observation_timestep_unique_gaps);
    swap(model_data_agg_level, rhs.model_data_agg_level);

    swap(e_A_ts, rhs.e_A_ts);
    swap(e_A_fourier_ts, rhs.e_A_fourier_ts);
    swap(num_modeling_timesteps_per_one_observation_timestep, rhs.num_modeling_timesteps_per_one_observation_timestep);

    swap(external_concepts, rhs.external_concepts);
    swap(concept_sample_pool, rhs.concept_sample_pool);
    swap(edge_sample_pool, rhs.edge_sample_pool);

    swap(t, rhs.t);
    swap(delta_t, rhs.delta_t);

    swap(log_likelihood, rhs.log_likelihood);
    swap(previous_log_likelihood, rhs.previous_log_likelihood);
    swap(log_likelihood_MAP, rhs.log_likelihood_MAP);
    swap(MAP_sample_number, rhs.MAP_sample_number);
    swap(log_likelihoods, rhs.log_likelihoods);

    swap(coin_flip, rhs.coin_flip);
    swap(coin_flip_thresh, rhs.coin_flip_thresh);

    swap(previous_theta, rhs.previous_theta);
    swap(changed_derivative, rhs.changed_derivative);
    swap(previous_derivative, rhs.previous_derivative);

    swap(s0, rhs.s0);
    swap(s0_prev, rhs.s0_prev);
    swap(derivative_prior_variance, rhs.derivative_prior_variance);

    swap(A_original, rhs.A_original);

    swap(head_node_model, rhs.head_node_model);
    swap(A_fourier_base, rhs.A_fourier_base);
    swap(s0_fourier, rhs.s0_fourier);

    swap(continuous, rhs.continuous);

    swap(current_latent_state, rhs.current_latent_state);

    swap(observed_state_sequence, rhs.observed_state_sequence);
    swap(predicted_latent_state_sequences, rhs.predicted_latent_state_sequences);
    swap(predicted_observed_state_sequences, rhs.predicted_observed_state_sequences);
    swap(test_observed_state_sequence, rhs.test_observed_state_sequence);

    swap(one_off_constraints, rhs.one_off_constraints);
    swap(head_node_one_off_constraints, rhs.head_node_one_off_constraints);
    swap(perpetual_constraints, rhs.perpetual_constraints);
    swap(is_one_off_constraints, rhs.is_one_off_constraints);
    swap(clamp_at_derivative, rhs.clamp_at_derivative);
    swap(rest_derivative_clamp_ts, rhs.rest_derivative_clamp_ts);

    swap(transition_matrix_collection, rhs.transition_matrix_collection);
    swap(initial_latent_state_collection, rhs.initial_latent_state_collection);
    swap(synthetic_latent_state_sequence, rhs.synthetic_latent_state_sequence);
    swap(synthetic_data_experiment, rhs.synthetic_data_experiment);

    swap(experiment_id, rhs.experiment_id);

    // Copying public members
    swap(id, rhs.id);
    swap(data_heuristic, rhs.data_heuristic);
    return *this;
}
