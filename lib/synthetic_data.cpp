#include "AnalysisGraph.hpp"
#include <range/v3/all.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/adaptors.hpp>
#include <sqlite3.h>
#include "dbg.h"
#include <unistd.h>

using namespace std;
using namespace delphi::utils;
using Eigen::VectorXd;
using fmt::print;


/*
 ============================================================================
 Private: Synthetic Data Experiment
 ============================================================================
*/

void AnalysisGraph::set_random_initial_latent_state() {
  int num_verts = this->num_vertices();

  this->set_default_initial_state();

  for (int v = 0; v < num_verts; v++) {
    this->s0(2 * v + 1) = 0.1 * this->uni_dist(this->rand_num_generator);
  }
}

void AnalysisGraph::generate_synthetic_latent_state_sequence() {
  int num_verts = this->num_vertices();

  // Allocate memory for synthetic_latent_state_sequence
  this->synthetic_latent_state_sequence.clear();
  this->synthetic_latent_state_sequence =
      vector<VectorXd>(this->n_timesteps, VectorXd(num_verts * 2));

  this->synthetic_latent_state_sequence[0] = this->s0;

  for (int ts = 1; ts < this->n_timesteps; ts++) {
    this->synthetic_latent_state_sequence[ts] =
        this->A_original * this->synthetic_latent_state_sequence[ts - 1];
  }
}

void AnalysisGraph::
    generate_synthetic_observed_state_sequence_from_synthetic_latent_state_sequence() {
  using ranges::to;
  using ranges::views::transform;
  // Allocate memory for observed_state_sequences
  this->test_observed_state_sequence.clear();
  this->test_observed_state_sequence = PredictedObservedStateSequence(
      this->n_timesteps, vector<vector<double>>());

  this->test_observed_state_sequence =
      this->synthetic_latent_state_sequence |
      transform([this](VectorXd latent_state) {
        return this->sample_observed_state(latent_state);
      }) |
      to<vector>();
}

vector<vector<double>>
AnalysisGraph::sample_observed_state(VectorXd latent_state) {
  using ranges::to;
  using ranges::views::transform;
  int num_verts = this->num_vertices();

  assert(num_verts == latent_state.size() / 2);

  vector<vector<double>> observed_state(num_verts);

  for (int v = 0; v < num_verts; v++) {
    vector<Indicator>& indicators = (*this)[v].indicators;

    observed_state[v] = vector<double>(indicators.size());

    // Sample observed value of each indicator around the mean of the
    // indicator
    // scaled by the value of the latent state that caused this observation.
    // TODO: Question - Is ind.mean * latent_state[ 2*v ] correct?
    //                  Shouldn't it be ind.mean + latent_state[ 2*v ]?
    observed_state[v] = indicators | transform([&](Indicator ind) {
                          normal_distribution<double> gaussian(
                              ind.mean * latent_state[2 * v], ind.stdev);

                          return gaussian(this->rand_num_generator);
                        }) |
                        to<vector>();
  }

  return observed_state;
}


/*
 ============================================================================
 Public: Synthetic data experiment
 ============================================================================
*/

pair<PredictedObservedStateSequence, Prediction>
AnalysisGraph::test_inference_with_synthetic_data(int start_year,
                                                  int start_month,
                                                  int end_year,
                                                  int end_month,
                                                  int res,
                                                  int burn,
                                                  string country,
                                                  string state,
                                                  string county,
                                                  map<string, string> units,
                                                  InitialBeta initial_beta,
                                                  InitialDerivative initial_derivative,
                                                  bool use_continuous) {
  synthetic_data_experiment = true;
  this->n_timesteps = this->calculate_num_timesteps(
      start_year, start_month, end_year, end_month);
  this->initialize_parameters(res, initial_beta, initial_derivative,
                              false, use_continuous);

  // Initialize the latent state vector at time 0
  this->set_random_initial_latent_state();
  this->generate_synthetic_latent_state_sequence();
  this->generate_synthetic_observed_state_sequence_from_synthetic_latent_state_sequence();

  for (vector<vector<double>> obs : this->test_observed_state_sequence) {
    print("({}, {})\n", obs[0][0], obs[1][0]);
  }

  this->train_model(start_year,
                    start_month,
                    end_year,
                    end_month,
                    res,
                    burn,
                    country,
                    state,
                    county,
                    units,
                    InitialBeta::ZERO);

  return make_pair(
      this->test_observed_state_sequence,
      this->generate_prediction(start_year, start_month, end_year, end_month));

  RNG::release_instance();
  synthetic_data_experiment = false;
}


AnalysisGraph AnalysisGraph::generate_random_CAG(unsigned int num_nodes,
                                                 unsigned int num_extra_edges) {
  AnalysisGraph G;

  G.initialize_random_number_generator();

  // Get all adjectives.
  vector<string> adjectives;
  AdjectiveResponseMap adjective_response_map =
      G.construct_adjective_response_map(G.rand_num_generator, G.uni_dist, G.norm_dist, 100);
  boost::range::copy(adjective_response_map | boost::adaptors::map_keys,
              back_inserter(adjectives));

  vector<int> cag_nodes = {0};
  vector<int> rand_node(2);
  vector<string> rand_adjectives(2);
  int polarity = 0;
  string source = "";
  string target = "";
  int src_idx = 0;

  for (rand_node[1] = 1; rand_node[1] < num_nodes; rand_node[1]++) {
    rand_node.clear();
    sample(cag_nodes.begin(), cag_nodes.end(), rand_node.begin(), 1, G.rand_num_generator);
    cag_nodes.push_back(rand_node[1]);

    src_idx = G.uni_dist(G.rand_num_generator) < 0.5? 0 : 1;
    source = to_string(rand_node[src_idx]);
    target = to_string(rand_node[1 - src_idx]);

    sample(adjectives.begin(), adjectives.end(), rand_adjectives.begin(), 2, G.rand_num_generator);
    polarity = G.uni_dist(G.rand_num_generator) < 0.5 ? 1 : -1;

    auto causal_fragment =
        CausalFragment({rand_adjectives[0], 1, source},
                       {rand_adjectives[1], polarity, target});
    G.add_edge(causal_fragment);
  }

  num_extra_edges = min(num_extra_edges, (num_nodes - 1) * (num_nodes - 1));

  pair<EdgeDescriptor, bool> edge;

  for (int _ = 0; _ < num_extra_edges; _++) {
    edge.second = true;
    while (edge.second) {
      sample(cag_nodes.begin(), cag_nodes.end(), rand_node.begin(), 2, G.rand_num_generator);
      src_idx = G.uni_dist(G.rand_num_generator) < 0.5? 0 : 1;
      source = to_string(rand_node[src_idx]);
      target = to_string(rand_node[1 - src_idx]);
      edge = boost::edge(G.get_vertex_id(source),
                   G.get_vertex_id(target), G.graph);
    }

    sample(adjectives.begin(), adjectives.end(), rand_adjectives.begin(), 2, G.rand_num_generator);
    polarity = G.uni_dist(G.rand_num_generator) < 0.5 ? 1 : -1;

    auto causal_fragment =
        CausalFragment({rand_adjectives[0], 1, source},
                       {rand_adjectives[1], polarity, target});
    G.add_edge(causal_fragment);
  }

  RNG::release_instance();
  return G;
}

/**
 * TODO: This is very similar to initialize_parameters() method defined in
 * parameter_initialization.cpp. Might be able to merge the two
 * @param kde_kernels Number of KDE kernels to use when constructing beta prior distributions
 * @param initial_beta How to initialize betas
 * @param initial_derivative How to initialize derivatives
 * @param use_continuous Whether to use matrix exponential or not
 */
void AnalysisGraph::initialize_random_CAG(unsigned int num_obs,
                                          unsigned int kde_kernels,
                                          InitialBeta initial_beta,
                                          InitialDerivative initial_derivative,
                                          bool use_continuous) {
  this->initialize_random_number_generator();
  this->set_default_initial_state(initial_derivative);
  this->set_res(kde_kernels);
  this->construct_theta_pdfs();
  this->init_betas_to(initial_beta);
  this->pred_timesteps = num_obs + 1;
  this->continuous = use_continuous;
  this->find_all_paths();
  this->set_transition_matrix_from_betas();
  this->transition_matrix_collection.clear();
  this->initial_latent_state_collection.clear();
  // TODO: We are using this variable for two different purposes.
  // create another variable.
  this->res = 1;
  this->transition_matrix_collection = vector<Eigen::MatrixXd>(this->res);
  this->initial_latent_state_collection = vector<Eigen::VectorXd>(this->res);
  this->transition_matrix_collection[0] = this->A_original;
  this->initial_latent_state_collection[0] = this->s0;
}


void AnalysisGraph::generate_synthetic_data(unsigned int num_obs,
                                            double noise_variance,
                                            unsigned int kde_kernels,
                                            InitialBeta initial_beta,
                                            InitialDerivative initial_derivative,
                                            bool use_continuous) {
  this->initialize_random_CAG(num_obs, kde_kernels, initial_beta, initial_derivative, use_continuous);
  vector<int> periods = {2, 3, 4, 6, 12};
  vector<int> rand_period(1);
  uniform_real_distribution<double> centers_dist(-100, 100);
  int max_samples_per_period = 12;
  for (int v : this->independent_nodes) {
    Node& n = (*this)[v];
    sample(periods.begin(), periods.end(), rand_period.begin(), 1, this->rand_num_generator);
    n.period = rand_period[0];
    int gap_size = max_samples_per_period / n.period;
    int offset = 0; // 0 <= offset < period
    vector<int> filled_months;
    for (int p = 0; p < n.period; p++) {
      double center = centers_dist(this->rand_num_generator);
      double spread = this->norm_dist(this->rand_num_generator) * 5;
      //dbg(spread);
      n.centers.push_back(center);
      n.spreads.push_back(spread);
      int month = offset + gap_size * p;
      n.generated_monthly_latent_centers_for_a_year[month] = center;
      n.generated_monthly_latent_spreads_for_a_year[month] = spread;
      filled_months.push_back(month);
    }
    this->interpolate_missing_months(filled_months, n);
    print("{0} - {1}\n", n.name, n.period);
  }

  for (int v = 0; v < this->num_vertices(); v++) {
    Node& n = (*this)[v];
    n.add_indicator("ind_" + n.name, "synthetic");
    n.mean = centers_dist(this->rand_num_generator);
    while (n.mean == 0) {
      n.mean = centers_dist(this->rand_num_generator);
    }
  }

  this->generate_latent_state_sequences(0);
  this->generate_observed_state_sequences();

  this->observed_state_sequence.clear();
  this->observation_timestep_gaps.clear();
  this->n_timesteps = num_obs;

  // Access (concept is a vertex in the CAG)
  // [ timestep ][ concept ][ indicator ][ observation ]
  this->observed_state_sequence = ObservedStateSequence(this->n_timesteps);
  this->observation_timestep_gaps = vector<double>(this->n_timesteps, 1);
  this->observation_timestep_gaps[0] = 0;

  int num_verts = this->num_vertices();
  // Fill in observed state sequence
  // NOTE: This code is very similar to the implementations in
  // set_observed_state_sequence_from_data and get_observed_state_from_data
  for (int ts = 0; ts < this->n_timesteps; ts++) {
    this->observed_state_sequence[ts] = vector<vector<vector<double>>>(num_verts);

    for (int v = 0; v < num_verts; v++) {
      Node& n = (*this)[v];
      this->observed_state_sequence[ts][v] = vector<vector<double>>(n.indicators.size());

      for (int i = 0; i < n.indicators.size(); i++) {
        this->observed_state_sequence[ts][v][i] = vector<double>();
        this->observed_state_sequence[ts][v][i].push_back(
            this->predicted_observed_state_sequences[0][ts + 1][v][i] +
            noise_variance * this->norm_dist(this->rand_num_generator));
      }
    }
  }
  RNG::release_instance();
}

void AnalysisGraph::interpolate_missing_months(vector<int> &filled_months, Node &n) {
  sort(filled_months.begin(), filled_months.end());

  // Interpolate values for the missing months
  if (filled_months.size() > 1) {
    for (int i = 0; i < filled_months.size(); i++) {
      int month_start = filled_months[i];
      int month_end = filled_months[(i + 1) % filled_months.size()];

      int num_missing_months = 0;
      if (month_end > month_start) {
        num_missing_months = month_end - month_start - 1;
      }
      else {
        num_missing_months = (11 - month_start) + month_end;
      }

      for (int month_missing = 1;
      month_missing <= num_missing_months;
      month_missing++) {
        n.generated_monthly_latent_centers_for_a_year
        [(month_start + month_missing) % 12] =
            ((num_missing_months - month_missing + 1) *
            n.generated_monthly_latent_centers_for_a_year
            [month_start] +
            (month_missing)*n
            .generated_monthly_latent_centers_for_a_year
            [month_end]) /
            (num_missing_months + 1);

        n.generated_monthly_latent_spreads_for_a_year
        [(month_start + month_missing) % 12] =
            ((num_missing_months - month_missing + 1) *
            n.generated_monthly_latent_spreads_for_a_year
            [month_start] +
            (month_missing)*n
            .generated_monthly_latent_spreads_for_a_year
            [month_end]) /
            (num_missing_months + 1);
      }
    }
  } else if (filled_months.size() == 1) {
    for (int month = 0; month < n.generated_monthly_latent_centers_for_a_year.size(); month++) {
      n.generated_monthly_latent_centers_for_a_year[month] =
          n.generated_monthly_latent_centers_for_a_year
          [filled_months[0]];
      n.generated_monthly_latent_spreads_for_a_year[month] =
          n.generated_monthly_latent_spreads_for_a_year
          [filled_months[0]];
    }
  }
}
