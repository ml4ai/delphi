#include "AnalysisGraph.hpp"
#include <range/v3/all.hpp>

using namespace std;
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
                                                  bool use_continuous) {
  synthetic_data_experiment = true;
  this->n_timesteps = this->calculate_num_timesteps(
      start_year, start_month, end_year, end_month);
  this->initialize_parameters(res, initial_beta, false, use_continuous);

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
