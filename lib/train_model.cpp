#include "AnalysisGraph.hpp"
#include "spdlog/spdlog.h"
#include <tqdm.hpp>

using namespace std;
using spdlog::debug;
using tq::trange;

void AnalysisGraph::train_model(int start_year,
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
                                bool use_heuristic) {

  this->initialize_random_number_generator();
  this->construct_beta_pdfs(this->rand_num_generator);
  this->find_all_paths();
  this->data_heuristic = use_heuristic;

  this->n_timesteps = this->calculate_num_timesteps(
      start_year, start_month, end_year, end_month);
  this->res = res;
  this->init_betas_to(initial_beta);
  this->sample_initial_transition_matrix_from_prior();
  this->parameterize(country, state, county, start_year, start_month, units);

  this->training_range = make_pair(make_pair(start_year, start_month),
                                   make_pair(end_year, end_month));

  if (!synthetic_data_experiment) {
    this->set_observed_state_sequence_from_data(
        start_year, start_month, end_year, end_month, country, state, county);
  }

  this->set_initial_latent_state_from_observed_state_sequence();

  this->set_log_likelihood();

  // Accumulates the transition matrices for accepted samples
  // Access: [ sample number ]
  // training_sampled_transition_matrix_sequence.clear();
  // training_sampled_transition_matrix_sequence =
  //    vector<Eigen::MatrixXd>(this->res);
  //
  // generate_prediction()      uses
  // sample_from_likelihood. It uses
  // transition_matrix_collection
  // So to keep things simple for the moment
  // I had to fall back to
  // transition_matrix_collection
  // HOWEVER: The purpose of transition_matrix_collection
  // seem to be different in the prevous code than here.
  // In the earlier code, (in sample_from_prior()) this is
  // populated with DEFAULT_N_SAMPLES of initial transition matrices.
  // Here we populate it with res number of sampler emitted transition
  // matrices.
  //
  this->transition_matrix_collection.clear();
  this->transition_matrix_collection = vector<Eigen::MatrixXd>(this->res);

  for (int i : trange(burn)) {
    this->sample_from_posterior();
  }

  for (int i : trange(this->res)) {
    this->sample_from_posterior();
    this->transition_matrix_collection[i] = this->A_original;
  }

  this->trained = true;
  RNG::release_instance();
  return;
}
