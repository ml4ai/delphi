#include "AnalysisGraph.hpp"
#include "spdlog/spdlog.h"
#include <tqdm.hpp>
#include "dbg.h"

using namespace std;
using tq::trange;
using spdlog::debug;

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

  this->find_all_paths();
  this->data_heuristic = use_heuristic;

  this->n_timesteps = this->calculate_num_timesteps(
      start_year, start_month, end_year, end_month);
  this->res = res;
  this->initialize_random_number_generator();
  this->init_betas_to(initial_beta);
  this->sample_initial_transition_matrix_from_prior();
  dbg("Parameterizing");
  this->parameterize(country, state, county, start_year, start_month, units);

  dbg("Setting training range");
  this->training_range = make_pair(make_pair(start_year,start_month), make_pair(end_year,end_month));

  dbg(synthetic_data_experiment);
  if (!synthetic_data_experiment) {
    dbg("Setting observed state sequence from data");
    this->set_observed_state_sequence_from_data(
        start_year, start_month, end_year, end_month, country, state, county);
  }

  dbg("Setting initial latent state from observed_state sequence");
  this->set_initial_latent_state_from_observed_state_sequence();

  dbg("Setting log_likelihood");
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
  dbg("Clearing transition_matrix_collection");
  this->transition_matrix_collection.clear();
  dbg("Initializing transition_matrix_collection");
  this->transition_matrix_collection = vector<Eigen::MatrixXd>(this->res);

  dbg("Burning samples");
  for (int i : trange(burn)) {
    this->sample_from_posterior();
  }

  for (int i : trange(this->res)) {
    this->sample_from_posterior();
    this->transition_matrix_collection[i] = this->A_original;
  }

  this->trained = true;
  return;
}
